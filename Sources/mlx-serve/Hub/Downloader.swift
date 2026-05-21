import Foundation
import HuggingFace
import Logging

/// Downloads model snapshots from the HuggingFace Hub into our own flat cache
/// directory at `~/.mlx-serve/models/<namespace>--<name>/<filename>`.
///
/// We use swift-huggingface only for listing the repo's files via the API.
/// The actual transport is `StreamingDownloader` — a thin URLSession wrapper
/// that streams via a session-level delegate so progress updates work, and
/// keeps `<file>.partial` on disk for byte-level resume across crashes.
struct HubDownloader: Sendable {

    enum DownloadError: Error, CustomStringConvertible {
        case invalidRepoId(String)
        case noWeights(repo: String)
        case underlying(Error)

        var description: String {
            switch self {
            case .invalidRepoId(let r):
                return "Invalid repo id '\(r)'. Expected '<namespace>/<name>'."
            case .noWeights(let r):
                return "No model weights found in '\(r)' (no .safetensors files)."
            case .underlying(let e):
                return "Download failed: \(e)"
            }
        }
    }

    enum Layout {
        /// Total size in bytes across all wanted files in a snapshot directory.
        static func diskSize(of directory: URL) -> Int64 {
            let fm = FileManager.default
            guard let enumerator = fm.enumerator(atPath: directory.path)
            else { return 0 }
            var total: Int64 = 0
            for case let relPath as String in enumerator {
                let fullPath = directory.appendingPathComponent(relPath).path
                guard let attrs = try? fm.attributesOfItem(atPath: fullPath) else { continue }
                guard (attrs[.type] as? FileAttributeType) == .typeRegular else { continue }
                if let size = attrs[.size] as? Int64 { total += size }
                else if let size = attrs[.size] as? Int { total += Int64(size) }
            }
            return total
        }
    }

    private let client: HubClient
    private let streamer: StreamingDownloader
    private let logger: Logger
    private let huggingFaceBase: URL

    init(
        client: HubClient = .default,
        streamer: StreamingDownloader = .shared,
        logger: Logger = Logger(label: "mlx-serve.hub"),
        huggingFaceBase: URL = URL(string: "https://huggingface.co")!
    ) {
        self.client = client
        self.streamer = streamer
        self.logger = logger
        self.huggingFaceBase = huggingFaceBase
    }

    /// Downloads the model snapshot in parallel and returns the local snapshot
    /// directory. Per-file progress is pushed into `tracker`; the caller is
    /// expected to wire a `ProgressRenderer` to read from the same tracker for
    /// display.
    func download(
        repoId: String,
        revision: String = "main",
        parallelism: Int = 4,
        retry: RetryPolicy = .default,
        tracker: ProgressTracker
    ) async throws -> URL {
        let repo = try Self.parseRepoId(repoId)
        logger.debug("Listing files in \(repoId)")

        let entries: [Git.TreeEntry]
        do {
            entries = try await client.listFiles(
                in: repo, kind: .model, revision: revision, recursive: true)
        } catch {
            throw DownloadError.underlying(error)
        }

        let wantedFiles = filterWantedFiles(entries)
        guard wantedFiles.contains(where: { $0.path.hasSuffix(".safetensors") }) else {
            throw DownloadError.noWeights(repo: repoId)
        }

        let modelDir = Paths.modelDirectory(namespace: repo.namespace, name: repo.name)
        try FileManager.default.createDirectory(
            at: modelDir, withIntermediateDirectories: true)

        // Pre-scan: for each wanted file, if `<filename>.partial` already
        // exists from a previous interrupted run, seed the tracker with the
        // resumed offset so the bar opens at the right place.
        for (index, entry) in wantedFiles.enumerated() {
            let total = Int64(entry.size ?? 0)
            let dest = modelDir.appendingPathComponent(entry.path)
            let partial = dest.appendingPathExtension("partial")
            let resumedFrom = (try? FileManager.default
                .attributesOfItem(atPath: partial.path)[.size] as? Int64) ?? 0
            await tracker.register(
                index: index, path: entry.path,
                total: total, resumedFrom: min(resumedFrom, total))
        }
        await tracker.start()

        logger.debug("Downloading \(wantedFiles.count) files from \(repoId) (parallel=\(parallelism))")

        try await runPool(
            wantedFiles: wantedFiles, repo: repo, revision: revision,
            modelDir: modelDir, parallelism: parallelism, retry: retry,
            streamer: streamer, base: huggingFaceBase, tracker: tracker)

        logger.debug("Snapshot ready at \(modelDir.path)")
        return modelDir
    }

    // MARK: - Pool

    private func runPool(
        wantedFiles: [Git.TreeEntry],
        repo: Repo.ID,
        revision: String,
        modelDir: URL,
        parallelism: Int,
        retry: RetryPolicy,
        streamer: StreamingDownloader,
        base: URL,
        tracker: ProgressTracker
    ) async throws {
        let pool = max(1, min(parallelism, wantedFiles.count))

        try await withThrowingTaskGroup(of: Void.self) { group in
            var nextIndex = 0
            while nextIndex < pool {
                let index = nextIndex
                let entry = wantedFiles[index]
                group.addTask {
                    try await Self.downloadOne(
                        entry: entry, index: index,
                        repo: repo, revision: revision,
                        modelDir: modelDir, retry: retry,
                        streamer: streamer, base: base, tracker: tracker)
                }
                nextIndex += 1
            }
            while try await group.next() != nil {
                if nextIndex < wantedFiles.count {
                    let index = nextIndex
                    let entry = wantedFiles[index]
                    group.addTask {
                        try await Self.downloadOne(
                            entry: entry, index: index,
                            repo: repo, revision: revision,
                            modelDir: modelDir, retry: retry,
                            streamer: streamer, base: base, tracker: tracker)
                    }
                    nextIndex += 1
                }
            }
        }
    }

    /// Downloads a single file with retry. The progress observer reads
    /// `progress.completedUnitCount` (which the StreamingDownloader's session
    /// delegate updates on every chunk) at 250 ms cadence.
    private static func downloadOne(
        entry: Git.TreeEntry,
        index: Int,
        repo: Repo.ID,
        revision: String,
        modelDir: URL,
        retry: RetryPolicy,
        streamer: StreamingDownloader,
        base: URL,
        tracker: ProgressTracker
    ) async throws {
        let destination = modelDir.appendingPathComponent(entry.path)

        // Fast path: if the final file already exists (from a previous run
        // that completed), skip the network entirely.
        if FileManager.default.fileExists(atPath: destination.path) {
            let size = ((try? FileManager.default
                .attributesOfItem(atPath: destination.path)[.size]) as? Int64)
                ?? Int64(entry.size ?? 0)
            await tracker.cached(index: index, bytes: size)
            return
        }

        let url = base
            .appendingPathComponent(repo.namespace)
            .appendingPathComponent(repo.name)
            .appendingPathComponent("resolve")
            .appendingPathComponent(revision)
            .appendingPathComponent(entry.path)

        do {
            try await retry.run(
                onAttempt: { attempt, _ in
                    if attempt > 1 {
                        Task {
                            await tracker.retrying(
                                index: index, attempt: attempt, delay: 0)
                        }
                    }
                }
            ) {
                await tracker.preparing(index: index)
                let totalBytes = Int64(entry.size ?? 0)
                let progress = Foundation.Progress(totalUnitCount: totalBytes)
                let pollerTask = Task {
                    var lastBytes: Int64 = -1
                    while !Task.isCancelled {
                        let current = progress.completedUnitCount
                        if current != lastBytes {
                            lastBytes = current
                            await tracker.update(index: index, bytes: current)
                        }
                        try? await Task.sleep(nanoseconds: 250_000_000)
                    }
                }
                defer { pollerTask.cancel() }

                try await streamer.download(
                    url: url, destination: destination, progress: progress)
            }
            let finalBytes = ((try? FileManager.default
                .attributesOfItem(atPath: destination.path)[.size]) as? Int64)
                ?? Int64(entry.size ?? 0)
            await tracker.completed(index: index, finalBytes: finalBytes)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            await tracker.failed(index: index, reason: String(describing: error))
            throw HubDownloader.DownloadError.underlying(error)
        }
    }

    // MARK: - Helpers

    private static func parseRepoId(_ s: String) throws -> Repo.ID {
        let parts = s.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            throw DownloadError.invalidRepoId(s)
        }
        return Repo.ID(namespace: parts[0], name: parts[1])
    }

    /// Keep only the files MLX models need. Excludes:
    ///   - non-file entries (directories)
    ///   - non-safetensors weight formats when safetensors are present
    ///   - example images, demos, docs we don't need at runtime
    private func filterWantedFiles(_ entries: [Git.TreeEntry]) -> [Git.TreeEntry] {
        let files = entries.filter { $0.type == .file }
        let hasSafetensors = files.contains { $0.path.hasSuffix(".safetensors") }

        return files.filter { entry in
            let path = entry.path.lowercased()
            if path.hasSuffix(".safetensors") { return true }
            if path.hasSuffix(".json") { return true }
            if path.hasSuffix(".txt") || path.hasSuffix(".jinja") { return true }
            let base = (path as NSString).lastPathComponent
            if base.hasPrefix("tokenizer") || base.hasPrefix("special_tokens") { return true }
            if base.hasPrefix("vocab") || base.hasPrefix("merges") { return true }
            if path.hasSuffix(".model") { return true }
            if hasSafetensors,
               path.hasSuffix(".bin") || path.hasSuffix(".gguf") || path.hasSuffix(".pt") {
                return false
            }
            return false
        }
    }
}
