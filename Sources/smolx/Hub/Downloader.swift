import Foundation
import HuggingFace
import Logging

/// Result returned by `HubDownloader.download` so callers can read the total
/// byte count without a second filesystem walk.
struct DownloadResult: Sendable {
    var snapshotURL: URL
    var totalBytes: Int64
}

/// Downloads model snapshots from the HuggingFace Hub into our own flat cache
/// directory at `~/.smolx/models/<namespace>--<name>/<filename>`.
///
/// We use swift-huggingface only for listing the repo's files via the API.
/// The actual transport is `StreamingDownloader` — a thin URLSession wrapper
/// that streams via a session-level delegate so progress updates work, and
/// keeps `<file>.partial` on disk for byte-level resume across crashes.
struct HubDownloader: Sendable {

    enum DownloadError: Error, CustomStringConvertible {
        case invalidRepoId(String)
        case noWeights(repo: String, format: ModelDescriptor.WeightFormat)
        case underlying(Error)

        var description: String {
            switch self {
            case .invalidRepoId(let r):
                return "Invalid repo id '\(r)'. Expected '<namespace>/<name>'."
            case .noWeights(let r, let format):
                return "No \(format.rawValue.uppercased()) model weights found in '\(r)'."
            case .underlying(let e):
                return "Download failed: \(e)"
            }
        }
    }

    private let client: HubClient
    private let streamer: StreamingDownloader
    private let logger: Logger
    private let huggingFaceBase: URL

    init(
        client: HubClient = .default,
        streamer: StreamingDownloader = .shared,
        logger: Logger = Logger(label: "smolx.hub"),
        huggingFaceBase: URL = URL(string: "https://huggingface.co")!
    ) {
        self.client = client
        self.streamer = streamer
        self.logger = logger
        self.huggingFaceBase = huggingFaceBase
    }

    /// Downloads the model snapshot in parallel and returns the local snapshot
    /// directory plus the total byte count from the manifest. Per-file progress
    /// is pushed into `progress`; the caller wires the renderer separately.
    func download(
        repoId: String,
        revision: String = "main",
        format: ModelDescriptor.WeightFormat = .mlx,
        weightFile: String? = nil,
        parallelism: Int = 4,
        retry: RetryPolicy = .default,
        progress: any DownloadProgressSink
    ) async throws -> DownloadResult {
        let repo = try Self.parseRepoId(repoId)
        logger.debug("Listing files in \(repoId)")

        let entries: [Git.TreeEntry]
        do {
            entries = try await client.listFiles(
                in: repo, kind: .model, revision: revision, recursive: true)
        } catch {
            throw DownloadError.underlying(error)
        }

        let wantedFiles = filterWantedFiles(entries, format: format, weightFile: weightFile)
        let hasExpectedWeights = switch format {
        case .mlx:
            wantedFiles.contains { $0.path.hasSuffix(".safetensors") }
        case .gguf:
            wantedFiles.contains { $0.path.lowercased().hasSuffix(".gguf") }
        }
        guard hasExpectedWeights else {
            throw DownloadError.noWeights(repo: repoId, format: format)
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
            let resumedFrom =
                (try? FileManager.default
                    .attributesOfItem(atPath: partial.path)[.size] as? Int64) ?? 0
            await progress.register(
                index: index, name: entry.path,
                totalBytes: total, resumedFrom: min(resumedFrom, total))
        }
        await progress.start()

        logger.debug("Downloading \(wantedFiles.count) files from \(repoId) (parallel=\(parallelism))")

        try await runPool(
            wantedFiles: wantedFiles, repo: repo, revision: revision,
            modelDir: modelDir, parallelism: parallelism, retry: retry,
            streamer: streamer, base: huggingFaceBase, progress: progress)

        logger.debug("Snapshot ready at \(modelDir.path)")
        let totalBytes = wantedFiles.reduce(into: Int64(0)) { $0 += Int64($1.size ?? 0) }
        return DownloadResult(snapshotURL: modelDir, totalBytes: totalBytes)
    }

    func downloadSidecars(
        repoId: String,
        revision: String = "main",
        paths: [String],
        into modelDir: URL,
        retry: RetryPolicy = .default
    ) async throws -> Int64 {
        guard !paths.isEmpty else { return 0 }
        let repo = try Self.parseRepoId(repoId)

        let entries: [Git.TreeEntry]
        do {
            entries = try await client.listFiles(
                in: repo, kind: .model, revision: revision, recursive: true)
        } catch {
            throw DownloadError.underlying(error)
        }

        let wanted = Set(paths)
        let sidecars = entries
            .filter { $0.type == .file && wanted.contains($0.path) }
            .sorted { $0.path < $1.path }

        var totalBytes: Int64 = 0
        for entry in sidecars {
            let destination = modelDir.appendingPathComponent(entry.path)
            if FileManager.default.fileExists(atPath: destination.path) {
                let size =
                    ((try? FileManager.default
                        .attributesOfItem(atPath: destination.path)[.size]) as? Int64)
                    ?? Int64(entry.size ?? 0)
                totalBytes += size
                continue
            }

            let url =
                huggingFaceBase
                .appendingPathComponent(repo.namespace)
                .appendingPathComponent(repo.name)
                .appendingPathComponent("resolve")
                .appendingPathComponent(revision)
                .appendingPathComponent(entry.path)

            do {
                try await retry.run {
                    try await streamer.download(
                        url: url,
                        destination: destination,
                        onBytes: { _ in })
                }
                totalBytes += Int64(entry.size ?? 0)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw DownloadError.underlying(error)
            }
        }
        return totalBytes
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
        progress: any DownloadProgressSink
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
                        streamer: streamer, base: base, progress: progress)
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
                            streamer: streamer, base: base, progress: progress)
                    }
                    nextIndex += 1
                }
            }
        }
    }

    private static func downloadOne(
        entry: Git.TreeEntry,
        index: Int,
        repo: Repo.ID,
        revision: String,
        modelDir: URL,
        retry: RetryPolicy,
        streamer: StreamingDownloader,
        base: URL,
        progress: any DownloadProgressSink
    ) async throws {
        let destination = modelDir.appendingPathComponent(entry.path)

        // Fast path: if the final file already exists (from a previous run
        // that completed), skip the network entirely.
        if FileManager.default.fileExists(atPath: destination.path) {
            let size =
                ((try? FileManager.default
                    .attributesOfItem(atPath: destination.path)[.size]) as? Int64)
                ?? Int64(entry.size ?? 0)
            await progress.cached(index: index, bytes: size)
            return
        }

        let url =
            base
            .appendingPathComponent(repo.namespace)
            .appendingPathComponent(repo.name)
            .appendingPathComponent("resolve")
            .appendingPathComponent(revision)
            .appendingPathComponent(entry.path)

        do {
            try await retry.run(
                onAttempt: { attempt, _ in
                    if attempt > 1 {
                        await progress.retrying(index: index, attempt: attempt, delay: 0)
                    }
                }
            ) {
                await progress.preparing(index: index)
                try await streamer.download(
                    url: url,
                    destination: destination,
                    onBytes: { absoluteBytes in
                        Task { await progress.update(index: index, bytes: absoluteBytes) }
                    })
            }
            await progress.completed(index: index)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            await progress.failed(index: index, reason: String(describing: error))
            throw HubDownloader.DownloadError.underlying(error)
        }
    }

    // MARK: - Helpers

    static func parseRepoId(_ s: String) throws -> Repo.ID {
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
    static func filterWantedPaths(
        _ paths: [String],
        format: ModelDescriptor.WeightFormat,
        weightFile: String?
    ) -> [String] {
        let hasSafetensors = paths.contains { $0.lowercased().hasSuffix(".safetensors") }
        let selectedGGUF = weightFile?.lowercased()

        return paths.filter { original in
            let path = original.lowercased()
            if path.hasSuffix(".safetensors") {
                return format == .mlx
            }
            if path.hasSuffix(".gguf") {
                guard format == .gguf else { return false }
                if let selectedGGUF {
                    return path == selectedGGUF
                }
                return true
            }
            if path.hasSuffix(".json") { return true }
            if path.hasSuffix(".txt") || path.hasSuffix(".jinja") { return true }
            let base = (path as NSString).lastPathComponent
            if base.hasPrefix("tokenizer") || base.hasPrefix("special_tokens") { return true }
            if base.hasPrefix("vocab") || base.hasPrefix("merges") { return true }
            if path.hasSuffix(".model") { return true }
            if format == .mlx, hasSafetensors,
                path.hasSuffix(".bin") || path.hasSuffix(".gguf") || path.hasSuffix(".pt")
            {
                return false
            }
            return false
        }
    }

    private func filterWantedFiles(
        _ entries: [Git.TreeEntry],
        format: ModelDescriptor.WeightFormat,
        weightFile: String?
    ) -> [Git.TreeEntry] {
        let files = entries.filter { $0.type == .file }
        let wanted = Set(Self.filterWantedPaths(
            files.map(\.path), format: format, weightFile: weightFile))
        return files.filter { wanted.contains($0.path) }
    }
}
