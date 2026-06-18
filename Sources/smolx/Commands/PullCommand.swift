import ArgumentParser
import Darwin
import Foundation
import HuggingFace

extension ModelDescriptor.WeightFormat: ExpressibleByArgument {
    init?(argument: String) {
        self.init(rawValue: argument.lowercased())
    }
}

struct PullCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a HuggingFace MLX model into the local cache and register it."
    )

    @Argument(
        help:
            "HuggingFace repo id (e.g. 'mlx-community/Qwen3.6-27B-4bit') or a substring to search for. Partial inputs open an interactive picker."
    )
    var repoId: String

    @Option(
        name: .long,
        help:
            "Short alias to address this model in the API (defaults to the repo's basename, lowercased).")
    var alias: String?

    @Option(name: .long, help: "Number of files to download concurrently.")
    var parallel: Int = 4

    @Option(name: .long, help: "Model weight format to install: mlx or gguf. Defaults to auto-detect.")
    var format: ModelDescriptor.WeightFormat?

    @Option(
        name: .long,
        help:
            "GGUF filename or substring to select when --format gguf is used, e.g. Q4_K_M.")
    var ggufFile: String?

    @Flag(name: .long, help: "Re-download even if the model is already registered.")
    var force: Bool = false

    @Flag(name: .long, help: "Show downloads + size + last-modified under each name in the picker.")
    var expand: Bool = false

    /// Outcome of matching the user's input against the HF search results.
    /// Kept as a pure value type so the branching logic is unit-testable
    /// without involving HF or termios.
    enum Decision: Equatable {
        /// One result's repo id is an exact (case-insensitive) match — pull it.
        case exact(repoId: String)
        /// Multiple matches; the user picks one interactively.
        case choose([String])
        /// HF returned nothing for the search.
        case empty
    }

    enum GGUFFileDecision: Equatable {
        case selected(String)
        case ambiguous([String])
        case empty
    }

    func run() async throws {
        let client = HubClient.default
        // Use `full=true` instead of `expand=...`. Two reasons:
        //   1) swift-huggingface serialises multi-value `expand` as the
        //      comma form `expand=A,B,C`, which HF's `/api/models`
        //      endpoint rejects (it only accepts repeated query params
        //      `expand=A&expand=B`). Single-field `expand` works, but
        //      we need three fields. `full=true` is the safe alternative.
        //   2) The list endpoint never populates per-sibling `size`
        //      anyway, so sizes need a per-model `getModel(id)` call —
        //      see Pager.fetchSizesIfNeeded which fires those in
        //      parallel only when the user passed `--expand`.
        let initialResponse = try await client.listModels(
            search: repoId,
            filter: format?.rawValue,
            limit: 20,
            full: true)
        let results = initialResponse.items

        let chosenRepoId: String
        switch Self.decide(input: repoId, results: results.map { $0.id.rawValue }) {
        case .empty:
            FileHandle.standardError.write(
                Data("No models match \"\(repoId)\".\n".utf8))
            throw ExitCode.failure

        case .exact(let id):
            chosenRepoId = id

        case .choose(let ids):
            // For the non-TTY fallback we want full metadata, so look
            // up each id in the results we already have.
            let pairs = ids.compactMap { id in
                results.first { $0.id.rawValue == id }.map { (id: id, model: $0) }
            }

            if isatty(STDIN_FILENO) == 0 {
                // Non-TTY: print the table and bail. We deliberately exit
                // non-zero so scripts that piped to `pull` don't silently
                // pick the wrong repo. Size is omitted here — fetching
                // per-row sizes for a fallback the user will copy/paste
                // from isn't worth 20 round-trips.
                FileHandle.standardError.write(
                    Data(
                        "Multiple models match \"\(repoId)\":\n"
                            .utf8))
                let now = Date()
                for p in pairs {
                    let meta = Self.formatRow(for: p.model, sizeBytes: nil, now: now)
                    FileHandle.standardError.write(Data("  \(p.id)  \(meta)\n".utf8))
                }
                FileHandle.standardError.write(
                    Data(
                        "error: rerun with an exact repo id.\n".utf8))
                throw ExitCode.failure
            }

            // Pager owns the page cursor + the id→Model lookup +
            // (optionally) the id→size map. Putting all three behind an
            // actor lets the `fetchMore` closure capture a single
            // Sendable handle.
            let pager = Pager(
                initialPage: initialResponse,
                client: client,
                fetchSizes: expand)
            // In expand mode this awaits 20 parallel `getModel(id)`
            // calls (~500ms total); in compact mode it just builds the
            // rows synchronously from the data we already have.
            let initialRows = await pager.primeAndBuildInitialRows()

            guard
                let selection = try await InteractivePicker.pick(
                    title: "Pick a model to pull (↑/↓, Enter, q to abort):",
                    initialRows: initialRows,
                    style: expand ? .expanded : .compact,
                    fetchMore: { try await pager.fetchNext() })
            else {
                // User aborted — exit silently with a non-zero code that
                // shells conventionally use for SIGINT-style cancellation.
                throw ExitCode(2)
            }
            chosenRepoId = selection.row.primary
        }

        try await pullResolved(repoId: chosenRepoId)
    }

    /// The download + registry path that runs once we've resolved the
    /// input down to a concrete repo id. Extracted so the search/picker
    /// branching above doesn't have to duplicate it.
    private func pullResolved(repoId resolvedRepoId: String) async throws {
        let registry = ModelRegistry()
        let alias = self.alias ?? Self.defaultAlias(forRepo: resolvedRepoId)

        if !force, let existing = try registry.find(alias) {
            print("Already installed: \(existing.name) -> \(existing.localPath)")
            print("Use --force to re-download.")
            return
        }

        let downloader = HubDownloader()
        let progress = DownloadProgress(repoId: resolvedRepoId, slots: parallel)
        let repoFiles = try await listRepoFiles(repoId: resolvedRepoId)
        let effectiveFormat = Self.decideWeightFormat(files: repoFiles, requested: format)
        let selectedWeightFile =
            effectiveFormat == .gguf
            ? try resolveGGUFFile(repoId: resolvedRepoId, files: repoFiles)
            : nil

        let result: DownloadResult
        do {
            result = try await downloader.download(
                repoId: resolvedRepoId,
                format: effectiveFormat,
                weightFile: selectedWeightFile,
                parallelism: parallel,
                progress: progress)
            await progress.stop(success: true)
        } catch {
            await progress.stop(success: false)
            throw error
        }

        var installedBytes = result.totalBytes
        if effectiveFormat == .gguf {
            installedBytes += try await hydrateGGUFRuntimeFiles(
                downloader: downloader,
                repoId: resolvedRepoId,
                snapshotURL: result.snapshotURL,
                weightFile: selectedWeightFile)
        }

        let descriptor = ModelDescriptor(
            name: alias,
            repoId: resolvedRepoId,
            localPath: result.snapshotURL.path,
            capability: ModelSnapshotInspector.capability(at: result.snapshotURL),
            diskSizeBytes: installedBytes,
            addedAt: Date(),
            weightFormat: effectiveFormat,
            weightFile: selectedWeightFile)
        try registry.upsert(descriptor)

        print(
            "Installed \(alias) (\(descriptor.capability.rawValue), \(descriptor.weightFormat.rawValue), \(Bytes.format(descriptor.diskSizeBytes)))"
        )
        print("Path: \(descriptor.localPath)")
    }

    private func hydrateGGUFRuntimeFiles(
        downloader: HubDownloader,
        repoId: String,
        snapshotURL: URL,
        weightFile: String?
    ) async throws -> Int64 {
        let repo = try HubDownloader.parseRepoId(repoId)
        let model = try await HubClient.default.getModel(repo, full: true, cardData: true)

        var sidecarBytes: Int64 = 0
        if let baseRepoId = GGUFRuntimeFiles.baseModelRepoId(from: model) {
            print("Hydrating GGUF tokenizer files from \(baseRepoId)")
            sidecarBytes = try await downloader.downloadSidecars(
                repoId: baseRepoId,
                paths: GGUFRuntimeFiles.tokenizerSidecarFilenames,
                into: snapshotURL)
        }

        let configURL = snapshotURL.appendingPathComponent("config.json")
        let configSizeBefore =
            ((try? FileManager.default.attributesOfItem(atPath: configURL.path)[.size]) as? Int64) ?? 0
        try GGUFRuntimeFiles.prepareSnapshot(at: snapshotURL, weightFile: weightFile)
        let configSizeAfter =
            ((try? FileManager.default.attributesOfItem(atPath: configURL.path)[.size]) as? Int64) ?? 0
        return sidecarBytes + max(0, configSizeAfter - configSizeBefore)
    }

    private func listRepoFiles(repoId: String) async throws -> [String] {
        let repo = try HubDownloader.parseRepoId(repoId)
        let entries = try await HubClient.default.listFiles(
            in: repo, kind: .model, revision: "main", recursive: true)
        return entries
            .filter { $0.type == .file }
            .map(\.path)
    }

    private func resolveGGUFFile(repoId: String, files: [String]) throws -> String {
        let isTTY = isatty(STDIN_FILENO) != 0

        switch Self.decideGGUFFile(files: files, pattern: ggufFile, isTTY: isTTY) {
        case .selected(let file):
            return file
        case .ambiguous(let files):
            FileHandle.standardError.write(
                Data("Multiple GGUF files match. Rerun with a more specific --gguf-file:\n".utf8))
            for file in files {
                FileHandle.standardError.write(Data("  \(file)\n".utf8))
            }
            throw ExitCode.failure
        case .empty:
            FileHandle.standardError.write(
                Data("No GGUF files found in \(repoId).\n".utf8))
            throw ExitCode.failure
        }
    }

    // MARK: - Decision

    /// Pure branching helper for the search-result handling. `input` is
    /// the raw argument the user typed; `results` is the ordered list of
    /// repo ids from HF (the search ranking is preserved). Case-folded
    /// match so a user typing `mlx-community/qwen3.6-27b-4bit` against
    /// `mlx-community/Qwen3.6-27B-4bit` still auto-pulls.
    static func decide(input: String, results: [String]) -> Decision {
        if results.isEmpty { return .empty }
        let folded = input.lowercased()
        if let exact = results.first(where: { $0.lowercased() == folded }) {
            return .exact(repoId: exact)
        }
        return .choose(results)
    }

    static func decideWeightFormat(
        files: [String],
        requested: ModelDescriptor.WeightFormat?
    ) -> ModelDescriptor.WeightFormat {
        if let requested { return requested }
        let lowercased = files.map { $0.lowercased() }
        if lowercased.contains(where: { $0.hasSuffix(".safetensors") }) {
            return .mlx
        }
        if lowercased.contains(where: { $0.hasSuffix(".gguf") }) {
            return .gguf
        }
        return .mlx
    }

    static func decideGGUFFile(
        files: [String],
        pattern: String?,
        isTTY: Bool
    ) -> GGUFFileDecision {
        let ggufs = files
            .filter { $0.lowercased().hasSuffix(".gguf") }
            .sorted()
        guard !ggufs.isEmpty else { return .empty }

        guard let pattern, !pattern.isEmpty else {
            return .selected(preferredGGUFFile(in: ggufs))
        }

        let matches: [String]
        let folded = pattern.lowercased()
        matches = ggufs.filter {
            let candidate = $0.lowercased()
            return candidate == folded || candidate.contains(folded)
        }

        guard !matches.isEmpty else { return .empty }
        if matches.count == 1 {
            return .selected(matches[0])
        }
        return .ambiguous(matches)
    }

    private static func preferredGGUFFile(in files: [String]) -> String {
        let quantPreference = [
            "q4_k_m",
            "q4_k_s",
            "q4_k",
            "q4_0",
            "q5_k_m",
            "q5_k_s",
            "q5_k",
            "q6_k",
            "q8_0",
            "q3_k_m",
            "q3_k_s",
            "q3_k",
            "q2_k",
        ]
        for quant in quantPreference {
            if let match = files.first(where: { $0.lowercased().contains(quant) }) {
                return match
            }
        }
        return files[0]
    }

    // MARK: - Naming

    private static func defaultAlias(forRepo repoId: String) -> String {
        let base = repoId.split(separator: "/").last.map(String.init) ?? repoId
        return base.lowercased()
    }

    // MARK: - Formatting

    /// Compact download count for the picker (`1.2M`, `12k`, `123`).
    static func formatDownloads(_ n: Int) -> String {
        if n >= 1_000_000 {
            return String(format: "%.1fM", Double(n) / 1_000_000)
        }
        if n >= 1_000 {
            // Round to int kilos for tighter display once we cross 10k.
            if n >= 10_000 {
                return "\(n / 1000)k"
            }
            return String(format: "%.1fk", Double(n) / 1000)
        }
        return "\(n)"
    }

    /// Human relative-time string: `"3 days ago"`, `"yesterday"`,
    /// `"2 weeks ago"`. `now:` is exposed so tests can pin the reference
    /// point and produce stable assertions.
    static func formatRelative(_ d: Date, now: Date = Date()) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        f.dateTimeStyle = .named
        f.locale = Locale(identifier: "en_US_POSIX")
        return f.localizedString(for: d, relativeTo: now)
    }

    /// Build the metadata line for an expand-mode picker row (and the
    /// non-TTY fallback). Size leads because disk cost is what the
    /// user is most likely to filter on when picking a quant; the
    /// downloads count is decoration. Size slot is included only when
    /// we actually have a value — compact mode passes nil so the
    /// column collapses. Format: `15.4 GB · 60k downloads · 3 days ago`.
    static func formatRow(
        for model: Model, sizeBytes: Int64?, now: Date
    ) -> String {
        var parts: [String] = []
        if let bytes = sizeBytes {
            parts.append(Bytes.formatShort(bytes))
        }
        if let dl = model.downloads {
            parts.append("\(formatDownloads(dl)) downloads")
        }
        if let modified = model.lastModified {
            parts.append(formatRelative(modified, now: now))
        }
        return parts.joined(separator: " · ")
    }
}

/// Wraps the pagination cursor + id→Model + id→size lookups for the
/// PullCommand picker. The `fetchMore` closure handed to
/// `InteractivePicker.pick` captures this single Sendable actor;
/// without it we'd be trying to capture mutable vars (`var page`,
/// `var byId`) from inside a `@Sendable` closure, which Swift 6
/// rejects.
private actor Pager {
    private var page: PaginatedResponse<Model>
    private let client: HubClient
    private(set) var byId: [String: Model]
    /// Per-model `usedStorage` cache. Only populated when the picker
    /// runs in expand mode (sizes aren't shown in compact, so paying
    /// for a per-row `getModel(id)` round-trip would be waste).
    private var sizes: [String: Int64] = [:]
    private let fetchSizes: Bool

    init(initialPage: PaginatedResponse<Model>, client: HubClient, fetchSizes: Bool) {
        self.page = initialPage
        self.client = client
        self.fetchSizes = fetchSizes
        self.byId = Dictionary(
            uniqueKeysWithValues:
                initialPage.items.map { ($0.id.rawValue, $0) })
    }

    /// Pre-fetch sizes for the initial page (no-op in compact mode) and
    /// return picker rows. Called by PullCommand before opening the
    /// picker so the first frame already has full metadata.
    func primeAndBuildInitialRows() async -> [InteractivePicker.Row] {
        await fetchSizesIfNeeded(for: page.items)
        return buildRows(page.items)
    }

    /// Fetch the next page (and its sizes, if applicable) and convert
    /// it into picker rows. Returns `nil` when HF has no more pages.
    func fetchNext() async throws -> [InteractivePicker.Row]? {
        guard let next = try await client.nextPage(after: page) else {
            return nil
        }
        page = next
        for m in next.items { byId[m.id.rawValue] = m }
        await fetchSizesIfNeeded(for: next.items)
        return buildRows(next.items)
    }

    /// Parallel `getModel(id)` lookups for the per-model `usedStorage`
    /// field. The list endpoint doesn't populate sibling sizes (verified
    /// against HF in 2026-05); the per-model endpoint does. We fire all
    /// requests at once — 20 parallel GETs against a CDN-backed API
    /// finish well under a second. Individual failures just drop the
    /// size for that row (formatRow handles nil).
    private func fetchSizesIfNeeded(for models: [Model]) async {
        guard fetchSizes else { return }
        let needed = models.filter { sizes[$0.id.rawValue] == nil }
        if needed.isEmpty { return }
        await withTaskGroup(of: (String, Int64?).self) { [client] group in
            for m in needed {
                group.addTask {
                    let detail = try? await client.getModel(m.id)
                    let bytes = detail?.usedStorage.map { Int64($0) }
                    return (m.id.rawValue, bytes)
                }
            }
            for await (id, bytes) in group {
                if let bytes { sizes[id] = bytes }
            }
        }
    }

    /// Project models into picker rows using the current state of the
    /// `sizes` cache. Called once per fetch with that fetch's models —
    /// the picker takes care of appending to its accumulated row list.
    private func buildRows(_ models: [Model]) -> [InteractivePicker.Row] {
        let now = Date()
        return models.map { m in
            InteractivePicker.Row(
                primary: m.id.rawValue,
                secondary: PullCommand.formatRow(
                    for: m,
                    sizeBytes: sizes[m.id.rawValue],
                    now: now))
        }
    }
}
