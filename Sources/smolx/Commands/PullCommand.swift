import ArgumentParser
import Darwin
import Foundation
import HuggingFace

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
            filter: "mlx",
            limit: 20,
            full: true)
        let results = initialResponse.items

        let chosenRepoId: String
        switch Self.decide(input: repoId, results: results.map { $0.id.rawValue }) {
        case .empty:
            FileHandle.standardError.write(
                Data("No MLX models match \"\(repoId)\".\n".utf8))
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
                        "Multiple MLX models match \"\(repoId)\":\n".utf8))
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
        let tracker = ProgressTracker()
        let renderer = ProgressRenderer(
            tracker: tracker, repoId: resolvedRepoId, slots: parallel)
        renderer.start()

        let snapshotDir: URL
        do {
            snapshotDir = try await downloader.download(
                repoId: resolvedRepoId,
                parallelism: parallel,
                tracker: tracker)
            await renderer.stop(success: true)
        } catch {
            await renderer.stop(success: false)
            throw error
        }

        let descriptor = ModelDescriptor(
            name: alias,
            repoId: resolvedRepoId,
            localPath: snapshotDir.path,
            capability: Self.detectCapability(at: snapshotDir),
            diskSizeBytes: HubDownloader.Layout.diskSize(of: snapshotDir),
            addedAt: Date())
        try registry.upsert(descriptor)

        print(
            "Installed \(alias) (\(descriptor.capability.rawValue), \(Self.formatBytes(descriptor.diskSizeBytes)))"
        )
        print("Path: \(descriptor.localPath)")
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

    // MARK: - Naming + capability

    private static func defaultAlias(forRepo repoId: String) -> String {
        let base = repoId.split(separator: "/").last.map(String.init) ?? repoId
        return base.lowercased()
    }

    /// VLM detection: a `preprocessor_config.json` (image processor) alongside
    /// the weights is the standard marker. Text-only LLM repos don't ship one.
    private static func detectCapability(at directory: URL) -> ModelDescriptor.Capability {
        let preprocessor = directory.appendingPathComponent("preprocessor_config.json")
        return FileManager.default.fileExists(atPath: preprocessor.path) ? .vision : .text
    }

    // MARK: - Formatting

    static func formatBytes(_ bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var unit = 0
        while value >= 1024 && unit < units.count - 1 {
            value /= 1024
            unit += 1
        }
        return String(format: "%.2f %@", value, units[unit])
    }

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

    /// Picker-friendly byte size: `1.2 GB`, `412 MB`, `9 KB`. Single
    /// fractional digit only when the leading value is < 10; otherwise
    /// integer.
    static func formatBytesShort(_ bytes: Int64) -> String {
        let units: [(suffix: String, divisor: Double)] = [
            ("TB", 1024 * 1024 * 1024 * 1024),
            ("GB", 1024 * 1024 * 1024),
            ("MB", 1024 * 1024),
            ("KB", 1024),
        ]
        let v = Double(bytes)
        for (suffix, div) in units where v >= div {
            let scaled = v / div
            if scaled >= 10 {
                return "\(Int(scaled.rounded())) \(suffix)"
            }
            return String(format: "%.1f %@", scaled, suffix)
        }
        return "\(bytes) B"
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
            parts.append(formatBytesShort(bytes))
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
