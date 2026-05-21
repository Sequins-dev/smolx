import ArgumentParser
import Foundation

struct PullCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a HuggingFace MLX model into the local cache and register it."
    )

    @Argument(help: "HuggingFace repository id, e.g. 'mlx-community/Llama-3.2-3B-Instruct-4bit'.")
    var repoId: String

    @Option(name: .long, help: "Short alias to address this model in the API (defaults to the repo's basename, lowercased).")
    var alias: String?

    @Option(name: .long, help: "Number of files to download concurrently.")
    var parallel: Int = 4

    @Flag(name: .long, help: "Re-download even if the model is already registered.")
    var force: Bool = false

    func run() async throws {
        let registry = ModelRegistry()
        let alias = self.alias ?? Self.defaultAlias(forRepo: repoId)

        if !force, let existing = try registry.find(alias) {
            print("Already installed: \(existing.name) -> \(existing.localPath)")
            print("Use --force to re-download.")
            return
        }

        let downloader = HubDownloader()
        let tracker = ProgressTracker()
        let renderer = ProgressRenderer(
            tracker: tracker, repoId: repoId, slots: parallel)
        renderer.start()

        let snapshotDir: URL
        do {
            snapshotDir = try await downloader.download(
                repoId: repoId,
                parallelism: parallel,
                tracker: tracker)
            await renderer.stop(success: true)
        } catch {
            await renderer.stop(success: false)
            throw error
        }

        let descriptor = ModelDescriptor(
            name: alias,
            repoId: repoId,
            localPath: snapshotDir.path,
            capability: Self.detectCapability(at: snapshotDir),
            diskSizeBytes: HubDownloader.Layout.diskSize(of: snapshotDir),
            addedAt: Date())
        try registry.upsert(descriptor)

        print("Installed \(alias) (\(descriptor.capability.rawValue), \(Self.formatBytes(descriptor.diskSizeBytes)))")
        print("Path: \(descriptor.localPath)")
    }

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
}
