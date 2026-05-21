import ArgumentParser
import Foundation

struct RemoveCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove an installed model (alias or repo id) and delete its cached files."
    )

    @Argument(help: "Alias or repo id of the model to remove.")
    var name: String

    @Flag(name: .long, help: "Also delete the cached snapshot from disk.")
    var purge: Bool = false

    func run() async throws {
        let registry = ModelRegistry()
        guard let removed = try registry.remove(name: name) else {
            print("Not installed: \(name)")
            throw ExitCode.failure
        }
        print("Removed registry entry for \(removed.name) (\(removed.repoId))")
        if purge {
            let url = URL(fileURLWithPath: removed.localPath)
            do {
                try FileManager.default.removeItem(at: url)
                print("Purged \(url.path)")
            } catch {
                print("Warning: could not delete \(url.path): \(error)")
            }
        } else {
            print("Snapshot still cached at \(removed.localPath) (use --purge to delete).")
        }
    }
}
