import ArgumentParser
import Foundation

struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List installed models."
    )

    @Flag(name: .long, help: "Emit JSON instead of a human-readable table.")
    var json: Bool = false

    func run() async throws {
        let registry = ModelRegistry()
        let models = try registry.load()

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(models)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
            return
        }

        if models.isEmpty {
            print("No models installed. Use `smolx pull <repo-id>` to add one.")
            return
        }

        // Manual column padding instead of `String(format: "%s", ...)` —
        // C-string format with Swift String causes a segfault on macOS as
        // soon as it has actual non-empty content to format.
        print(Self.row(name: "NAME", type: "TYPE", size: "SIZE", repo: "REPO"))
        for m in models {
            print(
                Self.row(
                    name: m.name,
                    type: m.capability.rawValue,
                    size: PullCommand.formatBytes(m.diskSizeBytes),
                    repo: m.repoId))
        }
    }

    private static func row(name: String, type: String, size: String, repo: String) -> String {
        let n = name.padding(toLength: 32, withPad: " ", startingAt: 0)
        let t = type.padding(toLength: 10, withPad: " ", startingAt: 0)
        let s = size.padding(toLength: 12, withPad: " ", startingAt: 0)
        return "\(n)\(t)\(s)\(repo)"
    }
}
