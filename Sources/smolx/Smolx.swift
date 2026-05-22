import ArgumentParser
import Foundation

@main
struct Smolx: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "smolx",
        abstract: "Serve local HuggingFace LLMs over OpenAI- and Anthropic-compatible HTTP APIs.",
        version: "0.0.1",
        subcommands: [
            ServeCommand.self,
            PullCommand.self,
            ModelsCommand.self,
            RemoveCommand.self,
            RunCommand.self,
        ]
    )
}
