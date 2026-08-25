import ArgumentParser
import Foundation
import Logging

struct ProxyCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "proxy",
        abstract: "Proxy a local OpenAI-compatible endpoint to another model server.",
        discussion: """
            Supported targets: \(ProxyTarget.allCases.map(\.rawValue).joined(separator: ", ")).

            Example:
              smolx proxy lm-studio --base-url http://192.168.1.55:1234
            """,
        subcommands: [LMStudioProxyCommand.self, ProxyLaunchdCommand.self]
    )

    static func normalizeBaseURL(_ rawValue: String) throws -> String {
        let trimmed = rawValue.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard
            var url = URL(string: trimmed),
            ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
            url.host != nil
        else {
            throw ValidationError("--base-url must be an absolute HTTP or HTTPS URL.")
        }

        if url.pathComponents.last == "v1" {
            url.deleteLastPathComponent()
        }
        return url.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }
}

struct LMStudioProxyCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: ProxyTarget.lmStudio.rawValue,
        abstract: "Proxy requests to an LM Studio server."
    )

    @Option(name: .long, help: "Base URL of the upstream model server.")
    var baseUrl: String

    @Option(name: .long, help: "Port to listen on locally.")
    var port: Int = 8080

    @Option(
        name: .long,
        help: "Address to bind. Use 0.0.0.0 for non-loopback access (requires --auth-token).")
    var bind: String = "127.0.0.1"

    @Option(
        name: .long,
        help: "Bearer token required on inbound requests. Mandatory when --bind is non-loopback.")
    var authToken: String?

    @Option(
        name: .long,
        help: "Bearer token sent to the upstream server instead of the inbound Authorization header.")
    var upstreamAuthToken: String?

    @Flag(name: .shortAndLong, help: "Increase log verbosity (-v: debug, -vv: trace).")
    var verbose: Int

    mutating func validate() throws {
        baseUrl = try ProxyCommand.normalizeBaseURL(baseUrl)
    }

    func run() async throws {
        let level: Logger.Level = verbose >= 2 ? .trace : (verbose >= 1 ? .debug : .info)
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = level
            return handler
        }

        let logger = Logger(label: "smolx")
        try await ProxyServer.run(
            config: .init(
                host: bind,
                port: port,
                authToken: authToken,
                upstreamBaseURL: URL(string: baseUrl)!,
                upstreamAuthToken: upstreamAuthToken,
                target: .lmStudio),
            logger: logger)
    }
}
