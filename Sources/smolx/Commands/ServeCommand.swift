import ArgumentParser
import Foundation
import Logging

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Run the HTTP server exposing OpenAI- and Anthropic-compatible endpoints."
    )

    @Option(name: .long, help: "Port to listen on.")
    var port: Int = 8080

    @Option(
        name: .long,
        help: "Address to bind. Use 0.0.0.0 to allow non-loopback access (requires --auth-token).")
    var bind: String = "127.0.0.1"

    @Option(
        name: .long,
        help: "Bearer token required on inbound requests. Mandatory when --bind is non-loopback.")
    var authToken: String?

    @Option(
        name: .long,
        help:
            "Evict loaded models whenever system available memory drops below this floor (e.g. '1GB', '512MB'). Default: 1GB."
    )
    var keepFree: String?

    @Option(
        name: .long,
        help:
            "Unload a model this long after the last active request releases it (e.g. '30s', '2m', '10m'). Default: 30s."
    )
    var idleTimeout: String?

    @Option(name: .long, help: "Cap on the number of models that can be resident at once.")
    var maxConcurrent: Int?

    @Flag(name: .shortAndLong, help: "Increase log verbosity (-v: debug, -vv: trace).")
    var verbose: Int

    func run() async throws {
        let level: Logger.Level = verbose >= 2 ? .trace : (verbose >= 1 ? .debug : .info)
        LoggingSystem.bootstrap { label in
            var h = StreamLogHandler.standardOutput(label: label)
            h.logLevel = level
            return h
        }

        let logger = Logger(label: "smolx")
        let registry = ModelRegistry()

        var settings = ModelManager.Settings.default
        if let s = keepFree, let bytes = ServeOptionParser.parseBytes(s) {
            settings.keepFreeBytes = bytes
        }
        if let raw = idleTimeout, let secs = ServeOptionParser.parseDuration(raw) {
            settings.idleTimeout = secs
        }
        settings.maxConcurrent = maxConcurrent

        let manager = ModelManager(
            registry: registry,
            factory: RoutingProviderFactory(
                mlx: MLXProviderFactory(),
                gguf: GGUFProviderFactory()),
            settings: settings,
            logger: logger)
        await manager.start()
        defer { Task { await manager.stop() } }

        try await HTTPServer.run(
            config: .init(host: bind, port: port, authToken: authToken),
            manager: manager,
            registry: registry,
            logger: logger)
    }
}
