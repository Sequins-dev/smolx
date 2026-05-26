import ArgumentParser
import Darwin
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
        help: "Maximum bytes resident across all loaded models. Accepts e.g. '32GB' or raw bytes.")
    var memoryBudget: String?

    @Option(
        name: .long,
        help:
            "Unload a model this long after the last active request releases it (e.g. '2m', '10m'). When unset, models stay resident until --memory-budget or --max-concurrent forces eviction."
    )
    var idleTimeout: String?

    @Option(name: .long, help: "Cap on the number of models that can be resident at once.")
    var maxConcurrent: Int?

    func run() async throws {
        // Install a hard SIGINT/SIGTERM handler before anything else so the
        // user can Ctrl+C the server reliably. Hummingbird's `runService`
        // does install signal handlers via swift-service-lifecycle, but they
        // were not bringing this process down in practice — our long-lived
        // ModelManager actor + MLX's internal threads were keeping the
        // process alive past `runService` returning. `Darwin.exit(0)` from
        // a signal handler is a sledgehammer but guaranteed to terminate the
        // process: it skips Swift-level deferred cleanup, but for a stateless
        // HTTP server with model state in MLX's GPU memory (reclaimed by the
        // OS on exit) that's fine. The handler is set on both SIGINT (Ctrl+C
        // from the controlling TTY) and SIGTERM (sent by `kill`).
        signal(SIGINT) { _ in
            // Stay async-signal-safe: write() not print(), then _exit.
            let msg = "\nsmolx: shutting down on SIGINT\n"
            _ = msg.withCString { write(STDERR_FILENO, $0, strlen($0)) }
            Darwin._exit(0)
        }
        signal(SIGTERM) { _ in
            let msg = "\nsmolx: shutting down on SIGTERM\n"
            _ = msg.withCString { write(STDERR_FILENO, $0, strlen($0)) }
            Darwin._exit(0)
        }

        let logger = Logger(label: "smolx")
        let registry = ModelRegistry()

        var settings = ModelManager.Settings.default
        if let b = memoryBudget, let bytes = SystemMemory.parse(b) {
            settings.memoryBudget = bytes
        }
        if let raw = idleTimeout, let secs = SystemMemory.parseDuration(raw) {
            settings.idleTimeout = secs
        }
        settings.maxConcurrent = maxConcurrent

        let manager = ModelManager(
            registry: registry,
            factory: MLXProviderFactory(),
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
