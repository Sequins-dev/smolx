import Foundation
import Hummingbird
import HummingbirdCore
import Logging

enum HTTPServer {

    struct Config {
        var host: String
        var port: Int
        var authToken: String?
    }

    enum StartError: Error, CustomStringConvertible {
        case nonLoopbackRequiresAuth(host: String)

        var description: String {
            switch self {
            case .nonLoopbackRequiresAuth(let h):
                return "Binding to non-loopback address (\(h)) requires --auth-token to be set."
            }
        }
    }

    /// Builds + runs the server. Returns when the server is stopped (e.g. via
    /// graceful-shutdown signal handler from Hummingbird).
    static func run(
        config: Config,
        manager: ModelManager,
        registry: ModelRegistry,
        logger: Logger
    ) async throws {
        if !isLoopback(config.host) && config.authToken == nil {
            throw StartError.nonLoopbackRequiresAuth(host: config.host)
        }

        let router = Router()
        router.add(middleware: BearerAuthMiddleware(token: config.authToken))

        HealthRoute.register(router)
        OpenAIRoutes.register(router, manager: manager, registry: registry, logger: logger)
        OpenAIResponsesRoute.register(router, manager: manager, registry: registry, logger: logger)
        AnthropicRoutes.register(router, manager: manager, registry: registry, logger: logger)

        var prepared = logger
        prepared[metadataKey: "component"] = "http"
        let serverLogger = prepared
        let host = config.host
        let port = config.port

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: "smolx"),
            onServerRunning: { _ in
                serverLogger.info("Listening on \(host):\(port)")
            },
            logger: serverLogger)

        try await app.runService()
    }

    private static func isLoopback(_ host: String) -> Bool {
        host == "127.0.0.1" || host == "::1" || host == "localhost"
    }
}
