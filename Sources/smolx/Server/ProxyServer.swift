import Foundation
import Hummingbird
import HummingbirdCore
import HTTPTypes
import Logging
import NIOCore

enum ProxyServer {
    struct Config: Sendable {
        var host: String
        var port: Int
        var authToken: String?
        var upstreamBaseURL: URL
        var upstreamAuthToken: String?
        var target: ProxyTarget
    }

    enum StartError: Error, CustomStringConvertible {
        case nonLoopbackRequiresAuth(host: String)

        var description: String {
            switch self {
            case .nonLoopbackRequiresAuth(let host):
                return "Binding to non-loopback address (\(host)) requires --auth-token to be set."
            }
        }
    }

    static func run(config: Config, logger: Logger) async throws {
        if !isLoopback(config.host) && config.authToken == nil {
            throw StartError.nonLoopbackRequiresAuth(host: config.host)
        }

        let router = Router()
        router.add(middleware: BearerAuthMiddleware(token: config.authToken))
        HealthRoute.register(router)

        let methods: [HTTPRequest.Method] = [
            .get, .post, .put, .delete, .patch, .head, .options,
        ]
        for method in methods {
            router.on("**", method: method) { request, _ -> Response in
                await forward(request, config: config, logger: logger)
            }
        }

        var prepared = logger
        prepared[metadataKey: "component"] = "proxy"
        let serverLogger = prepared
        let host = config.host
        let port = config.port
        let upstream = config.upstreamBaseURL.absoluteString

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: "smolx-proxy"),
            onServerRunning: { _ in
                serverLogger.info(
                    "Listening on \(host):\(port), proxying \(config.target.rawValue) at \(upstream)")
            },
            logger: serverLogger)

        try await app.runService()
    }

    private static func forward(
        _ request: Request,
        config: Config,
        logger: Logger
    ) async -> Response {
        do {
            let collected = try await request.body.collect(upTo: 64 * 1024 * 1024)
            let incomingBody = Data(collected.readableBytesView)
            let rewrite = try config.target.rewriteRequest(
                path: request.uri.path,
                body: incomingBody)

            if rewrite.translationCount > 0 {
                logger.debug(
                    "Applied \(rewrite.translationCount) \(config.target.rawValue) request translation(s)",
                    metadata: ["route": "\(request.method.rawValue) \(request.uri.path)"])
            }

            var upstreamRequest = URLRequest(
                url: upstreamURL(for: request, baseURL: config.upstreamBaseURL))
            upstreamRequest.httpMethod = request.method.rawValue
            upstreamRequest.httpBody = rewrite.body.isEmpty ? nil : rewrite.body
            upstreamRequest.timeoutInterval = 300

            for field in request.headers where !hopByHopRequestHeaders.contains(field.name.canonicalName) {
                // When inbound auth protects the proxy itself, that credential
                // must not leak to the upstream. Use --upstream-auth-token when
                // the target server also requires authentication.
                if field.name == .authorization, config.authToken != nil {
                    continue
                }
                upstreamRequest.addValue(field.value, forHTTPHeaderField: field.name.rawName)
            }
            upstreamRequest.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            if let token = config.upstreamAuthToken {
                upstreamRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            let (bytes, urlResponse) = try await URLSession.shared.bytes(for: upstreamRequest)
            guard let httpResponse = urlResponse as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }

            var headers = HTTPFields()
            for (rawName, rawValue) in httpResponse.allHeaderFields {
                guard
                    let nameString = rawName as? String,
                    !hopByHopResponseHeaders.contains(nameString.lowercased()),
                    let name = HTTPField.Name(nameString)
                else { continue }
                headers.append(HTTPField(name: name, value: String(describing: rawValue)))
            }

            let body = ResponseBody { writer in
                var buffer = ByteBufferAllocator().buffer(capacity: 16 * 1024)
                do {
                    for try await byte in bytes {
                        buffer.writeInteger(byte)
                        if buffer.readableBytes >= 16 * 1024 {
                            try await writer.write(buffer)
                            buffer.clear()
                        }
                    }
                    if buffer.readableBytes > 0 {
                        try await writer.write(buffer)
                    }
                    try await writer.finish(nil)
                } catch {
                    logger.error("Upstream response stream failed: \(error)")
                    try? await writer.finish(nil)
                }
            }

            return Response(
                status: .init(code: httpResponse.statusCode),
                headers: headers,
                body: body)
        } catch let error as ProxyTranslationError {
            return OpenAIRoutes.errorResponse(.badRequest, message: error.description)
        } catch {
            logger.error("Proxy request failed: \(error)")
            return OpenAIRoutes.errorResponse(
                .badGateway,
                message:
                    "Could not reach \(config.target.abstract) at \(config.upstreamBaseURL.absoluteString): \(error.localizedDescription)"
            )
        }
    }

    static func upstreamURL(for request: Request, baseURL: URL) -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        let basePath = components.percentEncodedPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let requestPath = request.uri.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components.percentEncodedPath =
            "/"
            + [basePath, requestPath]
            .filter { !$0.isEmpty }
            .joined(separator: "/")
        components.percentEncodedQuery = request.uri.query
        return components.url!
    }

    private static let hopByHopRequestHeaders: Set<String> = [
        "connection", "content-length", "host", "keep-alive", "proxy-authenticate",
        "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
    ]

    private static let hopByHopResponseHeaders: Set<String> = [
        "connection", "content-length", "keep-alive", "proxy-authenticate",
        "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
    ]

    private static func isLoopback(_ host: String) -> Bool {
        host == "127.0.0.1" || host == "::1" || host == "localhost"
    }
}
