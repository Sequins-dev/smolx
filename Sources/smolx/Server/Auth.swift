import Foundation
import Hummingbird
import HummingbirdCore

/// Middleware that enforces a bearer token on inbound requests when configured.
/// Requests bound to loopback with no `authToken` set pass through unauthenticated
/// — that's the v1 default.
struct BearerAuthMiddleware<Context: RequestContext>: MiddlewareProtocol {
    let token: String?

    func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard let token else {
            return try await next(request, context)
        }
        guard let header = request.headers[.authorization],
              header == "Bearer \(token)" || header == token
        else {
            return Response(
                status: .unauthorized,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(string: """
                    {"error":{"type":"unauthorized","message":"Invalid or missing bearer token"}}
                    """)))
        }
        return try await next(request, context)
    }
}
