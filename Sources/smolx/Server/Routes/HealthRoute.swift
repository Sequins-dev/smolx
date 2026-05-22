import Foundation
import Hummingbird

enum HealthRoute {
    static func register(_ router: Router<some RequestContext>) {
        router.get("/healthz") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(string: #"{"status":"ok"}"#)))
        }
    }
}
