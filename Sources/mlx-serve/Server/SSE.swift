import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdCore
import NIOCore

/// Helpers for writing Server-Sent Events. OpenAI streams `data: <json>\n\n`
/// frames terminated by `data: [DONE]\n\n`. Anthropic streams `event: <name>\n
/// data: <json>\n\n` frames including periodic `event: ping` keepalives.
enum SSE {

    static var headers: HTTPFields {
        var h = HTTPFields()
        h[.contentType] = "text/event-stream"
        h[.cacheControl] = "no-cache"
        if let name = HTTPField.Name("X-Accel-Buffering") {
            h.append(HTTPField(name: name, value: "no"))
        }
        return h
    }

    static func dataFrame(_ json: String) -> ByteBuffer {
        ByteBuffer(string: "data: \(json)\n\n")
    }

    static func namedEventFrame(event: String, json: String) -> ByteBuffer {
        ByteBuffer(string: "event: \(event)\ndata: \(json)\n\n")
    }

    static let openAIDone: ByteBuffer = ByteBuffer(string: "data: [DONE]\n\n")
}
