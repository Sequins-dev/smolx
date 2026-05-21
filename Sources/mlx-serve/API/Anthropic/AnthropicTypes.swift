import Foundation

/// Codable mirrors of the Anthropic Messages wire format.
enum Anthropic {

    // MARK: - Request

    struct MessagesRequest: Codable, Sendable {
        var model: String
        var maxTokens: Int
        var messages: [Message]
        var system: SystemValue?
        var stream: Bool?
        var temperature: Double?
        var topP: Double?
        var topK: Int?
        var stopSequences: [String]?
        var tools: [Tool]?
        var toolChoice: ToolChoice?

        enum CodingKeys: String, CodingKey {
            case model, messages, system, stream, temperature, tools
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case topK = "top_k"
            case stopSequences = "stop_sequences"
            case toolChoice = "tool_choice"
        }
    }

    struct Message: Codable, Sendable {
        var role: String  // "user" | "assistant"
        var content: Content
    }

    /// Anthropic accepts string content (sugar for `[{type:"text", ...}]`) or
    /// an array of content blocks. We round-trip whichever form arrived.
    enum Content: Codable, Sendable {
        case text(String)
        case blocks([Block])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .text(s); return }
            self = .blocks(try c.decode([Block].self))
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s): try c.encode(s)
            case .blocks(let b): try c.encode(b)
            }
        }
    }

    /// A request- or response-side content block. Wire types: `text`, `image`,
    /// `tool_use`, `tool_result`. We decode by inspecting `type`.
    enum Block: Codable, Sendable {
        case text(String)
        case image(ImageSource)
        case toolUse(id: String, name: String, input: JSONValue)
        case toolResult(toolUseId: String, content: String, isError: Bool)

        enum BlockType: String, Codable { case text, image, tool_use, tool_result }

        enum CodingKeys: String, CodingKey {
            case type, text, source, id, name, input
            case toolUseId = "tool_use_id"
            case content
            case isError = "is_error"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let type = try c.decode(BlockType.self, forKey: .type)
            switch type {
            case .text:
                self = .text(try c.decode(String.self, forKey: .text))
            case .image:
                self = .image(try c.decode(ImageSource.self, forKey: .source))
            case .tool_use:
                self = .toolUse(
                    id: try c.decode(String.self, forKey: .id),
                    name: try c.decode(String.self, forKey: .name),
                    input: try c.decode(JSONValue.self, forKey: .input))
            case .tool_result:
                let isError = (try? c.decode(Bool.self, forKey: .isError)) ?? false
                let content = try Block.decodeToolResultContent(c)
                self = .toolResult(
                    toolUseId: try c.decode(String.self, forKey: .toolUseId),
                    content: content,
                    isError: isError)
            }
        }

        /// `tool_result.content` may be a string or an array of text blocks.
        private static func decodeToolResultContent(
            _ c: KeyedDecodingContainer<CodingKeys>
        ) throws -> String {
            if let s = try? c.decode(String.self, forKey: .content) {
                return s
            }
            struct TextOnly: Decodable { let type: String; let text: String? }
            let parts = (try? c.decode([TextOnly].self, forKey: .content)) ?? []
            return parts.compactMap(\.text).joined()
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .text(let t):
                try c.encode("text", forKey: .type)
                try c.encode(t, forKey: .text)
            case .image(let src):
                try c.encode("image", forKey: .type)
                try c.encode(src, forKey: .source)
            case .toolUse(let id, let name, let input):
                try c.encode("tool_use", forKey: .type)
                try c.encode(id, forKey: .id)
                try c.encode(name, forKey: .name)
                try c.encode(input, forKey: .input)
            case .toolResult(let toolUseId, let content, let isError):
                try c.encode("tool_result", forKey: .type)
                try c.encode(toolUseId, forKey: .toolUseId)
                try c.encode(content, forKey: .content)
                if isError { try c.encode(true, forKey: .isError) }
            }
        }
    }

    struct ImageSource: Codable, Sendable {
        var type: String  // "base64" | "url"
        var mediaType: String?
        var data: String?  // base64-encoded bytes when type == "base64"
        var url: String?   // when type == "url"

        enum CodingKeys: String, CodingKey {
            case type, data, url
            case mediaType = "media_type"
        }
    }

    /// `system` may be a string or an array of text content blocks.
    enum SystemValue: Codable, Sendable {
        case text(String)
        case blocks([Block])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .text(s); return }
            self = .blocks(try c.decode([Block].self))
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s): try c.encode(s)
            case .blocks(let b): try c.encode(b)
            }
        }

        var asText: String {
            switch self {
            case .text(let s): return s
            case .blocks(let bs):
                return bs.compactMap {
                    if case .text(let t) = $0 { return t } else { return nil }
                }.joined(separator: "\n")
            }
        }
    }

    struct Tool: Codable, Sendable {
        var name: String
        var description: String?
        var inputSchema: JSONValue

        enum CodingKeys: String, CodingKey {
            case name, description
            case inputSchema = "input_schema"
        }
    }

    enum ToolChoice: Codable, Sendable {
        case auto
        case any
        case tool(name: String)
        case none

        enum CodingKeys: String, CodingKey { case type, name }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let type = try c.decode(String.self, forKey: .type)
            switch type {
            case "auto": self = .auto
            case "any": self = .any
            case "tool":
                self = .tool(name: try c.decode(String.self, forKey: .name))
            case "none": self = .none
            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type, in: c,
                    debugDescription: "Unknown tool_choice type: \(type)")
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .auto: try c.encode("auto", forKey: .type)
            case .any: try c.encode("any", forKey: .type)
            case .none: try c.encode("none", forKey: .type)
            case .tool(let name):
                try c.encode("tool", forKey: .type)
                try c.encode(name, forKey: .name)
            }
        }
    }

    // MARK: - Non-streaming response

    struct MessagesResponse: Codable, Sendable {
        var id: String
        var type: String = "message"
        var role: String = "assistant"
        var model: String
        var content: [Block]
        var stopReason: String?
        var stopSequence: String?
        var usage: ResponseUsage

        enum CodingKeys: String, CodingKey {
            case id, type, role, model, content, usage
            case stopReason = "stop_reason"
            case stopSequence = "stop_sequence"
        }
    }

    struct ResponseUsage: Codable, Sendable {
        var inputTokens: Int
        var outputTokens: Int

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
    }

    // MARK: - Streaming events
    //
    // The wire format is `event: <name>\ndata: <json>\n\n`. Each event below
    // produces one such frame. We model the `data:` payloads as Codable types;
    // the event name is carried alongside by the SSE writer.

    struct MessageStart: Codable, Sendable {
        var type: String = "message_start"
        var message: MessagesResponse  // content starts empty
    }

    struct ContentBlockStart: Codable, Sendable {
        var type: String = "content_block_start"
        var index: Int
        var contentBlock: Block

        enum CodingKeys: String, CodingKey {
            case type, index
            case contentBlock = "content_block"
        }
    }

    /// Two delta variants: `text_delta` (assistant text) and `input_json_delta`
    /// (partial JSON for a tool_use block's input).
    struct ContentBlockDelta: Codable, Sendable {
        var type: String = "content_block_delta"
        var index: Int
        var delta: Delta

        enum Delta: Codable, Sendable {
            case text(String)
            case inputJson(String)

            enum K: String, CodingKey { case type, text, partial_json }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: K.self)
                let t = try c.decode(String.self, forKey: .type)
                switch t {
                case "text_delta":
                    self = .text(try c.decode(String.self, forKey: .text))
                case "input_json_delta":
                    self = .inputJson(try c.decode(String.self, forKey: .partial_json))
                default:
                    throw DecodingError.dataCorruptedError(
                        forKey: .type, in: c,
                        debugDescription: "Unknown delta type: \(t)")
                }
            }

            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: K.self)
                switch self {
                case .text(let s):
                    try c.encode("text_delta", forKey: .type)
                    try c.encode(s, forKey: .text)
                case .inputJson(let s):
                    try c.encode("input_json_delta", forKey: .type)
                    try c.encode(s, forKey: .partial_json)
                }
            }
        }
    }

    struct ContentBlockStop: Codable, Sendable {
        var type: String = "content_block_stop"
        var index: Int
    }

    struct MessageDelta: Codable, Sendable {
        var type: String = "message_delta"
        var delta: Patch
        var usage: ResponseUsage

        struct Patch: Codable, Sendable {
            var stopReason: String?
            var stopSequence: String?

            enum CodingKeys: String, CodingKey {
                case stopReason = "stop_reason"
                case stopSequence = "stop_sequence"
            }
        }
    }

    struct MessageStop: Codable, Sendable {
        var type: String = "message_stop"
    }
}
