import Foundation

/// Codable mirrors of the OpenAI Chat Completions wire format. We accept the
/// fields callers actually send (Claude Code, aider, codex, the openai-python
/// SDK) and ignore everything else, so missing/unknown fields don't break us.
enum OpenAI {

    // MARK: - Request

    struct ChatCompletionRequest: Codable, Sendable {
        var model: String
        var messages: [Message]
        var stream: Bool?
        var temperature: Double?
        var topP: Double?
        var maxTokens: Int?
        var stop: StopValue?
        var tools: [Tool]?
        var toolChoice: ToolChoice?
        var responseFormat: ResponseFormat?
        var seed: UInt64?

        enum CodingKeys: String, CodingKey {
            case model, messages, stream, temperature, stop, tools, seed
            case topP = "top_p"
            case maxTokens = "max_tokens"
            case toolChoice = "tool_choice"
            case responseFormat = "response_format"
        }
    }

    struct Message: Codable, Sendable {
        var role: String  // "system" | "user" | "assistant" | "tool"
        var content: Content?
        var toolCalls: [ToolCall]?
        var toolCallId: String?  // only on role == "tool"
        var name: String?

        enum CodingKeys: String, CodingKey {
            case role, content, name
            case toolCalls = "tool_calls"
            case toolCallId = "tool_call_id"
        }
    }

    /// OpenAI permits message.content to be either a plain string or an array
    /// of parts. We preserve which form was sent so round-trips don't change
    /// the wire shape.
    enum Content: Codable, Sendable {
        case text(String)
        case parts([Part])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) {
                self = .text(s)
                return
            }
            self = .parts(try c.decode([Part].self))
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s): try c.encode(s)
            case .parts(let p): try c.encode(p)
            }
        }
    }

    struct Part: Codable, Sendable {
        var type: String  // "text" | "image_url"
        var text: String?
        var imageUrl: ImageURL?

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageUrl = "image_url"
        }
    }

    struct ImageURL: Codable, Sendable {
        /// Either a `data:image/...;base64,...` URI or an https URL.
        var url: String
        var detail: String?  // "auto" | "low" | "high"
    }

    struct ToolCall: Codable, Sendable {
        var id: String
        var type: String  // always "function"
        var function: FunctionCall
    }

    struct FunctionCall: Codable, Sendable {
        var name: String
        /// JSON-encoded arguments string (OpenAI's quirk — see Anthropic's parsed
        /// `input` for contrast). May be partial during streaming.
        var arguments: String
    }

    struct Tool: Codable, Sendable {
        var type: String  // "function"
        var function: ToolFunction
    }

    struct ToolFunction: Codable, Sendable {
        var name: String
        var description: String?
        var parameters: JSONValue?
    }

    /// `"auto"` / `"required"` / `"none"` or `{type:"function", function:{name:...}}`.
    enum ToolChoice: Codable, Sendable {
        case string(String)
        case named(name: String)

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) {
                self = .string(s)
                return
            }
            struct Named: Decodable {
                let function: Inner
                struct Inner: Decodable { let name: String }
            }
            let n = try c.decode(Named.self)
            self = .named(name: n.function.name)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .string(let s):
                try c.encode(s)
            case .named(let name):
                struct Out: Encodable {
                    let type = "function"
                    let function: Inner
                    struct Inner: Encodable { let name: String }
                }
                try c.encode(Out(function: .init(name: name)))
            }
        }
    }

    /// `stop` can be a single string or an array of strings.
    enum StopValue: Codable, Sendable {
        case one(String)
        case many([String])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) {
                self = .one(s)
                return
            }
            self = .many(try c.decode([String].self))
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .one(let s): try c.encode(s)
            case .many(let xs): try c.encode(xs)
            }
        }

        var asArray: [String] {
            switch self {
            case .one(let s): return [s]
            case .many(let xs): return xs
            }
        }
    }

    struct ResponseFormat: Codable, Sendable {
        var type: String  // "text" | "json_object" | "json_schema"
        var jsonSchema: JSONValue?

        enum CodingKeys: String, CodingKey {
            case type
            case jsonSchema = "json_schema"
        }
    }

    // MARK: - Non-streaming response

    struct ChatCompletionResponse: Codable, Sendable {
        var id: String
        var object: String = "chat.completion"
        var created: Int
        var model: String
        var choices: [Choice]
        var usage: ResponseUsage?
    }

    struct Choice: Codable, Sendable {
        var index: Int
        var message: ResponseMessage
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct ResponseMessage: Codable, Sendable {
        var role: String
        var content: String?
        var toolCalls: [ToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct ResponseUsage: Codable, Sendable {
        var promptTokens: Int
        var completionTokens: Int
        var totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }

    // MARK: - Streaming chunk

    struct ChatCompletionChunk: Codable, Sendable {
        var id: String
        var object: String = "chat.completion.chunk"
        var created: Int
        var model: String
        var choices: [ChunkChoice]
        var usage: ResponseUsage?
    }

    struct ChunkChoice: Codable, Sendable {
        var index: Int
        var delta: Delta
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Codable, Sendable {
        var role: String?
        var content: String?
        var toolCalls: [ChunkToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    /// Streaming tool-call deltas: `index` accumulates partial fields across
    /// chunks. `id` and `function.name` arrive on the first delta; later deltas
    /// only carry `function.arguments` fragments.
    struct ChunkToolCall: Codable, Sendable {
        var index: Int
        var id: String?
        var type: String?
        var function: ChunkFunctionCall?
    }

    struct ChunkFunctionCall: Codable, Sendable {
        var name: String?
        var arguments: String?
    }

    // MARK: - Models list

    struct ModelsList: Codable, Sendable {
        var object: String = "list"
        var data: [ModelInfo]
    }

    struct ModelInfo: Codable, Sendable {
        var id: String
        var object: String = "model"
        var created: Int
        var ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }
}
