import Foundation

/// Codable mirrors of OpenAI's Responses API wire format — the newer endpoint
/// (`POST /v1/responses`) that codex 0.128+ requires (it dropped support for
/// `wire_api = "chat"` in its provider config). This is a separate, richer
/// shape from Chat Completions: input/output are arrays of typed items, tool
/// calls are first-class items rather than fields on a message, and the
/// streaming protocol emits item-lifecycle events instead of plain deltas.
///
/// We implement only the subset codex actually parses (verified by reading
/// `codex-rs/codex-api/src/sse/responses.rs` on openai/codex@main): the
/// `response.created`, `output_item.added/.done`, `output_text.delta`,
/// `custom_tool_call_input.delta`, and `response.completed` events. Other
/// public-API events (reasoning summaries, refusals, etc.) are left out —
/// codex only `trace!`s them.
enum OpenAIResponses {

    // MARK: - Request

    struct Request: Codable, Sendable {
        var model: String
        var input: Input?
        var instructions: String?
        var tools: [Tool]?
        var toolChoice: ToolChoice?
        var stream: Bool?
        var temperature: Double?
        var topP: Double?
        var maxOutputTokens: Int?
        var previousResponseId: String?
        var store: Bool?
        var parallelToolCalls: Bool?
        var include: [String]?

        enum CodingKeys: String, CodingKey {
            case model, input, instructions, tools, stream, temperature, store, include
            case toolChoice = "tool_choice"
            case topP = "top_p"
            case maxOutputTokens = "max_output_tokens"
            case previousResponseId = "previous_response_id"
            case parallelToolCalls = "parallel_tool_calls"
        }
    }

    /// `input` is either a plain string (a single user prompt) or an array of
    /// typed items mirroring the conversation history.
    enum Input: Codable, Sendable {
        case text(String)
        case items([InputItem])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .text(s); return }
            self = .items(try c.decode([InputItem].self))
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s): try c.encode(s)
            case .items(let xs): try c.encode(xs)
            }
        }
    }

    /// A single entry in the input array. Item types we handle: `message`
    /// (user/assistant/system/developer with content parts), `function_call`
    /// (an earlier turn's tool invocation), and `function_call_output` (the
    /// result of running that tool, which the client returns on the next
    /// request). Any other `type` is decoded as `.other` and skipped.
    enum InputItem: Codable, Sendable {
        case message(role: String, content: [InputContentPart])
        case functionCall(callId: String, name: String, arguments: String)
        case functionCallOutput(callId: String, output: String)
        case other(type: String)

        enum K: String, CodingKey {
            case type, role, content
            case callId = "call_id"
            case name, arguments, output
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: K.self)
            let type = try c.decode(String.self, forKey: .type)
            switch type {
            case "message":
                let role = try c.decode(String.self, forKey: .role)
                let content = (try? c.decode([InputContentPart].self, forKey: .content)) ?? []
                self = .message(role: role, content: content)
            case "function_call":
                // All fields treated as optional with empty-string defaults
                // so a malformed/partial item from codex falls into `.other`
                // (via translation skipping) rather than failing the whole
                // request decode.
                let callId = (try? c.decode(String.self, forKey: .callId)) ?? ""
                let name = (try? c.decode(String.self, forKey: .name)) ?? ""
                let args = (try? c.decode(String.self, forKey: .arguments)) ?? ""
                if callId.isEmpty && name.isEmpty {
                    self = .other(type: type)
                } else {
                    self = .functionCall(callId: callId, name: name, arguments: args)
                }
            case "function_call_output":
                let callId = (try? c.decode(String.self, forKey: .callId)) ?? ""
                let output = (try? c.decode(String.self, forKey: .output)) ?? ""
                self = .functionCallOutput(callId: callId, output: output)
            default:
                self = .other(type: type)
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: K.self)
            switch self {
            case .message(let role, let content):
                try c.encode("message", forKey: .type)
                try c.encode(role, forKey: .role)
                try c.encode(content, forKey: .content)
            case .functionCall(let callId, let name, let args):
                try c.encode("function_call", forKey: .type)
                try c.encode(callId, forKey: .callId)
                try c.encode(name, forKey: .name)
                try c.encode(args, forKey: .arguments)
            case .functionCallOutput(let callId, let output):
                try c.encode("function_call_output", forKey: .type)
                try c.encode(callId, forKey: .callId)
                try c.encode(output, forKey: .output)
            case .other(let type):
                try c.encode(type, forKey: .type)
            }
        }
    }

    /// Content parts inside an `input` message. Most commonly `input_text`
    /// with a plain string. Codex also sends `input_image` for vision; we
    /// decode it so we don't error, but only pull the URL/data for VLM use.
    enum InputContentPart: Codable, Sendable {
        case text(String)
        case image(url: String?, data: String?, mimeType: String?)

        enum K: String, CodingKey {
            case type, text
            case imageUrl = "image_url"
            case data
            case mimeType = "media_type"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: K.self)
            let type = try c.decode(String.self, forKey: .type)
            switch type {
            case "input_text":
                self = .text(try c.decode(String.self, forKey: .text))
            case "input_image":
                self = .image(
                    url: try? c.decode(String.self, forKey: .imageUrl),
                    data: try? c.decode(String.self, forKey: .data),
                    mimeType: try? c.decode(String.self, forKey: .mimeType))
            default:
                self = .text("")
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: K.self)
            switch self {
            case .text(let t):
                try c.encode("input_text", forKey: .type)
                try c.encode(t, forKey: .text)
            case .image(let url, let data, let mt):
                try c.encode("input_image", forKey: .type)
                if let url { try c.encode(url, forKey: .imageUrl) }
                if let data { try c.encode(data, forKey: .data) }
                if let mt { try c.encode(mt, forKey: .mimeType) }
            }
        }
    }

    /// Tool shape in Responses API. Unlike Chat Completions which nests under
    /// `function: {name, parameters}`, Responses puts `name` / `parameters` at
    /// the top level of the tool object.
    ///
    /// `name` is marked optional here only so the request decoder doesn't
    /// throw on built-in tool types codex sends without a name (e.g. the
    /// `local_shell`, `web_search`, `file_search` builtins). Function tools
    /// always have a name; nameless entries are filtered out at translation
    /// time rather than failing the whole request.
    struct Tool: Codable, Sendable {
        var type: String  // "function" or a builtin like "local_shell"
        var name: String?
        var description: String?
        var parameters: JSONValue?
        var strict: Bool?
    }

    enum ToolChoice: Codable, Sendable {
        case auto
        case required
        case none
        case specific(name: String)

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) {
                switch s {
                case "auto": self = .auto; return
                case "required": self = .required; return
                case "none": self = .none; return
                default: self = .auto; return
                }
            }
            // Object form: {"type":"function","name":"foo"}
            struct Named: Decodable { let name: String }
            let n = try c.decode(Named.self)
            self = .specific(name: n.name)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .auto: try c.encode("auto")
            case .required: try c.encode("required")
            case .none: try c.encode("none")
            case .specific(let name):
                struct Out: Encodable {
                    let type = "function"
                    let name: String
                }
                try c.encode(Out(name: name))
            }
        }
    }

    // MARK: - Non-streaming response

    /// The full Response object returned for non-streaming requests AND
    /// embedded in the `response` field of `response.created` /
    /// `response.completed` streaming events.
    struct Response: Codable, Sendable {
        var id: String
        var object: String = "response"
        var createdAt: Int
        var status: String
        var model: String
        var output: [OutputItem]
        var usage: Usage?
        var instructions: String?
        var previousResponseId: String?

        enum CodingKeys: String, CodingKey {
            case id, object, status, model, output, usage, instructions
            case createdAt = "created_at"
            case previousResponseId = "previous_response_id"
        }
    }

    /// One entry in `response.output`. Codex parses these via its
    /// `ResponseItem` enum on `output_item.added` / `.done` events.
    enum OutputItem: Codable, Sendable {
        case message(id: String, role: String, content: [OutputContentPart], status: String)
        case functionCall(id: String, callId: String, name: String, arguments: String, status: String)

        enum K: String, CodingKey {
            case type, id, role, content, status, name, arguments
            case callId = "call_id"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: K.self)
            let type = try c.decode(String.self, forKey: .type)
            switch type {
            case "message":
                self = .message(
                    id: try c.decode(String.self, forKey: .id),
                    role: try c.decode(String.self, forKey: .role),
                    content: (try? c.decode([OutputContentPart].self, forKey: .content)) ?? [],
                    status: (try? c.decode(String.self, forKey: .status)) ?? "completed")
            case "function_call":
                self = .functionCall(
                    id: try c.decode(String.self, forKey: .id),
                    callId: try c.decode(String.self, forKey: .callId),
                    name: try c.decode(String.self, forKey: .name),
                    arguments: (try? c.decode(String.self, forKey: .arguments)) ?? "",
                    status: (try? c.decode(String.self, forKey: .status)) ?? "completed")
            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type, in: c,
                    debugDescription: "Unsupported output item type: \(type)")
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: K.self)
            switch self {
            case .message(let id, let role, let content, let status):
                try c.encode("message", forKey: .type)
                try c.encode(id, forKey: .id)
                try c.encode(role, forKey: .role)
                try c.encode(content, forKey: .content)
                try c.encode(status, forKey: .status)
            case .functionCall(let id, let callId, let name, let args, let status):
                try c.encode("function_call", forKey: .type)
                try c.encode(id, forKey: .id)
                try c.encode(callId, forKey: .callId)
                try c.encode(name, forKey: .name)
                try c.encode(args, forKey: .arguments)
                try c.encode(status, forKey: .status)
            }
        }
    }

    /// Content of a `message` output item. Currently only `output_text` —
    /// the Responses API also has `refusal` but we don't generate that.
    struct OutputContentPart: Codable, Sendable {
        var type: String  // "output_text"
        var text: String
        var annotations: [JSONValue]?

        init(text: String) {
            self.type = "output_text"
            self.text = text
            self.annotations = []
        }
    }

    struct Usage: Codable, Sendable {
        var inputTokens: Int
        var outputTokens: Int
        var totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case totalTokens = "total_tokens"
        }
    }

    // MARK: - Streaming events

    /// SSE event payloads. Each is serialised under an `event: <name>` line
    /// followed by `data: <json>`. The `type` field inside the JSON
    /// duplicates the event name (codex parses by `type`, the `event:` line
    /// is convention).
    enum EventName: String, Sendable {
        case created                 = "response.created"
        case outputItemAdded         = "response.output_item.added"
        case outputItemDone          = "response.output_item.done"
        case outputTextDelta         = "response.output_text.delta"
        case customToolCallInputDelta = "response.custom_tool_call_input.delta"
        case completed               = "response.completed"
        case failed                  = "response.failed"
    }

    struct CreatedEvent: Encodable, Sendable {
        var type: String = EventName.created.rawValue
        var response: Response
        var sequenceNumber: Int

        enum CodingKeys: String, CodingKey {
            case type, response
            case sequenceNumber = "sequence_number"
        }
    }

    struct OutputItemAddedEvent: Encodable, Sendable {
        var type: String = EventName.outputItemAdded.rawValue
        var outputIndex: Int
        var item: OutputItem
        var sequenceNumber: Int

        enum CodingKeys: String, CodingKey {
            case type, item
            case outputIndex = "output_index"
            case sequenceNumber = "sequence_number"
        }
    }

    struct OutputItemDoneEvent: Encodable, Sendable {
        var type: String = EventName.outputItemDone.rawValue
        var outputIndex: Int
        var item: OutputItem
        var sequenceNumber: Int

        enum CodingKeys: String, CodingKey {
            case type, item
            case outputIndex = "output_index"
            case sequenceNumber = "sequence_number"
        }
    }

    struct OutputTextDeltaEvent: Encodable, Sendable {
        var type: String = EventName.outputTextDelta.rawValue
        var itemId: String
        var outputIndex: Int
        var contentIndex: Int
        var delta: String
        var sequenceNumber: Int

        enum CodingKeys: String, CodingKey {
            case type, delta
            case itemId = "item_id"
            case outputIndex = "output_index"
            case contentIndex = "content_index"
            case sequenceNumber = "sequence_number"
        }
    }

    /// Streaming tool-arguments delta. The event name `custom_tool_call_input`
    /// (not `function_call_arguments`) matches what codex's SSE parser
    /// dispatches on for non-OpenAI providers.
    struct CustomToolCallInputDeltaEvent: Encodable, Sendable {
        var type: String = EventName.customToolCallInputDelta.rawValue
        var itemId: String
        var callId: String
        var outputIndex: Int
        var delta: String
        var sequenceNumber: Int

        enum CodingKeys: String, CodingKey {
            case type, delta
            case itemId = "item_id"
            case callId = "call_id"
            case outputIndex = "output_index"
            case sequenceNumber = "sequence_number"
        }
    }

    struct CompletedEvent: Encodable, Sendable {
        var type: String = EventName.completed.rawValue
        var response: Response
        var sequenceNumber: Int

        enum CodingKeys: String, CodingKey {
            case type, response
            case sequenceNumber = "sequence_number"
        }
    }

    struct FailedEvent: Encodable, Sendable {
        var type: String = EventName.failed.rawValue
        var response: FailedResponse
        var sequenceNumber: Int

        struct FailedResponse: Encodable, Sendable {
            var id: String
            var object: String = "response"
            var status: String = "failed"
            var error: ErrorBody
        }

        struct ErrorBody: Encodable, Sendable {
            var type: String
            var message: String
            var code: String?
        }

        enum CodingKeys: String, CodingKey {
            case type, response
            case sequenceNumber = "sequence_number"
        }
    }
}
