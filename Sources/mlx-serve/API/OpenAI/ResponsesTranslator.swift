import Foundation

/// Translation between the OpenAI Responses wire format and our internal
/// domain types. The streaming state machine differs significantly from
/// Chat Completions: events are item-lifecycle (`output_item.added` /
/// `output_text.delta` / `output_item.done`) rather than per-token deltas,
/// and tool calls become first-class items rather than nested fields on a
/// message. Conceptually the state machine mirrors `AnthropicTranslator`
/// but with Responses-shaped wrappers.
enum ResponsesTranslator {

    struct DecodedRequest {
        var model: String
        var messages: [ChatMessage]
        var tools: [ToolDefinition]
        var toolChoice: ToolChoice?
        var params: GenerationParams
    }

    enum TranslationError: Error, CustomStringConvertible {
        case unsupportedRole(String)
        var description: String {
            switch self {
            case .unsupportedRole(let r): return "Unsupported message role: \(r)"
            }
        }
    }

    // MARK: - Wire ⇒ Domain

    static func decode(_ req: OpenAIResponses.Request) throws -> DecodedRequest {
        var messages: [ChatMessage] = []
        // `instructions` is the Responses-API equivalent of a system prompt
        // (kept separate from `input` because it's typically immutable across
        // a multi-turn agent loop).
        if let instructions = req.instructions, !instructions.isEmpty {
            messages.append(ChatMessage(role: .system, text: instructions))
        }

        switch req.input {
        case .text(let s):
            messages.append(ChatMessage(role: .user, text: s))
        case .items(let items):
            for item in items {
                if let m = try Self.itemToMessage(item) {
                    messages.append(m)
                }
            }
        case .none:
            break
        }

        // Filter out tools without names — those are codex's built-in
        // helpers (`local_shell`, `web_search`, …) that we don't surface to
        // the model. Only nameable function tools make it through.
        let tools: [ToolDefinition] = (req.tools ?? []).compactMap { tool in
            guard let name = tool.name, !name.isEmpty else { return nil }
            return ToolDefinition(
                name: name,
                description: tool.description,
                inputSchema: tool.parameters ?? .object([:]))
        }
        let choice: ToolChoice? = req.toolChoice.map {
            switch $0 {
            case .auto: return .auto
            case .required: return .required
            case .none: return ToolChoice.none
            case .specific(let n): return .specific(name: n)
            }
        }
        let params = GenerationParams(
            temperature: req.temperature,
            topP: req.topP,
            topK: nil,
            maxTokens: req.maxOutputTokens,
            stopSequences: [],
            seed: nil,
            stream: req.stream ?? false)
        return DecodedRequest(
            model: req.model, messages: messages, tools: tools,
            toolChoice: choice, params: params)
    }

    private static func itemToMessage(_ item: OpenAIResponses.InputItem) throws -> ChatMessage? {
        switch item {
        case .message(let role, let content):
            let roleEnum: ChatMessage.Role
            switch role {
            case "user": roleEnum = .user
            case "assistant": roleEnum = .assistant
            case "system", "developer": roleEnum = .system
            case "tool": roleEnum = .tool
            default: throw TranslationError.unsupportedRole(role)
            }
            var blocks: [ContentBlock] = []
            for part in content {
                switch part {
                case .text(let t):
                    if !t.isEmpty { blocks.append(.text(t)) }
                case .image(let url, let data, let mt):
                    if let b64 = data, let bytes = Data(base64Encoded: b64) {
                        blocks.append(.image(.init(data: bytes, mimeType: mt ?? "image/png")))
                    } else if let url, url.hasPrefix("data:") {
                        // data: URI form
                        let comma = url.firstIndex(of: ",") ?? url.endIndex
                        let after = comma < url.endIndex ? String(url[url.index(after: comma)...]) : ""
                        if let bytes = Data(base64Encoded: after) {
                            blocks.append(.image(.init(data: bytes, mimeType: mt ?? "image/png")))
                        }
                    }
                }
            }
            return ChatMessage(role: roleEnum, content: blocks)

        case .functionCall(let callId, let name, let args):
            // A previous-turn tool invocation. Render as an assistant message
            // containing a tool_use block so the model sees what it asked for.
            let input: JSONValue = (try? JSONDecoder().decode(
                JSONValue.self, from: Data(args.utf8))) ?? .object([:])
            return ChatMessage(
                role: .assistant,
                content: [.toolUse(.init(id: callId, name: name, input: input))])

        case .functionCallOutput(let callId, let output):
            // Result of running the tool. Goes back to the model as a tool
            // message keyed by call_id.
            return ChatMessage(
                role: .tool,
                content: [.toolResult(.init(
                    toolUseId: callId, content: output, isError: false))])

        case .other:
            return nil
        }
    }

    // MARK: - Domain ⇒ Wire (non-streaming)

    static func finalResponse(
        id: String,
        model: String,
        assistantText: String,
        toolCalls: [ToolUse],
        usage: Usage?
    ) -> OpenAIResponses.Response {
        var output: [OpenAIResponses.OutputItem] = []
        if !assistantText.isEmpty {
            output.append(.message(
                id: "msg_" + UUID().uuidString.prefix(20).lowercased(),
                role: "assistant",
                content: [.init(text: assistantText)],
                status: "completed"))
        }
        for tc in toolCalls {
            let argsJSON = Self.encodeJSON(tc.input)
            output.append(.functionCall(
                id: "fc_" + UUID().uuidString.prefix(20).lowercased(),
                callId: tc.id,
                name: tc.name,
                arguments: argsJSON,
                status: "completed"))
        }
        let usageOut = usage.map {
            OpenAIResponses.Usage(
                inputTokens: $0.promptTokens,
                outputTokens: $0.completionTokens,
                totalTokens: $0.totalTokens)
        }
        return OpenAIResponses.Response(
            id: id,
            createdAt: Int(Date().timeIntervalSince1970),
            status: "completed",
            model: model,
            output: output,
            usage: usageOut)
    }

    // MARK: - Domain ⇒ Wire (streaming)

    /// State machine driving the streaming-event sequence. Tracks the current
    /// open output item (message or function_call), accumulated content,
    /// and the sequence counter codex expects on every event.
    final class StreamState {
        let responseId: String
        let model: String
        private(set) var sequence: Int = 0
        private(set) var nextOutputIndex: Int = 0
        var openMessage: PendingMessage?
        var openFunctionCall: PendingFunctionCall?
        /// Completed items, accumulated so `response.completed` can ship the
        /// full final output array.
        var completedOutput: [OpenAIResponses.OutputItem] = []

        init(responseId: String, model: String) {
            self.responseId = responseId
            self.model = model
        }

        func nextSequence() -> Int {
            sequence += 1
            return sequence
        }

        func consumeOutputIndex() -> Int {
            let v = nextOutputIndex
            nextOutputIndex += 1
            return v
        }

        struct PendingMessage {
            var id: String
            var outputIndex: Int
            var text: String
        }

        struct PendingFunctionCall {
            var id: String
            var callId: String
            var name: String
            var outputIndex: Int
            var arguments: String
        }
    }

    struct Frame: Sendable, Equatable {
        var event: String
        var jsonData: String
    }

    static func startFrames(state: StreamState) throws -> [Frame] {
        let createdAt = Int(Date().timeIntervalSince1970)
        let response = OpenAIResponses.Response(
            id: state.responseId,
            createdAt: createdAt,
            status: "in_progress",
            model: state.model,
            output: [],
            usage: nil)
        let event = OpenAIResponses.CreatedEvent(
            response: response,
            sequenceNumber: state.nextSequence())
        return [try frame(name: .created, payload: event)]
    }

    static func frames(for event: StreamEvent, state: StreamState) throws -> [Frame] {
        switch event {
        case .textDelta(let s):
            return try handleTextDelta(s, state: state)
        case .toolUseStart(let id, let name):
            return try handleToolUseStart(id: id, name: name, state: state)
        case .toolUseInputDelta(let chunk):
            return try handleToolInputDelta(chunk, state: state)
        case .toolUseStop:
            return try handleToolUseStop(state: state)
        case .done(_, let usage):
            return try handleDone(state: state, usage: usage)
        }
    }

    // MARK: - Per-event handlers

    private static func handleTextDelta(
        _ s: String, state: StreamState
    ) throws -> [Frame] {
        var frames: [Frame] = []
        // If a tool call is in flight, close it first — text after a tool
        // call should be a fresh message item.
        if let fc = state.openFunctionCall {
            frames.append(contentsOf: try closeFunctionCall(fc, state: state))
        }
        if state.openMessage == nil {
            let id = "msg_" + UUID().uuidString.prefix(20).lowercased()
            let idx = state.consumeOutputIndex()
            state.openMessage = .init(id: id, outputIndex: idx, text: "")
            let opening = OpenAIResponses.OutputItemAddedEvent(
                outputIndex: idx,
                item: .message(id: id, role: "assistant", content: [], status: "in_progress"),
                sequenceNumber: state.nextSequence())
            frames.append(try frame(name: .outputItemAdded, payload: opening))
        }
        state.openMessage!.text += s
        let delta = OpenAIResponses.OutputTextDeltaEvent(
            itemId: state.openMessage!.id,
            outputIndex: state.openMessage!.outputIndex,
            contentIndex: 0,
            delta: s,
            sequenceNumber: state.nextSequence())
        frames.append(try frame(name: .outputTextDelta, payload: delta))
        return frames
    }

    private static func handleToolUseStart(
        id: String, name: String, state: StreamState
    ) throws -> [Frame] {
        var frames: [Frame] = []
        if let msg = state.openMessage {
            frames.append(contentsOf: try closeMessage(msg, state: state))
        }
        let itemId = "fc_" + UUID().uuidString.prefix(20).lowercased()
        let idx = state.consumeOutputIndex()
        state.openFunctionCall = .init(
            id: itemId, callId: id, name: name, outputIndex: idx, arguments: "")
        let opening = OpenAIResponses.OutputItemAddedEvent(
            outputIndex: idx,
            item: .functionCall(id: itemId, callId: id, name: name, arguments: "", status: "in_progress"),
            sequenceNumber: state.nextSequence())
        frames.append(try frame(name: .outputItemAdded, payload: opening))
        return frames
    }

    private static func handleToolInputDelta(
        _ chunk: String, state: StreamState
    ) throws -> [Frame] {
        guard let fc = state.openFunctionCall else { return [] }
        state.openFunctionCall!.arguments += chunk
        let event = OpenAIResponses.CustomToolCallInputDeltaEvent(
            itemId: fc.id, callId: fc.callId, outputIndex: fc.outputIndex,
            delta: chunk, sequenceNumber: state.nextSequence())
        return [try frame(name: .customToolCallInputDelta, payload: event)]
    }

    private static func handleToolUseStop(state: StreamState) throws -> [Frame] {
        guard let fc = state.openFunctionCall else { return [] }
        return try closeFunctionCall(fc, state: state)
    }

    private static func handleDone(state: StreamState, usage: Usage?) throws -> [Frame] {
        var frames: [Frame] = []
        if let msg = state.openMessage {
            frames.append(contentsOf: try closeMessage(msg, state: state))
        }
        if let fc = state.openFunctionCall {
            frames.append(contentsOf: try closeFunctionCall(fc, state: state))
        }
        let usageOut = usage.map {
            OpenAIResponses.Usage(
                inputTokens: $0.promptTokens,
                outputTokens: $0.completionTokens,
                totalTokens: $0.totalTokens)
        }
        let response = OpenAIResponses.Response(
            id: state.responseId,
            createdAt: Int(Date().timeIntervalSince1970),
            status: "completed",
            model: state.model,
            output: state.completedOutput,
            usage: usageOut)
        let completed = OpenAIResponses.CompletedEvent(
            response: response, sequenceNumber: state.nextSequence())
        frames.append(try frame(name: .completed, payload: completed))
        return frames
    }

    private static func closeMessage(
        _ msg: StreamState.PendingMessage, state: StreamState
    ) throws -> [Frame] {
        let finalItem: OpenAIResponses.OutputItem = .message(
            id: msg.id, role: "assistant",
            content: [.init(text: msg.text)],
            status: "completed")
        let done = OpenAIResponses.OutputItemDoneEvent(
            outputIndex: msg.outputIndex,
            item: finalItem,
            sequenceNumber: state.nextSequence())
        state.openMessage = nil
        state.completedOutput.append(finalItem)
        return [try frame(name: .outputItemDone, payload: done)]
    }

    private static func closeFunctionCall(
        _ fc: StreamState.PendingFunctionCall, state: StreamState
    ) throws -> [Frame] {
        let finalItem: OpenAIResponses.OutputItem = .functionCall(
            id: fc.id, callId: fc.callId, name: fc.name,
            arguments: fc.arguments, status: "completed")
        let done = OpenAIResponses.OutputItemDoneEvent(
            outputIndex: fc.outputIndex,
            item: finalItem,
            sequenceNumber: state.nextSequence())
        state.openFunctionCall = nil
        state.completedOutput.append(finalItem)
        return [try frame(name: .outputItemDone, payload: done)]
    }

    // MARK: - Encoding

    private static func frame<T: Encodable>(name: OpenAIResponses.EventName, payload: T) throws -> Frame {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        let data = try enc.encode(payload)
        return Frame(
            event: name.rawValue,
            jsonData: String(data: data, encoding: .utf8) ?? "{}")
    }

    private static func encodeJSON(_ v: JSONValue) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        let data = (try? enc.encode(v)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
