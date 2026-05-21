import Foundation

enum AnthropicTranslator {

    struct DecodedRequest {
        var model: String
        var messages: [ChatMessage]
        var tools: [ToolDefinition]
        var toolChoice: ToolChoice?
        var params: GenerationParams
    }

    enum TranslationError: Error, CustomStringConvertible {
        case invalidImageSource
        case unsupportedImageScheme(String)

        var description: String {
            switch self {
            case .invalidImageSource:
                return "Image source missing both `data` and `url`."
            case .unsupportedImageScheme(let s):
                return "Image url scheme not supported: \(s). Provide base64 source instead."
            }
        }
    }

    // MARK: - Wire ⇒ Domain

    static func decode(_ req: Anthropic.MessagesRequest) throws -> DecodedRequest {
        var messages: [ChatMessage] = []
        if let sys = req.system {
            messages.append(ChatMessage(role: .system, text: sys.asText))
        }
        for m in req.messages {
            messages.append(try toDomain(m))
        }
        let tools = (req.tools ?? []).map {
            ToolDefinition(name: $0.name, description: $0.description, inputSchema: $0.inputSchema)
        }
        let choice: ToolChoice? = req.toolChoice.map {
            switch $0 {
            case .auto: return .auto
            case .any: return .required
            case .tool(let name): return .specific(name: name)
            case .none: return .none
            }
        }
        let params = GenerationParams(
            temperature: req.temperature,
            topP: req.topP,
            topK: req.topK,
            maxTokens: req.maxTokens,
            stopSequences: req.stopSequences ?? [],
            seed: nil,
            stream: req.stream ?? false)
        return DecodedRequest(
            model: req.model, messages: messages, tools: tools,
            toolChoice: choice, params: params)
    }

    private static func toDomain(_ m: Anthropic.Message) throws -> ChatMessage {
        let role: ChatMessage.Role = (m.role == "assistant") ? .assistant : .user
        var blocks: [ContentBlock] = []
        switch m.content {
        case .text(let s):
            if !s.isEmpty { blocks.append(.text(s)) }
        case .blocks(let bs):
            for b in bs {
                switch b {
                case .text(let t):
                    blocks.append(.text(t))
                case .image(let src):
                    blocks.append(.image(try decodeImage(src)))
                case .toolUse(let id, let name, let input):
                    blocks.append(.toolUse(.init(id: id, name: name, input: input)))
                case .toolResult(let id, let content, let isError):
                    blocks.append(.toolResult(
                        .init(toolUseId: id, content: content, isError: isError)))
                }
            }
        }
        return ChatMessage(role: role, content: blocks)
    }

    private static func decodeImage(_ src: Anthropic.ImageSource) throws -> ImagePayload {
        switch src.type {
        case "base64":
            guard let payload = src.data, let bytes = Data(base64Encoded: payload) else {
                throw TranslationError.invalidImageSource
            }
            return ImagePayload(data: bytes, mimeType: src.mediaType ?? "image/png")
        case "url":
            throw TranslationError.unsupportedImageScheme("url")
        default:
            throw TranslationError.invalidImageSource
        }
    }

    // MARK: - Domain ⇒ Wire (non-streaming)

    static func finalResponse(
        id: String,
        model: String,
        assistantText: String,
        toolCalls: [ToolUse],
        finishReason: FinishReason,
        usage: Usage?
    ) -> Anthropic.MessagesResponse {
        var content: [Anthropic.Block] = []
        if !assistantText.isEmpty { content.append(.text(assistantText)) }
        for tc in toolCalls {
            content.append(.toolUse(id: tc.id, name: tc.name, input: tc.input))
        }
        let usageOut = Anthropic.ResponseUsage(
            inputTokens: usage?.promptTokens ?? 0,
            outputTokens: usage?.completionTokens ?? 0)
        return Anthropic.MessagesResponse(
            id: id, model: model, content: content,
            stopReason: finishReason.anthropicWire,
            stopSequence: nil,
            usage: usageOut)
    }

    // MARK: - Domain ⇒ Wire (streaming)
    //
    // Anthropic's streaming choreography:
    //   1. message_start (entire shell with empty content)
    //   2. for the assistant text:
    //        content_block_start (index 0, text)
    //        content_block_delta (text_delta) ... N times
    //        content_block_stop  (index 0)
    //   3. for each tool_use:
    //        content_block_start (next index, tool_use)
    //        content_block_delta (input_json_delta) ... N times
    //        content_block_stop  (that index)
    //   4. message_delta (with stop_reason, output_tokens)
    //   5. message_stop
    //
    // Tracking this is enough state that we hold it in an iterator that
    // produces SSE frames lazily as upstream StreamEvents arrive.

    final class StreamState {
        var messageId: String
        var model: String
        var nextIndex = 0
        /// Index of the currently-open text block, if any.
        var openTextIndex: Int?
        /// Index of the currently-open tool_use block, if any.
        var openToolIndex: Int?
        var outputTokens = 0

        init(messageId: String, model: String) {
            self.messageId = messageId
            self.model = model
        }
    }

    /// One translated upstream event may produce several SSE frames (e.g. opening
    /// a new content block before emitting its first delta).
    struct Frame: Sendable, Equatable {
        var event: String
        var jsonData: String
    }

    static func startFrames(state: StreamState) throws -> [Frame] {
        let start = Anthropic.MessageStart(
            message: Anthropic.MessagesResponse(
                id: state.messageId, model: state.model,
                content: [], stopReason: nil, stopSequence: nil,
                usage: .init(inputTokens: 0, outputTokens: 0)))
        return [try frame(event: "message_start", payload: start)]
    }

    static func frames(for event: StreamEvent, state: StreamState) throws -> [Frame] {
        switch event {
        case .textDelta(let s):
            var frames: [Frame] = []
            if state.openTextIndex == nil {
                let idx = state.nextIndex; state.nextIndex += 1
                state.openTextIndex = idx
                let open = Anthropic.ContentBlockStart(
                    index: idx, contentBlock: .text(""))
                frames.append(try frame(event: "content_block_start", payload: open))
            }
            let delta = Anthropic.ContentBlockDelta(
                index: state.openTextIndex!, delta: .text(s))
            frames.append(try frame(event: "content_block_delta", payload: delta))
            return frames

        case .toolUseStart(let tid, let name):
            var frames: [Frame] = []
            if let textIdx = state.openTextIndex {
                frames.append(try frame(
                    event: "content_block_stop",
                    payload: Anthropic.ContentBlockStop(index: textIdx)))
                state.openTextIndex = nil
            }
            let idx = state.nextIndex; state.nextIndex += 1
            state.openToolIndex = idx
            let open = Anthropic.ContentBlockStart(
                index: idx,
                contentBlock: .toolUse(id: tid, name: name, input: .object([:])))
            frames.append(try frame(event: "content_block_start", payload: open))
            return frames

        case .toolUseInputDelta(let chunk):
            guard let idx = state.openToolIndex else { return [] }
            let delta = Anthropic.ContentBlockDelta(
                index: idx, delta: .inputJson(chunk))
            return [try frame(event: "content_block_delta", payload: delta)]

        case .toolUseStop:
            guard let idx = state.openToolIndex else { return [] }
            state.openToolIndex = nil
            return [try frame(
                event: "content_block_stop",
                payload: Anthropic.ContentBlockStop(index: idx))]

        case .done(let reason, let usage):
            var frames: [Frame] = []
            if let textIdx = state.openTextIndex {
                frames.append(try frame(
                    event: "content_block_stop",
                    payload: Anthropic.ContentBlockStop(index: textIdx)))
                state.openTextIndex = nil
            }
            if let toolIdx = state.openToolIndex {
                frames.append(try frame(
                    event: "content_block_stop",
                    payload: Anthropic.ContentBlockStop(index: toolIdx)))
                state.openToolIndex = nil
            }
            let md = Anthropic.MessageDelta(
                delta: .init(stopReason: reason.anthropicWire, stopSequence: nil),
                usage: .init(
                    inputTokens: usage?.promptTokens ?? 0,
                    outputTokens: usage?.completionTokens ?? state.outputTokens))
            frames.append(try frame(event: "message_delta", payload: md))
            frames.append(try frame(event: "message_stop", payload: Anthropic.MessageStop()))
            return frames
        }
    }

    static func pingFrame() throws -> Frame {
        struct Ping: Encodable { let type = "ping" }
        return try frame(event: "ping", payload: Ping())
    }

    private static func frame<T: Encodable>(event: String, payload: T) throws -> Frame {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        let data = try enc.encode(payload)
        return Frame(
            event: event,
            jsonData: String(data: data, encoding: .utf8) ?? "{}")
    }
}

extension FinishReason {
    var anthropicWire: String {
        switch self {
        case .stop: return "end_turn"
        case .length: return "max_tokens"
        case .toolCalls: return "tool_use"
        case .contentFilter: return "stop_sequence"
        }
    }
}
