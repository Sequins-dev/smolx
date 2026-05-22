import Foundation

enum OpenAITranslator {

    struct DecodedRequest {
        var model: String
        var messages: [ChatMessage]
        var tools: [ToolDefinition]
        var toolChoice: ToolChoice?
        var params: GenerationParams
    }

    enum TranslationError: Error, CustomStringConvertible {
        case invalidRole(String)
        case unsupportedImageScheme(String)
        case invalidBase64Image

        var description: String {
            switch self {
            case .invalidRole(let r): return "Unknown message role: \(r)"
            case .unsupportedImageScheme(let s):
                return "Image URL scheme not supported: \(s). Use a data: URI."
            case .invalidBase64Image: return "Image data: URI payload was not valid base64."
            }
        }
    }

    // MARK: - Wire ⇒ Domain

    static func decode(_ req: OpenAI.ChatCompletionRequest) throws -> DecodedRequest {
        let messages = try req.messages.map { try toDomain($0) }
        let tools = (req.tools ?? []).map {
            ToolDefinition(
                name: $0.function.name,
                description: $0.function.description,
                inputSchema: $0.function.parameters ?? .object([:]))
        }
        let choice: ToolChoice? = req.toolChoice.flatMap { tc -> ToolChoice? in
            switch tc {
            case .string(let s):
                switch s {
                case "auto": return ToolChoice.auto
                case "none": return ToolChoice.none
                case "required": return ToolChoice.required
                default: return nil
                }
            case .named(let n): return ToolChoice.specific(name: n)
            }
        }
        let params = GenerationParams(
            temperature: req.temperature,
            topP: req.topP,
            topK: nil,
            maxTokens: req.maxTokens,
            stopSequences: req.stop?.asArray ?? [],
            seed: req.seed,
            stream: req.stream ?? false)
        return DecodedRequest(
            model: req.model,
            messages: messages,
            tools: tools,
            toolChoice: choice,
            params: params)
    }

    private static func toDomain(_ m: OpenAI.Message) throws -> ChatMessage {
        guard let role = ChatMessage.Role(rawValue: m.role) else {
            throw TranslationError.invalidRole(m.role)
        }
        var blocks: [ContentBlock] = []
        switch m.content {
        case .some(.text(let s)):
            if !s.isEmpty { blocks.append(.text(s)) }
        case .some(.parts(let parts)):
            for p in parts {
                switch p.type {
                case "text":
                    if let t = p.text { blocks.append(.text(t)) }
                case "image_url":
                    if let url = p.imageUrl?.url {
                        blocks.append(.image(try decodeImage(url)))
                    }
                default: break
                }
            }
        case .none: break
        }
        for tc in m.toolCalls ?? [] {
            let parsed = parseArguments(tc.function.arguments)
            blocks.append(.toolUse(.init(id: tc.id, name: tc.function.name, input: parsed)))
        }
        if role == .tool, let id = m.toolCallId, case .some(.text(let s)) = m.content {
            // Replace the bare text with a tool_result block keyed by id.
            blocks = [.toolResult(.init(toolUseId: id, content: s, isError: false))]
        }
        return ChatMessage(role: role, content: blocks)
    }

    private static func decodeImage(_ url: String) throws -> ImagePayload {
        if url.hasPrefix("data:") {
            // data:[<mediatype>][;base64],<data>
            guard let comma = url.firstIndex(of: ",") else {
                throw TranslationError.invalidBase64Image
            }
            let meta = url[url.index(url.startIndex, offsetBy: 5)..<comma]
            let payload = String(url[url.index(after: comma)...])
            let mime = meta.split(separator: ";").first.map(String.init) ?? "image/png"
            guard let data = Data(base64Encoded: payload) else {
                throw TranslationError.invalidBase64Image
            }
            return ImagePayload(data: data, mimeType: mime)
        }
        throw TranslationError.unsupportedImageScheme(
            URL(string: url)?.scheme ?? url)
    }

    private static func parseArguments(_ s: String) -> JSONValue {
        guard let data = s.data(using: .utf8),
            let value = try? JSONDecoder().decode(JSONValue.self, from: data)
        else { return .object([:]) }
        return value
    }

    // MARK: - Domain ⇒ Wire (non-streaming)

    static func finalResponse(
        id: String,
        model: String,
        assistantText: String,
        toolCalls: [ToolUse],
        finishReason: FinishReason,
        usage: Usage?
    ) -> OpenAI.ChatCompletionResponse {
        let calls: [OpenAI.ToolCall] = toolCalls.map {
            OpenAI.ToolCall(
                id: $0.id, type: "function",
                function: .init(name: $0.name, arguments: encodeJSON($0.input)))
        }
        let msg = OpenAI.ResponseMessage(
            role: "assistant",
            content: assistantText.isEmpty ? nil : assistantText,
            toolCalls: calls.isEmpty ? nil : calls)
        let choice = OpenAI.Choice(index: 0, message: msg, finishReason: finishReason.openAIWire)
        let usageOut = usage.map {
            OpenAI.ResponseUsage(
                promptTokens: $0.promptTokens,
                completionTokens: $0.completionTokens,
                totalTokens: $0.totalTokens)
        }
        return OpenAI.ChatCompletionResponse(
            id: id,
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [choice],
            usage: usageOut)
    }

    // MARK: - Domain ⇒ Wire (streaming)

    /// Produces the SSE-payload chunks for a stream of StreamEvents. Returns
    /// `nil` for events that don't map onto a chunk (e.g. internal markers).
    /// `state` tracks tool-call indices so each tool_use gets its own
    /// accumulating slot per OpenAI's wire convention.
    final class StreamState {
        var nextToolIndex = 0
        var activeToolIndex: Int?
    }

    static func chunkFor(
        event: StreamEvent,
        id: String,
        model: String,
        state: StreamState,
        isFirst: Bool
    ) -> OpenAI.ChatCompletionChunk? {
        let created = Int(Date().timeIntervalSince1970)
        switch event {
        case .textDelta(let s):
            var delta = OpenAI.Delta(role: isFirst ? "assistant" : nil, content: s, toolCalls: nil)
            if !isFirst { delta.role = nil }
            return OpenAI.ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [.init(index: 0, delta: delta, finishReason: nil)])
        case .toolUseStart(let tid, let name):
            let idx = state.nextToolIndex
            state.nextToolIndex += 1
            state.activeToolIndex = idx
            let tc = OpenAI.ChunkToolCall(
                index: idx, id: tid, type: "function",
                function: .init(name: name, arguments: ""))
            return OpenAI.ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [
                    .init(
                        index: 0,
                        delta: .init(role: isFirst ? "assistant" : nil, content: nil, toolCalls: [tc]),
                        finishReason: nil)
                ])
        case .toolUseInputDelta(let chunk):
            guard let idx = state.activeToolIndex else { return nil }
            let tc = OpenAI.ChunkToolCall(
                index: idx, id: nil, type: nil,
                function: .init(name: nil, arguments: chunk))
            return OpenAI.ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [
                    .init(
                        index: 0,
                        delta: .init(role: nil, content: nil, toolCalls: [tc]),
                        finishReason: nil)
                ])
        case .toolUseStop:
            state.activeToolIndex = nil
            return nil
        case .done(let reason, let usage):
            let usageOut = usage.map {
                OpenAI.ResponseUsage(
                    promptTokens: $0.promptTokens,
                    completionTokens: $0.completionTokens,
                    totalTokens: $0.totalTokens)
            }
            return OpenAI.ChatCompletionChunk(
                id: id, created: created, model: model,
                choices: [
                    .init(
                        index: 0,
                        delta: .init(role: nil, content: nil, toolCalls: nil),
                        finishReason: reason.openAIWire)
                ],
                usage: usageOut)
        }
    }

    private static func encodeJSON(_ v: JSONValue) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        let data = (try? enc.encode(v)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}

extension FinishReason {
    var openAIWire: String {
        switch self {
        case .stop: return "stop"
        case .length: return "length"
        case .toolCalls: return "tool_calls"
        case .contentFilter: return "content_filter"
        }
    }
}
