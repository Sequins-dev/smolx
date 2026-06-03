import Foundation
import MLXLMCommon

/// Converts our domain `ChatMessage` history into the model-specific dictionary
/// format that MLX's `UserInput(messages:)` passes through to the model's
/// chat template. The dict shape matches what HuggingFace chat templates
/// expect at the `messages = [...]` level: each message is `[String: Any]`
/// with `role` + `content` + optional `tool_calls` / `tool_call_id`.
///
/// This is the proper fix for tool-history rendering. MLX's `Chat.Message`
/// is just `role + content` — no `tool_calls` field — which is why the
/// chat template's "tool message must follow assistant.tool_calls" sanity
/// check throws `TemplateException` on every multi-turn agent loop. By
/// going through `UserInput(messages:)` instead of `UserInput(chat:)` we
/// hand the template a structured tool_calls array exactly the way a
/// vanilla OpenAI client would.
enum PromptBuilder {

    /// Build the model-dict array for the chat template.
    static func messageDicts(from messages: [ChatMessage]) -> [MLXLMCommon.Message] {
        // Some models (e.g. Qwen3) require the system message to be first and
        // reject any system message that appears after position 0. Agentic
        // clients like Codex replay full conversation history on every request
        // and may inject system messages mid-sequence. Normalise by collecting
        // all system content into a single leading system message so the
        // template never sees a system message in the wrong position.
        var systemParts: [String] = []
        var rest: [MLXLMCommon.Message] = []

        for msg in messages {
            if msg.role == .system {
                let text = msg.content.compactMap {
                    if case .text(let t) = $0 { return t } else { return nil }
                }.joined(separator: "\n")
                if !text.isEmpty { systemParts.append(text) }
            } else {
                let d = dict(forMessage: msg)
                if !d.isEmpty { rest.append(d) }
            }
        }

        var out: [MLXLMCommon.Message] = []
        if !systemParts.isEmpty {
            out.append(["role": "system", "content": systemParts.joined(separator: "\n\n")])
        }
        out.append(contentsOf: rest)
        return out
    }

    private static func dict(forMessage msg: ChatMessage) -> [String: any Sendable] {
        var d: [String: any Sendable] = [:]
        d["role"] = role(msg.role)

        var text = ""
        var toolCalls: [[String: any Sendable]] = []
        var toolCallId: String?
        var isError = false

        for block in msg.content {
            switch block {
            case .text(let t):
                if !text.isEmpty { text += "\n" }
                text += t
            case .toolUse(let tu):
                // Standard OpenAI / Hugging Face shape — `function.arguments`
                // is a JSON-encoded *string*, not a parsed object, because
                // that's what every chat template that supports tools
                // (Qwen, Llama, gpt-oss, Mistral, …) expects.
                let argsString = encodeArgs(tu.input)
                toolCalls.append([
                    "id": tu.id,
                    "type": "function",
                    "function": [
                        "name": tu.name,
                        "arguments": argsString,
                    ] as [String: any Sendable],
                ])
            case .toolResult(let r):
                toolCallId = r.toolUseId
                if !text.isEmpty { text += "\n" }
                text += r.content
                isError = isError || r.isError
            case .image:
                // Images travel via `UserInput.images`, not through the
                // text content of the template message — `MessageGenerator`
                // is responsible for splicing image placeholders in.
                continue
            }
        }

        d["content"] = text
        if !toolCalls.isEmpty {
            d["tool_calls"] = toolCalls
        }
        if msg.role == .tool, let id = toolCallId {
            d["tool_call_id"] = id
        }
        if msg.role == .tool, isError {
            d["is_error"] = true
        }
        return d
    }

    private static func role(_ r: ChatMessage.Role) -> String {
        switch r {
        case .system: return "system"
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "tool"
        }
    }

    private static func encodeArgs(_ v: JSONValue) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        guard let data = try? enc.encode(v),
            let str = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return str
    }
}
