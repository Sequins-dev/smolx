import Testing
import Foundation
@testable import mlx_serve

@Suite("PromptBuilder")
struct PromptBuilderTests {

    @Test func emitsRoleAndTextForBasicMessages() {
        let messages: [ChatMessage] = [
            ChatMessage(role: .system, text: "You are helpful."),
            ChatMessage(role: .user, text: "Hello"),
            ChatMessage(role: .assistant, text: "Hi there"),
        ]
        let dicts = PromptBuilder.messageDicts(from: messages)
        #expect(dicts.count == 3)
        #expect(dicts[0]["role"] as? String == "system")
        #expect(dicts[0]["content"] as? String == "You are helpful.")
        #expect(dicts[1]["role"] as? String == "user")
        #expect(dicts[2]["role"] as? String == "assistant")
        #expect(dicts[2]["content"] as? String == "Hi there")
        // Plain messages have no tool_calls / tool_call_id keys.
        #expect(dicts[0]["tool_calls"] == nil)
        #expect(dicts[2]["tool_calls"] == nil)
    }

    @Test func assistantWithToolCallProducesStructuredCalls() {
        let toolUse = ToolUse(id: "call_abc", name: "get_weather",
                              input: .object(["location": .string("SF")]))
        let messages: [ChatMessage] = [
            ChatMessage(role: .user, text: "weather?"),
            ChatMessage(role: .assistant, content: [.toolUse(toolUse)]),
        ]
        let dicts = PromptBuilder.messageDicts(from: messages)
        #expect(dicts.count == 2)
        let assistant = dicts[1]
        #expect(assistant["role"] as? String == "assistant")
        // Content is an empty string when the assistant only emitted a tool
        // call — every HF chat template tolerates an empty `content` key
        // alongside `tool_calls`.
        #expect(assistant["content"] as? String == "")
        let calls = assistant["tool_calls"] as? [[String: any Sendable]]
        #expect(calls != nil)
        #expect(calls?.count == 1)
        let call = calls?.first
        #expect(call?["id"] as? String == "call_abc")
        #expect(call?["type"] as? String == "function")
        let fn = call?["function"] as? [String: any Sendable]
        #expect(fn?["name"] as? String == "get_weather")
        // Arguments are a JSON-string per OpenAI / HF convention, not parsed.
        let args = fn?["arguments"] as? String
        #expect(args != nil)
        #expect(args?.contains("\"location\"") == true)
        #expect(args?.contains("\"SF\"") == true)
    }

    @Test func toolMessageHasToolCallIdAndContent() {
        let messages: [ChatMessage] = [
            ChatMessage(role: .tool, content: [
                .toolResult(.init(toolUseId: "call_abc", content: "sunny", isError: false))
            ]),
        ]
        let dicts = PromptBuilder.messageDicts(from: messages)
        let tool = dicts[0]
        #expect(tool["role"] as? String == "tool")
        #expect(tool["tool_call_id"] as? String == "call_abc")
        #expect(tool["content"] as? String == "sunny")
        #expect(tool["is_error"] == nil)
    }

    @Test func toolErrorMarksMessage() {
        let messages: [ChatMessage] = [
            ChatMessage(role: .tool, content: [
                .toolResult(.init(toolUseId: "call_1", content: "fail", isError: true))
            ]),
        ]
        let dict = PromptBuilder.messageDicts(from: messages)[0]
        #expect(dict["is_error"] as? Bool == true)
    }

    @Test func mixedTextAndToolUseInOneAssistantMessage() {
        // The model sometimes emits a sentence + a tool call in the same turn
        // ("Let me check the weather. <call>"). Both should render — text in
        // `content`, call in `tool_calls`.
        let toolUse = ToolUse(id: "id_1", name: "f", input: .object([:]))
        let messages: [ChatMessage] = [
            ChatMessage(role: .assistant, content: [
                .text("Let me check."),
                .toolUse(toolUse),
            ]),
        ]
        let dict = PromptBuilder.messageDicts(from: messages)[0]
        #expect(dict["content"] as? String == "Let me check.")
        let calls = dict["tool_calls"] as? [[String: any Sendable]]
        #expect(calls?.count == 1)
    }
}
