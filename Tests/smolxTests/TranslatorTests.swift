import Testing
import Foundation
@testable import smolx

@Suite("OpenAI translator")
struct OpenAITranslatorTests {
    @Test func decodeSimpleTextRequest() throws {
        let json = """
        {"model":"foo","messages":[{"role":"user","content":"hi"}]}
        """
        let req = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        let dec = try OpenAITranslator.decode(req)
        #expect(dec.model == "foo")
        #expect(dec.messages.count == 1)
        #expect(dec.messages[0].role == .user)
        #expect(dec.messages[0].content == [.text("hi")])
        #expect(dec.params.stream == false)
    }

    @Test func decodeMultiPartContentWithImage() throws {
        let pngB64 = Data([137, 80, 78, 71]).base64EncodedString()  // PNG magic
        let json = """
        {
          "model": "qwen-vl",
          "messages": [{
            "role": "user",
            "content": [
              {"type":"text","text":"what is here"},
              {"type":"image_url","image_url":{"url":"data:image/png;base64,\(pngB64)"}}
            ]
          }]
        }
        """
        let req = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        let dec = try OpenAITranslator.decode(req)
        #expect(dec.messages[0].content.count == 2)
        if case .image(let img) = dec.messages[0].content[1] {
            #expect(img.mimeType == "image/png")
            #expect(img.data.first == 137)
        } else {
            Issue.record("expected an image block")
        }
    }

    @Test func streamingChunkFraming() throws {
        let state = OpenAITranslator.StreamState()
        let first = OpenAITranslator.chunkFor(
            event: .textDelta("Hello"),
            id: "chatcmpl-1", model: "foo", state: state, isFirst: true)
        #expect(first?.choices.first?.delta.role == "assistant")
        #expect(first?.choices.first?.delta.content == "Hello")

        let mid = OpenAITranslator.chunkFor(
            event: .textDelta(" world"),
            id: "chatcmpl-1", model: "foo", state: state, isFirst: false)
        #expect(mid?.choices.first?.delta.role == nil)

        let end = OpenAITranslator.chunkFor(
            event: .done(finishReason: .stop, usage: nil),
            id: "chatcmpl-1", model: "foo", state: state, isFirst: false)
        #expect(end?.choices.first?.finishReason == "stop")
    }

    @Test func streamingToolCallAccumulator() throws {
        let state = OpenAITranslator.StreamState()
        _ = OpenAITranslator.chunkFor(
            event: .textDelta("ok"),
            id: "x", model: "m", state: state, isFirst: true)
        let start = OpenAITranslator.chunkFor(
            event: .toolUseStart(id: "call_1", name: "get_weather"),
            id: "x", model: "m", state: state, isFirst: false)
        let firstTc = start?.choices.first?.delta.toolCalls?.first
        #expect(firstTc?.index == 0)
        #expect(firstTc?.id == "call_1")
        #expect(firstTc?.function?.name == "get_weather")

        let mid = OpenAITranslator.chunkFor(
            event: .toolUseInputDelta("{\"loc\":\"SF\"}"),
            id: "x", model: "m", state: state, isFirst: false)
        #expect(mid?.choices.first?.delta.toolCalls?.first?.function?.arguments == "{\"loc\":\"SF\"}")
    }
}

@Suite("Anthropic translator")
struct AnthropicTranslatorTests {
    @Test func decodeWithSystemAndToolUse() throws {
        let json = """
        {
          "model": "claude-x",
          "max_tokens": 64,
          "system": "Be brief.",
          "messages": [
            {"role":"user","content":[{"type":"text","text":"hi"}]},
            {"role":"assistant","content":[
              {"type":"tool_use","id":"toolu_1","name":"add","input":{"a":1,"b":2}}
            ]},
            {"role":"user","content":[
              {"type":"tool_result","tool_use_id":"toolu_1","content":"3"}
            ]}
          ]
        }
        """
        let req = try JSONDecoder().decode(
            Anthropic.MessagesRequest.self, from: Data(json.utf8))
        let dec = try AnthropicTranslator.decode(req)
        #expect(dec.messages.count == 4)  // system + 3 originals
        #expect(dec.messages[0].role == .system)
        #expect(dec.messages[0].content == [.text("Be brief.")])
        if case .toolUse(let tu) = dec.messages[2].content[0] {
            #expect(tu.id == "toolu_1")
            #expect(tu.name == "add")
        } else {
            Issue.record("expected tool_use block")
        }
    }

    @Test func streamChoreography() throws {
        let state = AnthropicTranslator.StreamState(messageId: "msg_1", model: "m")
        let opening = try AnthropicTranslator.startFrames(state: state)
        #expect(opening.map(\.event) == ["message_start"])

        let textFrames = try AnthropicTranslator.frames(
            for: .textDelta("Hi"), state: state)
        #expect(textFrames.map(\.event) == ["content_block_start", "content_block_delta"])

        let toolFrames = try AnthropicTranslator.frames(
            for: .toolUseStart(id: "t1", name: "f"), state: state)
        #expect(toolFrames.map(\.event) == ["content_block_stop", "content_block_start"])

        let inputDelta = try AnthropicTranslator.frames(
            for: .toolUseInputDelta("{\"a\":1}"), state: state)
        #expect(inputDelta.map(\.event) == ["content_block_delta"])

        let toolStop = try AnthropicTranslator.frames(
            for: .toolUseStop, state: state)
        #expect(toolStop.map(\.event) == ["content_block_stop"])

        let done = try AnthropicTranslator.frames(
            for: .done(finishReason: .toolCalls, usage: nil), state: state)
        #expect(done.map(\.event) == ["message_delta", "message_stop"])
    }

    @Test func finishReasonMapping() {
        #expect(FinishReason.stop.anthropicWire == "end_turn")
        #expect(FinishReason.length.anthropicWire == "max_tokens")
        #expect(FinishReason.toolCalls.anthropicWire == "tool_use")
    }
}
