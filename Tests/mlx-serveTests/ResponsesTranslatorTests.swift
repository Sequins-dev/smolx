import Testing
import Foundation
@testable import mlx_serve

@Suite("Responses translator")
struct ResponsesTranslatorTests {

    // MARK: - Request decoding

    @Test func decodeStringInputBecomesUserMessage() throws {
        let json = #"""
        {"model":"foo","input":"hello world"}
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        #expect(dec.model == "foo")
        #expect(dec.messages.count == 1)
        #expect(dec.messages[0].role == .user)
        #expect(dec.messages[0].content == [.text("hello world")])
        #expect(dec.params.stream == false)
        #expect(dec.tools.isEmpty)
    }

    @Test func decodeInstructionsBecomesSystemMessage() throws {
        let json = #"""
        {"model":"foo","instructions":"You are terse.","input":"hi"}
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        #expect(dec.messages.count == 2)
        #expect(dec.messages[0].role == .system)
        #expect(dec.messages[0].content == [.text("You are terse.")])
        #expect(dec.messages[1].role == .user)
    }

    @Test func decodeArrayInputWithMessageItems() throws {
        let json = #"""
        {
          "model": "foo",
          "input": [
            {"type":"message","role":"user","content":[{"type":"input_text","text":"first"}]},
            {"type":"message","role":"assistant","content":[{"type":"input_text","text":"reply"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"second"}]}
          ]
        }
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        #expect(dec.messages.count == 3)
        #expect(dec.messages.map(\.role) == [.user, .assistant, .user])
        #expect(dec.messages[2].content == [.text("second")])
    }

    @Test func decodeArrayInputWithFunctionCallAndOutput() throws {
        // Round-trip a prior tool invocation back into the conversation —
        // function_call becomes assistant.tool_use, function_call_output
        // becomes a tool-role message. This is what codex sends back when
        // resuming a multi-turn agent loop.
        let json = #"""
        {
          "model": "foo",
          "input": [
            {"type":"message","role":"user","content":[{"type":"input_text","text":"ls"}]},
            {"type":"function_call","call_id":"call_abc","name":"shell","arguments":"{\"cmd\":\"ls\"}"},
            {"type":"function_call_output","call_id":"call_abc","output":"a.txt\nb.txt"}
          ]
        }
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        #expect(dec.messages.count == 3)
        // [0] user; [1] assistant with tool_use; [2] tool with tool_result.
        #expect(dec.messages[1].role == .assistant)
        guard case .toolUse(let tu) = dec.messages[1].content.first else {
            Issue.record("expected tool_use block, got \(dec.messages[1].content)")
            return
        }
        #expect(tu.id == "call_abc")
        #expect(tu.name == "shell")
        if case .object(let dict) = tu.input, case .string(let s) = dict["cmd"] {
            #expect(s == "ls")
        } else {
            Issue.record("args didn't decode: \(tu.input)")
        }
        #expect(dec.messages[2].role == .tool)
        guard case .toolResult(let r) = dec.messages[2].content.first else {
            Issue.record("expected tool_result block")
            return
        }
        #expect(r.toolUseId == "call_abc")
        #expect(r.content == "a.txt\nb.txt")
    }

    @Test func decodeFunctionCallWithMissingFieldsFallsThrough() throws {
        // Codex occasionally streams partial items mid-replay. The decoder
        // must NOT fail the whole request when call_id and name are absent;
        // the item is silently dropped (becomes `.other`).
        let json = #"""
        {
          "model":"foo",
          "input":[
            {"type":"function_call"},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"x"}]}
          ]
        }
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        // The malformed function_call item is filtered; user message survives.
        #expect(dec.messages.count == 1)
        #expect(dec.messages[0].role == .user)
    }

    @Test func decodeFiltersNamelessBuiltinTools() throws {
        // Codex's built-in tools (`local_shell`, `web_search`) come in
        // without a name — translation must drop them so the model only
        // sees real function tools.
        let json = #"""
        {
          "model":"foo",
          "input":"hi",
          "tools":[
            {"type":"local_shell"},
            {"type":"function","name":"read_file","description":"read","parameters":{"type":"object"}}
          ]
        }
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        #expect(dec.tools.count == 1)
        #expect(dec.tools[0].name == "read_file")
    }

    @Test func decodeStreamFlagAndGenerationParams() throws {
        let json = #"""
        {"model":"foo","input":"hi","stream":true,"temperature":0.3,"top_p":0.9,"max_output_tokens":128}
        """#
        let req = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(json.utf8))
        let dec = try ResponsesTranslator.decode(req)
        #expect(dec.params.stream == true)
        #expect(dec.params.temperature == 0.3)
        #expect(dec.params.topP == 0.9)
        #expect(dec.params.maxTokens == 128)
    }

    @Test func decodeToolChoiceVariants() throws {
        // Three accepted forms: "auto", "required", and {type:function,name:X}.
        let autoJSON = #"{"model":"f","input":"x","tool_choice":"auto"}"#
        let reqAuto = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(autoJSON.utf8))
        if case .auto = try ResponsesTranslator.decode(reqAuto).toolChoice {} else {
            Issue.record("expected auto")
        }

        let specificJSON = #"{"model":"f","input":"x","tool_choice":{"type":"function","name":"read_file"}}"#
        let reqSpec = try JSONDecoder().decode(
            OpenAIResponses.Request.self, from: Data(specificJSON.utf8))
        if case .specific(let n) = try ResponsesTranslator.decode(reqSpec).toolChoice {
            #expect(n == "read_file")
        } else {
            Issue.record("expected specific")
        }
    }

    // MARK: - Stream-event sequence (domain → wire)

    /// Helper: run a sequence of domain events through the translator,
    /// collecting all emitted frame event-names so the call-site can assert
    /// the high-level lifecycle.
    private func runStream(
        _ events: [StreamEvent],
        responseId: String = "resp_test",
        model: String = "m"
    ) throws -> (frames: [ResponsesTranslator.Frame], state: ResponsesTranslator.StreamState) {
        let state = ResponsesTranslator.StreamState(responseId: responseId, model: model)
        var out: [ResponsesTranslator.Frame] = []
        out.append(contentsOf: try ResponsesTranslator.startFrames(state: state))
        for e in events {
            out.append(contentsOf: try ResponsesTranslator.frames(for: e, state: state))
        }
        return (out, state)
    }

    @Test func textOnlyStreamHasExpectedEventOrder() throws {
        let (frames, _) = try runStream([
            .textDelta("Hello"),
            .textDelta(", "),
            .textDelta("world."),
            .done(finishReason: .stop, usage: nil),
        ])
        let names = frames.map(\.event)
        // Lifecycle: created → output_item.added (the message) →
        // output_text.delta ×3 → output_item.done → response.completed
        #expect(names == [
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_item.done",
            "response.completed",
        ])
    }

    @Test func textOnlyStreamSequenceNumbersAreMonotonic() throws {
        struct Envelope: Decodable { let sequence_number: Int }
        let (frames, _) = try runStream([
            .textDelta("a"),
            .textDelta("b"),
            .done(finishReason: .stop, usage: nil),
        ])
        let seqs: [Int] = frames.map {
            let env = try! JSONDecoder().decode(
                Envelope.self, from: Data($0.jsonData.utf8))
            return env.sequence_number
        }
        // Each event must increase the sequence_number by 1.
        #expect(seqs == Array(1...seqs.count))
    }

    @Test func toolCallOnlyStreamEmitsFunctionCallLifecycle() throws {
        let (frames, state) = try runStream([
            .toolUseStart(id: "call_1", name: "shell"),
            .toolUseInputDelta(#"{"cmd":"#),
            .toolUseInputDelta(#""ls"}"#),
            .toolUseStop,
            .done(finishReason: .toolCalls, usage: nil),
        ])
        let names = frames.map(\.event)
        #expect(names == [
            "response.created",
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.delta",
            "response.output_item.done",
            "response.completed",
        ])

        // The completed output items must include the assembled function_call
        // with the FULL concatenated arguments string — that's what codex
        // reads off `response.completed` if it missed the streamed deltas.
        guard case .functionCall(_, let callId, let name, let args, let status)
                = state.completedOutput.first
        else {
            Issue.record("expected function_call as first completed item")
            return
        }
        #expect(callId == "call_1")
        #expect(name == "shell")
        #expect(args == #"{"cmd":"ls"}"#)
        #expect(status == "completed")
    }

    @Test func mixedTextThenToolClosesTextItemBeforeOpeningTool() throws {
        // Real gpt-oss output sometimes streams a sentence and THEN emits
        // a tool call. The message item must be closed (output_item.done)
        // before the function_call item opens.
        let (frames, state) = try runStream([
            .textDelta("Let me check."),
            .toolUseStart(id: "c1", name: "shell"),
            .toolUseInputDelta(#"{"cmd":"ls"}"#),
            .toolUseStop,
            .done(finishReason: .toolCalls, usage: nil),
        ])
        let names = frames.map(\.event)
        #expect(names == [
            "response.created",
            "response.output_item.added",          // message
            "response.output_text.delta",
            "response.output_item.done",           // message closes here
            "response.output_item.added",          // function_call
            "response.custom_tool_call_input.delta",
            "response.output_item.done",           // function_call closes
            "response.completed",
        ])
        // Two completed items: message then function_call, in that order.
        #expect(state.completedOutput.count == 2)
        guard case .message = state.completedOutput[0],
              case .functionCall = state.completedOutput[1]
        else {
            Issue.record("wrong order of completed output items")
            return
        }
    }

    @Test func toolThenTextClosesToolBeforeOpeningTextItem() throws {
        // Symmetric: a tool call followed by text. The tool must close,
        // then a fresh message item opens for the text.
        let (frames, state) = try runStream([
            .toolUseStart(id: "c1", name: "shell"),
            .toolUseInputDelta(#"{"cmd":"pwd"}"#),
            .toolUseStop,
            .textDelta("Done."),
            .done(finishReason: .stop, usage: nil),
        ])
        let names = frames.map(\.event)
        #expect(names == [
            "response.created",
            "response.output_item.added",          // function_call
            "response.custom_tool_call_input.delta",
            "response.output_item.done",           // function_call closes
            "response.output_item.added",          // message
            "response.output_text.delta",
            "response.output_item.done",           // message closes
            "response.completed",
        ])
        #expect(state.completedOutput.count == 2)
        guard case .functionCall = state.completedOutput[0],
              case .message = state.completedOutput[1]
        else {
            Issue.record("wrong order of completed output items")
            return
        }
    }

    @Test func completedFrameJSONRoundTripsThroughDecoder() throws {
        // The final `response.completed` event JSON must round-trip through
        // the same Codable shapes we use for non-streaming responses — codex
        // parses the embedded `response` object identically in both paths.
        let (frames, _) = try runStream([
            .textDelta("ok"),
            .done(finishReason: .stop, usage: Usage(promptTokens: 5, completionTokens: 1)),
        ])
        guard let completed = frames.last(where: { $0.event == "response.completed" }) else {
            Issue.record("no completed frame")
            return
        }
        struct Envelope: Decodable {
            let type: String
            let response: OpenAIResponses.Response
            let sequence_number: Int
        }
        let env = try JSONDecoder().decode(
            Envelope.self, from: Data(completed.jsonData.utf8))
        #expect(env.type == "response.completed")
        #expect(env.response.status == "completed")
        #expect(env.response.usage?.totalTokens == 6)
        // One message item should be in the final output.
        #expect(env.response.output.count == 1)
        guard case .message(_, let role, let content, _) = env.response.output.first else {
            Issue.record("expected message")
            return
        }
        #expect(role == "assistant")
        #expect(content.first?.text == "ok")
    }

    @Test func customToolCallInputDeltaCarriesCallIdAndDelta() throws {
        // The custom_tool_call_input.delta event must reference the same
        // call_id we got in toolUseStart — codex matches deltas to their
        // function_call item via this id, NOT via the output_index.
        let (frames, _) = try runStream([
            .toolUseStart(id: "the_call_id", name: "shell"),
            .toolUseInputDelta(#"{"x":1}"#),
            .toolUseStop,
            .done(finishReason: .toolCalls, usage: nil),
        ])
        guard let deltaFrame = frames.first(where: {
            $0.event == "response.custom_tool_call_input.delta"
        }) else {
            Issue.record("no delta frame")
            return
        }
        struct DeltaEvent: Decodable {
            let type: String
            let item_id: String
            let call_id: String
            let output_index: Int
            let delta: String
        }
        let event = try JSONDecoder().decode(
            DeltaEvent.self, from: Data(deltaFrame.jsonData.utf8))
        #expect(event.call_id == "the_call_id")
        #expect(event.delta == #"{"x":1}"#)
        #expect(event.output_index == 0)
    }
}
