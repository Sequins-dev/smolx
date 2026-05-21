import Testing
import Foundation
@testable import mlx_serve

@Suite("HarmonyParser")
struct HarmonyParserTests {

    /// Feeds the full string in one chunk, then flushes. Matches what a
    /// post-mortem reconstruction of the stream sees.
    private func parseFull(_ s: String) -> [HarmonyParser.Event] {
        let p = HarmonyParser()
        let events = p.feed(s) + p.flush()
        return events
    }

    /// Feeds one character at a time to exercise the partial-marker
    /// safety margin. Should produce equivalent semantics to a single feed.
    private func parseCharByChar(_ s: String) -> [HarmonyParser.Event] {
        let p = HarmonyParser()
        var events: [HarmonyParser.Event] = []
        for ch in s {
            events.append(contentsOf: p.feed(String(ch)))
        }
        events.append(contentsOf: p.flush())
        return events
    }

    @Test func analysisChannelIsSuppressed() {
        let input = "<|channel|>analysis<|message|>thinking out loud<|end|>"
        #expect(parseFull(input).isEmpty)
    }

    @Test func finalChannelBecomesTextDelta() {
        let input = "<|channel|>final<|message|>hello world<|return|>"
        let events = parseFull(input)
        #expect(events == [.textDelta("hello world")])
    }

    @Test func analysisThenFinalEmitsOnlyFinal() {
        let input = """
            <|channel|>analysis<|message|>let me think<|end|>\
            <|start|>assistant<|channel|>final<|message|>here's the answer<|return|>
            """
        let events = parseFull(input)
        #expect(events == [.textDelta("here's the answer")])
    }

    @Test func commentaryWithToolEmitsToolUseSequence() {
        let input = """
            <|channel|>commentary to=functions.list_dir<|constrain|>json<|message|>\
            {"path":"."}<|end|>
            """
        let events = parseFull(input)
        guard events.count == 3 else {
            Issue.record("expected 3 events, got \(events.count): \(events)")
            return
        }
        if case .toolUseStart(_, let name) = events[0] {
            #expect(name == "list_dir")
        } else {
            Issue.record("first event should be toolUseStart, got \(events[0])")
        }
        #expect(events[1] == .toolUseInputDelta(#"{"path":"."}"#))
        #expect(events[2] == .toolUseStop)
    }

    @Test func commentaryWithoutToolIsSuppressed() {
        // Plain commentary without `to=functions.X` is inter-message notes —
        // not visible output, not a tool call. Drop it.
        let input = "<|channel|>commentary<|message|>a note<|end|>"
        #expect(parseFull(input).isEmpty)
    }

    @Test func toolCallAndFinalInSameStream() {
        let input = """
            <|channel|>analysis<|message|>plan...<|end|>\
            <|start|>assistant<|channel|>commentary to=functions.bash<|message|>{"cmd":"ls"}<|end|>\
            <|start|>assistant<|channel|>final<|message|>done.<|return|>
            """
        let events = parseFull(input)
        let kinds: [String] = events.map {
            switch $0 {
            case .textDelta: return "text"
            case .toolUseStart: return "toolStart"
            case .toolUseInputDelta: return "toolDelta"
            case .toolUseStop: return "toolStop"
            }
        }
        #expect(kinds == ["toolStart", "toolDelta", "toolStop", "text"])
    }

    @Test func streamingCharByCharMatchesSingleFeed() {
        // The whole point of the parser is that it survives being fed at
        // arbitrary chunk boundaries. Worst case is one char at a time.
        let input = """
            <|channel|>final<|message|>line one
            line two<|return|>
            """
        let full = parseFull(input)
        let stepped = parseCharByChar(input)
        // Char-by-char may split textDelta across multiple events, but
        // concatenating them must equal the single-feed text.
        let fullText = full.compactMap {
            if case .textDelta(let t) = $0 { return t } else { return nil }
        }.joined()
        let steppedText = stepped.compactMap {
            if case .textDelta(let t) = $0 { return t } else { return nil }
        }.joined()
        #expect(fullText == steppedText)
        #expect(fullText.contains("line one"))
        #expect(fullText.contains("line two"))
    }

    @Test func toolNameWithSurroundingDirectives() {
        // Real headers can have `<|constrain|>json` between the recipient
        // and `<|message|>`. The tool name extractor must not consume the
        // constraint as part of the name.
        let input = """
            <|channel|>commentary to=functions.read_file<|constrain|>json<|message|>\
            {"path":"x.txt"}<|end|>
            """
        let events = parseFull(input)
        if case .toolUseStart(_, let name) = events.first {
            #expect(name == "read_file")
        } else {
            Issue.record("expected toolUseStart with name read_file")
        }
    }

    @Test func unterminatedFinalChannelFlushesResidue() {
        // If the upstream stream ends without `<|end|>` (e.g. max_tokens
        // hit mid-sentence), the combined feed+flush output should reconstruct
        // the user-visible text. feed() emits everything except a safety-margin
        // tail; flush() emits that tail.
        let p = HarmonyParser()
        let fed = p.feed("<|channel|>final<|message|>partial answer")
        let flushed = p.flush()
        let text = (fed + flushed).compactMap {
            if case .textDelta(let t) = $0 { return t } else { return nil }
        }.joined()
        #expect(text == "partial answer")
    }

    @Test func unterminatedToolEmitsStop() {
        // Same idea for tool calls — flush must close the block AND drain
        // residual JSON. Dropping the safety-margin tail surfaced as
        // "Unterminated string" parse errors in agent clients because the
        // closing `}` and the last few key/value pairs vanished.
        let p = HarmonyParser()
        let fed = p.feed("<|channel|>commentary to=functions.foo<|message|>{\"a\":1,\"b\":2}")
        let flushed = p.flush()
        let combined = fed + flushed
        #expect(combined.last == .toolUseStop)
        let json = combined.compactMap {
            if case .toolUseInputDelta(let s) = $0 { return s } else { return nil }
        }.joined()
        #expect(json == "{\"a\":1,\"b\":2}")
    }

    @Test func toolCallTerminatedByCallMarker() {
        // gpt-oss uses `<|call|>` (not `<|end|>`) to terminate function-call
        // commentary. The parser must accept it as a terminator or the JSON
        // never closes and the safety-margin tail gets dropped on flush.
        let input = """
            <|channel|>commentary to=functions.list_dir<|constrain|>json<|message|>\
            {"path":"./"}<|call|>
            """
        let events = parseFull(input)
        guard events.count == 3 else {
            Issue.record("expected 3 events, got \(events.count): \(events)")
            return
        }
        #expect(events[1] == .toolUseInputDelta(#"{"path":"./"}"#))
        #expect(events[2] == .toolUseStop)
    }

    @Test func toolCallCharByCharWithCallMarker() {
        // Verify that the char-by-char streaming case (the realistic one,
        // since tokens arrive one at a time) doesn't truncate JSON args
        // when `<|call|>` is the terminator. This is the actual user-visible
        // bug: codex reported `{"path":"./",".` — the last ~13 chars
        // ("ecursive":true}<|call|>) were dropped because flush() bailed
        // out without draining the buffer.
        let input = """
            <|channel|>commentary to=functions.list_dir<|message|>\
            {"path":"./","recursive":true}<|call|>
            """
        let events = parseCharByChar(input)
        let json = events.compactMap {
            if case .toolUseInputDelta(let s) = $0 { return s } else { return nil }
        }.joined()
        #expect(json == #"{"path":"./","recursive":true}"#,
                "JSON was truncated to \(json.debugDescription)")
        #expect(events.last == .toolUseStop)
    }
}
