import Foundation
import Testing

@testable import smolx

@Suite("ProxyTarget")
struct ProxyTargetTests {
    @Test func lmStudioPromotesCodexAdditionalTools() throws {
        let body = Data(
            """
            {
              "model": "qwen3.8-27b-mlx",
              "input": [
                {
                  "type": "additional_tools",
                  "role": "developer",
                  "tools": [
                    {"type": "function", "name": "shell", "parameters": {"type": "object"}}
                  ]
                },
                {
                  "type": "message",
                  "role": "user",
                  "content": [{"type": "input_text", "text": "hello"}]
                }
              ],
              "stream": true,
              "unknown_codex_field": "preserved"
            }
            """.utf8)

        let result = try ProxyTarget.lmStudio.rewriteRequest(
            path: "/v1/responses", body: body)
        let json = try #require(
            JSONSerialization.jsonObject(with: result.body) as? [String: Any])
        let input = try #require(json["input"] as? [[String: Any]])
        let tools = try #require(json["tools"] as? [[String: Any]])

        #expect(result.translationCount == 1)
        #expect(input.count == 1)
        #expect(input.first?["type"] as? String == "message")
        #expect(tools.count == 1)
        #expect(tools.first?["name"] as? String == "shell")
        #expect(json["unknown_codex_field"] as? String == "preserved")
    }

    @Test func lmStudioAppendsPromotedToolsToExistingTools() throws {
        let body = Data(
            """
            {
              "model": "model",
              "tools": [{"type": "function", "name": "existing"}],
              "input": [
                {"type": "additional_tools", "tools": [{"type": "function", "name": "deferred"}]}
              ]
            }
            """.utf8)

        let result = try ProxyTarget.lmStudio.rewriteRequest(
            path: "/v1/responses", body: body)
        let json = try #require(
            JSONSerialization.jsonObject(with: result.body) as? [String: Any])
        let tools = try #require(json["tools"] as? [[String: Any]])

        #expect(tools.compactMap { $0["name"] as? String } == ["existing", "deferred"])
    }

    @Test func standardResponsesBodyPassesThroughByteForByte() throws {
        let body = Data(#"{"model":"model","input":"hello"}"#.utf8)

        let result = try ProxyTarget.lmStudio.rewriteRequest(
            path: "/v1/responses", body: body)

        #expect(result.translationCount == 0)
        #expect(result.body == body)
    }

    @Test func lmStudioAddsMissingTextFormatWithoutReplacingOtherTextSettings() throws {
        let body = Data(
            #"{"model":"model","input":"hello","text":{"verbosity":"medium"}}"#.utf8)

        let result = try ProxyTarget.lmStudio.rewriteRequest(
            path: "/v1/responses", body: body)
        let json = try #require(
            JSONSerialization.jsonObject(with: result.body) as? [String: Any])
        let text = try #require(json["text"] as? [String: Any])
        let format = try #require(text["format"] as? [String: Any])

        #expect(result.translationCount == 1)
        #expect(text["verbosity"] as? String == "medium")
        #expect(format["type"] as? String == "text")
    }

    @Test func nonResponsesRoutePassesThroughByteForByte() throws {
        let body = Data(#"{"input":[{"type":"additional_tools"}]}"#.utf8)

        let result = try ProxyTarget.lmStudio.rewriteRequest(
            path: "/v1/chat/completions", body: body)

        #expect(result.translationCount == 0)
        #expect(result.body == body)
    }

    @Test func malformedResponsesBodyIsRejectedLocally() {
        #expect(throws: ProxyTranslationError.self) {
            _ = try ProxyTarget.lmStudio.rewriteRequest(
                path: "/v1/responses", body: Data("[]".utf8))
        }
    }
}
