import ArgumentParser
import Foundation

enum ProxyTarget: String, CaseIterable, ExpressibleByArgument, Sendable {
    case lmStudio = "lm-studio"

    var abstract: String {
        switch self {
        case .lmStudio:
            return "LM Studio's OpenAI-compatible server"
        }
    }

    func rewriteRequest(path: String, body: Data) throws -> ProxyRewriteResult {
        switch self {
        case .lmStudio:
            return try LMStudioProxyTranslator.rewriteRequest(path: path, body: body)
        }
    }
}

struct ProxyRewriteResult: Sendable {
    var body: Data
    var translationCount: Int
}

enum ProxyTranslationError: Error, CustomStringConvertible {
    case invalidResponsesBody

    var description: String {
        switch self {
        case .invalidResponsesBody:
            return "The /v1/responses request body must be a JSON object."
        }
    }
}

enum LMStudioProxyTranslator {
    /// Codex can place deferred tool definitions in an `additional_tools`
    /// input item. That item is outside the public Responses API input union
    /// and LM Studio rejects the entire request. Promote those definitions to
    /// the standard top-level `tools` array, remove the pseudo-item, and leave
    /// every other request field untouched.
    static func rewriteRequest(path: String, body: Data) throws -> ProxyRewriteResult {
        guard path == "/v1/responses" else {
            return ProxyRewriteResult(body: body, translationCount: 0)
        }

        guard
            let object = try? JSONSerialization.jsonObject(with: body),
            var request = object as? [String: Any]
        else {
            throw ProxyTranslationError.invalidResponsesBody
        }

        var translationCount = 0

        if let input = request["input"] as? [Any] {
            var retainedInput: [Any] = []
            var promotedTools = (request["tools"] as? [Any]) ?? []

            for item in input {
                guard
                    let dictionary = item as? [String: Any],
                    dictionary["type"] as? String == "additional_tools"
                else {
                    retainedInput.append(item)
                    continue
                }

                translationCount += 1
                if let tools = dictionary["tools"] as? [Any] {
                    promotedTools.append(contentsOf: tools)
                }
            }

            if retainedInput.count != input.count {
                request["input"] = retainedInput
                if !promotedTools.isEmpty {
                    request["tools"] = promotedTools
                }
            }
        }

        // Codex includes a `text` configuration object, but does not always
        // include its optional public-API `format` field. LM Studio treats the
        // field as required whenever `text` is present, so supply the ordinary
        // text default without changing any other text settings.
        if var text = request["text"] as? [String: Any], text["format"] == nil {
            text["format"] = ["type": "text"]
            request["text"] = text
            translationCount += 1
        }

        guard translationCount > 0 else {
            return ProxyRewriteResult(body: body, translationCount: 0)
        }

        let rewritten = try JSONSerialization.data(
            withJSONObject: request,
            options: [.withoutEscapingSlashes])
        return ProxyRewriteResult(
            body: rewritten,
            translationCount: translationCount)
    }
}
