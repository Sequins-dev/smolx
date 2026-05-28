import Foundation
import Logging

/// Eight-character request ID stamped into every debug/trace log line for a request.
func reqId() -> String {
    String(UUID().uuidString.prefix(8))
}

/// One-liner summary of non-default generation params for `.debug` log lines.
func summarize(_ params: GenerationParams) -> String {
    var parts: [String] = []
    if let t = params.temperature { parts.append("temp=\(t)") }
    if let p = params.topP { parts.append("top_p=\(p)") }
    if let k = params.topK { parts.append("top_k=\(k)") }
    if let m = params.maxTokens { parts.append("max_tokens=\(m)") }
    if let s = params.seed { parts.append("seed=\(s)") }
    if !params.stopSequences.isEmpty { parts.append("stop_seqs=\(params.stopSequences.count)") }
    return parts.isEmpty ? "defaults" : parts.joined(separator: " ")
}

/// Full message-list dump for `.trace` log lines; clips each content block at 2 KB.
func summarize(_ messages: [ChatMessage]) -> String {
    messages.enumerated().map { (i, msg) in
        let blocks = msg.content.map { block -> String in
            switch block {
            case .text(let s):
                return "text:\(clip(s))"
            case .image(let p):
                return "image(\(p.mimeType) \(p.data.count)B)"
            case .toolUse(let t):
                return "toolUse(id=\(t.id) name=\(t.name))"
            case .toolResult(let r):
                return "toolResult(id=\(r.toolUseId) error=\(r.isError) \(clip(r.content)))"
            }
        }.joined(separator: " | ")
        return "  [\(i)] \(msg.role.rawValue): \(blocks)"
    }.joined(separator: "\n")
}

/// Short description of a `StreamEvent` for per-event `.trace` log lines.
func describe(_ event: StreamEvent) -> String {
    switch event {
    case .textDelta(let s):
        return "textDelta(\(clip(s, limit: 120)))"
    case .toolUseStart(let id, let name):
        return "toolUseStart(id=\(id) name=\(name))"
    case .toolUseInputDelta(let d):
        return "toolUseInputDelta(\(clip(d, limit: 120)))"
    case .toolUseStop:
        return "toolUseStop"
    case .done(let r, let u):
        if let u {
            return "done(reason=\(r.rawValue) prompt=\(u.promptTokens) completion=\(u.completionTokens))"
        }
        return "done(reason=\(r.rawValue))"
    }
}

private func clip(_ s: String, limit: Int = 2048) -> String {
    guard s.count > limit else { return s }
    return String(s.prefix(limit)) + "…"
}
