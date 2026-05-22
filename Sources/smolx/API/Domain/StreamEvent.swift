import Foundation

/// Provider-emitted incremental events. Translators map these onto either the
/// OpenAI `chat.completion.chunk` shape or the Anthropic SSE event sequence.
enum StreamEvent: Sendable, Equatable {
    /// A piece of assistant text.
    case textDelta(String)

    /// The model has decided to call a tool; subsequent `.toolUseInputDelta`
    /// events accumulate that tool's JSON arguments until `.toolUseStop`.
    case toolUseStart(id: String, name: String)
    case toolUseInputDelta(String)
    case toolUseStop

    /// Final marker. `finishReason` mirrors the OpenAI vocabulary and is mapped
    /// to Anthropic's `stop_reason` by that translator.
    case done(finishReason: FinishReason, usage: Usage?)
}

enum FinishReason: String, Sendable, Equatable {
    case stop  // natural end / stop sequence hit
    case length  // hit max_tokens
    case toolCalls  // model emitted tool_use
    case contentFilter  // safety stop
}

struct Usage: Sendable, Equatable {
    var promptTokens: Int
    var completionTokens: Int
    var totalTokens: Int { promptTokens + completionTokens }
}
