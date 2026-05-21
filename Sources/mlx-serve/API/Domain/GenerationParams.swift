import Foundation

struct GenerationParams: Sendable, Equatable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var maxTokens: Int?
    var stopSequences: [String]
    var seed: UInt64?
    /// `true` ⇒ provider must emit text incrementally; `false` ⇒ caller wants the
    /// full final completion and translator can buffer.
    var stream: Bool

    static let `default` = GenerationParams(
        temperature: nil, topP: nil, topK: nil, maxTokens: nil,
        stopSequences: [], seed: nil, stream: false)
}
