import Foundation

enum GGUFPromptWindow {
    static let defaultMaxTokens = 256

    struct Adjustment: Equatable, Sendable {
        var promptTokenLimit: Int
        var maxTokens: Int
        var shouldTrim: Bool
    }

    static func adjustment(
        promptTokens: Int,
        contextWindow: Int,
        requestedMaxTokens: Int?
    ) -> Adjustment {
        let window = max(1, contextWindow)
        let requested = requestedMaxTokens ?? defaultMaxTokens
        let cappedRequest =
            requested >= window
            ? defaultMaxTokens
            : requested
        let maxTokens = min(max(1, cappedRequest), max(1, window - 1))
        let promptLimit = max(1, window - maxTokens)

        return Adjustment(
            promptTokenLimit: min(promptTokens, promptLimit),
            maxTokens: maxTokens,
            shouldTrim: promptTokens > promptLimit)
    }

    static func compactMessagesForOversizedPrompt(_ messages: [ChatMessage]) -> [ChatMessage] {
        if let latestUser = messages.last(where: { $0.role == .user }) {
            return [latestUser]
        }
        if let last = messages.last {
            return [last]
        }
        return []
    }
}
