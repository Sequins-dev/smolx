import Foundation

enum GGUFPromptWindow {
    static let defaultMaxTokens = 512
    static let hardMaxTokens = 512
    private static let smallContextCompletionDivisor = 4

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
        let defaultBudget = min(defaultMaxTokens, max(1, window / smallContextCompletionDivisor))
        let requested = requestedMaxTokens ?? defaultBudget
        let cappedRequest =
            requested >= window
            ? defaultBudget
            : min(requested, hardMaxTokens)
        var maxTokens = min(max(1, cappedRequest), max(1, window - 1))
        if promptTokens > 0, promptTokens < window, promptTokens + maxTokens > window {
            maxTokens = max(1, window - promptTokens)
        }
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
