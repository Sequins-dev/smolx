import Testing

@testable import smolx

@Suite("GGUFPromptWindow")
struct GGUFPromptWindowTests {
    @Test func leavesShortPromptUnchanged() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 128,
            contextWindow: 1024,
            requestedMaxTokens: 64)

        #expect(adjustment.promptTokenLimit == 128)
        #expect(adjustment.maxTokens == 64)
        #expect(adjustment.shouldTrim == false)
    }

    @Test func reservesRequestedCompletionBudget() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 1400,
            contextWindow: 1024,
            requestedMaxTokens: 64)

        #expect(adjustment.promptTokenLimit == 960)
        #expect(adjustment.maxTokens == 64)
        #expect(adjustment.shouldTrim == true)
    }

    @Test func capsOversizedCompletionBudget() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 1400,
            contextWindow: 1024,
            requestedMaxTokens: 2048)

        #expect(adjustment.promptTokenLimit == 768)
        #expect(adjustment.maxTokens == 256)
        #expect(adjustment.shouldTrim == true)
    }

    @Test func usesSmallDefaultCompletionBudgetWhenRequestOmitsMaxTokens() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 1400,
            contextWindow: 1024,
            requestedMaxTokens: nil)

        #expect(adjustment.promptTokenLimit == 768)
        #expect(adjustment.maxTokens == 256)
        #expect(adjustment.shouldTrim == true)
    }

    @Test func usesLargeDefaultCompletionBudgetForLargeContext() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 8000,
            contextWindow: 262144,
            requestedMaxTokens: nil)

        #expect(adjustment.promptTokenLimit == 8000)
        #expect(adjustment.maxTokens == 512)
        #expect(adjustment.shouldTrim == false)
    }

    @Test func reservesLargeRequestedCompletionBudgetForLargeContext() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 260000,
            contextWindow: 262144,
            requestedMaxTokens: 8192)

        #expect(adjustment.promptTokenLimit == 260000)
        #expect(adjustment.maxTokens == 512)
        #expect(adjustment.shouldTrim == false)
    }

    @Test func capsOversizedCompletionBudgetBeforeTrimmingSmallPrompt() {
        let adjustment = GGUFPromptWindow.adjustment(
            promptTokens: 15,
            contextWindow: 262144,
            requestedMaxTokens: 262143)

        #expect(adjustment.promptTokenLimit == 15)
        #expect(adjustment.maxTokens == 512)
        #expect(adjustment.shouldTrim == false)
    }

    @Test func oversizedPromptFallbackKeepsLatestUserMessageOnly() {
        let messages = [
            ChatMessage(role: .system, text: String(repeating: "policy ", count: 500)),
            ChatMessage(role: .user, text: "old request"),
            ChatMessage(role: .assistant, text: "old answer"),
            ChatMessage(role: .user, text: "hello"),
        ]

        #expect(GGUFPromptWindow.compactMessagesForOversizedPrompt(messages) == [
            ChatMessage(role: .user, text: "hello")
        ])
    }

    @Test func oversizedPromptFallbackUsesLastMessageWhenNoUserMessageExists() {
        let messages = [
            ChatMessage(role: .system, text: "policy"),
            ChatMessage(role: .assistant, text: "partial answer"),
        ]

        #expect(GGUFPromptWindow.compactMessagesForOversizedPrompt(messages) == [
            ChatMessage(role: .assistant, text: "partial answer")
        ])
    }
}
