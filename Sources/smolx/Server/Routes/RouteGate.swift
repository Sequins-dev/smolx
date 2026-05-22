import Foundation

/// Shared route-level predicates that don't fit in any one route file.
enum RouteGate {
    /// True if any message in the conversation carries an image content block.
    /// Used to reject image inputs sent to text-only models.
    static func hasImages(_ messages: [ChatMessage]) -> Bool {
        for m in messages {
            for block in m.content {
                if case .image = block { return true }
            }
        }
        return false
    }
}
