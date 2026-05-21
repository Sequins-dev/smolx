import Foundation

/// Provider-agnostic message format. The OpenAI and Anthropic translators both
/// convert their respective wire formats into this representation before the
/// request reaches a `ModelProvider`.
struct ChatMessage: Equatable, Sendable {
    enum Role: String, Sendable, Equatable {
        case system
        case user
        case assistant
        case tool
    }

    var role: Role
    var content: [ContentBlock]

    /// Convenience init for plain-text messages.
    init(role: Role, text: String) {
        self.role = role
        self.content = [.text(text)]
    }

    init(role: Role, content: [ContentBlock]) {
        self.role = role
        self.content = content
    }
}

/// A single content block within a message. OpenAI's multi-part content and
/// Anthropic's content blocks both map to this enum.
enum ContentBlock: Equatable, Sendable {
    case text(String)
    case image(ImagePayload)
    case toolUse(ToolUse)
    case toolResult(ToolResult)
}

struct ImagePayload: Equatable, Sendable {
    /// Raw image bytes. Translators decode base64 or fetch URLs into this form
    /// at the request boundary so providers never deal with the wire format.
    var data: Data
    var mimeType: String
}

struct ToolUse: Equatable, Sendable {
    var id: String
    var name: String
    /// Parsed input arguments as JSON. Always an object at the top level.
    var input: JSONValue
}

struct ToolResult: Equatable, Sendable {
    /// References the `ToolUse.id` from a previous assistant message.
    var toolUseId: String
    /// Resulting content from running the tool — text in the common case.
    var content: String
    var isError: Bool
}
