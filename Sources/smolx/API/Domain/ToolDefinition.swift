import Foundation

struct ToolDefinition: Sendable, Equatable {
    var name: String
    var description: String?
    /// JSON Schema describing the tool's input shape. Both OpenAI and Anthropic
    /// accept JSON Schema; we keep the parsed value and re-emit it as-is.
    var inputSchema: JSONValue
}

enum ToolChoice: Sendable, Equatable {
    case auto
    case none
    case required
    case specific(name: String)
}
