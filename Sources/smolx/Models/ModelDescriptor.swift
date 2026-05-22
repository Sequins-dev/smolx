import Foundation

/// Persistent description of an installed model. Persisted to
/// `~/.smolx/registry.json`.
struct ModelDescriptor: Codable, Sendable, Equatable {
    /// User-facing identifier used in API requests (e.g. "llama-3.2-3b").
    var name: String
    /// Original HuggingFace repository id (e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit").
    var repoId: String
    /// Absolute path of the model snapshot on disk.
    var localPath: String
    /// Whether this model accepts image inputs.
    var capability: Capability
    /// Size of weights on disk in bytes — used as the initial budget hint
    /// before MLX's runtime memory accounting kicks in after load.
    var diskSizeBytes: Int64
    var addedAt: Date

    enum Capability: String, Codable, Sendable {
        case text
        case vision
    }
}
