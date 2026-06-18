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
    /// Weight format used to choose the provider/runtime path.
    var weightFormat: WeightFormat
    /// Selected weight artifact for formats that have multiple candidate files
    /// in one snapshot, currently GGUF.
    var weightFile: String?

    enum Capability: String, Codable, Sendable {
        case text
        case vision
    }

    enum WeightFormat: String, Codable, Sendable {
        case mlx
        case gguf
    }

    init(
        name: String,
        repoId: String,
        localPath: String,
        capability: Capability,
        diskSizeBytes: Int64,
        addedAt: Date,
        weightFormat: WeightFormat = .mlx,
        weightFile: String? = nil
    ) {
        self.name = name
        self.repoId = repoId
        self.localPath = localPath
        self.capability = capability
        self.diskSizeBytes = diskSizeBytes
        self.addedAt = addedAt
        self.weightFormat = weightFormat
        self.weightFile = weightFile
    }

    func matches(_ identifier: String) -> Bool {
        name == identifier || repoId == identifier
    }
}

extension ModelDescriptor {
    private enum CodingKeys: String, CodingKey {
        case name
        case repoId
        case localPath
        case capability
        case diskSizeBytes
        case addedAt
        case weightFormat
        case weightFile
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            name: try container.decode(String.self, forKey: .name),
            repoId: try container.decode(String.self, forKey: .repoId),
            localPath: try container.decode(String.self, forKey: .localPath),
            capability: try container.decode(Capability.self, forKey: .capability),
            diskSizeBytes: try container.decode(Int64.self, forKey: .diskSizeBytes),
            addedAt: try container.decode(Date.self, forKey: .addedAt),
            weightFormat: try container.decodeIfPresent(WeightFormat.self, forKey: .weightFormat)
                ?? .mlx,
            weightFile: try container.decodeIfPresent(String.self, forKey: .weightFile))
    }
}
