import Foundation

/// Builds new provider instances on demand. Keeps `ModelManager` agnostic to
/// the concrete provider type, which makes the manager unit-testable with a
/// stub provider and lets us add new backends without touching the manager.
protocol ProviderFactory: Sendable {
    func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider
}

struct RoutingProviderFactory: ProviderFactory {
    private let mlx: any ProviderFactory
    private let gguf: any ProviderFactory

    init(mlx: any ProviderFactory, gguf: any ProviderFactory) {
        self.mlx = mlx
        self.gguf = gguf
    }

    func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider {
        let descriptor = Self.normalizedDescriptor(descriptor)
        switch descriptor.weightFormat {
        case .mlx:
            return try await mlx.make(descriptor)
        case .gguf:
            return try await gguf.make(descriptor)
        }
    }

    private static func normalizedDescriptor(_ descriptor: ModelDescriptor) -> ModelDescriptor {
        guard descriptor.weightFormat == .mlx else {
            return descriptor
        }

        let directory = URL(fileURLWithPath: descriptor.localPath)
        guard let ggufFile = firstGGUFFile(in: directory) else {
            return descriptor
        }

        var normalized = descriptor
        normalized.weightFormat = .gguf
        normalized.weightFile = ggufFile
        return normalized
    }

    private static func firstGGUFFile(in directory: URL) -> String? {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil)
        else { return nil }

        return urls
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .map(\.lastPathComponent)
            .sorted()
            .first
    }
}
