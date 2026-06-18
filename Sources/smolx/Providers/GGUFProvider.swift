import Foundation
import Logging
import MLX

/// Factory for GGUF-backed MLX models.
///
/// The runtime intentionally goes through the same `ModelProvider` surface as
/// native MLX models. Once the mlx-swift-lm fork exposes GGUF-aware
/// `loadModelContainer` support, this factory can keep returning the same
/// provider shape while the fork owns GGUF tensor loading and quantized layer
/// construction.
struct GGUFProviderFactory: ProviderFactory {
    func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider {
        guard descriptor.weightFormat == .gguf else {
            throw ProviderError.unsupportedFeature(
                "GGUFProviderFactory can only load GGUF descriptors")
        }
        guard descriptor.weightFile != nil else {
            throw ProviderError.loadFailed("GGUF descriptor is missing weightFile")
        }

        try GGUFRuntimeFiles.prepareSnapshot(
            at: URL(fileURLWithPath: descriptor.localPath),
            weightFile: descriptor.weightFile)

        Memory.cacheLimit = 512 * 1024 * 1024
        let provider = MLXProvider(
            descriptor: descriptor,
            logger: Logger(label: "smolx.gguf"))
        try await provider.load()
        return provider
    }
}
