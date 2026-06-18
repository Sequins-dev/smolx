import Foundation
import Testing

@testable import smolx

@Suite("MLXGenerationParameterBuilder")
struct MLXGenerationParameterBuilderTests {
    @Test func ggufModelsDoNotForceQuantizedKVCache() {
        let descriptor = descriptor(weightFormat: .gguf)

        let params = MLXGenerationParameterBuilder.make(
            GenerationParams.default, descriptor: descriptor)

        #expect(params.kvBits == nil)
    }

    @Test func ggufModelsAvoidChunkedPrefillByDefault() {
        let descriptor = descriptor(weightFormat: .gguf)

        let params = MLXGenerationParameterBuilder.make(
            GenerationParams.default, descriptor: descriptor)

        #expect(params.prefillStepSize >= 32_768)
    }

    @Test func mlxModelsKeepQuantizedKVCacheDefault() {
        let descriptor = descriptor(weightFormat: .mlx)

        let params = MLXGenerationParameterBuilder.make(
            GenerationParams.default, descriptor: descriptor)

        #expect(params.kvBits == 4)
    }

    private func descriptor(weightFormat: ModelDescriptor.WeightFormat) -> ModelDescriptor {
        ModelDescriptor(
            name: "model",
            repoId: "owner/model",
            localPath: "/tmp/model",
            capability: .text,
            diskSizeBytes: 1,
            addedAt: Date(timeIntervalSince1970: 0),
            weightFormat: weightFormat)
    }
}
