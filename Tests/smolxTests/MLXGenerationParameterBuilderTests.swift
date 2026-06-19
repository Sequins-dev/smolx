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

    @Test func ggufModelsUseStableSamplingDefaults() {
        let descriptor = descriptor(weightFormat: .gguf)

        let params = MLXGenerationParameterBuilder.make(
            GenerationParams.default, descriptor: descriptor)

        #expect(params.temperature == 0)
        #expect(params.topP == 1)
        #expect(params.topK == 0)
        #expect(params.repetitionPenalty == 1.08)
        #expect(params.repetitionContextSize == 128)
    }

    @Test func ggufModelsHonorExplicitSamplingParams() {
        let descriptor = descriptor(weightFormat: .gguf)
        let generationParams = GenerationParams(
            temperature: 0.4,
            topP: 0.8,
            topK: 20,
            maxTokens: nil,
            stopSequences: [],
            seed: nil,
            stream: false)

        let params = MLXGenerationParameterBuilder.make(
            generationParams, descriptor: descriptor)

        #expect(params.temperature == 0.4)
        #expect(params.topP == 0.8)
        #expect(params.topK == 20)
        #expect(params.repetitionPenalty == 1.08)
        #expect(params.repetitionContextSize == 128)
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
