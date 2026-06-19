import Foundation
import MLXLMCommon
import Testing

@testable import smolx

@Suite("ModelSamplingDefaults")
struct ModelSamplingDefaultsTests {
    @Test func ggufMetadataProvidesSamplingDefaults() {
        let defaults = ModelSamplingDefaults.fromGGUFMetadata([
            "general.sampling.temp": .float32(0.8),
            "general.sampling.top_p": .float64(0.95),
            "general.sampling.top_k": .uint32(64),
        ])

        #expect(defaults?.temperature == Double(Float(0.8)))
        #expect(defaults?.topP == 0.95)
        #expect(defaults?.topK == 64)
    }

    @Test func generationConfigProvidesSamplingDefaults() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"temperature":0.7,"top_p":0.9,"top_k":50}"#
            .write(to: dir.appendingPathComponent("generation_config.json"), atomically: true, encoding: .utf8)

        let defaults = ModelSamplingDefaults.fromGenerationConfig(in: dir)

        #expect(defaults?.temperature == 0.7)
        #expect(defaults?.topP == 0.9)
        #expect(defaults?.topK == 50)
    }

    @Test func samplingDefaultsMergePrefersPrimaryValues() {
        let primary = ModelSamplingDefaults(temperature: 0.8, topP: nil, topK: 64)
        let fallback = ModelSamplingDefaults(temperature: 0.6, topP: 0.9, topK: 32)

        let merged = primary.mergingMissingValues(from: fallback)

        #expect(merged.temperature == 0.8)
        #expect(merged.topP == 0.9)
        #expect(merged.topK == 64)
    }

    @Test func builderUsesSamplingDefaultsWhenRequestOmitsValues() {
        let descriptor = descriptor(weightFormat: .gguf)
        let defaults = ModelSamplingDefaults(temperature: 0.75, topP: 0.92, topK: 40)

        let params = MLXGenerationParameterBuilder.make(
            GenerationParams.default,
            descriptor: descriptor,
            samplingDefaults: defaults)

        #expect(params.temperature == 0.75)
        #expect(params.topP == 0.92)
        #expect(params.topK == 40)
    }

    @Test func builderRequestValuesOverrideSamplingDefaults() {
        let descriptor = descriptor(weightFormat: .gguf)
        let defaults = ModelSamplingDefaults(temperature: 0.75, topP: 0.92, topK: 40)
        let request = GenerationParams(
            temperature: 0.2,
            topP: 0.7,
            topK: 10,
            maxTokens: nil,
            stopSequences: [],
            seed: nil,
            stream: false)

        let params = MLXGenerationParameterBuilder.make(
            request,
            descriptor: descriptor,
            samplingDefaults: defaults)

        #expect(params.temperature == 0.2)
        #expect(params.topP == 0.7)
        #expect(params.topK == 10)
    }

    @Test func builderFallsBackToMLXLMDefaultsWhenFilesHaveNoSamplingDefaults() {
        let descriptor = descriptor(weightFormat: .gguf)

        let params = MLXGenerationParameterBuilder.make(
            GenerationParams.default,
            descriptor: descriptor,
            samplingDefaults: nil)

        #expect(params.temperature == 0.6)
        #expect(params.topP == 1)
        #expect(params.topK == 0)
    }

    private func tempDir() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("smolx-sampling-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
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
