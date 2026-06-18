import Foundation
import Testing

@testable import smolx

@Suite("ProviderFactory")
struct ProviderFactoryTests {
    actor StubProvider: ModelProvider {
        nonisolated let descriptor: ModelDescriptor
        var residentBytes: Int64 { descriptor.diskSizeBytes }
        var lastUsedAt: Date { Date() }

        init(descriptor: ModelDescriptor) {
            self.descriptor = descriptor
        }

        nonisolated func generate(
            messages: [ChatMessage],
            tools: [ToolDefinition],
            toolChoice: ToolChoice?,
            params: GenerationParams
        ) -> AsyncThrowingStream<StreamEvent, Error> {
            AsyncThrowingStream { continuation in
                continuation.finish()
            }
        }

        func unload() async {}
    }

    struct RecordingFactory: ProviderFactory {
        let label: String
        let calls: CallLog

        func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider {
            await calls.append(label)
            return StubProvider(descriptor: descriptor)
        }
    }

    actor CallLog {
        private var values: [String] = []

        func append(_ value: String) {
            values.append(value)
        }

        var all: [String] { values }
    }

    private func descriptor(format: ModelDescriptor.WeightFormat) -> ModelDescriptor {
        var descriptor = ModelDescriptor(
            name: "model",
            repoId: "repo/model",
            localPath: "/tmp/model",
            capability: .text,
            diskSizeBytes: 1,
            addedAt: Date())
        descriptor.weightFormat = format
        if format == .gguf {
            descriptor.weightFile = "model.Q4_K_M.gguf"
        }
        return descriptor
    }

    @Test func routingFactoryUsesMLXFactoryForMLXDescriptors() async throws {
        let calls = CallLog()
        let factory = RoutingProviderFactory(
            mlx: RecordingFactory(label: "mlx", calls: calls),
            gguf: RecordingFactory(label: "gguf", calls: calls))

        _ = try await factory.make(descriptor(format: .mlx))

        #expect(await calls.all == ["mlx"])
    }

    @Test func routingFactoryUsesGGUFFactoryForGGUFDescriptors() async throws {
        let calls = CallLog()
        let factory = RoutingProviderFactory(
            mlx: RecordingFactory(label: "mlx", calls: calls),
            gguf: RecordingFactory(label: "gguf", calls: calls))

        _ = try await factory.make(descriptor(format: .gguf))

        #expect(await calls.all == ["gguf"])
    }

    @Test func routingFactoryDetectsGGUFWeightsInStaleMLXDescriptor() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("smolx-provider-routing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data().write(to: directory.appendingPathComponent("model.Q4_K_M.gguf"))

        var stale = descriptor(format: .mlx)
        stale.localPath = directory.path

        let calls = CallLog()
        let factory = RoutingProviderFactory(
            mlx: RecordingFactory(label: "mlx", calls: calls),
            gguf: RecordingFactory(label: "gguf", calls: calls))

        _ = try await factory.make(stale)

        #expect(await calls.all == ["gguf"])
    }
}
