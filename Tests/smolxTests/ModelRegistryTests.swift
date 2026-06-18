import Foundation
import Testing

@testable import smolx

@Suite("ModelRegistry")
struct ModelRegistryTests {
    private func descriptor(name: String, repoId: String) -> ModelDescriptor {
        ModelDescriptor(
            name: name,
            repoId: repoId,
            localPath: "/tmp/\(name)",
            capability: .text,
            diskSizeBytes: 1,
            addedAt: Date())
    }

    private func registryWith(_ models: [ModelDescriptor]) throws -> ModelRegistry {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("smolx-registry-\(UUID().uuidString).json")
        let registry = ModelRegistry(url: url)
        try registry.save(models)
        return registry
    }

    @Test func descriptorMatchesNameOrRepoId() {
        let model = descriptor(name: "local-name", repoId: "mlx-community/model")
        #expect(model.matches("local-name"))
        #expect(model.matches("mlx-community/model"))
        #expect(!model.matches("other"))
    }

    @Test func findAndRemoveUseTheSameIdentifierMatching() throws {
        let registry = try registryWith([
            descriptor(name: "local-name", repoId: "mlx-community/model")
        ])
        defer { try? FileManager.default.removeItem(at: registry.url) }

        #expect(try registry.find("mlx-community/model")?.name == "local-name")
        #expect(try registry.remove(name: "mlx-community/model")?.name == "local-name")
        #expect(try registry.load().isEmpty)
    }

    @Test func oldRegistryEntriesDefaultToMLXWeightFormat() throws {
        let json = """
            [
              {
                "addedAt": "2026-01-01T00:00:00Z",
                "capability": "text",
                "diskSizeBytes": 123,
                "localPath": "/tmp/model",
                "name": "model",
                "repoId": "mlx-community/model"
              }
            ]
            """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let models = try decoder.decode([ModelDescriptor].self, from: Data(json.utf8))

        #expect(models.first?.weightFormat == .mlx)
        #expect(models.first?.weightFile == nil)
    }

    @Test func ggufRegistryEntriesRoundTripSelectedWeightFile() throws {
        var model = descriptor(name: "llama-gguf", repoId: "bartowski/llama")
        model.weightFormat = .gguf
        model.weightFile = "llama.Q4_K_M.gguf"
        let registry = try registryWith([model])
        defer { try? FileManager.default.removeItem(at: registry.url) }

        let loaded = try #require(try registry.find("llama-gguf"))
        #expect(loaded.weightFormat == .gguf)
        #expect(loaded.weightFile == "llama.Q4_K_M.gguf")
    }
}
