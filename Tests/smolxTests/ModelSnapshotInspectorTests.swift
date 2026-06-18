import Foundation
import Testing

@testable import smolx

@Suite("ModelSnapshotInspector")
struct ModelSnapshotInspectorTests {
    private func tempDir() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("smolx-snapshot-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func contextLengthReadsTopLevelConfig() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"max_position_embeddings":4096}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        #expect(ModelSnapshotInspector.contextLength(at: dir) == 4096)
    }

    @Test func contextLengthReadsVisionLanguageModelConfig() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"language_model":{"max_position_embeddings":8192}}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        #expect(ModelSnapshotInspector.contextLength(at: dir) == 8192)
    }

    @Test func contextLengthReadsTextConfig() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"text_config":{"max_position_embeddings":262144}}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        #expect(ModelSnapshotInspector.contextLength(at: dir) == 262144)
    }

    @Test func ggufDescriptorUsesSlidingWindowEffectiveContext() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"text_config":{"max_position_embeddings":262144,"sliding_window":1024}}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let descriptor = ModelDescriptor(
            name: "gemma4-q4",
            repoId: "owner/model-GGUF",
            localPath: dir.path,
            capability: .text,
            diskSizeBytes: 1,
            addedAt: Date(timeIntervalSince1970: 0),
            weightFormat: .gguf,
            weightFile: "model.gguf")

        #expect(ModelSnapshotInspector.contextLength(for: descriptor) == 1024)
    }

    @Test func contextLengthReturnsNilWhenMissing() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(ModelSnapshotInspector.contextLength(at: dir) == nil)
    }

    @Test func capabilityUsesPreprocessorMarker() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelSnapshotInspector.capability(at: dir) == .text)

        try "{}".write(
            to: dir.appendingPathComponent("preprocessor_config.json"),
            atomically: true,
            encoding: .utf8)

        #expect(ModelSnapshotInspector.capability(at: dir) == .vision)
    }
}
