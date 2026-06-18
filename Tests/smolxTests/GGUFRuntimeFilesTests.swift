import Foundation
import HuggingFace
import Testing

@testable import smolx

@Suite("GGUFRuntimeFiles")
struct GGUFRuntimeFilesTests {
    @Test func baseModelPrefersCardDataString() throws {
        let model = try makeModel(
            id: "owner/model-GGUF",
            cardData: #"{"base_model":"google/gemma-4-12B-it"}"#,
            tags: ["base_model:other/repo"])

        #expect(GGUFRuntimeFiles.baseModelRepoId(from: model) == "google/gemma-4-12B-it")
    }

    @Test func baseModelReadsFirstRepoIdFromCardDataArray() throws {
        let model = try makeModel(
            id: "owner/model-GGUF",
            cardData: #"{"base_model":["not-a-repo","google/gemma-4-12B-it"]}"#,
            tags: [])

        #expect(GGUFRuntimeFiles.baseModelRepoId(from: model) == "google/gemma-4-12B-it")
    }

    @Test func baseModelFallsBackToTags() throws {
        let model = try makeModel(
            id: "owner/model-GGUF",
            cardData: nil,
            tags: ["gguf", "base_model:google/gemma-4-12B-it"])

        #expect(GGUFRuntimeFiles.baseModelRepoId(from: model) == "google/gemma-4-12B-it")
    }

    @Test func tokenizerSidecarPathsKeepOnlyRuntimeTokenizerFiles() {
        let paths = GGUFRuntimeFiles.tokenizerSidecarPaths(in: [
            "README.md",
            "config.json",
            "tokenizer.json",
            "nested/tokenizer_config.json",
            "chat_template.jinja",
            "model.safetensors",
            "model-Q4_K_M.gguf",
        ])

        #expect(paths == [
            "chat_template.jinja",
            "nested/tokenizer_config.json",
            "tokenizer.json",
        ])
    }

    @Test func tokenizerEOSHelpersReadEOTTokenId() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("smolx-gguf-runtime-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try #"{"eot_token":"<turn|>"}"#.write(
            to: directory.appendingPathComponent("tokenizer_config.json"),
            atomically: true,
            encoding: .utf8)
        try #"{"model":{"vocab":{"<turn|>":106}}}"#.write(
            to: directory.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8)

        #expect(GGUFRuntimeFiles.tokenizerEOSTokens(in: directory) == ["<turn|>"])
        #expect(GGUFRuntimeFiles.tokenizerTokenId("<turn|>", in: directory) == 106)
    }

    private func makeModel(id: String, cardData: String?, tags: [String]) throws -> Model {
        let tagsJSON = tags.map { #""\#($0)""# }.joined(separator: ",")
        let cardDataJSON = cardData.map { #","cardData":\#($0)"# } ?? ""
        let json = """
            {
              "id": "\(id)",
              "tags": [\(tagsJSON)]\(cardDataJSON)
            }
            """
        return try JSONDecoder().decode(Model.self, from: Data(json.utf8))
    }
}
