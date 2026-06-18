import Testing

@testable import smolx

@Suite("HubDownloader")
struct HubDownloaderTests {
    @Test func mlxFilterKeepsSafetensorsAndRuntimeSidecars() {
        let paths = [
            "model.safetensors",
            "model.gguf",
            "config.json",
            "tokenizer.json",
            "README.md",
        ]

        let wanted = HubDownloader.filterWantedPaths(
            paths, format: .mlx, weightFile: nil)

        #expect(wanted == ["model.safetensors", "config.json", "tokenizer.json"])
    }

    @Test func ggufFilterKeepsSelectedGGUFAndRuntimeSidecars() {
        let paths = [
            "model.Q4_K_M.gguf",
            "model.Q5_K_M.gguf",
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
        ]

        let wanted = HubDownloader.filterWantedPaths(
            paths, format: .gguf, weightFile: "model.Q4_K_M.gguf")

        #expect(wanted == [
            "model.Q4_K_M.gguf",
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
        ])
    }
}
