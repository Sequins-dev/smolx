import Foundation
import Logging

/// Generates the model catalog file that Codex reads to resolve local model
/// metadata (context window, modalities, tool support). Without this, Codex
/// falls back to conservative defaults that starve thinking-heavy models.
///
/// The catalog is written to ~/.smolx/codex-catalog.json. Point Codex at it
/// with a one-time addition to ~/.codex/config.toml:
///   model_catalog_json = "/Users/<you>/.smolx/codex-catalog.json"
enum CodexCatalog {

    static var outputPath: URL {
        Paths.appRoot.appendingPathComponent("codex-catalog.json")
    }

    static func write(_ models: [ModelDescriptor], logger: Logger) {
        let entries = models.map { entry(for: $0) }
        let catalog: [String: Any] = ["models": entries]
        guard let data = try? JSONSerialization.data(
            withJSONObject: catalog,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        else { return }

        try? Paths.ensureAppRoot()

        if (try? data.write(to: outputPath, options: .atomic)) != nil {
            let msg: Logger.Message =
                "Codex catalog written to \(outputPath.path) — add to ~/.codex/config.toml: model_catalog_json = \"\(outputPath.path)\""
            logger.info(msg)
        }
    }

    private static func entry(for d: ModelDescriptor) -> [String: Any] {
        let ctx = d.contextLength ?? 131_072
        var modalities: [String] = ["text"]
        if d.capability == .vision { modalities.append("image") }

        return [
            "slug": d.name,
            "display_name": d.name,
            "context_window": ctx,
            "apply_patch_tool_type": "function",
            "shell_type": "default",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "truncation_policy": [
                "mode": "bytes",
                "limit": 10_000,
            ] as [String: Any],
            "input_modalities": modalities,
            "base_instructions": "",
            "support_verbosity": true,
            "default_verbosity": "low",
            "supports_parallel_tool_calls": false,
            "supports_reasoning_summaries": false,
            "supported_reasoning_levels": [] as [String],
            "experimental_supported_tools": [] as [String],
        ]
    }
}
