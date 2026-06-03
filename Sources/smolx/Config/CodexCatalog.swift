import Foundation
import Logging

/// Generates a merged model catalog for Codex that combines the existing
/// ~/.codex/models_cache.json (OpenAI models) with smolx's installed models.
///
/// Codex's `model_catalog_json` config key replaces the built-in catalog
/// entirely, so we must include the existing models to preserve access to
/// OpenAI models. The merged file is written to ~/.smolx/codex-catalog.json.
///
/// One-time setup: add this to ~/.codex/config.toml:
///   model_catalog_json = "/Users/<you>/.smolx/codex-catalog.json"
enum CodexCatalog {

    static var outputPath: URL {
        Paths.appRoot.appendingPathComponent("codex-catalog.json")
    }

    static var cacheSourcePath: URL {
        Paths.home.appendingPathComponent(".codex/models_cache.json")
    }

    static func write(_ models: [ModelDescriptor], logger: Logger) {
        // Load Codex's own model cache so we can merge into it. If it's
        // absent (fresh install), start with an empty models array.
        var existing: [[String: Any]] = []
        if let data = try? Data(contentsOf: cacheSourcePath),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let cached = json["models"] as? [[String: Any]]
        {
            existing = cached
        }

        // Remove any previously-injected smolx entries so re-runs don't
        // accumulate duplicates.
        let smolxSlugs = Set(models.map { $0.name })
        var merged = existing.filter { entry in
            guard let slug = entry["slug"] as? String else { return true }
            return !smolxSlugs.contains(slug)
        }

        // Append current smolx models.
        for model in models {
            merged.append(entry(for: model))
        }

        let catalog: [String: Any] = ["models": merged]
        guard let data = try? JSONSerialization.data(
            withJSONObject: catalog,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        else { return }

        try? Paths.ensureAppRoot()
        guard (try? data.write(to: outputPath, options: .atomic)) != nil else { return }

        let n = models.count
        let p = outputPath.path
        logger.info("Codex catalog written (\(n) smolx model(s) merged) — add to ~/.codex/config.toml: model_catalog_json = \"\(p)\"")
    }

    private static func entry(for d: ModelDescriptor) -> [String: Any] {
        let ctx = d.contextLength ?? 131_072
        var modalities: [String] = ["text"]
        if d.capability == .vision { modalities.append("image") }

        return [
            "slug": d.name,
            "display_name": d.name,
            "context_window": ctx,
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "input_modalities": modalities,
            "supports_parallel_tool_calls": false,
            "supported_reasoning_levels": [] as [String],
        ]
    }
}
