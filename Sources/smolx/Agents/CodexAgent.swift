import Foundation
import Logging

struct CodexAgent: AgentPlugin {
    let commandName = "codex"

    func plan(
        baseURL: String,
        models: UserConfig,
        authToken: String,
        installedModels: [ModelDescriptor]
    ) -> AgentPlan {
        // codex 0.128.x prompts for sign-in (ChatGPT OAuth or API key) on
        // every fresh CODEX_HOME. We pre-write `auth.json` to satisfy the
        // login state, then pin model + provider via repeated `-c` CLI
        // overrides because codex rewrites its own config.toml on startup
        // (it strips unknown / unrecognised top-level keys and persists
        // its own state under `[projects.*]` and `[tui.*]`). `-c` flags
        // are applied AFTER config-file loading and can't be erased by
        // that rewrite, so the local provider stays pinned every run.
        //
        // codex has no native fast/small split — its "fast" knob is a
        // service_tier, not a separate model. We use only `models.smart ?? ""`
        // here and ignore the other tiers.
        let codexHome = Paths.appRoot
            .appendingPathComponent("codex-home", isDirectory: true).path
        let authJson = """
            {
              "OPENAI_API_KEY": "\(authToken)",
              "auth_mode": "apikey",
              "tokens": null,
              "last_refresh": null
            }
            """
        // Each `-c key=value` is parsed as TOML; values that look like
        // strings need shell-safe quoting. Dotted paths land at the
        // right nested key.
        //
        // codex 0.128 reserves `openai` (and likely other built-in
        // provider ids) — it refuses to load if we try to override them.
        // So we declare `smolx` as a fresh custom provider and point
        // codex at it via `model_provider`. Underscored name avoids any
        // identifier-validation surprises.
        let providerId = "smolx"
        let catalogPath = catalogOutputPath.path
        let prefixArgs: [String] = [
            "-c", "model=\"\(models.smart ?? "")\"",
            "-c", "model_provider=\"\(providerId)\"",
            "-c", "model_catalog_json=\"\(catalogPath)\"",
            "-c", "model_providers.\(providerId).name=\"smolx\"",
            "-c", "model_providers.\(providerId).base_url=\"\(baseURL)/v1\"",
            "-c", "model_providers.\(providerId).env_key=\"OPENAI_API_KEY\"",
            "-c", "model_providers.\(providerId).wire_api=\"responses\"",
            "-c", "model_providers.\(providerId).request_max_retries=2",
            "-c", "model_providers.\(providerId).stream_idle_timeout_ms=120000",
        ]
        return AgentPlan(
            executable: "codex",
            env: [
                "OPENAI_BASE_URL": baseURL + "/v1",
                "OPENAI_API_KEY": authToken,
                "CODEX_HOME": codexHome,
            ],
            prefixArgs: prefixArgs,
            files: [.init(path: codexHome + "/auth.json", contents: authJson)])
    }

    // MARK: - Catalog

    func setup(installedModels: [ModelDescriptor], logger: Logger) {
        writeCatalog(installedModels, logger: logger)
    }

    private var catalogOutputPath: URL {
        Paths.appRoot.appendingPathComponent("codex-catalog.json")
    }

    private var cacheSourcePath: URL {
        Paths.home.appendingPathComponent(".codex/models_cache.json")
    }

    private func writeCatalog(_ models: [ModelDescriptor], logger: Logger) {
        // Load Codex's own model cache so we can merge into it. If it's
        // absent (fresh install), start with an empty models array.
        var existing: [[String: Any]] = []
        if let data = try? Data(contentsOf: cacheSourcePath),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let cached = json["models"] as? [[String: Any]]
        {
            existing = cached
        }

        // Use the first existing entry as a template so we inherit all
        // required fields (base_instructions, supports_reasoning_summaries,
        // etc.) without hardcoding them. Falls back to a minimal stub when
        // the cache is absent (fresh Codex install).
        let template: [String: Any] = existing.first ?? [
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "base_instructions": "You are a helpful coding assistant.",
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "supports_parallel_tool_calls": false,
            "supports_search_tool": false,
            "additional_speed_tiers": [] as [String],
            "service_tiers": [] as [String],
            "supported_reasoning_levels": [] as [String],
            "experimental_supported_tools": [] as [String],
        ]

        // Remove any previously-injected smolx entries so re-runs don't
        // accumulate duplicates.
        let smolxSlugs = Set(models.map { $0.name })
        var merged = existing.filter { entry in
            guard let slug = entry["slug"] as? String else { return true }
            return !smolxSlugs.contains(slug)
        }

        for model in models {
            merged.append(catalogEntry(for: model, template: template))
        }

        let catalog: [String: Any] = ["models": merged]
        guard let data = try? JSONSerialization.data(
            withJSONObject: catalog,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        else { return }

        try? Paths.ensureAppRoot()
        guard (try? data.write(to: catalogOutputPath, options: .atomic)) != nil else { return }

        let n = models.count
        let p = catalogOutputPath.path
        logger.info(
            "Codex catalog written (\(n) smolx model(s) merged) — add to ~/.codex/config.toml: model_catalog_json = \"\(p)\""
        )
    }

    private func catalogEntry(for d: ModelDescriptor, template: [String: Any]) -> [String: Any] {
        let ctx = ModelSnapshotInspector.contextLength(for: d) ?? 131_072
        var modalities: [String] = ["text"]
        if d.capability == .vision { modalities.append("image") }

        var entry = template
        entry["slug"] = d.name
        entry["display_name"] = d.name
        entry["description"] = ""
        entry["context_window"] = ctx
        entry["max_context_window"] = ctx
        entry["effective_context_window_percent"] = 95
        entry["input_modalities"] = modalities
        entry["supports_parallel_tool_calls"] = false
        entry["supported_reasoning_levels"] = [] as [String]
        entry["additional_speed_tiers"] = [] as [String]
        entry["service_tiers"] = [] as [String]
        entry["availability_nux"] = nil as Any? as Any
        entry["upgrade"] = nil as Any? as Any
        entry["default_reasoning_level"] = nil as Any? as Any
        return entry
    }
}
