import Foundation

/// Static knowledge about how each supported agent CLI is configured to talk
/// to a non-default endpoint. The `run` subcommand looks up the named agent
/// here, derives env vars / config files, and execs the binary.
enum AgentEnvironment {

    struct Plan: Sendable {
        var executable: String
        var env: [String: String]
        /// Variables to *remove* from the inherited environment before exec.
        /// Used to clear pre-existing values that conflict with the ones we
        /// set — e.g. Claude Code refuses to run cleanly when both
        /// `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are set, so the
        /// claude plan adds `ANTHROPIC_API_KEY` here.
        var unsetEnv: [String]
        /// Args injected before the user's `--` passthrough. Used when the
        /// agent has CLI overrides we want to apply unconditionally — e.g.
        /// codex's `-c key=value` flags pin the model + provider config on
        /// every invocation, surviving codex's habit of rewriting its
        /// config.toml on startup.
        var prefixArgs: [String]
        /// Files to write before exec (e.g. a generated opencode config).
        var files: [TempFile]

        init(
            executable: String,
            env: [String: String],
            unsetEnv: [String] = [],
            prefixArgs: [String] = [],
            files: [TempFile] = []
        ) {
            self.executable = executable
            self.env = env
            self.unsetEnv = unsetEnv
            self.prefixArgs = prefixArgs
            self.files = files
        }

        struct TempFile: Sendable {
            var path: String
            var contents: String
        }
    }

    enum Kind: String, CaseIterable, Sendable {
        case claude
        case codex
        case aider
        case opencode
        case pi
        case crush
    }

    enum ResolveError: Error, CustomStringConvertible {
        case unknownAgent(String)

        var description: String {
            switch self {
            case .unknownAgent(let n):
                return "Unknown agent: \(n). Supported: \(Kind.allCases.map(\.rawValue).joined(separator: ", "))"
            }
        }
    }

    static func plan(
        agentName: String,
        baseURL: String,
        modelName: String,
        authToken: String?,
        installedModels: [ModelDescriptor] = []
    ) throws -> Plan {
        guard let kind = Kind(rawValue: agentName) else {
            throw ResolveError.unknownAgent(agentName)
        }
        let token = authToken ?? "mlx-serve-local"
        switch kind {
        case .claude:
            // Claude Code reads three credential sources:
            //   1. ANTHROPIC_API_KEY env var
            //   2. ANTHROPIC_AUTH_TOKEN env var (bearer-style, for proxies)
            //   3. a cached /login key on disk under CLAUDE_CONFIG_DIR (default ~/.claude)
            // We want #2 only, against our local server. To prevent #3 from
            // also being present (which triggers the "/login managed key
            // conflict" warning), we point CLAUDE_CONFIG_DIR at a directory
            // under ~/.mlx-serve/ that's isolated from the user's regular
            // Claude install. Settings made there persist across mlx-serve
            // runs but never see the real-Anthropic creds.
            let configDir = Paths.appRoot
                .appendingPathComponent("claude-config", isDirectory: true).path
            return Plan(
                executable: "claude",
                env: [
                    "ANTHROPIC_BASE_URL": baseURL,
                    "ANTHROPIC_AUTH_TOKEN": token,
                    "ANTHROPIC_MODEL": modelName,
                    "CLAUDE_CONFIG_DIR": configDir,
                ],
                unsetEnv: ["ANTHROPIC_API_KEY"],
                files: [.init(path: configDir + "/.keep", contents: "")])
        case .codex:
            // codex 0.128.x prompts for sign-in (ChatGPT OAuth or API key) on
            // every fresh CODEX_HOME. We pre-write `auth.json` to satisfy the
            // login state, then pin model + provider via repeated `-c` CLI
            // overrides because codex rewrites its own config.toml on startup
            // (it strips unknown / unrecognised top-level keys and persists
            // its own state under `[projects.*]` and `[tui.*]`). `-c` flags
            // are applied AFTER config-file loading and can't be erased by
            // that rewrite, so the local provider stays pinned every run.
            let codexHome = Paths.appRoot
                .appendingPathComponent("codex-home", isDirectory: true).path
            let authJson = """
                {
                  "OPENAI_API_KEY": "\(token)",
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
            // So we declare `mlx_serve` as a fresh custom provider and point
            // codex at it via `model_provider`. Underscored name avoids any
            // identifier-validation surprises.
            let providerId = "mlx_serve"
            let prefix: [String] = [
                "-c", "model=\"\(modelName)\"",
                "-c", "model_provider=\"\(providerId)\"",
                "-c", "model_providers.\(providerId).name=\"mlx-serve\"",
                "-c", "model_providers.\(providerId).base_url=\"\(baseURL)/v1\"",
                "-c", "model_providers.\(providerId).env_key=\"OPENAI_API_KEY\"",
                "-c", "model_providers.\(providerId).wire_api=\"responses\"",
                "-c", "model_providers.\(providerId).request_max_retries=2",
                "-c", "model_providers.\(providerId).stream_idle_timeout_ms=120000",
            ]
            return Plan(
                executable: "codex",
                env: [
                    "OPENAI_BASE_URL": baseURL + "/v1",
                    "OPENAI_API_KEY": token,
                    "CODEX_HOME": codexHome,
                ],
                prefixArgs: prefix,
                files: [.init(path: codexHome + "/auth.json", contents: authJson)])
        case .aider:
            // aider reads OPENAI_API_BASE (legacy) for compatibility — we set
            // both so aider works regardless of its internal preference.
            return Plan(
                executable: "aider",
                env: [
                    "OPENAI_API_BASE": baseURL + "/v1",
                    "OPENAI_BASE_URL": baseURL + "/v1",
                    "OPENAI_API_KEY": token,
                ],
                files: [])
        case .opencode:
            // Opencode supports three config-injection mechanisms:
            //   1. ~/.config/opencode/opencode.json (and project .opencode.json) — user-owned
            //   2. OPENCODE_CONFIG=/path/file.json — "load an additional config" (merged in)
            //   3. OPENCODE_CONFIG_CONTENT='{...}' — "inject inline JSON as a final
            //      local-scope merge" (verbatim from opencode's own --help)
            //
            // We use (3) so there's no file involved at all and our overrides
            // win over the user's existing user/project config without
            // touching their disk state.
            //
            // The model entries MUST include `limit.context` and `limit.output`
            // — those are required by opencode's schema and the entire
            // provider gets silently dropped from the UI if they're missing
            // (this is the bug the user reported as "opencode only shows
            // OpenCode Zen and OpenAI"). We use 32K context / 4K output as
            // a sensible default; users can override per-model later.
            let modelEntries = (installedModels.isEmpty
                ? [ModelDescriptor(name: modelName, repoId: "", localPath: "",
                                   capability: .text, diskSizeBytes: 0, addedAt: Date())]
                : installedModels)
            let modelsJSON = modelEntries.map { m in
                """
                  "\(m.name)": {
                    "id": "\(m.name)",
                    "name": "\(m.name)",
                    "tool_call": true,
                    "temperature": true,
                    "attachment": \(m.capability == .vision ? "true" : "false"),
                    "limit": { "context": 32768, "output": 4096 }
                  }
                """
            }.joined(separator: ",\n")
            let inlineConfig = """
                {
                  "$schema": "https://opencode.ai/config.json",
                  "model": "mlx-serve/\(modelName)",
                  "provider": {
                    "mlx-serve": {
                      "npm": "@ai-sdk/openai-compatible",
                      "name": "mlx-serve",
                      "options": {
                        "baseURL": "\(baseURL)/v1",
                        "apiKey": "\(token)"
                      },
                      "models": {
                \(modelsJSON)
                      }
                    }
                  }
                }
                """
            // Collapse to one line so the env-var value stays single-line.
            let oneLineConfig = inlineConfig
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "  ", with: " ")
            return Plan(
                executable: "opencode",
                env: ["OPENCODE_CONFIG_CONTENT": oneLineConfig])
        case .pi:
            // Pi (pi.dev / @earendil-works/pi-coding-agent) reads providers
            // from ~/.pi/agent/models.json by default. PI_CODING_AGENT_DIR
            // overrides that directory, so we point it at a sandbox under
            // ~/.mlx-serve/ to avoid touching the user's real Pi state
            // (skills, sessions, auth.json, the user's own models.json).
            //
            // Pi's schema is `providers.<id>` with `api: "openai-completions"`
            // for OpenAI-compatible servers. We list every installed mlx-serve
            // model so they appear in Pi's `/model` picker; the one passed
            // via --model is listed first so it surfaces as the default.
            let piHome = Paths.appRoot
                .appendingPathComponent("pi-agent", isDirectory: true).path
            let piModelEntries = (installedModels.isEmpty
                ? [ModelDescriptor(name: modelName, repoId: "", localPath: "",
                                   capability: .text, diskSizeBytes: 0, addedAt: Date())]
                : installedModels)
            let piOrdered = piModelEntries.sorted { a, _ in a.name == modelName }
            let piModelsJSON = piOrdered.map { m in
                let inputs = m.capability == .vision
                    ? "[\"text\", \"image\"]" : "[\"text\"]"
                return """
                          { "id": "\(m.name)", "name": "\(m.name)", "input": \(inputs), "contextWindow": 32768, "maxTokens": 4096 }
                    """
            }.joined(separator: ",\n")
            let piModelsConfig = """
                {
                  "providers": {
                    "mlx-serve": {
                      "baseUrl": "\(baseURL)/v1",
                      "api": "openai-completions",
                      "apiKey": "\(token)",
                      "models": [
                \(piModelsJSON)
                      ]
                    }
                  }
                }
                """
            return Plan(
                executable: "pi",
                env: ["PI_CODING_AGENT_DIR": piHome],
                files: [.init(path: piHome + "/models.json",
                              contents: piModelsConfig)])
        case .crush:
            // Charm's Crush reads its config from .crush.json (CWD),
            // crush.json (CWD), or $HOME/.config/crush/crush.json. The env
            // var CRUSH_GLOBAL_CONFIG points at the *directory* containing
            // crush.json (NOT the file itself — Crush appends `/crush.json`
            // internally; setting it to a file path produces a path like
            // `…/crush.json/crush.json: not a directory` error). We sandbox
            // it under ~/.mlx-serve/ and declare mlx-serve as an
            // `openai-compat` provider with every installed model listed.
            //
            // The model list uses Crush's documented keys (`context_window`,
            // `default_max_tokens`). Without them Crush either fails to load
            // the provider or falls back to broken defaults.
            let crushConfigDir = Paths.appRoot
                .appendingPathComponent("crush", isDirectory: true).path
            let crushConfigPath = crushConfigDir + "/crush.json"
            let crushModelEntries = (installedModels.isEmpty
                ? [ModelDescriptor(name: modelName, repoId: "", localPath: "",
                                   capability: .text, diskSizeBytes: 0, addedAt: Date())]
                : installedModels)
            let crushModelsJSON = crushModelEntries.map { m in
                """
                          { "id": "\(m.name)", "name": "\(m.name)", "context_window": 32768, "default_max_tokens": 4096 }
                    """
            }.joined(separator: ",\n")
            let crushConfig = """
                {
                  "$schema": "https://charm.land/crush.json",
                  "providers": {
                    "mlx-serve": {
                      "type": "openai-compat",
                      "base_url": "\(baseURL)/v1",
                      "api_key": "\(token)",
                      "models": [
                \(crushModelsJSON)
                      ]
                    }
                  }
                }
                """
            return Plan(
                executable: "crush",
                env: ["CRUSH_GLOBAL_CONFIG": crushConfigDir],
                files: [.init(path: crushConfigPath, contents: crushConfig)])
        }
    }
}
