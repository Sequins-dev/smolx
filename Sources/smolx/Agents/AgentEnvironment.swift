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
        models: ModelTuple,
        authToken: String?,
        installedModels: [ModelDescriptor] = []
    ) throws -> Plan {
        guard let kind = Kind(rawValue: agentName) else {
            throw ResolveError.unknownAgent(agentName)
        }
        let token = authToken ?? "smolx-local"
        switch kind {
        case .claude:
            // Claude Code reads three credential sources:
            //   1. ANTHROPIC_API_KEY env var
            //   2. ANTHROPIC_AUTH_TOKEN env var (bearer-style, for proxies)
            //   3. a cached /login key on disk under CLAUDE_CONFIG_DIR (default ~/.claude)
            // We want #2 only, against our local server. To prevent #3 from
            // also being present (which triggers the "/login managed key
            // conflict" warning), we point CLAUDE_CONFIG_DIR at a directory
            // under ~/.smolx/ that's isolated from the user's regular
            // Claude install. Settings made there persist across smolx
            // runs but never see the real-Anthropic creds.
            //
            // Tier mapping (per Claude Code's model-config docs):
            //   smart → ANTHROPIC_DEFAULT_OPUS_MODEL   (highest tier alias)
            //   fast  → ANTHROPIC_DEFAULT_SONNET_MODEL (default tier alias)
            //   small → ANTHROPIC_DEFAULT_HAIKU_MODEL  (background-task alias)
            //         + ANTHROPIC_SMALL_FAST_MODEL    (legacy name, still
            //                                          honoured for version-skew safety)
            // ANTHROPIC_MODEL is set to the `fast` value so claude's `/model`
            // picker opens on the user's mid-tier choice (claude's own default
            // selection is `sonnet`, which our `fast` tier overrides).
            let configDir = Paths.appRoot
                .appendingPathComponent("claude-config", isDirectory: true).path
            return Plan(
                executable: "claude",
                env: [
                    "ANTHROPIC_BASE_URL": baseURL,
                    "ANTHROPIC_AUTH_TOKEN": token,
                    "ANTHROPIC_MODEL": models.fast,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": models.smart,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": models.fast,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": models.small,
                    "ANTHROPIC_SMALL_FAST_MODEL": models.small,
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
            //
            // codex has no native fast/small split — its "fast" knob is a
            // service_tier, not a separate model. We use only `models.smart`
            // here and ignore the other tiers.
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
            // So we declare `smolx` as a fresh custom provider and point
            // codex at it via `model_provider`. Underscored name avoids any
            // identifier-validation surprises.
            let providerId = "smolx"
            let prefix: [String] = [
                "-c", "model=\"\(models.smart)\"",
                "-c", "model_provider=\"\(providerId)\"",
                "-c", "model_providers.\(providerId).name=\"smolx\"",
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
            //
            // aider has a three-way native split via CLI flags:
            //   --model           main / coding model      (smart)
            //   --editor-model    diff-generation model    (fast)
            //   --weak-model      commit-message / summary (small)
            // Pin all three via prefixArgs so the resolved tuple lands
            // unambiguously regardless of any .aider.conf.yml the user has.
            return Plan(
                executable: "aider",
                env: [
                    "OPENAI_API_BASE": baseURL + "/v1",
                    "OPENAI_BASE_URL": baseURL + "/v1",
                    "OPENAI_API_KEY": token,
                ],
                prefixArgs: [
                    "--model", models.smart,
                    "--editor-model", models.fast,
                    "--weak-model", models.small,
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
            // provider gets silently dropped from the UI if they're missing.
            // We use 32K context / 4K output as sensible defaults.
            //
            // Tier mapping: top-level `model` = smart, top-level `small_model`
            // = small. Opencode has no middle slot so `fast` is unused here.
            let modelEntries = (installedModels.isEmpty
                ? [ModelDescriptor(name: models.smart, repoId: "", localPath: "",
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
                  "model": "smolx/\(models.smart)",
                  "small_model": "smolx/\(models.small)",
                  "provider": {
                    "smolx": {
                      "npm": "@ai-sdk/openai-compatible",
                      "name": "smolx",
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
            // ~/.smolx/ to avoid touching the user's real Pi state.
            //
            // Pi has no native fast/small split — it's a single picker.
            // We use `models.smart` as the default (listed first); the
            // other tiers are ignored.
            let piHome = Paths.appRoot
                .appendingPathComponent("pi-agent", isDirectory: true).path
            let piModelEntries = (installedModels.isEmpty
                ? [ModelDescriptor(name: models.smart, repoId: "", localPath: "",
                                   capability: .text, diskSizeBytes: 0, addedAt: Date())]
                : installedModels)
            let piOrdered = piModelEntries.sorted { a, _ in a.name == models.smart }
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
                    "smolx": {
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
            // crush.json (NOT the file itself).
            //
            // Crush has two distinct model-related sections in its config:
            //   - `providers.<id>.models[]`  — the picker list (what's
            //                                  available to choose from).
            //   - `models.large` / `models.small` — explicit tier
            //                                  assignments that pin which
            //                                  provider-qualified model id
            //                                  fills each tier role.
            //
            // We populate both: the picker carries all installed models
            // so the user can switch with `/model`, and the tier
            // assignments pin our `smart`/`small` choices on launch.
            // (Crush has no middle slot, so `fast` is unused here.)
            let crushConfigDir = Paths.appRoot
                .appendingPathComponent("crush", isDirectory: true).path
            let crushConfigPath = crushConfigDir + "/crush.json"
            let crushModelEntries = (installedModels.isEmpty
                ? [ModelDescriptor(name: models.smart, repoId: "", localPath: "",
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
                    "smolx": {
                      "type": "openai-compat",
                      "base_url": "\(baseURL)/v1",
                      "api_key": "\(token)",
                      "models": [
                \(crushModelsJSON)
                      ]
                    }
                  },
                  "models": {
                    "large": { "model": "\(models.smart)", "provider": "smolx" },
                    "small": { "model": "\(models.small)", "provider": "smolx" }
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
