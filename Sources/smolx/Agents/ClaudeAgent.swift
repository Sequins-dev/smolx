import Foundation

struct ClaudeAgent: AgentPlugin {
    let commandName = "claude"

    func plan(
        baseURL: String,
        models: UserConfig,
        authToken: String,
        installedModels: [ModelDescriptor]
    ) -> AgentPlan {
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
        return AgentPlan(
            executable: "claude",
            env: [
                "ANTHROPIC_BASE_URL": baseURL,
                "ANTHROPIC_AUTH_TOKEN": authToken,
                "ANTHROPIC_MODEL": models.fast ?? "",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": models.smart ?? "",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": models.fast ?? "",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": models.small ?? "",
                "ANTHROPIC_SMALL_FAST_MODEL": models.small ?? "",
                "CLAUDE_CONFIG_DIR": configDir,
            ],
            unsetEnv: ["ANTHROPIC_API_KEY"],
            files: [.init(path: configDir + "/.keep", contents: "")])
    }
}
