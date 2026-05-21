import Testing
import Foundation
@testable import mlx_serve

@Suite("AgentEnvironment")
struct AgentEnvironmentTests {
    @Test func claudePlanSetsAnthropicEnvVars() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "claude",
            baseURL: "http://127.0.0.1:8080",
            modelName: "llama-3.2-3b",
            authToken: "tok")
        #expect(plan.executable == "claude")
        #expect(plan.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080")
        #expect(plan.env["ANTHROPIC_AUTH_TOKEN"] == "tok")
        #expect(plan.env["ANTHROPIC_MODEL"] == "llama-3.2-3b")
        // Don't set ANTHROPIC_API_KEY — Claude Code warns about auth conflict
        // when both are set. Do unset it so a pre-existing shell export
        // doesn't bleed into the child.
        #expect(plan.env["ANTHROPIC_API_KEY"] == nil)
        #expect(plan.unsetEnv.contains("ANTHROPIC_API_KEY"))
        // CLAUDE_CONFIG_DIR is redirected so the launched session doesn't see
        // a real ~/.claude /login key (which would also trigger an auth conflict
        // warning even though it lives on disk rather than in env).
        let configDir = plan.env["CLAUDE_CONFIG_DIR"]
        #expect(configDir != nil)
        #expect(configDir?.contains(".mlx-serve") == true)
        #expect(plan.files.count == 1)
        #expect(plan.files.first?.path.hasPrefix(configDir ?? "") == true)
    }

    @Test func codexPlanSandboxesConfigDir() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "codex",
            baseURL: "http://127.0.0.1:8080",
            modelName: "x",
            authToken: nil)
        #expect(plan.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1")
        #expect(plan.env["OPENAI_API_KEY"] == "mlx-serve-local")
        // CODEX_HOME points away from the user's real ~/.codex so a
        // pre-existing config can't break the launched session.
        let home = plan.env["CODEX_HOME"]
        #expect(home != nil)
        #expect(home?.contains(".mlx-serve") == true)
        // Pre-written auth.json bypasses the onboarding sign-in screen.
        // auth_mode must be "apikey" (lowercase) — codex 0.128 rejects "ApiKey".
        let auth = plan.files.first { $0.path.hasSuffix("auth.json") }
        #expect(auth != nil)
        #expect(auth?.contents.contains("\"auth_mode\": \"apikey\"") == true)
        #expect(auth?.contents.contains("\"OPENAI_API_KEY\": \"mlx-serve-local\"") == true)
        // Model + provider config goes through `-c key=value` CLI overrides
        // (not config.toml) because codex rewrites its own config.toml on
        // startup and strips unknown top-level keys, which would otherwise
        // erase our local-provider definition every run.
        let prefix = plan.prefixArgs
        #expect(prefix.contains("-c"))
        #expect(prefix.contains("model=\"x\""))
        // Custom provider id (not `openai` — codex 0.128 forbids overriding
        // built-in provider ids and refuses to start with "model_providers
        // contains reserved built-in provider IDs").
        #expect(prefix.contains("model_provider=\"mlx_serve\""))
        #expect(prefix.contains(where: { $0.contains("model_providers.mlx_serve.base_url") }))
        #expect(prefix.contains(where: { $0.contains("model_providers.mlx_serve.wire_api=\"responses\"") }))
    }

    @Test func aiderPlanSetsOpenAIEnvVars() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "aider", baseURL: "http://127.0.0.1:8080",
            modelName: "x", authToken: nil)
        #expect(plan.env["OPENAI_API_BASE"] == "http://127.0.0.1:8080/v1")
        #expect(plan.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1")
        // Falls back to a placeholder token when none is supplied.
        #expect(plan.env["OPENAI_API_KEY"] == "mlx-serve-local")
    }

    @Test func opencodePlanInjectsInlineConfigWithAllModels() throws {
        let installed = [
            ModelDescriptor(name: "a", repoId: "x/a", localPath: "/p/a",
                            capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(name: "b", repoId: "x/b", localPath: "/p/b",
                            capability: .vision, diskSizeBytes: 200, addedAt: Date()),
        ]
        let plan = try AgentEnvironment.plan(
            agentName: "opencode", baseURL: "http://127.0.0.1:8080",
            modelName: "a", authToken: "secret",
            installedModels: installed)
        let inline = plan.env["OPENCODE_CONFIG_CONTENT"]
        #expect(inline != nil)
        let cfg = inline ?? ""
        #expect(cfg.contains("http://127.0.0.1:8080/v1"))
        #expect(cfg.contains("secret"))
        // Both registered models appear in the models map so opencode's UI
        // surfaces them as options — not just the one passed as --model.
        #expect(cfg.contains("\"a\""))
        #expect(cfg.contains("\"b\""))
        // The top-level model selector follows the `<provider>/<id>` form.
        #expect(cfg.contains("\"mlx-serve/a\""))
        // SDK declared as openai-compatible — what opencode recommends for
        // self-hosted OpenAI-API-compatible servers like ours.
        #expect(cfg.contains("@ai-sdk/openai-compatible"))
        // `limit` is required by opencode's schema; without it the provider
        // is silently dropped from the UI.
        #expect(cfg.contains("\"context\":"))
        #expect(cfg.contains("\"output\":"))
        // Vision-capable model gets attachment: true.
        #expect(cfg.contains("\"attachment\": true"))
    }

    @Test func unknownAgentThrows() {
        #expect(throws: AgentEnvironment.ResolveError.self) {
            _ = try AgentEnvironment.plan(
                agentName: "doesnotexist",
                baseURL: "x", modelName: "y", authToken: nil)
        }
    }
}
