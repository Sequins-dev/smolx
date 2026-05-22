import Testing
import Foundation
@testable import smolx

@Suite("AgentEnvironment")
struct AgentEnvironmentTests {
    @Test func claudePlanSetsAnthropicEnvVars() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "claude",
            baseURL: "http://127.0.0.1:8080",
            models: ModelTuple(single: "llama-3.2-3b"),
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
        #expect(configDir?.contains(".smolx") == true)
        #expect(plan.files.count == 1)
        #expect(plan.files.first?.path.hasPrefix(configDir ?? "") == true)
    }

    @Test func codexPlanSandboxesConfigDir() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "codex",
            baseURL: "http://127.0.0.1:8080",
            models: ModelTuple(single: "x"),
            authToken: nil)
        #expect(plan.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1")
        #expect(plan.env["OPENAI_API_KEY"] == "smolx-local")
        // CODEX_HOME points away from the user's real ~/.codex so a
        // pre-existing config can't break the launched session.
        let home = plan.env["CODEX_HOME"]
        #expect(home != nil)
        #expect(home?.contains(".smolx") == true)
        // Pre-written auth.json bypasses the onboarding sign-in screen.
        // auth_mode must be "apikey" (lowercase) — codex 0.128 rejects "ApiKey".
        let auth = plan.files.first { $0.path.hasSuffix("auth.json") }
        #expect(auth != nil)
        #expect(auth?.contents.contains("\"auth_mode\": \"apikey\"") == true)
        #expect(auth?.contents.contains("\"OPENAI_API_KEY\": \"smolx-local\"") == true)
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
        #expect(prefix.contains("model_provider=\"smolx\""))
        #expect(prefix.contains(where: { $0.contains("model_providers.smolx.base_url") }))
        #expect(prefix.contains(where: { $0.contains("model_providers.smolx.wire_api=\"responses\"") }))
    }

    @Test func aiderPlanSetsOpenAIEnvVars() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "aider", baseURL: "http://127.0.0.1:8080",
            models: ModelTuple(single: "x"), authToken: nil)
        #expect(plan.env["OPENAI_API_BASE"] == "http://127.0.0.1:8080/v1")
        #expect(plan.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1")
        // Falls back to a placeholder token when none is supplied.
        #expect(plan.env["OPENAI_API_KEY"] == "smolx-local")
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
            models: ModelTuple(single: "a"), authToken: "secret",
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
        #expect(cfg.contains("\"smolx/a\""))
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

    @Test func piPlanSandboxesAgentDirAndListsAllModels() throws {
        let installed = [
            ModelDescriptor(name: "a", repoId: "x/a", localPath: "/p/a",
                            capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(name: "b", repoId: "x/b", localPath: "/p/b",
                            capability: .vision, diskSizeBytes: 200, addedAt: Date()),
        ]
        let plan = try AgentEnvironment.plan(
            agentName: "pi", baseURL: "http://127.0.0.1:8080",
            models: ModelTuple(single: "b"), authToken: "secret",
            installedModels: installed)
        #expect(plan.executable == "pi")
        // PI_CODING_AGENT_DIR points away from the user's real ~/.pi/agent
        // so a pre-existing Pi install can't bleed in (or get clobbered).
        let dir = plan.env["PI_CODING_AGENT_DIR"]
        #expect(dir != nil)
        #expect(dir?.contains(".smolx") == true)
        // models.json gets written under that sandboxed dir.
        let models = plan.files.first { $0.path.hasSuffix("models.json") }
        #expect(models != nil)
        #expect(models?.path.hasPrefix(dir ?? "") == true)
        let cfg = models?.contents ?? ""
        // openai-completions is Pi's API tag for OpenAI Chat Completions —
        // what smolx actually serves.
        #expect(cfg.contains("\"api\": \"openai-completions\""))
        #expect(cfg.contains("http://127.0.0.1:8080/v1"))
        #expect(cfg.contains("\"apiKey\": \"secret\""))
        // Both installed models surface in Pi's /model picker.
        #expect(cfg.contains("\"a\""))
        #expect(cfg.contains("\"b\""))
        // The --model entry is listed first so it's Pi's default selection.
        let posB = cfg.range(of: "\"id\": \"b\"")?.lowerBound
        let posA = cfg.range(of: "\"id\": \"a\"")?.lowerBound
        #expect(posB != nil && posA != nil)
        #expect(posB! < posA!)
        // Vision model gets "image" in its input array.
        #expect(cfg.contains("\"image\""))
    }

    @Test func crushPlanInjectsOpenAICompatProvider() throws {
        let installed = [
            ModelDescriptor(name: "m1", repoId: "x/m1", localPath: "/p/m1",
                            capability: .text, diskSizeBytes: 100, addedAt: Date()),
        ]
        let plan = try AgentEnvironment.plan(
            agentName: "crush", baseURL: "http://127.0.0.1:8080",
            models: ModelTuple(single: "m1"), authToken: "tok",
            installedModels: installed)
        #expect(plan.executable == "crush")
        // CRUSH_GLOBAL_CONFIG points at the config *directory*, not the
        // crush.json file — Crush appends `/crush.json` internally. Setting
        // it to a file path produces `…/crush.json/crush.json: not a
        // directory` at load time.
        let cfgDir = plan.env["CRUSH_GLOBAL_CONFIG"]
        #expect(cfgDir != nil)
        #expect(cfgDir?.contains(".smolx") == true)
        #expect(cfgDir?.hasSuffix("crush.json") == false)
        // crush.json is written inside that directory.
        let file = plan.files.first { $0.path == (cfgDir ?? "") + "/crush.json" }
        #expect(file != nil)
        let cfg = file?.contents ?? ""
        // `openai-compat` is Crush's documented type for OpenAI-shaped
        // self-hosted endpoints (vs `openai` which proxies through OpenAI).
        #expect(cfg.contains("\"type\": \"openai-compat\""))
        #expect(cfg.contains("\"base_url\": \"http://127.0.0.1:8080/v1\""))
        #expect(cfg.contains("\"api_key\": \"tok\""))
        #expect(cfg.contains("\"id\": \"m1\""))
        // `context_window` is required by Crush's schema; the provider is
        // silently dropped from the UI if it's missing.
        #expect(cfg.contains("\"context_window\":"))
        #expect(cfg.contains("\"default_max_tokens\":"))
    }

    @Test func unknownAgentThrows() {
        #expect(throws: AgentEnvironment.ResolveError.self) {
            _ = try AgentEnvironment.plan(
                agentName: "doesnotexist",
                baseURL: "x", models: ModelTuple(single: "y"), authToken: nil)
        }
    }

    // MARK: - Three-tier wiring per agent

    /// A three-tier tuple where each tier carries a recognisable string
    /// so tests can assert which tier landed where.
    private static let tieredTuple = ModelTuple(
        smart: "S-smart", fast: "F-fast", small: "T-small")

    @Test func claudePlanWiresThreeTierEnvVars() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "claude", baseURL: "http://localhost:8080",
            models: Self.tieredTuple, authToken: "tok")
        // The three native Claude Code override env vars line up with the
        // smart/fast/small tiers per Claude Code's model-config docs.
        #expect(plan.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "S-smart")
        #expect(plan.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "F-fast")
        #expect(plan.env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "T-small")
        // ANTHROPIC_SMALL_FAST_MODEL is the deprecated alias for haiku —
        // setting both survives version skew between Claude Code releases.
        #expect(plan.env["ANTHROPIC_SMALL_FAST_MODEL"] == "T-small")
        // ANTHROPIC_MODEL is the top-level "default tier" choice — we
        // point it at `fast` (= sonnet) so the /model picker opens on
        // the user's mid-tier pick, matching claude's own default.
        #expect(plan.env["ANTHROPIC_MODEL"] == "F-fast")
    }

    @Test func aiderPlanWiresThreeTierFlags() throws {
        let plan = try AgentEnvironment.plan(
            agentName: "aider", baseURL: "http://localhost:8080",
            models: Self.tieredTuple, authToken: nil)
        // aider's three CLI flags map 1:1 onto our tiers:
        //   --model         smart  (main editing model)
        //   --editor-model  fast   (diff-generation)
        //   --weak-model    small  (commit messages / history summaries)
        let args = plan.prefixArgs
        #expect(args.contains("--model"))
        #expect(args.contains("S-smart"))
        #expect(args.contains("--editor-model"))
        #expect(args.contains("F-fast"))
        #expect(args.contains("--weak-model"))
        #expect(args.contains("T-small"))
        // Each flag is followed by its value (positional in argv).
        if let i = args.firstIndex(of: "--model") {
            #expect(args[args.index(after: i)] == "S-smart")
        } else { Issue.record("missing --model") }
        if let i = args.firstIndex(of: "--weak-model") {
            #expect(args[args.index(after: i)] == "T-small")
        } else { Issue.record("missing --weak-model") }
    }

    @Test func opencodePlanWiresSmartAndSmallModel() throws {
        let installed = [
            ModelDescriptor(name: "S-smart", repoId: "x/s", localPath: "/p/s",
                            capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(name: "T-small", repoId: "x/t", localPath: "/p/t",
                            capability: .text, diskSizeBytes: 50, addedAt: Date()),
        ]
        let plan = try AgentEnvironment.plan(
            agentName: "opencode", baseURL: "http://localhost:8080",
            models: Self.tieredTuple, authToken: "tok", installedModels: installed)
        let cfg = plan.env["OPENCODE_CONFIG_CONTENT"] ?? ""
        // Top-level `model` picks the smart tier; `small_model` picks the
        // small tier. opencode has no middle slot so `fast` doesn't land
        // anywhere here.
        #expect(cfg.contains("\"model\": \"smolx/S-smart\""))
        #expect(cfg.contains("\"small_model\": \"smolx/T-small\""))
    }

    @Test func crushPlanWiresLargeAndSmallTiers() throws {
        let installed = [
            ModelDescriptor(name: "S-smart", repoId: "x/s", localPath: "/p/s",
                            capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(name: "T-small", repoId: "x/t", localPath: "/p/t",
                            capability: .text, diskSizeBytes: 50, addedAt: Date()),
        ]
        let plan = try AgentEnvironment.plan(
            agentName: "crush", baseURL: "http://localhost:8080",
            models: Self.tieredTuple, authToken: "tok", installedModels: installed)
        let file = plan.files.first { $0.path.hasSuffix("crush.json") }
        let cfg = file?.contents ?? ""
        // Crush's two-slot split lands at the root `models` object —
        // `large` for our smart tier, `small` for our small tier. The
        // `fast` tier has no native home on crush.
        #expect(cfg.contains("\"large\": { \"model\": \"S-smart\""))
        #expect(cfg.contains("\"small\": { \"model\": \"T-small\""))
    }
}
