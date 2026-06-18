import Foundation
import Testing

@testable import smolx

@Suite("AgentPlugins")
struct AgentEnvironmentTests {

    // A three-tier config where each tier carries a recognisable string.
    private static let tiered = UserConfig(smart: "S-smart", fast: "F-fast", small: "T-small")
    private static let single = UserConfig(smart: "x", fast: "x", small: "x")

    // MARK: - Claude

    @Test func claudePlanSetsAnthropicEnvVars() {
        let plan = ClaudeAgent().plan(
            baseURL: "http://127.0.0.1:8080",
            models: Self.single,
            authToken: "tok",
            installedModels: [])
        #expect(plan.executable == "claude")
        #expect(plan.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080")
        #expect(plan.env["ANTHROPIC_AUTH_TOKEN"] == "tok")
        #expect(plan.env["ANTHROPIC_MODEL"] == "x")
        #expect(plan.env["ANTHROPIC_API_KEY"] == nil)
        #expect(plan.unsetEnv.contains("ANTHROPIC_API_KEY"))
        let configDir = plan.env["CLAUDE_CONFIG_DIR"]
        #expect(configDir != nil)
        #expect(configDir?.contains(".smolx") == true)
        #expect(plan.files.count == 1)
        #expect(plan.files.first?.path.hasPrefix(configDir ?? "") == true)
    }

    @Test func claudePlanWiresThreeTierEnvVars() {
        let plan = ClaudeAgent().plan(
            baseURL: "http://localhost:8080",
            models: Self.tiered,
            authToken: "tok",
            installedModels: [])
        #expect(plan.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "S-smart")
        #expect(plan.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "F-fast")
        #expect(plan.env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "T-small")
        #expect(plan.env["ANTHROPIC_SMALL_FAST_MODEL"] == "T-small")
        #expect(plan.env["ANTHROPIC_MODEL"] == "F-fast")
    }

    // MARK: - Codex

    @Test func codexPlanSandboxesConfigDir() {
        let plan = CodexAgent().plan(
            baseURL: "http://127.0.0.1:8080",
            models: Self.single,
            authToken: "smolx-local",
            installedModels: [])
        #expect(plan.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1")
        #expect(plan.env["OPENAI_API_KEY"] == "smolx-local")
        let home = plan.env["CODEX_HOME"]
        #expect(home != nil)
        #expect(home?.contains(".smolx") == true)
        let auth = plan.files.first { $0.path.hasSuffix("auth.json") }
        #expect(auth != nil)
        #expect(auth?.contents.contains("\"auth_mode\": \"apikey\"") == true)
        #expect(auth?.contents.contains("\"OPENAI_API_KEY\": \"smolx-local\"") == true)
        let prefix = plan.prefixArgs
        #expect(prefix.contains("-c"))
        #expect(prefix.contains("model=\"x\""))
        #expect(prefix.contains("model_provider=\"smolx\""))
        #expect(prefix.contains(where: { $0.contains("model_providers.smolx.base_url") }))
        #expect(prefix.contains(where: { $0.contains("model_providers.smolx.wire_api=\"responses\"") }))
    }

    // MARK: - Opencode

    @Test func opencodePlanInjectsInlineConfigWithAllModels() {
        let installed = [
            ModelDescriptor(
                name: "a", repoId: "x/a", localPath: "/p/a",
                capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(
                name: "b", repoId: "x/b", localPath: "/p/b",
                capability: .vision, diskSizeBytes: 200, addedAt: Date()),
        ]
        let plan = OpencodeAgent().plan(
            baseURL: "http://127.0.0.1:8080",
            models: UserConfig(smart: "a", fast: "a", small: "a"),
            authToken: "secret",
            installedModels: installed)
        let inline = plan.env["OPENCODE_CONFIG_CONTENT"]
        #expect(inline != nil)
        let cfg = inline ?? ""
        #expect(cfg.contains("http://127.0.0.1:8080/v1"))
        #expect(cfg.contains("secret"))
        #expect(cfg.contains("\"a\""))
        #expect(cfg.contains("\"b\""))
        #expect(cfg.contains("\"smolx/a\""))
        #expect(cfg.contains("@ai-sdk/openai-compatible"))
        #expect(cfg.contains("\"context\":"))
        #expect(cfg.contains("\"output\":"))
        #expect(cfg.contains("\"attachment\": true"))
    }

    @Test func opencodePlanWiresSmartAndSmallModel() {
        let installed = [
            ModelDescriptor(
                name: "S-smart", repoId: "x/s", localPath: "/p/s",
                capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(
                name: "T-small", repoId: "x/t", localPath: "/p/t",
                capability: .text, diskSizeBytes: 50, addedAt: Date()),
        ]
        let plan = OpencodeAgent().plan(
            baseURL: "http://localhost:8080",
            models: Self.tiered,
            authToken: "tok",
            installedModels: installed)
        let cfg = plan.env["OPENCODE_CONFIG_CONTENT"] ?? ""
        #expect(cfg.contains("\"model\": \"smolx/S-smart\""))
        #expect(cfg.contains("\"small_model\": \"smolx/T-small\""))
    }

    // MARK: - Pi

    @Test func piPlanSandboxesAgentDirAndListsAllModels() {
        let installed = [
            ModelDescriptor(
                name: "a", repoId: "x/a", localPath: "/p/a",
                capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(
                name: "b", repoId: "x/b", localPath: "/p/b",
                capability: .vision, diskSizeBytes: 200, addedAt: Date()),
        ]
        let plan = PiAgent().plan(
            baseURL: "http://127.0.0.1:8080",
            models: UserConfig(smart: "b", fast: "b", small: "b"),
            authToken: "secret",
            installedModels: installed)
        #expect(plan.executable == "pi")
        let dir = plan.env["PI_CODING_AGENT_DIR"]
        #expect(dir != nil)
        #expect(dir?.contains(".smolx") == true)
        let models = plan.files.first { $0.path.hasSuffix("models.json") }
        #expect(models != nil)
        #expect(models?.path.hasPrefix(dir ?? "") == true)
        let cfg = models?.contents ?? ""
        #expect(cfg.contains("\"api\": \"openai-completions\""))
        #expect(cfg.contains("http://127.0.0.1:8080/v1"))
        #expect(cfg.contains("\"apiKey\": \"secret\""))
        #expect(cfg.contains("\"a\""))
        #expect(cfg.contains("\"b\""))
        let posB = cfg.range(of: "\"id\": \"b\"")?.lowerBound
        let posA = cfg.range(of: "\"id\": \"a\"")?.lowerBound
        #expect(posB != nil && posA != nil)
        #expect(posB! < posA!)
        #expect(cfg.contains("\"image\""))
    }

    // MARK: - Crush

    @Test func crushPlanInjectsOpenAICompatProvider() {
        let installed = [
            ModelDescriptor(
                name: "m1", repoId: "x/m1", localPath: "/p/m1",
                capability: .text, diskSizeBytes: 100, addedAt: Date())
        ]
        let plan = CrushAgent().plan(
            baseURL: "http://127.0.0.1:8080",
            models: UserConfig(smart: "m1", fast: "m1", small: "m1"),
            authToken: "tok",
            installedModels: installed)
        #expect(plan.executable == "crush")
        let cfgDir = plan.env["CRUSH_GLOBAL_CONFIG"]
        #expect(cfgDir != nil)
        #expect(cfgDir?.contains(".smolx") == true)
        #expect(cfgDir?.hasSuffix("crush.json") == false)
        let file = plan.files.first { $0.path == (cfgDir ?? "") + "/crush.json" }
        #expect(file != nil)
        let cfg = file?.contents ?? ""
        #expect(cfg.contains("\"type\": \"openai-compat\""))
        #expect(cfg.contains("\"base_url\": \"http://127.0.0.1:8080/v1\""))
        #expect(cfg.contains("\"api_key\": \"tok\""))
        #expect(cfg.contains("\"id\": \"m1\""))
        #expect(cfg.contains("\"context_window\":"))
        #expect(cfg.contains("\"default_max_tokens\":"))
    }

    @Test func crushPlanWiresLargeAndSmallTiers() {
        let installed = [
            ModelDescriptor(
                name: "S-smart", repoId: "x/s", localPath: "/p/s",
                capability: .text, diskSizeBytes: 100, addedAt: Date()),
            ModelDescriptor(
                name: "T-small", repoId: "x/t", localPath: "/p/t",
                capability: .text, diskSizeBytes: 50, addedAt: Date()),
        ]
        let plan = CrushAgent().plan(
            baseURL: "http://localhost:8080",
            models: Self.tiered,
            authToken: "tok",
            installedModels: installed)
        // Check both config and state files for the tier assignments.
        let cfg = plan.files.compactMap { $0.contents }.joined()
        #expect(cfg.contains("\"large\": { \"model\": \"S-smart\""))
        #expect(cfg.contains("\"small\": { \"model\": \"T-small\""))
    }

    // MARK: - Registry

    @Test func unknownAgentReturnsNil() {
        #expect(AgentRegistry.find(named: "doesnotexist") == nil)
    }
}
