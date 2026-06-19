import Foundation

struct OpencodeAgent: AgentPlugin {
    let commandName = "opencode"

    func plan(
        baseURL: String,
        models: UserConfig,
        authToken: String,
        installedModels: [ModelDescriptor]
    ) -> AgentPlan {
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
        //
        // Tier mapping: top-level `model` = smart, top-level `small_model`
        // = small. Opencode has no middle slot so `fast` is unused here.
        let modelEntries =
            installedModels.isEmpty
            ? [
                ModelDescriptor(
                    name: models.smart ?? "", repoId: "", localPath: "",
                    capability: .text, diskSizeBytes: 0, addedAt: Date())
            ]
            : installedModels
        let modelsJSON = modelEntries.map { m in
            let context = Self.contextLimit(for: m)
            let output = Self.outputLimit(for: m, context: context)
            return """
              "\(m.name)": {
                "id": "\(m.name)",
                "name": "\(m.name)",
                "tool_call": true,
                "temperature": true,
                "attachment": \(m.capability == .vision ? "true" : "false"),
                "limit": { "context": \(context), "output": \(output) }
              }
            """
        }.joined(separator: ",\n")
        let inlineConfig = """
            {
              "$schema": "https://opencode.ai/config.json",
              "model": "smolx/\(models.smart ?? "")",
              "small_model": "smolx/\(models.small ?? "")",
              "provider": {
                "smolx": {
                  "npm": "@ai-sdk/openai-compatible",
                  "name": "smolx",
                  "options": {
                    "baseURL": "\(baseURL)/v1",
                    "apiKey": "\(authToken)"
                  },
                  "models": {
            \(modelsJSON)
                  }
                }
              }
            }
            """
        // Collapse to one line so the env-var value stays single-line.
        let oneLineConfig =
            inlineConfig
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
        return AgentPlan(
            executable: "opencode",
            env: ["OPENCODE_CONFIG_CONTENT": oneLineConfig])
    }

    private static func contextLimit(for descriptor: ModelDescriptor) -> Int {
        ModelSnapshotInspector.contextLength(for: descriptor) ?? 32768
    }

    private static func outputLimit(for descriptor: ModelDescriptor, context: Int) -> Int {
        let contextLimit = max(1, context - 1)
        guard descriptor.weightFormat == .gguf else { return contextLimit }
        return min(GGUFPromptWindow.defaultMaxTokens, contextLimit)
    }
}
