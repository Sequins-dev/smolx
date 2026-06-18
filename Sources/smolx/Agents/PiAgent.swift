import Foundation

struct PiAgent: AgentPlugin {
    let commandName = "pi"

    func plan(
        baseURL: String,
        models: UserConfig,
        authToken: String,
        installedModels: [ModelDescriptor]
    ) -> AgentPlan {
        // Pi (pi.dev / @earendil-works/pi-coding-agent) reads providers
        // from ~/.pi/agent/models.json by default. PI_CODING_AGENT_DIR
        // overrides that directory, so we point it at a sandbox under
        // ~/.smolx/ to avoid touching the user's real Pi state.
        //
        // Pi has no native fast/small split — it's a single picker.
        // We use `models.smart ?? ""` as the default via --provider/--model;
        // the other tiers are ignored.
        let piHome = Paths.appRoot
            .appendingPathComponent("pi-agent", isDirectory: true).path
        let modelEntries =
            installedModels.isEmpty
            ? [
                ModelDescriptor(
                    name: models.smart ?? "", repoId: "", localPath: "",
                    capability: .text, diskSizeBytes: 0, addedAt: Date())
            ]
            : installedModels
        let piOrdered = modelEntries.sorted { a, _ in a.name == models.smart ?? "" }
        let piModelsJSON = piOrdered.map { m in
            let inputs = m.capability == .vision ? "[\"text\", \"image\"]" : "[\"text\"]"
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
                  "apiKey": "\(authToken)",
                  "models": [
            \(piModelsJSON)
                  ]
                }
              }
            }
            """
        return AgentPlan(
            executable: "pi",
            env: ["PI_CODING_AGENT_DIR": piHome],
            prefixArgs: ["--provider", "smolx", "--model", models.smart ?? ""],
            files: [.init(path: piHome + "/models.json", contents: piModelsConfig)])
    }
}
