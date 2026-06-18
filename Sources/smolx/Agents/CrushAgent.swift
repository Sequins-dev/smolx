import Foundation

struct CrushAgent: AgentPlugin {
    let commandName = "crush"

    func plan(
        baseURL: String,
        models: UserConfig,
        authToken: String,
        installedModels: [ModelDescriptor]
    ) -> AgentPlan {
        // Charm's Crush reads its config from .crush.json (CWD),
        // crush.json (CWD), or $HOME/.config/crush/crush.json. The env
        // var CRUSH_GLOBAL_CONFIG points at the *directory* containing
        // crush.json (NOT the file itself).
        //
        // Crush has two config-layer concepts:
        //   - `providers.<id>.models[]`   picker list shown in /model
        //   - `models.large` / `models.small ?? ""`  active tier assignments
        //
        // CRUSH_GLOBAL_CONFIG only redirects the config dir. Crush also
        // reads persisted model-selection state from a separate data dir
        // (default: ~/.local/share/crush/). That saved state overrides
        // the config's models.large/small assignments, so the user's
        // previous selection would win. --data-dir isolates the data dir
        // too, giving us a fresh state where our config pins apply.
        // We also write a state file to that data dir with the API key
        // and tier assignments so Crush can authenticate our provider.
        let crushConfigDir = Paths.appRoot
            .appendingPathComponent("crush", isDirectory: true).path
        let crushDataDir = Paths.appRoot
            .appendingPathComponent("crush-data", isDirectory: true).path

        let modelEntries =
            installedModels.isEmpty
            ? [
                ModelDescriptor(
                    name: models.smart ?? "", repoId: "", localPath: "",
                    capability: .text, diskSizeBytes: 0, addedAt: Date())
            ]
            : installedModels
        let crushModelsJSON = modelEntries.map { m in
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
                  "api_key": "\(authToken)",
                  "models": [
            \(crushModelsJSON)
                  ]
                }
              },
              "models": {
                "large": { "model": "\(models.smart ?? "")", "provider": "smolx" },
                "small": { "model": "\(models.small ?? "")", "provider": "smolx" }
              }
            }
            """

        // State file: Crush reads API keys and the active model selection
        // from the data dir's crush.json, not the config. We write it
        // fresh each run so the smolx provider is credentialed and both
        // tier slots point at our models.
        let crushState = """
            {
              "providers": {
                "smolx": { "api_key": "\(authToken)" }
              },
              "models": {
                "large": { "model": "\(models.smart ?? "")", "provider": "smolx" },
                "small": { "model": "\(models.small ?? "")", "provider": "smolx" }
              }
            }
            """

        return AgentPlan(
            executable: "crush",
            env: ["CRUSH_GLOBAL_CONFIG": crushConfigDir],
            prefixArgs: ["--data-dir", crushDataDir],
            files: [
                .init(path: crushConfigDir + "/crush.json", contents: crushConfig),
                .init(path: crushDataDir + "/crush.json", contents: crushState),
            ])
    }
}
