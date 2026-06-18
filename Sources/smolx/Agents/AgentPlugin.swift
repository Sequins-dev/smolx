import Foundation
import Logging

/// Contract every agent plugin must satisfy.
protocol AgentPlugin: Sendable {
    /// The name the user types: `smolx run <commandName>`.
    var commandName: String { get }

    /// Build the execution plan for this agent.
    func plan(
        baseURL: String,
        models: UserConfig,
        authToken: String,
        installedModels: [ModelDescriptor]
    ) -> AgentPlan

    /// Optional pre-exec hook (e.g. writing a catalog file). Default: no-op.
    func setup(installedModels: [ModelDescriptor], logger: Logger)
}

extension AgentPlugin {
    func setup(installedModels: [ModelDescriptor], logger: Logger) {}
}

// MARK: - Plan

struct AgentPlan: Sendable {
    var executable: String
    /// Environment variables to set before exec.
    var env: [String: String]
    /// Variables to remove from the inherited environment before exec.
    var unsetEnv: [String]
    /// Args prepended before the user's `--` passthrough.
    var prefixArgs: [String]
    /// Files written to disk before exec.
    var files: [TempFile]

    struct TempFile: Sendable {
        var path: String
        var contents: String
    }

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
}

// MARK: - Registry

enum AgentRegistry {
    static let all: [any AgentPlugin] = [
        ClaudeAgent(),
        CodexAgent(),
        OpencodeAgent(),
        PiAgent(),
        CrushAgent(),
    ]

    static func find(named name: String) -> (any AgentPlugin)? {
        all.first { $0.commandName == name }
    }
}
