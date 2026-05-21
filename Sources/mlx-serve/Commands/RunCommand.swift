import ArgumentParser
import Darwin
import Foundation

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Launch an agent CLI with env vars pointing at the local mlx-serve instance.",
        discussion: """
            Supported agents: \(AgentEnvironment.Kind.allCases.map(\.rawValue).joined(separator: ", ")).
            Requires `mlx-serve serve` to be already running (pass --no-check to skip the health probe).
            Pass agent-specific arguments after `--`, e.g.
              mlx-serve run claude --model llama-3.2-3b -- "summarise this README"
            """
    )

    @Argument(help: "Agent CLI to launch.")
    var agent: String

    @Option(name: .long, help: "Model to use; defaults to the first registered model.")
    var model: String?

    @Option(name: .long, help: "URL of an already-running mlx-serve. Defaults to http://127.0.0.1:8080.")
    var baseUrl: String = "http://127.0.0.1:8080"

    @Option(name: .long, help: "Bearer token to send with requests (only needed if the server requires one).")
    var authToken: String?

    @Flag(name: .long, help: "Skip the /healthz probe that verifies the server is running before launching the agent.")
    var noCheck: Bool = false

    // `.postTerminator` only captures args appearing AFTER a literal `--`,
    // so `mlx-serve run claude --base-url X -- some prompt` parses correctly:
    // `--base-url X` is consumed by mlx-serve, `some prompt` goes to claude.
    // With the old `.captureForPassthrough` strategy, everything after the
    // agent name (including our own flags) was greedily forwarded to the
    // child, which then errored out on flags it didn't recognise.
    @Argument(parsing: .postTerminator, help: "Arguments forwarded to the agent CLI. Place after `--`.")
    var passthrough: [String] = []

    func run() async throws {
        let registry = ModelRegistry()
        let installedModels = (try? registry.load()) ?? []
        let modelName: String
        if let m = model {
            modelName = m
        } else if let first = installedModels.first?.name {
            modelName = first
        } else {
            FileHandle.standardError.write(Data(
                "No model installed. Use `mlx-serve pull <repo-id>` first, or pass --model.\n".utf8))
            throw ExitCode.failure
        }

        let plan: AgentEnvironment.Plan
        do {
            plan = try AgentEnvironment.plan(
                agentName: agent,
                baseURL: baseUrl,
                modelName: modelName,
                authToken: authToken,
                installedModels: installedModels)
        } catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            throw ExitCode.failure
        }

        // Health-check the server first so we fail fast with a clear error
        // instead of letting the agent hang waiting for a non-existent endpoint.
        if !noCheck {
            if !(await Self.serverReachable(baseUrl)) {
                FileHandle.standardError.write(Data("""
                    mlx-serve is not reachable at \(baseUrl).
                    Start it in another shell: `mlx-serve serve` — or pass --no-check to skip this probe.

                    """.utf8))
                throw ExitCode.failure
            }
        }

        // Write any per-agent temp config files (e.g. opencode config,
        // claude config-dir placeholder). These persist after we exec into
        // the agent — `/tmp` cleanup is left to the OS. No `defer { remove }`
        // here because exec replaces our process and defer wouldn't run anyway.
        for file in plan.files {
            let parent = (file.path as NSString).deletingLastPathComponent
            do {
                if !parent.isEmpty {
                    try FileManager.default.createDirectory(
                        atPath: parent, withIntermediateDirectories: true)
                }
                try file.contents.write(
                    toFile: file.path, atomically: true, encoding: .utf8)
            } catch {
                FileHandle.standardError.write(Data(
                    "Failed to write \(file.path): \(error)\n".utf8))
                throw ExitCode.failure
            }
        }

        guard let executablePath = Self.resolveExecutable(plan.executable) else {
            FileHandle.standardError.write(Data(
                "Could not find `\(plan.executable)` on PATH. Install it and try again.\n".utf8))
            throw ExitCode.failure
        }

        // execve replaces our process with the agent so the terminal is
        // inherited intact — Foundation's Process attaches stdin/stdout but
        // doesn't give the child a controlling TTY, which is why
        // `mlx-serve run claude` used to hang at the interactive prompt.
        Self.execAgent(
            executablePath: executablePath,
            arguments: plan.prefixArgs + passthrough,
            envOverrides: plan.env,
            envUnset: plan.unsetEnv)
        // execAgent is `Never`-returning on success and exits non-zero on
        // failure, so anything past this point is unreachable.
    }

    // MARK: - Helpers

    private static func resolveExecutable(_ name: String) -> String? {
        let env = ProcessInfo.processInfo.environment
        let path = env["PATH"] ?? "/usr/bin:/bin:/usr/local/bin"
        let home = env["HOME"] ?? NSHomeDirectory()

        // Skip known agent-CLI wrapper directories when resolving the four
        // agent binaries. The Supacode bash wrapper at ~/.superset/bin/codex
        // (and analogous wrappers for the other agents) unconditionally adds
        // flags like `--enable hooks` that newer agent versions reject, so
        // we fall through to whatever the wrapper itself would have located:
        // the underlying real binary further down PATH. This matches the
        // wrapper's own `find_real_binary` strategy.
        let isAgentName = AgentEnvironment.Kind(rawValue: name) != nil
        let supersetBin = "\(home)/.superset/bin"

        for dir in path.split(separator: ":") {
            let d = String(dir)
            if isAgentName {
                if d == supersetBin || d.hasPrefix("\(home)/.superset-") {
                    continue
                }
            }
            let candidate = "\(d)/\(name)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func serverReachable(_ baseURL: String) async -> Bool {
        guard let url = URL(string: baseURL + "/healthz") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    /// Replaces the current process with the agent binary. We use `execve`
    /// (path + env explicit) rather than `execvp` so the child sees exactly
    /// the merged environment we constructed. `envUnset` is applied *after*
    /// the overrides so plans can explicitly drop pre-existing inherited
    /// vars that would conflict with what they set (e.g. Claude Code
    /// rejecting both `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`).
    private static func execAgent(
        executablePath: String,
        arguments: [String],
        envOverrides: [String: String],
        envUnset: [String]
    ) -> Never {
        var env = ProcessInfo.processInfo.environment
        for (k, v) in envOverrides { env[k] = v }
        for k in envUnset { env.removeValue(forKey: k) }

        // Build the C-string argv and envp. We deliberately strdup each
        // entry so the buffers outlive any Swift String autorelease pool;
        // execve doesn't return on success so leaks here are irrelevant.
        let argvStrings: [String] = [executablePath] + arguments
        let envStrings: [String] = env.map { "\($0.key)=\($0.value)" }

        let cArgv: [UnsafeMutablePointer<CChar>?] =
            argvStrings.map { strdup($0) } + [nil]
        let cEnvp: [UnsafeMutablePointer<CChar>?] =
            envStrings.map { strdup($0) } + [nil]

        // Keep both buffers alive across the execve call by holding them in
        // contiguous storage and passing the base pointers.
        var argvStorage = cArgv
        var envpStorage = cEnvp
        argvStorage.withUnsafeMutableBufferPointer { argvBuf in
            envpStorage.withUnsafeMutableBufferPointer { envpBuf in
                _ = execve(executablePath, argvBuf.baseAddress, envpBuf.baseAddress)
            }
        }

        // Only reached if execve failed.
        let err = errno
        let message = String(cString: strerror(err))
        FileHandle.standardError.write(Data(
            "execve(\(executablePath)) failed (errno \(err)): \(message)\n".utf8))
        Darwin.exit(1)
    }
}
