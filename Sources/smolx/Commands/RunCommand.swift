import ArgumentParser
import Darwin
import Foundation
import Logging

struct RunCommand: AsyncParsableCommand {
    static let defaultBaseURL = "http://127.0.0.1:8080"

    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Launch an agent CLI connected to a local or remote smolx server.",
        discussion: """
            Supported agents: \(AgentRegistry.all.map(\.commandName).joined(separator: ", ")).
            Requires `smolx serve` to be running locally, or pass --base-url for a remote server.
            Pass --no-check to skip the health probe.
            Pass agent-specific arguments after `--`, e.g.
              smolx run claude --model llama-3.2-3b -- "summarise this README"
            """
    )

    @Argument(help: "Agent CLI to launch.")
    var agent: String

    @Option(
        name: .long,
        help:
            "Model to use for ALL tiers for this invocation. Per-tier flags like --smart override this for their tier."
    )
    var model: String?

    @Option(name: .long, help: "Override the `smart` tier for this invocation only.")
    var smart: String?

    @Option(name: .long, help: "Override the `fast` tier for this invocation only.")
    var fast: String?

    @Option(name: .long, help: "Override the `small` tier for this invocation only.")
    var small: String?

    @Option(name: .long, help: "URL of an already-running smolx. Defaults to http://127.0.0.1:8080.")
    var baseUrl: String = Self.defaultBaseURL

    @Option(
        name: .long,
        help: "Bearer token to send with requests (only needed if the server requires one).")
    var authToken: String?

    @Flag(
        name: .long,
        help: "Skip the /healthz probe that verifies the server is running before launching the agent.")
    var noCheck: Bool = false

    // `.postTerminator` only captures args appearing AFTER a literal `--`,
    // so `smolx run claude --base-url X -- some prompt` parses correctly:
    // `--base-url X` is consumed by smolx, `some prompt` goes to claude.
    // With the old `.captureForPassthrough` strategy, everything after the
    // agent name (including our own flags) was greedily forwarded to the
    // child, which then errored out on flags it didn't recognise.
    @Argument(
        parsing: .postTerminator, help: "Arguments forwarded to the agent CLI. Place after `--`.")
    var passthrough: [String] = []

    var usesRemoteServer: Bool { baseUrl != Self.defaultBaseURL }
    var modelsURL: URL? { endpointURL(path: "v1/models") }
    var healthURL: URL? { endpointURL(path: "healthz") }

    mutating func validate() throws {
        baseUrl = baseUrl.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard
            let url = URL(string: baseUrl),
            ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
            url.host != nil
        else {
            throw ValidationError("--base-url must be an absolute HTTP or HTTPS URL.")
        }
    }

    func run() async throws {
        let registry = ModelRegistry()
        let installedModels: [ModelDescriptor]
        if usesRemoteServer {
            installedModels = try await remoteModels()
        } else {
            installedModels = (try? registry.load()) ?? []
        }
        let userConfig = (try? UserConfig.load()) ?? UserConfig()

        guard
            let models = userConfig.resolve(
                smartOverride: smart,
                fastOverride: fast,
                smallOverride: small,
                modelSugar: model,
                firstInstalled: installedModels.first?.name)
        else {
            let guidance =
                usesRemoteServer
                ? "Pass --model, or configure at least one model on the remote server."
                : "Use `smolx pull <repo-id>` first, or pass --model."
            FileHandle.standardError.write(
                Data("No model is available. \(guidance)\n".utf8))
            throw ExitCode.failure
        }

        // Validate every resolved tier exists in the selected server's catalog — refuse
        // to launch with a typo'd alias rather than letting the agent
        // surface "model not found" mid-session.
        for alias in Set([models.smart, models.fast, models.small].compactMap { $0 }) {
            if !installedModels.contains(where: { $0.matches(alias) }) {
                let location = usesRemoteServer ? " on \(baseUrl)" : ""
                FileHandle.standardError.write(
                    Data(
                        "Unknown model alias '\(alias)'\(location).\n".utf8))
                throw ExitCode.failure
            }
        }

        guard let plugin = AgentRegistry.find(named: agent) else {
            let supported = AgentRegistry.all.map(\.commandName).joined(separator: ", ")
            FileHandle.standardError.write(
                Data("Unknown agent: \(agent). Supported: \(supported)\n".utf8))
            throw ExitCode.failure
        }

        let plan = plugin.plan(
            baseURL: baseUrl,
            models: models,
            authToken: authToken ?? "smolx-local",
            installedModels: installedModels)

        // Health-check the server first so we fail fast with a clear error
        // instead of letting the agent hang waiting for a non-existent endpoint.
        if !noCheck {
            if !(await serverReachable()) {
                FileHandle.standardError.write(
                    Data(
                        """
                        smolx is not reachable at \(baseUrl).
                        Start it in another shell: `smolx serve` — or pass --no-check to skip this probe.

                        """.utf8))
                throw ExitCode.failure
            }
        }

        // Write any per-agent temp config files before exec.
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
                FileHandle.standardError.write(
                    Data(
                        "Failed to write \(file.path): \(error)\n".utf8))
                throw ExitCode.failure
            }
        }

        // Run the plugin's pre-exec setup hook (e.g. catalog generation).
        let logger = Logger(label: "smolx")
        plugin.setup(installedModels: installedModels, logger: logger)

        guard let executablePath = Self.resolveExecutable(plan.executable, isAgent: true) else {
            FileHandle.standardError.write(
                Data(
                    "Could not find `\(plan.executable)` on PATH. Install it and try again.\n".utf8))
            throw ExitCode.failure
        }

        // execve replaces our process with the agent so the terminal is
        // inherited intact — Foundation's Process attaches stdin/stdout but
        // doesn't give the child a controlling TTY, which is why
        // `smolx run claude` used to hang at the interactive prompt.
        Self.execAgent(
            executablePath: executablePath,
            arguments: plan.prefixArgs + passthrough,
            envOverrides: plan.env,
            envUnset: plan.unsetEnv)
        // execAgent is `Never`-returning on success and exits non-zero on
        // failure, so anything past this point is unreachable.
    }

    // MARK: - Helpers

    private static func resolveExecutable(_ name: String, isAgent: Bool) -> String? {
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
        let supersetBin = "\(home)/.superset/bin"

        for dir in path.split(separator: ":") {
            let d = String(dir)
            if isAgent {
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

    private func endpointURL(path: String) -> URL? {
        URL(string: baseUrl + "/" + path)
    }

    private func authenticatedRequest(url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        if let authToken {
            request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func remoteModels() async throws -> [ModelDescriptor] {
        guard let url = modelsURL else { throw ValidationError("Invalid --base-url.") }
        var request = authenticatedRequest(url: url)
        request.timeoutInterval = 5
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw ValidationError("Could not read the model catalog at \(url.absoluteString).")
            }
            return try JSONDecoder().decode(OpenAI.ModelsList.self, from: data).data.map { model in
                ModelDescriptor(
                    name: model.id,
                    repoId: model.id,
                    localPath: "",
                    capability: .text,
                    diskSizeBytes: 0,
                    addedAt: Date(timeIntervalSince1970: TimeInterval(model.created)))
            }
        } catch let error as ValidationError {
            throw error
        } catch {
            throw ValidationError(
                "Could not connect to the smolx server at \(baseUrl): \(error.localizedDescription)")
        }
    }

    private func serverReachable() async -> Bool {
        guard let url = healthURL else { return false }
        var request = authenticatedRequest(url: url)
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
        FileHandle.standardError.write(
            Data(
                "execve(\(executablePath)) failed (errno \(err)): \(message)\n".utf8))
        Darwin.exit(1)
    }
}
