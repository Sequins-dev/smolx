import ArgumentParser
import Foundation

enum ProxyLaunchdService {
    static let label = "dev.sequins.smolx.proxy"

    struct Configuration: Equatable {
        var executableURL: URL
        var target: ProxyTarget
        var baseURL: String
        var bind: String = "127.0.0.1"
        var port: Int = 8080
        var authToken: String?
        var upstreamAuthToken: String?
        var verbose: Int = 0
        var standardOutputURL: URL = ProxyLaunchdService.standardOutputURL
        var standardErrorURL: URL = ProxyLaunchdService.standardErrorURL

        var programArguments: [String] {
            var arguments = [
                executableURL.path,
                "proxy",
                target.rawValue,
                "--base-url",
                baseURL,
                "--bind",
                bind,
                "--port",
                String(port),
            ]
            if let authToken {
                arguments.append(contentsOf: ["--auth-token", authToken])
            }
            if let upstreamAuthToken {
                arguments.append(contentsOf: ["--upstream-auth-token", upstreamAuthToken])
            }
            arguments.append(contentsOf: Array(repeating: "-v", count: verbose))
            return arguments
        }
    }

    static var launchAgentsDirectory: URL {
        Paths.home
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("LaunchAgents", isDirectory: true)
    }

    static var defaultPropertyListURL: URL {
        launchAgentsDirectory.appendingPathComponent("\(label).plist")
    }

    static var logsDirectory: URL {
        Paths.appRoot.appendingPathComponent("logs", isDirectory: true)
    }

    static var standardOutputURL: URL {
        logsDirectory.appendingPathComponent("proxy.log")
    }

    static var standardErrorURL: URL {
        logsDirectory.appendingPathComponent("proxy.error.log")
    }

    static func currentExecutableURL() throws -> URL {
        let executableURL =
            Bundle.main.executableURL
            ?? URL(fileURLWithPath: CommandLine.arguments[0])
        let resolved = executableURL.standardizedFileURL.resolvingSymlinksInPath()
        guard resolved.path.hasPrefix("/"), FileManager.default.isExecutableFile(atPath: resolved.path) else {
            throw ValidationError("Could not resolve the current smolx executable to an absolute executable path.")
        }
        return resolved
    }

    static func propertyList(for configuration: Configuration) -> [String: Any] {
        [
            "Label": label,
            "Program": configuration.executableURL.path,
            "ProgramArguments": configuration.programArguments,
            "RunAtLoad": true,
            "KeepAlive": true,
            "ProcessType": "Background",
            "ThrottleInterval": 5,
            "StandardOutPath": configuration.standardOutputURL.path,
            "StandardErrorPath": configuration.standardErrorURL.path,
        ]
    }

    static func write(
        _ configuration: Configuration,
        to outputURL: URL,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try fileManager.createDirectory(
            at: configuration.standardOutputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try fileManager.createDirectory(
            at: configuration.standardErrorURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        let data = try PropertyListSerialization.data(
            fromPropertyList: propertyList(for: configuration),
            format: .xml,
            options: 0)
        try data.write(to: outputURL, options: .atomic)
        try fileManager.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: outputURL.path)
    }
}
