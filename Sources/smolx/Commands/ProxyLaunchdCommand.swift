import ArgumentParser
import Foundation

struct ProxyLaunchdCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "launchd",
        abstract: "Generate a macOS LaunchAgent plist for the proxy."
    )

    @Argument(help: "Upstream server type.")
    var target: ProxyTarget

    @Option(name: .long, help: "Base URL of the upstream model server.")
    var baseUrl: String

    @Option(name: .long, help: "Port to listen on locally.")
    var port: Int = 8080

    @Option(
        name: .long,
        help: "Address to bind. Use 0.0.0.0 for non-loopback access (requires --auth-token).")
    var bind: String = "127.0.0.1"

    @Option(
        name: .long,
        help: "Bearer token required on inbound requests. Mandatory when --bind is non-loopback.")
    var authToken: String?

    @Option(
        name: .long,
        help: "Bearer token sent to the upstream server instead of the inbound Authorization header.")
    var upstreamAuthToken: String?

    @Option(name: .long, help: "Destination plist path. Defaults to the user's LaunchAgents directory.")
    var output: String?

    @Flag(name: .shortAndLong, help: "Increase service log verbosity (-v: debug, -vv: trace).")
    var verbose: Int

    mutating func validate() throws {
        baseUrl = try ProxyCommand.normalizeBaseURL(baseUrl)
    }

    func run() async throws {
        let configuration = ProxyLaunchdService.Configuration(
            executableURL: try ProxyLaunchdService.currentExecutableURL(),
            target: target,
            baseURL: baseUrl,
            bind: bind,
            port: port,
            authToken: authToken,
            upstreamAuthToken: upstreamAuthToken,
            verbose: verbose)
        let outputURL =
            output.map {
                URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
            } ?? ProxyLaunchdService.defaultPropertyListURL
        try ProxyLaunchdService.write(configuration, to: outputURL)
        print("Wrote \(outputURL.path)")
    }
}
