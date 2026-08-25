import Foundation
import Testing

@testable import smolx

@Suite("ProxyLaunchdService")
struct ProxyLaunchdServiceTests {
    @Test func buildsLaunchdProgramArgumentsFromProxyOptions() {
        let configuration = ProxyLaunchdService.Configuration(
            executableURL: URL(fileURLWithPath: "/Users/example/.local/bin/smolx"),
            target: .lmStudio,
            baseURL: "http://192.168.1.55:1234",
            bind: "127.0.0.1",
            port: 9090,
            authToken: "inbound-secret",
            upstreamAuthToken: "upstream-secret",
            verbose: 2)

        #expect(
            configuration.programArguments == [
                "/Users/example/.local/bin/smolx",
                "proxy",
                "lm-studio",
                "--base-url",
                "http://192.168.1.55:1234",
                "--bind",
                "127.0.0.1",
                "--port",
                "9090",
                "--auth-token",
                "inbound-secret",
                "--upstream-auth-token",
                "upstream-secret",
                "-v",
                "-v",
            ])
    }

    @Test func propertyListKeepsServiceAliveAndWritesSeparateLogs() throws {
        let configuration = ProxyLaunchdService.Configuration(
            executableURL: URL(fileURLWithPath: "/Users/example/.local/bin/smolx"),
            target: .lmStudio,
            baseURL: "http://localhost:1234")
        let propertyList = ProxyLaunchdService.propertyList(for: configuration)

        #expect(propertyList["Label"] as? String == ProxyLaunchdService.label)
        #expect(propertyList["Program"] as? String == "/Users/example/.local/bin/smolx")
        #expect(propertyList["RunAtLoad"] as? Bool == true)
        #expect(propertyList["KeepAlive"] as? Bool == true)
        #expect(propertyList["StandardOutPath"] as? String == ProxyLaunchdService.standardOutputURL.path)
        #expect(propertyList["StandardErrorPath"] as? String == ProxyLaunchdService.standardErrorURL.path)

        _ = try PropertyListSerialization.data(
            fromPropertyList: propertyList,
            format: .xml,
            options: 0)
    }

    @Test func writesModeRestrictedPropertyList() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let outputURL = directory.appendingPathComponent("proxy.plist")
        defer { try? FileManager.default.removeItem(at: directory) }

        let configuration = ProxyLaunchdService.Configuration(
            executableURL: URL(fileURLWithPath: "/Users/example/.local/bin/smolx"),
            target: .lmStudio,
            baseURL: "http://localhost:1234",
            standardOutputURL: directory.appendingPathComponent("proxy.log"),
            standardErrorURL: directory.appendingPathComponent("proxy.error.log"))
        try ProxyLaunchdService.write(configuration, to: outputURL)

        let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
        #expect(attributes[.posixPermissions] as? Int == 0o600)
    }
}
