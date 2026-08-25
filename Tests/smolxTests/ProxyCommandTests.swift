import ArgumentParser
import Foundation
import Testing

@testable import smolx

@Suite("ProxyCommand")
struct ProxyCommandTests {
    @Test func parsesExplicitLMStudioTarget() throws {
        let root = try ProxyCommand.parseAsRoot([
            "lm-studio", "--base-url", "http://192.168.1.55:1234",
        ])
        let command = try #require(root as? LMStudioProxyCommand)

        #expect(command.baseUrl == "http://192.168.1.55:1234")
        #expect(command.bind == "127.0.0.1")
        #expect(command.port == 8080)
    }

    @Test func acceptsAndNormalizesVersionedBaseURL() throws {
        let command = try LMStudioProxyCommand.parse([
            "--base-url", "http://localhost:1234/v1/",
        ])

        #expect(command.baseUrl == "http://localhost:1234")
    }

    @Test func rejectsUnknownTarget() {
        #expect(throws: (any Error).self) {
            _ = try ProxyCommand.parseAsRoot([
                "ollama", "--base-url", "http://localhost:11434",
            ])
        }
    }

    @Test func rejectsInvalidBaseURL() {
        #expect(throws: (any Error).self) {
            _ = try LMStudioProxyCommand.parse([
                "--base-url", "localhost:1234",
            ])
        }
    }

    @Test func parsesLaunchdGeneratorWithoutConfusingItForATarget() throws {
        let root = try ProxyCommand.parseAsRoot([
            "launchd", "lm-studio", "--base-url", "http://localhost:1234/v1/",
        ])
        let command = try #require(root as? ProxyLaunchdCommand)

        #expect(command.target == .lmStudio)
        #expect(command.baseUrl == "http://localhost:1234")
    }
}
