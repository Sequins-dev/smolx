import ArgumentParser
import Testing

@testable import smolx

@Suite("RunCommand")
struct RunCommandTests {
    @Test func customBaseURLUsesServerCatalog() throws {
        let command = try RunCommand.parse([
            "claude", "--base-url", "http://smolx.local:8080", "--model", "remote-model",
        ])

        #expect(command.baseUrl == "http://smolx.local:8080")
        #expect(command.modelsURL?.absoluteString == "http://smolx.local:8080/v1/models")
    }

    @Test func defaultBaseURLUsesServerCatalog() throws {
        let command = try RunCommand.parse(["claude", "--model", "local-model"])

        #expect(command.baseUrl == RunCommand.defaultBaseURL)
        #expect(command.modelsURL?.absoluteString == "http://127.0.0.1:8080/v1/models")
    }

    @Test func normalizesTrailingSlashBeforeBuildingEndpoints() throws {
        let command = try RunCommand.parse([
            "claude", "--base-url", "http://smolx.local:8080/", "--model", "remote-model",
        ])

        #expect(command.baseUrl == "http://smolx.local:8080")
        #expect(command.modelsURL?.absoluteString == "http://smolx.local:8080/v1/models")
        #expect(command.healthURL?.absoluteString == "http://smolx.local:8080/healthz")
    }

    @Test func rejectsBaseURLWithoutHTTPTransport() {
        #expect(throws: (any Error).self) {
            _ = try RunCommand.parse([
                "claude", "--base-url", "smolx.local:8080", "--model", "remote-model",
            ])
        }
    }
}
