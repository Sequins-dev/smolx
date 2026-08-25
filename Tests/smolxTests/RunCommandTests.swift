import ArgumentParser
import Foundation
import Testing

@testable import smolx

@Suite("RunCommand")
struct RunCommandTests {
    @Test func usesLocalServerCatalog() throws {
        let command = try RunCommand.parse(["claude", "--model", "local-model"])

        #expect(command.baseUrl == RunCommand.serverBaseURL)
        #expect(command.modelsURL?.absoluteString == "http://127.0.0.1:8080/v1/models")
        #expect(command.healthURL?.absoluteString == "http://127.0.0.1:8080/healthz")
    }

    @Test func decodesCatalogWithoutCreationMetadata() throws {
        let data = Data(
            """
            {
              "object": "list",
              "data": [
                {
                  "id": "qwen3.8-27b-mlx",
                  "object": "model",
                  "owned_by": "organization_owner"
                }
              ]
            }
            """.utf8)

        let models = try RunCommand.decodeServerModels(data)

        #expect(models.map(\.name) == ["qwen3.8-27b-mlx"])
        #expect(models.first?.addedAt == Date(timeIntervalSince1970: 0))
    }

    @Test func stalePersistedAliasesFallBackToSelectedServerCatalog() throws {
        let installed = [remoteModel("remote-model")]

        let resolved = RunCommand.resolveModels(
            persisted: UserConfig(
                smart: "local-smart", fast: "local-fast", small: "local-small"),
            installedModels: installed)

        #expect(
            resolved
                == UserConfig(
                    smart: "remote-model", fast: "remote-model", small: "remote-model"))
    }

    @Test func persistedAliasesAvailableOnSelectedServerArePreserved() throws {
        let installed = [remoteModel("remote-smart"), remoteModel("remote-fast")]

        let resolved = RunCommand.resolveModels(
            persisted: UserConfig(
                smart: "remote-smart", fast: "remote-fast", small: "local-small"),
            installedModels: installed)

        #expect(
            resolved
                == UserConfig(
                    smart: "remote-smart", fast: "remote-fast", small: "remote-fast"))
    }

    @Test func explicitModelOverrideIsNotSilentlyDiscarded() throws {
        let installed = [remoteModel("remote-model")]

        let resolved = RunCommand.resolveModels(
            persisted: UserConfig(smart: "local-model"),
            installedModels: installed,
            modelSugar: "explicit-model")

        #expect(
            resolved
                == UserConfig(
                    smart: "explicit-model", fast: "explicit-model", small: "explicit-model"))
    }

    @Test func rejectsBaseURLBecauseUpstreamSelectionBelongsToProxy() {
        #expect(throws: (any Error).self) {
            _ = try RunCommand.parse([
                "claude", "--base-url", "http://smolx.local:8080", "--model", "remote-model",
            ])
        }
    }

    private func remoteModel(_ name: String) -> ModelDescriptor {
        ModelDescriptor(
            name: name,
            repoId: name,
            localPath: "",
            capability: .text,
            diskSizeBytes: 0,
            addedAt: Date(timeIntervalSince1970: 0))
    }
}
