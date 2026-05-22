import Foundation
import Testing

@testable import smolx

@Suite("UserConfig")
struct UserConfigTests {

    /// Each test gets its own tmpdir-backed config file so we never touch
    /// the user's real `~/.smolx/config.json`.
    private func tmpConfigURL() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("smolx-userconfig-\(UUID().uuidString).json")
    }

    @Test func missingFileReturnsEmptyConfig() throws {
        let url = tmpConfigURL()
        // Don't create the file.
        let cfg = try UserConfig.load(from: url)
        #expect(cfg.isEmpty)
        #expect(cfg.smart == nil)
        #expect(cfg.fast == nil)
        #expect(cfg.small == nil)
    }

    @Test func saveThenLoadRoundTrips() throws {
        let url = tmpConfigURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let original = UserConfig(smart: "a", fast: "b", small: "c")
        try original.save(to: url)
        let loaded = try UserConfig.load(from: url)
        #expect(loaded == original)
    }

    @Test func saveWritesOnlyTheSetTiers() throws {
        let url = tmpConfigURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let cfg = UserConfig(smart: "only-this", fast: nil, small: nil)
        try cfg.save(to: url)
        let contents = try String(contentsOf: url, encoding: .utf8)
        #expect(contents.contains("\"smart\""))
        // Nil tiers don't get serialised — keeps the file readable and
        // doesn't bake-in a `null` that future readers might misinterpret.
        #expect(!contents.contains("\"fast\""))
        #expect(!contents.contains("\"small\""))
    }

    @Test func savingEmptyConfigDeletesExistingFile() throws {
        let url = tmpConfigURL()
        defer { try? FileManager.default.removeItem(at: url) }
        // Seed with a populated config.
        try UserConfig(smart: "x").save(to: url)
        #expect(FileManager.default.fileExists(atPath: url.path))
        // Now save an empty config — file should disappear.
        try UserConfig().save(to: url)
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func savingEmptyConfigWhenFileAbsentIsNoop() throws {
        let url = tmpConfigURL()
        // No file exists yet; saving empty should not error and should not
        // create the file either.
        try UserConfig().save(to: url)
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}
