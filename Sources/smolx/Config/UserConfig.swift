import Foundation

/// Persisted user preferences stored at `~/.smolx/config.json`.
/// Each tier is optional; `resolve(...)` returns a fully-filled copy where
/// all three tiers are guaranteed non-nil.
struct UserConfig: Codable, Sendable, Equatable {
    var smart: String?
    var fast: String?
    var small: String?

    init(smart: String? = nil, fast: String? = nil, small: String? = nil) {
        self.smart = smart
        self.fast = fast
        self.small = small
    }

    /// True when every tier is nil — used by `save(_:)` to delete the
    /// file rather than leave an empty JSON object behind.
    var isEmpty: Bool { smart == nil && fast == nil && small == nil }

    // MARK: - Resolution

    /// Resolve CLI overrides and installed-model fallback on top of this
    /// config and return a fully-filled copy (all tiers non-nil).
    ///
    /// Resolution order (highest precedence first):
    ///   1. Persisted config (self.smart / .fast / .small).
    ///   2. `modelSugar` (i.e. `--model X`) replaces all tiers for this invocation.
    ///   3. Per-tier CLI overrides replace the value for that tier.
    ///   4. Cascade downward: nil `fast` inherits `smart`; nil `small` inherits `fast`.
    ///   5. Last resort: `firstInstalled` fills `smart` and re-cascades.
    ///
    /// Returns `nil` only when `smart` cannot be filled by any source.
    func resolve(
        smartOverride: String? = nil,
        fastOverride: String? = nil,
        smallOverride: String? = nil,
        modelSugar: String? = nil,
        firstInstalled: String? = nil
    ) -> UserConfig? {
        var s = smart
        var f = fast
        var t = small
        if let modelSugar {
            s = modelSugar
            f = modelSugar
            t = modelSugar
        }
        if let smartOverride { s = smartOverride }
        if let fastOverride { f = fastOverride }
        if let smallOverride { t = smallOverride }
        if f == nil { f = s }
        if t == nil { t = f }
        if s == nil, let firstInstalled {
            s = firstInstalled
            if f == nil { f = firstInstalled }
            if t == nil { t = firstInstalled }
        }
        guard s != nil, f != nil, t != nil else { return nil }
        return UserConfig(smart: s, fast: f, small: t)
    }

    // MARK: - I/O

    /// Read the persisted config. A missing file is not an error — returns
    /// an empty config.
    static func load(from url: URL = Paths.configFile) throws -> UserConfig {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return UserConfig()
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(UserConfig.self, from: data)
    }

    /// Write atomically. When the config is empty, remove the file instead
    /// of writing `{}`.
    func save(to url: URL = Paths.configFile) throws {
        if isEmpty {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            return
        }
        try Paths.ensureAppRoot()
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try enc.encode(self).write(to: url, options: .atomic)
    }
}
