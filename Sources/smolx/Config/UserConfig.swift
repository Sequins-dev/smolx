import Foundation

/// Persisted user preferences stored at `~/.smolx/config.json`. Today it
/// only carries the model-tier defaults; future additions land here too.
///
/// Each tier is optional. The cascade resolution in `ModelTuple.resolve`
/// fills missing tiers from larger configured ones, so a user who only
/// sets `smart` gets the "one model for everything" behavior automatically.
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

    // MARK: - I/O

    /// Read the persisted config. A missing file is not an error — it
    /// just means the user hasn't configured anything yet; we return an
    /// empty config.
    static func load(from url: URL = Paths.configFile) throws -> UserConfig {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return UserConfig()
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(UserConfig.self, from: data)
    }

    /// Write atomically. When the config is empty, remove the file
    /// instead of writing `{}` — keeps `~/.smolx/` tidy and makes it
    /// trivial to spot whether the user has ever configured anything.
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
