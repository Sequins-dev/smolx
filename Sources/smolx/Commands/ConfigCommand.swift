import ArgumentParser
import Foundation

/// `smolx config` — surface for inspecting and editing
/// `~/.smolx/config.json`. Currently just exposes the three model tiers
/// (`smart`, `fast`, `small`); future config additions land here too.
///
/// Subcommands:
///   smolx config set <tier> <model>   set one tier
///   smolx config get [<tier>]         print one tier or all
///   smolx config unset <tier>         clear one tier
struct ConfigCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "View and edit smolx's persisted config.",
        subcommands: [Set.self, Get.self, Unset.self]
    )

    // MARK: - Tier name parsing

    /// String → field-on-UserConfig mapping. Kept in one place so
    /// `set` / `get` / `unset` share the same validation + error text.
    fileprivate enum Tier: String, CaseIterable {
        case smart, fast, small

        static func parse(_ raw: String) throws -> Tier {
            guard let t = Tier(rawValue: raw.lowercased()) else {
                throw ValidationError(
                    "unknown tier '\(raw)' — expected one of: \(allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return t
        }

        func read(from cfg: UserConfig) -> String? {
            switch self {
            case .smart: return cfg.smart
            case .fast: return cfg.fast
            case .small: return cfg.small
            }
        }

        func write(_ value: String?, into cfg: inout UserConfig) {
            switch self {
            case .smart: cfg.smart = value
            case .fast: cfg.fast = value
            case .small: cfg.small = value
            }
        }
    }

    // MARK: - set

    struct Set: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "set",
            abstract: "Set a config value. Aliases must be registered via `smolx pull`."
        )

        @Argument(help: "Config tier: smart | fast | small.")
        var tier: String

        @Argument(help: "Model alias (must already be installed).")
        var value: String

        func run() async throws {
            let tier = try Tier.parse(tier)
            // Validate the alias against the registry — refusing unknown
            // aliases at set-time is friendlier than surfacing them later
            // at `smolx run` time as a load failure.
            let registry = ModelRegistry()
            guard try registry.find(value) != nil else {
                throw ValidationError(
                    "model alias '\(value)' isn't registered — run `smolx models` to see installed models, or `smolx pull <repo>` to add one.")
            }
            var cfg = try UserConfig.load()
            tier.write(value, into: &cfg)
            try cfg.save()
            print("\(tier.rawValue) = \(value)")
        }
    }

    // MARK: - get

    struct Get: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "get",
            abstract: "Print a config value, or all values if no tier is given."
        )

        @Argument(help: "Optional tier: smart | fast | small. Omit to print everything.")
        var tier: String?

        func run() async throws {
            let cfg = try UserConfig.load()
            let unsetLabel = "<unset>"
            if let raw = tier {
                let t = try Tier.parse(raw)
                print(t.read(from: cfg) ?? unsetLabel)
            } else {
                for t in Tier.allCases {
                    print("\(t.rawValue) = \(t.read(from: cfg) ?? unsetLabel)")
                }
            }
        }
    }

    // MARK: - unset

    struct Unset: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "unset",
            abstract: "Clear one tier. The file is removed when all tiers are unset."
        )

        @Argument(help: "Config tier: smart | fast | small.")
        var tier: String

        func run() async throws {
            let tier = try Tier.parse(tier)
            var cfg = try UserConfig.load()
            tier.write(nil, into: &cfg)
            try cfg.save()
            print("\(tier.rawValue) unset")
        }
    }
}
