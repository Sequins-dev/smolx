import Foundation

enum Paths {
    static var home: URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
    }

    /// `~/.mlx-serve/` — top-level state directory for the tool.
    static var appRoot: URL {
        home.appendingPathComponent(".mlx-serve", isDirectory: true)
    }

    /// `~/.mlx-serve/config.json`
    static var configFile: URL {
        appRoot.appendingPathComponent("config.json")
    }

    /// `~/.mlx-serve/registry.json` — list of installed models + aliases.
    static var registryFile: URL {
        appRoot.appendingPathComponent("registry.json")
    }

    /// `~/.mlx-serve/models/` — root for our own model snapshots.
    /// Each model lives at `<namespace>--<name>/<filename>` (flat layout —
    /// no symlinks or blob deduplication, just the files MLX needs to load).
    static var modelsRoot: URL {
        appRoot.appendingPathComponent("models", isDirectory: true)
    }

    static func modelDirectory(namespace: String, name: String) -> URL {
        modelsRoot.appendingPathComponent("\(namespace)--\(name)", isDirectory: true)
    }

    /// `~/.cache/huggingface/hub/` — shared HF Hub cache (Python-compatible layout).
    /// Honours `HF_HUB_CACHE` then `HF_HOME` env vars.
    static var huggingFaceCache: URL {
        let env = ProcessInfo.processInfo.environment
        if let override = env["HF_HUB_CACHE"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }
        return home
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
    }

    static func ensureAppRoot() throws {
        try FileManager.default.createDirectory(
            at: appRoot, withIntermediateDirectories: true)
    }
}
