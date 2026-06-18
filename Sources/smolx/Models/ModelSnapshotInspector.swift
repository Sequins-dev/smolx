import Foundation

/// Reads metadata from a downloaded model snapshot directory.
enum ModelSnapshotInspector {
    /// VLM detection: a `preprocessor_config.json` (image processor) alongside
    /// the weights is the standard marker. Text-only LLM repos don't ship one.
    static func capability(at directory: URL) -> ModelDescriptor.Capability {
        let preprocessor = directory.appendingPathComponent("preprocessor_config.json")
        return FileManager.default.fileExists(atPath: preprocessor.path) ? .vision : .text
    }

    static func contextLength(for descriptor: ModelDescriptor) -> Int? {
        contextLength(at: URL(fileURLWithPath: descriptor.localPath))
    }

    /// Maximum context length read from the model's `config.json`.
    /// Returns `nil` when the file is absent or the key is missing.
    static func contextLength(at directory: URL) -> Int? {
        guard let json = configJSON(at: directory) else { return nil }

        if let lm = json["language_model"] as? [String: Any],
            let value = lm["max_position_embeddings"] as? Int
        {
            return value
        }
        if let textConfig = json["text_config"] as? [String: Any],
            let value = textConfig["max_position_embeddings"] as? Int
        {
            return value
        }
        return json["max_position_embeddings"] as? Int
    }

    private static func configJSON(at directory: URL) -> [String: Any]? {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json
    }
}
