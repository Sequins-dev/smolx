import Foundation
import HuggingFace
import MLXLMCommon

struct GGUFRuntimeFiles {
    static let tokenizerSidecarFilenames = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]

    static func baseModelRepoId(from model: Model) -> String? {
        if let value = model.cardData?["base_model"] {
            switch value {
            case .string(let repoId):
                return repoId.contains("/") ? repoId : nil
            case .array(let values):
                return values.compactMap(\.stringValue).first { $0.contains("/") }
            default:
                break
            }
        }

        return model.tags?
            .lazy
            .compactMap { tag -> String? in
                let prefix = "base_model:"
                guard tag.hasPrefix(prefix) else { return nil }
                let repoId = String(tag.dropFirst(prefix.count))
                return repoId.contains("/") ? repoId : nil
            }
            .first
    }

    static func tokenizerSidecarPaths(in files: [String]) -> [String] {
        let wanted = Set(tokenizerSidecarFilenames)
        return files
            .filter { wanted.contains(($0 as NSString).lastPathComponent) }
            .sorted()
    }

    static func prepareSnapshot(at directory: URL, weightFile: String?) throws {
        let ggufURL = try locateGGUF(in: directory, weightFile: weightFile)
        let reader = try GGUFReader(url: ggufURL)
        try synthesizeConfigIfNeeded(from: reader, in: directory)
    }

    private static func locateGGUF(in directory: URL, weightFile: String?) throws -> URL {
        if let weightFile {
            return directory.appendingPathComponent(weightFile)
        }
        let urls = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted { $0.path < $1.path }
        guard let url = urls.first else {
            throw ProviderError.loadFailed("No GGUF file found in \(directory.path)")
        }
        return url
    }

    private static func synthesizeConfigIfNeeded(from reader: GGUFReader, in directory: URL) throws {
        let url = directory.appendingPathComponent("config.json")
        guard reader.metadata["general.architecture"]?.stringValue == "gemma4" else {
            return
        }

        let config = try gemma4TextConfig(from: reader, in: directory)
        if FileManager.default.fileExists(atPath: url.path),
            configContainsQuantization(at: url),
            configMatchesTensorTying(at: url, reader: reader),
            configMatchesEOS(at: url, expected: config["eos_token_id"] as? [Int] ?? [])
        {
            return
        }

        let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func gemma4TextConfig(from reader: GGUFReader, in directory: URL) throws -> [String: Any] {
        let metadata = reader.metadata

        let layerTypes = try boolArray(metadata, "gemma4.attention.sliding_window_pattern")
            .map { $0 ? "sliding_attention" : "full_attention" }
        let kvHeads = try intArray(metadata, "gemma4.attention.head_count_kv")
        let slidingKVHeads = zip(layerTypes, kvHeads)
            .first { $0.0 == "sliding_attention" }?.1 ?? kvHeads.first ?? 1
        let globalKVHeads = zip(layerTypes, kvHeads)
            .first { $0.0 == "full_attention" }?.1

        let vocabSize = tokenCount(from: metadata) ?? tensorColumns(named: "token_embd.weight", in: reader) ?? 0
        guard vocabSize > 0 else {
            throw ProviderError.loadFailed("Unable to infer Gemma4 vocab size from GGUF metadata")
        }

        let hasOutputHead = reader.tensors.contains { $0.name == "output.weight" }

        var textConfig: [String: Any] = [
            "model_type": "gemma4_text",
            "hidden_size": try int(metadata, "gemma4.embedding_length"),
            "num_hidden_layers": try int(metadata, "gemma4.block_count"),
            "intermediate_size": try int(metadata, "gemma4.feed_forward_length"),
            "num_attention_heads": try int(metadata, "gemma4.attention.head_count"),
            "head_dim": try int(metadata, "gemma4.attention.key_length_swa"),
            "global_head_dim": try int(metadata, "gemma4.attention.key_length"),
            "rms_norm_eps": try float(metadata, "gemma4.attention.layer_norm_rms_epsilon"),
            "vocab_size": vocabSize,
            "vocab_size_per_layer_input": vocabSize,
            "num_key_value_heads": slidingKVHeads,
            "num_kv_shared_layers": try int(metadata, "gemma4.attention.shared_kv_layers"),
            "hidden_size_per_layer_input": try int(metadata, "gemma4.embedding_length_per_layer_input"),
            "sliding_window": try int(metadata, "gemma4.attention.sliding_window"),
            "max_position_embeddings": try int(metadata, "gemma4.context_length"),
            "attention_k_eq_v": hasFullAttentionWithoutValueProjection(reader: reader, layerTypes: layerTypes),
            "final_logit_softcapping": try float(metadata, "gemma4.final_logit_softcapping"),
            "layer_types": layerTypes,
            "tie_word_embeddings": !hasOutputHead,
            "rope_parameters": [
                "sliding_attention": [
                    "rope_theta": try float(metadata, "gemma4.rope.freq_base_swa"),
                    "rope_type": "default",
                ],
                "full_attention": [
                    "rope_theta": try float(metadata, "gemma4.rope.freq_base"),
                    "partial_rotary_factor": try fullPartialRotaryFactor(metadata),
                    "rope_type": "proportional",
                ],
            ],
        ]
        if let globalKVHeads {
            textConfig["num_global_key_value_heads"] = globalKVHeads
        }

        return [
            "model_type": "gemma4",
            "vocab_size": vocabSize,
            "eos_token_id": eosTokenIds(from: metadata, in: directory),
            "quantization": quantizationConfig(from: reader),
            "text_config": textConfig,
        ]
    }

    private static func configContainsQuantization(at url: URL) -> Bool {
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return json["quantization"] != nil
    }

    private static func configMatchesTensorTying(at url: URL, reader: GGUFReader) -> Bool {
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let textConfig = json["text_config"] as? [String: Any],
            let tied = textConfig["tie_word_embeddings"] as? Bool
        else { return false }

        let hasOutputHead = reader.tensors.contains { $0.name == "output.weight" }
        return tied == !hasOutputHead
    }

    private static func configMatchesEOS(at url: URL, expected: [Int]) -> Bool {
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let value = json["eos_token_id"]
        else { return false }

        let actual: [Int]
        if let int = value as? Int {
            actual = [int]
        } else if let array = value as? [Int] {
            actual = array
        } else if let array = value as? [Any] {
            actual = array.compactMap { $0 as? Int }
        } else {
            return false
        }
        return actual == expected
    }

    private static func quantizationConfig(from reader: GGUFReader) -> [String: Any] {
        var result: [String: Any] = [
            "group_size": 32,
            "bits": 4,
        ]

        for tensor in reader.tensors {
            let option: [String: Any]?
            switch tensor.type {
            case .q4K:
                option = ["group_size": 32, "bits": 4]
            case .q6K:
                option = ["group_size": 16, "bits": 6]
            default:
                option = nil
            }
            guard let option else { continue }

            let mapped = GGUFReader.mapTensorName(tensor.name, architecture: "gemma4")
            guard mapped.hasSuffix(".weight") else { continue }
            result[String(mapped.dropLast(".weight".count))] = option
        }

        return result
    }

    private static func eosTokenIds(from metadata: [String: GGUFReader.MetadataValue], in directory: URL) -> [Int] {
        var result: [Int] = []
        if let eos = metadata["tokenizer.ggml.eos_token_id"]?.uint32Value {
            result.append(Int(eos))
        } else {
            result.append(1)
        }

        for token in tokenizerEOSTokens(in: directory) {
            if let id = tokenizerTokenId(token, in: directory), !result.contains(id) {
                result.append(id)
            }
        }
        return result
    }

    static func tokenizerEOSTokens(in directory: URL) -> [String] {
        let url = directory.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [] }

        return ["eot_token"]
            .compactMap { json[$0] }
            .compactMap(tokenString)
    }

    static func tokenizerTokenId(_ token: String, in directory: URL) -> Int? {
        let url = directory.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        if let model = json["model"] as? [String: Any],
            let vocab = model["vocab"] as? [String: Any],
            let id = vocab[token] as? Int
        {
            return id
        }

        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for added in addedTokens
            where added["content"] as? String == token {
                return added["id"] as? Int
            }
        }
        return nil
    }

    private static func tokenString(from value: Any) -> String? {
        if let string = value as? String {
            return string
        }
        if let object = value as? [String: Any] {
            return object["content"] as? String
        }
        return nil
    }

    private static func tokenCount(from metadata: [String: GGUFReader.MetadataValue]) -> Int? {
        guard case .array(let tokens)? = metadata["tokenizer.ggml.tokens"] else { return nil }
        return tokens.count
    }

    private static func tensorColumns(named name: String, in reader: GGUFReader) -> Int? {
        reader.tensors.first { $0.name == name }?.shape.last
    }

    private static func hasFullAttentionWithoutValueProjection(
        reader: GGUFReader,
        layerTypes: [String]
    ) -> Bool {
        for (index, layerType) in layerTypes.enumerated() where layerType == "full_attention" {
            if !reader.tensors.contains(where: { $0.name == "blk.\(index).attn_v.weight" }) {
                return true
            }
        }
        return false
    }

    private static func fullPartialRotaryFactor(_ metadata: [String: GGUFReader.MetadataValue]) throws -> Float {
        let rotary = try float(metadata, "gemma4.rope.dimension_count")
        let head = try float(metadata, "gemma4.attention.key_length")
        return head == 0 ? 1 : rotary / head
    }

    private static func int(_ metadata: [String: GGUFReader.MetadataValue], _ key: String) throws -> Int {
        switch metadata[key] {
        case .uint32(let value):
            return Int(value)
        case .int32(let value):
            return Int(value)
        case .uint64(let value):
            return Int(value)
        case .int64(let value):
            return Int(value)
        default:
            throw ProviderError.loadFailed("Missing GGUF metadata key \(key)")
        }
    }

    private static func float(_ metadata: [String: GGUFReader.MetadataValue], _ key: String) throws -> Float {
        switch metadata[key] {
        case .float32(let value):
            return value
        case .float64(let value):
            return Float(value)
        case .uint32(let value):
            return Float(value)
        case .int32(let value):
            return Float(value)
        default:
            throw ProviderError.loadFailed("Missing GGUF metadata key \(key)")
        }
    }

    private static func boolArray(
        _ metadata: [String: GGUFReader.MetadataValue],
        _ key: String
    ) throws -> [Bool] {
        guard case .array(let values)? = metadata[key] else {
            throw ProviderError.loadFailed("Missing GGUF metadata key \(key)")
        }
        return try values.map {
            guard case .bool(let value) = $0 else {
                throw ProviderError.loadFailed("Invalid boolean array metadata key \(key)")
            }
            return value
        }
    }

    private static func intArray(
        _ metadata: [String: GGUFReader.MetadataValue],
        _ key: String
    ) throws -> [Int] {
        guard case .array(let values)? = metadata[key] else {
            throw ProviderError.loadFailed("Missing GGUF metadata key \(key)")
        }
        return try values.map {
            switch $0 {
            case .uint32(let value):
                return Int(value)
            case .int32(let value):
                return Int(value)
            case .uint64(let value):
                return Int(value)
            case .int64(let value):
                return Int(value)
            default:
                throw ProviderError.loadFailed("Invalid integer array metadata key \(key)")
            }
        }
    }
}
