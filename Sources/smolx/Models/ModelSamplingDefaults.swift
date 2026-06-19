import Foundation
import MLXLMCommon

struct ModelSamplingDefaults: Equatable, Sendable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?

    static func load(for descriptor: ModelDescriptor) -> ModelSamplingDefaults? {
        let directory = URL(fileURLWithPath: descriptor.localPath)
        let generationConfig = fromGenerationConfig(in: directory)

        guard descriptor.weightFormat == .gguf else {
            return generationConfig
        }

        guard let ggufMetadata = fromGGUFFile(in: directory, weightFile: descriptor.weightFile) else {
            return generationConfig
        }

        return ggufMetadata.mergingMissingValues(from: generationConfig)
    }

    static func fromGenerationConfig(in directory: URL) -> ModelSamplingDefaults? {
        let url = directory.appendingPathComponent("generation_config.json")
        guard let data = try? Data(contentsOf: url),
            let config = try? JSONDecoder().decode(GenerationConfig.self, from: data)
        else { return nil }

        return ModelSamplingDefaults(
            temperature: config.temperature,
            topP: config.topP,
            topK: config.topK)
        .nilIfEmpty
    }

    static func fromGGUFMetadata(_ metadata: [String: GGUFReader.MetadataValue]) -> ModelSamplingDefaults? {
        ModelSamplingDefaults(
            temperature: double(metadata["general.sampling.temp"]),
            topP: double(metadata["general.sampling.top_p"]),
            topK: int(metadata["general.sampling.top_k"]))
        .nilIfEmpty
    }

    func mergingMissingValues(from fallback: ModelSamplingDefaults?) -> ModelSamplingDefaults {
        guard let fallback else { return self }
        return ModelSamplingDefaults(
            temperature: temperature ?? fallback.temperature,
            topP: topP ?? fallback.topP,
            topK: topK ?? fallback.topK)
    }

    private var nilIfEmpty: ModelSamplingDefaults? {
        temperature == nil && topP == nil && topK == nil ? nil : self
    }

    private static func fromGGUFFile(in directory: URL, weightFile: String?) -> ModelSamplingDefaults? {
        guard let url = locateGGUF(in: directory, weightFile: weightFile),
            let reader = try? GGUFReader(url: url)
        else { return nil }
        return fromGGUFMetadata(reader.metadata)
    }

    private static func locateGGUF(in directory: URL, weightFile: String?) -> URL? {
        if let weightFile {
            return directory.appendingPathComponent(weightFile)
        }
        return try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted { $0.path < $1.path }
            .first
    }

    private static func double(_ value: GGUFReader.MetadataValue?) -> Double? {
        switch value {
        case .float32(let value): Double(value)
        case .float64(let value): value
        case .uint8(let value): Double(value)
        case .int8(let value): Double(value)
        case .uint16(let value): Double(value)
        case .int16(let value): Double(value)
        case .uint32(let value): Double(value)
        case .int32(let value): Double(value)
        case .uint64(let value): Double(value)
        case .int64(let value): Double(value)
        default: nil
        }
    }

    private static func int(_ value: GGUFReader.MetadataValue?) -> Int? {
        switch value {
        case .uint8(let value): Int(value)
        case .int8(let value): Int(value)
        case .uint16(let value): Int(value)
        case .int16(let value): Int(value)
        case .uint32(let value): Int(value)
        case .int32(let value): Int(value)
        case .uint64(let value): Int(value)
        case .int64(let value): Int(value)
        case .float32(let value): Int(value)
        case .float64(let value): Int(value)
        default: nil
        }
    }

    private struct GenerationConfig: Decodable {
        var temperature: Double?
        var topP: Double?
        var topK: Int?

        enum CodingKeys: String, CodingKey {
            case temperature
            case topP = "top_p"
            case topK = "top_k"
        }
    }
}
