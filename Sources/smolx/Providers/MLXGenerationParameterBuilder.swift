import Foundation
import MLXLMCommon

enum MLXGenerationParameterBuilder {
    static func make(
        _ p: GenerationParams,
        descriptor: ModelDescriptor,
        samplingDefaults: ModelSamplingDefaults? = nil
    ) -> GenerateParameters {
        var params = GenerateParameters()
        if let t = p.temperature ?? samplingDefaults?.temperature { params.temperature = Float(t) }
        if let tp = p.topP ?? samplingDefaults?.topP { params.topP = Float(tp) }
        if let k = p.topK ?? samplingDefaults?.topK { params.topK = k }
        if let mt = p.maxTokens { params.maxTokens = mt }

        if descriptor.weightFormat == .mlx {
            params.kvBits = 4
            params.prefillStepSize = 1024
        } else {
            params.prefillStepSize = 32_768
            params.repetitionPenalty = 1.08
            params.repetitionContextSize = 128
        }
        return params
    }
}
