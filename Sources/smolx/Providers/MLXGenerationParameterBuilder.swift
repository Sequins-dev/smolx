import Foundation
import MLXLMCommon

enum MLXGenerationParameterBuilder {
    static func make(
        _ p: GenerationParams,
        descriptor: ModelDescriptor
    ) -> GenerateParameters {
        var params = GenerateParameters()
        if let t = p.temperature { params.temperature = Float(t) }
        if let tp = p.topP { params.topP = Float(tp) }
        if let k = p.topK { params.topK = k }
        if let mt = p.maxTokens { params.maxTokens = mt }

        if descriptor.weightFormat == .mlx {
            params.kvBits = 4
            params.prefillStepSize = 1024
        } else {
            params.prefillStepSize = 32_768
        }
        return params
    }
}
