import Foundation

/// Generic interface a backend (MLX text, MLX vision, or future llama.cpp etc.)
/// must satisfy. Concrete providers conform via `actor` types so that
/// `ModelManager` can hold them as `any ModelProvider` while still letting
/// generation execute concurrently with eviction/inspection.
protocol ModelProvider: Actor {
    nonisolated var descriptor: ModelDescriptor { get }
    /// Best-effort resident-bytes hint for the budget calculator. Set after
    /// `load()` based on actual GPU memory consumption when available, falling
    /// back to `descriptor.diskSizeBytes`.
    var residentBytes: Int64 { get }
    var lastUsedAt: Date { get }

    /// Run inference. Implementations must propagate cancellation through the
    /// returned stream so HTTP client disconnects free MLX resources promptly.
    /// `nonisolated` so route handlers can kick off generation without hopping
    /// to the actor; the actual work happens inside a Task spawned by the
    /// implementation.
    nonisolated func generate(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        toolChoice: ToolChoice?,
        params: GenerationParams
    ) -> AsyncThrowingStream<StreamEvent, Error>

    /// Drop the loaded weights. Subsequent `generate` calls must re-load (or
    /// the manager creates a new provider instance).
    func unload() async
}

enum ProviderError: Error, CustomStringConvertible {
    case modelNotFound(String)
    case unsupportedFeature(String)
    case loadFailed(String)
    case generationFailed(String)

    var description: String {
        switch self {
        case .modelNotFound(let n): return "Model not found: \(n)"
        case .unsupportedFeature(let s): return "Unsupported feature: \(s)"
        case .loadFailed(let s): return "Failed to load model: \(s)"
        case .generationFailed(let s): return "Generation failed: \(s)"
        }
    }
}
