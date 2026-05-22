import CoreImage
import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Concrete provider that loads MLX-quantised weights from a local snapshot
/// directory and drives generation through `MLXLMCommon.ChatSession`.
actor MLXProvider: ModelProvider {

    nonisolated let descriptor: ModelDescriptor
    private(set) var residentBytes: Int64
    private(set) var lastUsedAt: Date
    private var container: ModelContainer?
    /// In-flight load task. Belt-and-suspenders dedup: `MLXProviderFactory.make`
    /// is the only caller today and it's serialized through
    /// `ModelManager.inflightLoads`, but trapping the race inside this actor
    /// too means a misbehaving caller can't ever trigger a parallel
    /// `loadModelContainer` (which would silently allocate two copies of the
    /// weights in GPU memory). The earlier `ensureLoaded()` approach raced
    /// because actor isolation releases at the `await loadModelContainer(...)`
    /// suspension — two concurrent callers both saw `container == nil`.
    private var loadTask: Task<ModelContainer, Error>?
    private let logger: Logger

    init(descriptor: ModelDescriptor, logger: Logger = Logger(label: "smolx.mlx")) {
        self.descriptor = descriptor
        self.residentBytes = descriptor.diskSizeBytes
        self.lastUsedAt = Date()
        self.logger = logger
    }

    // MARK: - Loading

    /// Public load entry point used by `MLXProviderFactory.make`. Idempotent
    /// and dedup-safe: concurrent callers join the same load task. Returns
    /// once the container is ready; from then on `generate()` can read
    /// `requireContainer()` without racing the load.
    func load() async throws {
        _ = try await loadIfNeeded()
    }

    private func loadIfNeeded() async throws -> ModelContainer {
        if let c = container { return c }
        if let t = loadTask {
            return try await t.value
        }
        let task = Task<ModelContainer, Error> { [descriptor, logger] in
            let directory = URL(fileURLWithPath: descriptor.localPath)
            logger.info("Loading MLX model from \(directory.path)")
            let before = MLX.Memory.activeMemory
            let tokenizerLoader = #huggingFaceTokenizerLoader()
            do {
                let c = try await loadModelContainer(from: directory, using: tokenizerLoader)
                let after = MLX.Memory.activeMemory
                let delta = Int64(after - before)
                logger.info("Loaded \(descriptor.name) — resident: \(delta / (1024 * 1024)) MB")
                return c
            } catch {
                throw ProviderError.loadFailed("\(error)")
            }
        }
        loadTask = task
        do {
            let c = try await task.value
            container = c
            let delta = Int64(MLX.Memory.activeMemory)
            if delta > 0 { residentBytes = max(residentBytes, delta) }
            loadTask = nil
            return c
        } catch {
            loadTask = nil
            throw error
        }
    }

    /// Read the loaded container. Used by `generate()` which assumes the
    /// factory's eager-load contract has already populated it.
    private func requireContainer() throws -> ModelContainer {
        guard let c = container else {
            throw ProviderError.loadFailed(
                "MLXProvider used before load completed for \(descriptor.name)")
        }
        return c
    }

    func unload() async {
        // Dropping the container releases the MLX weights — they're owned by
        // ModelContainer and freed when the last reference goes away.
        if container != nil {
            logger.info("Unloading \(descriptor.name)")
        }
        container = nil
        loadTask = nil
    }

    // MARK: - Generation

    nonisolated func generate(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        toolChoice: ToolChoice?,
        params: GenerationParams
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        let descriptor = self.descriptor
        let logger = self.logger
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let container = try await self.requireContainer()
                    await self.markUsed()

                    // Build dict-form messages with structured tool_calls /
                    // tool_call_id fields so the model's chat template can
                    // render past tool history correctly. `ChatSession` only
                    // accepts the bare `role + content` Chat.Message form
                    // (no tool_calls field) which is why we bypass it and
                    // drive `processor.prepare` + `TokenIterator` directly.
                    let dictMessages = PromptBuilder.messageDicts(from: messages)
                    let toolSpecs = Self.makeToolSpecs(tools)
                    let generateParameters = Self.makeGenerateParameters(params)

                    // All MLX-side work (image construction, processor prep,
                    // generation loop, Harmony post-processing) happens inside
                    // `container.perform { ... }` because `UserInput.Image`
                    // and the language model itself are non-Sendable —
                    // capturing them across @Sendable boundaries is rejected
                    // by Swift 6 strict concurrency. The closure returns the
                    // single Sendable bit we need outside (whether any tool
                    // call fired, which decides the finish_reason).
                    let messagesCopy = messages
                    let capability = descriptor.capability
                    let repoId = descriptor.repoId

                    let emittedToolCall: Bool = try await container.perform { context in
                        let collectedImages = Self.collectImages(
                            from: messagesCopy, capability: capability)
                        let userInput = UserInput(
                            messages: dictMessages,
                            images: collectedImages,
                            tools: toolSpecs)
                        let input = try await context.processor.prepare(input: userInput)

                        let kvCache = context.model.newCache(parameters: generateParameters)
                        let iterator = try TokenIterator(
                            input: input, model: context.model,
                            cache: kvCache, parameters: generateParameters)

                        let (stream, generationTask) = MLXLMCommon.generateTask(
                            promptTokenCount: input.text.tokens.size,
                            modelConfiguration: context.configuration,
                            tokenizer: context.tokenizer,
                            iterator: iterator)

                        let harmony: HarmonyParser? =
                            repoId.lowercased().contains("gpt-oss")
                                ? HarmonyParser() : nil
                        var emittedToolCall = false
                        let debug = Self.debugRawEnabled

                        for await detail in stream {
                            if Task.isCancelled { break }
                            switch detail {
                            case .chunk(let s) where !s.isEmpty:
                                if debug { logger.info("RAW chunk: \(s.debugDescription)") }
                                if let harmony {
                                    for event in harmony.feed(s) {
                                        if debug { logger.info("HARMONY event: \(String(describing: event))") }
                                        Self.relay(event, &emittedToolCall, continuation)
                                    }
                                } else {
                                    continuation.yield(.textDelta(s))
                                }
                            case .chunk:
                                break
                            case .toolCall(let tc):
                                if debug { logger.info("RAW toolCall: \(tc.function.name) args=\(tc.function.arguments)") }
                                emittedToolCall = true
                                let id = "toolu_\(UUID().uuidString.prefix(16))"
                                continuation.yield(.toolUseStart(id: String(id), name: tc.function.name))
                                if let json = Self.encodeArguments(tc.function.arguments) {
                                    continuation.yield(.toolUseInputDelta(json))
                                }
                                continuation.yield(.toolUseStop)
                            case .info:
                                break
                            }
                        }
                        await generationTask.value

                        if let harmony {
                            for event in harmony.flush() {
                                if debug { logger.info("HARMONY flush event: \(String(describing: event))") }
                                Self.relay(event, &emittedToolCall, continuation)
                            }
                        }
                        return emittedToolCall
                    }

                    let reason: FinishReason = emittedToolCall ? .toolCalls : .stop
                    continuation.yield(.done(finishReason: reason, usage: nil))
                    continuation.finish()
                } catch {
                    logger.error("Generation failed for \(descriptor.name): \(error)")
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Set `MLX_SERVE_DEBUG_RAW=1` in the server's env to dump every raw
    /// chunk + Harmony parser event to the logger. The value is read once
    /// at first access so toggling it requires a server restart — that's
    /// fine; it's a developer-only knob for diagnosing tool-call surfacing
    /// when the model output format doesn't match parser assumptions.
    private static let debugRawEnabled: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_SERVE_DEBUG_RAW"] ?? ""
        return v == "1" || v.lowercased() == "true"
    }()

    /// Pulls `UserInput.Image` instances out of every image content block
    /// across all messages (only when the model declares vision capability).
    /// Replaces the per-message image plumbing from the old `splitMessages`
    /// path now that we ship a flat dict-message array through `UserInput`.
    private static func collectImages(
        from messages: [ChatMessage],
        capability: ModelDescriptor.Capability
    ) -> [UserInput.Image] {
        guard capability == .vision else { return [] }
        return messages.flatMap { msg in
            msg.content.compactMap { imageFor($0) }
        }
    }

    /// Bump lastUsedAt from inside the actor so external callers don't race.
    private func markUsed() { lastUsedAt = Date() }

    /// Translates a HarmonyParser event to the domain StreamEvent shape and
    /// pumps it into the generation stream. Pulled out so the inner
    /// generation loop and the post-loop flush() drain don't duplicate the
    /// mapping logic. `emittedToolCall` is updated by reference because the
    /// finish-reason calculation downstream needs to know whether any tool
    /// fired so it can pick `.toolCalls` vs `.stop`.
    private static func relay(
        _ event: HarmonyParser.Event,
        _ emittedToolCall: inout Bool,
        _ continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    ) {
        switch event {
        case .textDelta(let t):
            continuation.yield(.textDelta(t))
        case .toolUseStart(let id, let name):
            emittedToolCall = true
            continuation.yield(.toolUseStart(id: id, name: name))
        case .toolUseInputDelta(let d):
            continuation.yield(.toolUseInputDelta(d))
        case .toolUseStop:
            continuation.yield(.toolUseStop)
        }
    }

    // MARK: - Domain mapping

    /// Turn one of our `ContentBlock.image` payloads into MLX's `UserInput.Image`.
    /// Returns `nil` for non-image blocks or when the bytes won't decode as an
    /// image — in that case the block is just dropped from this turn.
    private static func imageFor(_ block: ContentBlock) -> UserInput.Image? {
        guard case .image(let payload) = block else { return nil }
        guard let ci = CIImage(data: payload.data) else { return nil }
        return .ciImage(ci)
    }

    private static func makeGenerateParameters(_ p: GenerationParams) -> GenerateParameters {
        var params = GenerateParameters()
        if let t = p.temperature { params.temperature = Float(t) }
        if let tp = p.topP { params.topP = Float(tp) }
        if let mt = p.maxTokens { params.maxTokens = mt }
        return params
    }

    /// Convert our domain `ToolDefinition`s to MLX's `ToolSpec` dict format,
    /// which is the same shape OpenAI uses: `{type: function, function: {...}}`.
    /// MLX's prompt-time tool injection reads name/description/parameters from
    /// this dict, so we just need to round-trip the JSON schema through.
    private static func makeToolSpecs(_ tools: [ToolDefinition]) -> [ToolSpec]? {
        guard !tools.isEmpty else { return nil }
        return tools.compactMap { tool -> ToolSpec? in
            guard let parameters = jsonValueToSendable(tool.inputSchema) as? [String: any Sendable] else {
                return nil
            }
            var function: [String: any Sendable] = [
                "name": tool.name,
                "parameters": parameters,
            ]
            if let d = tool.description { function["description"] = d }
            return [
                "type": "function",
                "function": function,
            ]
        }
    }

    /// Encode MLX's tool-call arguments dict to a JSON string for our domain's
    /// `toolUseInputDelta` event. MLX emits the entire argument map at once
    /// (single-shot), so we emit a single non-streaming delta.
    private static func encodeArguments(
        _ args: [String: MLXLMCommon.JSONValue]
    ) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        guard let data = try? encoder.encode(args) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Recursively convert our `JSONValue` to plain Swift types compatible with
    /// MLX's `ToolSpec` dictionary requirement (`any Sendable`).
    private static func jsonValueToSendable(_ v: JSONValue) -> any Sendable {
        switch v {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return Int(i)
        case .double(let d): return d
        case .string(let s): return s
        case .array(let a): return a.map(jsonValueToSendable) as [any Sendable]
        case .object(let o):
            var out: [String: any Sendable] = [:]
            for (k, val) in o { out[k] = jsonValueToSendable(val) }
            return out
        }
    }
}

/// Default factory used by `ModelManager.start()` in `smolx serve`.
/// `make` performs the heavy `loadModelContainer` here so the returned
/// provider is fully ready to serve traffic; concurrent generate() calls
/// on the resulting provider can read its container directly without
/// racing a lazy ensureLoaded path. Dedup of *concurrent factory.make
/// invocations* is the caller's responsibility — `ModelManager` does this
/// via `inflightLoads`.
struct MLXProviderFactory: ProviderFactory {
    func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider {
        let provider = MLXProvider(descriptor: descriptor)
        try await provider.load()
        return provider
    }
}
