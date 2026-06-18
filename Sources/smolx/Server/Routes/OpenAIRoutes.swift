import Foundation
import Hummingbird
import HummingbirdCore
import Logging
import NIOCore

enum OpenAIRoutes {

    static func register<Context: RequestContext>(
        _ router: Router<Context>,
        manager: ModelManager,
        registry: ModelRegistry,
        logger: Logger
    ) {
        // GET /v1/models — list installed models in OpenAI shape
        router.get("/v1/models") { _, _ -> Response in
            let models = (try? registry.load()) ?? []
            let now = Int(Date().timeIntervalSince1970)
            let payload = OpenAI.ModelsList(
                data: models.map { modelInfo(for: $0) })
            return try jsonResponse(payload, status: .ok, fallbackTime: now)
        }

        // GET /v1/models/:id — single model info (queried by Codex and other clients)
        router.get("/v1/models/:id") { request, _ -> Response in
            let id = request.uri.path.components(separatedBy: "/").last ?? ""
            let models = (try? registry.load()) ?? []
            guard let descriptor = models.first(where: { $0.name == id || $0.repoId == id }) else {
                return errorResponse(.notFound, message: "Model '\(id)' not found")
            }
            let now = Int(Date().timeIntervalSince1970)
            return try jsonResponse(modelInfo(for: descriptor), status: .ok, fallbackTime: now)
        }

        // POST /v1/chat/completions
        router.post("/v1/chat/completions") { request, context -> Response in
            let rid = reqId()
            let req: OpenAI.ChatCompletionRequest
            do {
                req = try await request.decode(
                    as: OpenAI.ChatCompletionRequest.self, context: context)
            } catch {
                return errorResponse(.badRequest, message: "Invalid request body: \(error)")
            }
            let decoded: OpenAITranslator.DecodedRequest
            do {
                decoded = try OpenAITranslator.decode(req)
            } catch {
                return errorResponse(.badRequest, message: "\(error)")
            }

            logger.debug(
                "request_started",
                metadata: [
                    "req_id": "\(rid)", "route": "POST /v1/chat/completions",
                    "model": "\(decoded.model)",
                    "messages": "\(decoded.messages.count)",
                    "tools": "\(decoded.tools.count)",
                    "stream": "\(decoded.params.stream)",
                    "params": "\(summarize(decoded.params))",
                ])
            logger.trace("request_messages req_id=\(rid)\n\(summarize(decoded.messages))")

            let lease: ModelLease
            do {
                lease = try await manager.acquire(decoded.model)
            } catch let e as ProviderError {
                return errorResponse(.notFound, message: e.description)
            } catch {
                return errorResponse(.internalServerError, message: "\(error)")
            }

            if RouteGate.hasImages(decoded.messages),
                lease.provider.descriptor.capability != .vision
            {
                await lease.release()
                return errorResponse(
                    .badRequest,
                    message: "Model '\(decoded.model)' does not support image inputs.")
            }

            let completionId = "chatcmpl-\(UUID().uuidString)"
            let model = decoded.model

            if decoded.params.stream {
                return streamingResponse(
                    lease: lease,
                    decoded: decoded,
                    completionId: completionId,
                    model: model,
                    rid: rid,
                    logger: logger)
            } else {
                return await bufferedResponse(
                    lease: lease,
                    decoded: decoded,
                    completionId: completionId,
                    model: model,
                    rid: rid,
                    logger: logger)
            }
        }
    }

    // MARK: - Streaming

    private static func streamingResponse(
        lease: ModelLease,
        decoded: OpenAITranslator.DecodedRequest,
        completionId: String,
        model: String,
        rid: String,
        logger: Logger
    ) -> Response {
        let stream = lease.provider.generate(
            messages: decoded.messages,
            tools: decoded.tools,
            toolChoice: decoded.toolChoice,
            params: decoded.params)
        let body = ResponseBody { writer in
            let start = Date()
            var finishReason: FinishReason = .stop
            var toolCallCount = 0
            do {
                let state = OpenAITranslator.StreamState()
                var isFirst = true
                for try await event in stream {
                    logger.trace("event req_id=\(rid) \(describe(event))")
                    switch event {
                    case .toolUseStart: toolCallCount += 1
                    case .done(let r, _): finishReason = r
                    default: break
                    }
                    if let chunk = OpenAITranslator.chunkFor(
                        event: event, id: completionId, model: model,
                        state: state, isFirst: isFirst)
                    {
                        isFirst = false
                        let data = try encodeJSON(chunk)
                        try await writer.write(SSE.dataFrame(data))
                    }
                }
                try await writer.write(SSE.openAIDone)
                try await writer.finish(nil)
            } catch {
                logger.error("Streaming failed: \(error)")
                let payload = #"{"error":{"type":"internal_error","message":"\#(error)"}}"#
                try? await writer.write(SSE.dataFrame(payload))
                try? await writer.finish(nil)
            }
            let ms = Int(Date().timeIntervalSince(start) * 1000)
            logger.debug(
                "request_completed",
                metadata: [
                    "req_id": "\(rid)", "finish_reason": "\(finishReason.rawValue)",
                    "tool_calls": "\(toolCallCount)", "total_ms": "\(ms)",
                ])
            await lease.release()
        }
        return Response(status: .ok, headers: SSE.headers, body: body)
    }

    // MARK: - Non-streaming

    private static func bufferedResponse(
        lease: ModelLease,
        decoded: OpenAITranslator.DecodedRequest,
        completionId: String,
        model: String,
        rid: String,
        logger: Logger
    ) async -> Response {
        // Detached so the sync defer can perform the async release. Captures
        // `lease` (Sendable); runs once on any function-exit path.
        let start = Date()
        defer { Task { await lease.release() } }

        var fullText = ""
        var toolCalls: [ToolUse] = []
        var currentTool: (id: String, name: String, jsonBuffer: String)?
        var finishReason: FinishReason = .stop
        var usage: Usage?

        let stream = lease.provider.generate(
            messages: decoded.messages,
            tools: decoded.tools,
            toolChoice: decoded.toolChoice,
            params: decoded.params)
        do {
            for try await event in stream {
                logger.trace("event req_id=\(rid) \(describe(event))")
                switch event {
                case .textDelta(let s): fullText += s
                case .toolUseStart(let id, let name):
                    currentTool = (id, name, "")
                case .toolUseInputDelta(let chunk):
                    currentTool?.jsonBuffer += chunk
                case .toolUseStop:
                    if let t = currentTool {
                        let input: JSONValue =
                            (try? JSONDecoder().decode(
                                JSONValue.self, from: Data(t.jsonBuffer.utf8))) ?? .object([:])
                        toolCalls.append(.init(id: t.id, name: t.name, input: input))
                        currentTool = nil
                    }
                case .done(let reason, let u):
                    finishReason = reason
                    usage = u
                }
            }
        } catch {
            logger.error("Generation failed: \(error)")
            return errorResponse(.internalServerError, message: "\(error)")
        }

        let ms = Int(Date().timeIntervalSince(start) * 1000)
        logger.debug(
            "request_completed",
            metadata: [
                "req_id": "\(rid)", "finish_reason": "\(finishReason.rawValue)",
                "tool_calls": "\(toolCalls.count)", "total_ms": "\(ms)",
            ])

        let payload = OpenAITranslator.finalResponse(
            id: completionId, model: model,
            assistantText: fullText, toolCalls: toolCalls,
            finishReason: finishReason, usage: usage)
        do {
            return try jsonResponse(payload, status: .ok)
        } catch {
            return errorResponse(.internalServerError, message: "JSON encoding failed")
        }
    }

    // MARK: - Model info helpers

    private static func modelInfo(for descriptor: ModelDescriptor) -> OpenAI.ModelInfo {
        let ctx = ModelSnapshotInspector.contextLength(for: descriptor)
        return OpenAI.ModelInfo(
            id: descriptor.name,
            created: Int(descriptor.addedAt.timeIntervalSince1970),
            ownedBy: "smolx",
            contextWindow: ctx,
            maxOutputTokens: ctx)
    }

    // MARK: - JSON helpers (shared across routes)

    static func encodeJSON<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let data = try encoder.encode(value)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    static func jsonResponse<T: Encodable>(
        _ value: T, status: HTTPResponse.Status, fallbackTime: Int = 0
    ) throws -> Response {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let data = try encoder.encode(value)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data)))
    }

    static func errorResponse(_ status: HTTPResponse.Status, message: String) -> Response {
        let escaped = message.replacingOccurrences(of: "\"", with: "\\\"")
        let payload = #"{"error":{"type":"\#(status.code)","message":"\#(escaped)"}}"#
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(string: payload)))
    }
}
