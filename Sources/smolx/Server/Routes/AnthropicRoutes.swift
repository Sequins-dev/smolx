import Foundation
import Hummingbird
import HummingbirdCore
import Logging
import NIOCore

enum AnthropicRoutes {

    static func register<Context: RequestContext>(
        _ router: Router<Context>,
        manager: ModelManager,
        registry: ModelRegistry,
        logger: Logger
    ) {
        router.post("/v1/messages") { request, context -> Response in
            let rid = reqId()
            let req: Anthropic.MessagesRequest
            do {
                req = try await request.decode(
                    as: Anthropic.MessagesRequest.self, context: context)
            } catch {
                return OpenAIRoutes.errorResponse(.badRequest, message: "Invalid request body: \(error)")
            }
            let decoded: AnthropicTranslator.DecodedRequest
            do {
                decoded = try AnthropicTranslator.decode(req)
            } catch {
                return OpenAIRoutes.errorResponse(.badRequest, message: "\(error)")
            }

            logger.debug(
                "request_started",
                metadata: [
                    "req_id": "\(rid)", "route": "POST /v1/messages",
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
                return OpenAIRoutes.errorResponse(.notFound, message: e.description)
            } catch {
                return OpenAIRoutes.errorResponse(.internalServerError, message: "\(error)")
            }

            if RouteGate.hasImages(decoded.messages),
                lease.provider.descriptor.capability != .vision
            {
                await lease.release()
                return OpenAIRoutes.errorResponse(
                    .badRequest,
                    message: "Model '\(decoded.model)' does not support image inputs.")
            }

            let messageId = "msg_\(UUID().uuidString)"

            if decoded.params.stream {
                return streamingResponse(
                    lease: lease, decoded: decoded,
                    messageId: messageId, rid: rid, logger: logger)
            } else {
                return await bufferedResponse(
                    lease: lease, decoded: decoded,
                    messageId: messageId, rid: rid, logger: logger)
            }
        }
    }

    // MARK: - Streaming

    private static func streamingResponse(
        lease: ModelLease,
        decoded: AnthropicTranslator.DecodedRequest,
        messageId: String,
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
                let state = AnthropicTranslator.StreamState(
                    messageId: messageId, model: decoded.model)
                for frame in try AnthropicTranslator.startFrames(state: state) {
                    try await writer.write(
                        SSE.namedEventFrame(
                            event: frame.event, json: frame.jsonData))
                }
                for try await event in stream {
                    logger.trace("event req_id=\(rid) \(describe(event))")
                    switch event {
                    case .toolUseStart: toolCallCount += 1
                    case .done(let r, _): finishReason = r
                    default: break
                    }
                    for frame in try AnthropicTranslator.frames(for: event, state: state) {
                        try await writer.write(
                            SSE.namedEventFrame(
                                event: frame.event, json: frame.jsonData))
                    }
                }
                try await writer.finish(nil)
            } catch {
                logger.error("Streaming failed: \(error)")
                let ping = try? AnthropicTranslator.pingFrame()
                if let ping {
                    try? await writer.write(
                        SSE.namedEventFrame(
                            event: ping.event, json: ping.jsonData))
                }
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
        decoded: AnthropicTranslator.DecodedRequest,
        messageId: String,
        rid: String,
        logger: Logger
    ) async -> Response {
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
            return OpenAIRoutes.errorResponse(.internalServerError, message: "\(error)")
        }

        let ms = Int(Date().timeIntervalSince(start) * 1000)
        logger.debug(
            "request_completed",
            metadata: [
                "req_id": "\(rid)", "finish_reason": "\(finishReason.rawValue)",
                "tool_calls": "\(toolCalls.count)", "total_ms": "\(ms)",
            ])

        let payload = AnthropicTranslator.finalResponse(
            id: messageId, model: decoded.model,
            assistantText: fullText, toolCalls: toolCalls,
            finishReason: finishReason, usage: usage)
        do {
            return try OpenAIRoutes.jsonResponse(payload, status: .ok)
        } catch {
            return OpenAIRoutes.errorResponse(.internalServerError, message: "JSON encoding failed")
        }
    }
}
