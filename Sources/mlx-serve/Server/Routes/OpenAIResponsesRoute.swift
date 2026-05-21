import Foundation
import Hummingbird
import HummingbirdCore
import Logging
import NIOCore

/// POST /v1/responses — OpenAI's Responses API. codex 0.128+ uses this
/// exclusively (`wire_api = "chat"` was removed). Streaming and buffered
/// branches share decoding + provider acquisition; the streaming branch
/// drives `ResponsesTranslator.StreamState` to emit the item-lifecycle
/// event sequence codex's SSE parser expects.
enum OpenAIResponsesRoute {

    static func register<Context: RequestContext>(
        _ router: Router<Context>,
        manager: ModelManager,
        registry: ModelRegistry,
        logger: Logger
    ) {
        router.post("/v1/responses") { request, context -> Response in
            let req: OpenAIResponses.Request
            do {
                req = try await request.decode(
                    as: OpenAIResponses.Request.self, context: context)
            } catch {
                return OpenAIRoutes.errorResponse(.badRequest, message: "Invalid request body: \(error)")
            }
            let decoded: ResponsesTranslator.DecodedRequest
            do {
                decoded = try ResponsesTranslator.decode(req)
            } catch {
                return OpenAIRoutes.errorResponse(.badRequest, message: "\(error)")
            }

            let lease: ModelLease
            do {
                lease = try await manager.acquire(decoded.model)
            } catch let e as ProviderError {
                return OpenAIRoutes.errorResponse(.notFound, message: e.description)
            } catch {
                return OpenAIRoutes.errorResponse(.internalServerError, message: "\(error)")
            }

            if RouteGate.hasImages(decoded.messages),
               lease.provider.descriptor.capability != .vision {
                await lease.release()
                return OpenAIRoutes.errorResponse(.badRequest,
                    message: "Model '\(decoded.model)' does not support image inputs.")
            }

            let responseId = "resp_" + UUID().uuidString.prefix(24).lowercased()

            if decoded.params.stream {
                return streamingResponse(
                    lease: lease, decoded: decoded,
                    responseId: responseId, logger: logger)
            } else {
                return await bufferedResponse(
                    lease: lease, decoded: decoded,
                    responseId: responseId, logger: logger)
            }
        }
    }

    // MARK: - Streaming

    private static func streamingResponse(
        lease: ModelLease,
        decoded: ResponsesTranslator.DecodedRequest,
        responseId: String,
        logger: Logger
    ) -> Response {
        let stream = lease.provider.generate(
            messages: decoded.messages,
            tools: decoded.tools,
            toolChoice: decoded.toolChoice,
            params: decoded.params)
        let body = ResponseBody { writer in
            do {
                let state = ResponsesTranslator.StreamState(
                    responseId: responseId, model: decoded.model)
                for frame in try ResponsesTranslator.startFrames(state: state) {
                    try await writer.write(SSE.namedEventFrame(
                        event: frame.event, json: frame.jsonData))
                }
                for try await event in stream {
                    for frame in try ResponsesTranslator.frames(for: event, state: state) {
                        try await writer.write(SSE.namedEventFrame(
                            event: frame.event, json: frame.jsonData))
                    }
                }
                try await writer.finish(nil)
            } catch {
                logger.error("Streaming failed: \(error)")
                try? await writer.finish(nil)
            }
            await lease.release()
        }
        return Response(status: .ok, headers: SSE.headers, body: body)
    }

    // MARK: - Non-streaming

    private static func bufferedResponse(
        lease: ModelLease,
        decoded: ResponsesTranslator.DecodedRequest,
        responseId: String,
        logger: Logger
    ) async -> Response {
        defer { Task { await lease.release() } }

        var fullText = ""
        var toolCalls: [ToolUse] = []
        var currentTool: (id: String, name: String, jsonBuffer: String)?
        var usage: Usage?

        let stream = lease.provider.generate(
            messages: decoded.messages,
            tools: decoded.tools,
            toolChoice: decoded.toolChoice,
            params: decoded.params)
        do {
            for try await event in stream {
                switch event {
                case .textDelta(let s): fullText += s
                case .toolUseStart(let id, let name):
                    currentTool = (id, name, "")
                case .toolUseInputDelta(let chunk):
                    currentTool?.jsonBuffer += chunk
                case .toolUseStop:
                    if let t = currentTool {
                        let input: JSONValue = (try? JSONDecoder().decode(
                            JSONValue.self, from: Data(t.jsonBuffer.utf8))) ?? .object([:])
                        toolCalls.append(.init(id: t.id, name: t.name, input: input))
                        currentTool = nil
                    }
                case .done(_, let u):
                    usage = u
                }
            }
        } catch {
            logger.error("Generation failed: \(error)")
            return OpenAIRoutes.errorResponse(.internalServerError, message: "\(error)")
        }

        let payload = ResponsesTranslator.finalResponse(
            id: responseId, model: decoded.model,
            assistantText: fullText, toolCalls: toolCalls,
            usage: usage)
        do {
            return try OpenAIRoutes.jsonResponse(payload, status: .ok)
        } catch {
            return OpenAIRoutes.errorResponse(.internalServerError, message: "JSON encoding failed")
        }
    }
}
