import Foundation

/// Streaming parser for OpenAI's Harmony channel format, which is what
/// gpt-oss models emit instead of normalised text + tool-call markers.
///
/// Raw format looks like:
///
///     <|start|>assistant<|channel|>analysis<|message|>chain of thought<|end|>
///     <|start|>assistant<|channel|>commentary to=functions.NAME<|constrain|>json<|message|>{"a":1}<|call|>
///     <|start|>assistant<|channel|>final<|message|>visible answer<|return|>
///
/// Note the per-channel terminator: analysis uses `<|end|>`, function-call
/// commentary uses `<|call|>` (NOT `<|end|>` — that's a foot-gun in OpenAI's
/// spec), and the final answer uses `<|return|>`. We accept any of the three
/// as a terminator and let the channel context decide what to do at the cut.
///
/// MLX runs gpt-oss inference fine, but mlx-swift-lm doesn't ship a Harmony
/// post-processor — so without us doing it here, those markers leak straight
/// through to the client and tool calls never fire. We swallow the analysis
/// channel, extract tool calls from `commentary to=functions.X`, and emit
/// plain text from the final channel.
final class HarmonyParser: @unchecked Sendable {

    enum Event: Equatable, Sendable {
        case textDelta(String)
        case toolUseStart(id: String, name: String)
        case toolUseInputDelta(String)
        case toolUseStop
    }

    private enum State {
        case idle  // between blocks; suppress
        case analysis  // chain-of-thought; suppress
        case commentary(id: String, name: String)  // tool args; emit as input delta
        case finalChannel  // user-visible; emit as text
    }

    /// Longest trailing prefix we have to hold back to avoid splitting a
    /// `<|...|>` marker across emit boundaries. `<|constrain|>` is 13 chars
    /// — the largest marker we care about — so this is the floor.
    private static let markerSafetyMargin = 13

    private var state: State = .idle
    private var buffer = ""

    /// Feed a chunk of raw model output. Returns any events that became
    /// emittable as a result.
    func feed(_ chunk: String) -> [Event] {
        if chunk.isEmpty { return [] }
        buffer.append(chunk)
        return drain()
    }

    /// Called when the upstream stream finishes. Flushes any in-flight
    /// channel state and emits residual content. After this the parser is
    /// reset and ready for a fresh stream.
    func flush() -> [Event] {
        var events = drain()
        switch state {
        case .commentary:
            // Tool call never saw its terminator — drain the residual JSON
            // (the safety-margin tail that `drainContentful` was holding
            // back for terminator detection) and close the tool_use block.
            // Dropping this tail silently truncates the args JSON — e.g.
            // `{"path":"./",` lost its final `"recursive":true}` — which
            // surfaces in clients as "Unterminated string" parse errors.
            if !buffer.isEmpty {
                events.append(.toolUseInputDelta(buffer))
            }
            events.append(.toolUseStop)
        case .finalChannel:
            // Emit residual final-channel text. analysis residue is dropped.
            if !buffer.isEmpty {
                events.append(.textDelta(buffer))
            }
        case .analysis, .idle:
            break
        }
        buffer.removeAll()
        state = .idle
        return events
    }

    // MARK: - Internals

    private func drain() -> [Event] {
        var events: [Event] = []
        while drainStep(&events) { /* keep going while progress is made */  }
        return events
    }

    /// One pass of the state machine. Returns true if progress was made so
    /// the caller can re-enter (e.g. a channel just opened, we should now
    /// try to drain its content).
    private func drainStep(_ events: inout [Event]) -> Bool {
        switch state {
        case .idle:
            return drainIdle(&events)
        case .analysis:
            return drainSuppress()
        case .commentary:
            return drainContentful(&events, asText: false)
        case .finalChannel:
            return drainContentful(&events, asText: true)
        }
    }

    /// Look for a complete `<|channel|>...<|message|>` header. If we have
    /// one, parse it and transition into the corresponding channel state.
    private func drainIdle(_ events: inout [Event]) -> Bool {
        guard let chRange = buffer.range(of: "<|channel|>") else {
            // No channel start yet — drop everything except a tail that
            // might be a partial `<|channel|>` marker.
            shrinkBufferToSafetyMargin()
            return false
        }
        guard
            let msgRange = buffer.range(
                of: "<|message|>", range: chRange.upperBound..<buffer.endIndex)
        else {
            // Header is incomplete; wait for more input. Don't consume yet.
            return false
        }

        let header = String(buffer[chRange.upperBound..<msgRange.lowerBound])
        buffer.removeSubrange(buffer.startIndex..<msgRange.upperBound)

        let trimmed = header.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("analysis") {
            state = .analysis
        } else if trimmed.hasPrefix("commentary"), let toolName = Self.toolName(in: trimmed) {
            let id = "toolu_" + UUID().uuidString.prefix(16).lowercased()
            events.append(.toolUseStart(id: id, name: toolName))
            state = .commentary(id: id, name: toolName)
        } else if trimmed.hasPrefix("final") {
            state = .finalChannel
        } else {
            // Unknown channel (including plain commentary without a tool
            // recipient — that's just inter-message notes). Suppress.
            state = .analysis
        }
        return true
    }

    /// Discard input until the current channel terminates.
    private func drainSuppress() -> Bool {
        if let term = firstTerminator() {
            buffer.removeSubrange(buffer.startIndex..<term.upperBound)
            state = .idle
            return true
        }
        // Keep just enough tail to recognise an incoming terminator.
        shrinkBufferToSafetyMargin()
        return false
    }

    /// Emit text or tool-input deltas until the current channel terminates.
    private func drainContentful(_ events: inout [Event], asText: Bool) -> Bool {
        if let term = firstTerminator() {
            let content = String(buffer[buffer.startIndex..<term.lowerBound])
            if !content.isEmpty {
                events.append(asText ? .textDelta(content) : .toolUseInputDelta(content))
            }
            if !asText {
                events.append(.toolUseStop)
            }
            buffer.removeSubrange(buffer.startIndex..<term.upperBound)
            state = .idle
            return true
        }
        // No terminator yet — emit everything except a tail that could be
        // a partial marker, leave the tail in the buffer for next round.
        let tailCount = min(buffer.count, Self.markerSafetyMargin)
        let safeCount = buffer.count - tailCount
        if safeCount > 0 {
            let emit = String(buffer.prefix(safeCount))
            events.append(asText ? .textDelta(emit) : .toolUseInputDelta(emit))
            buffer.removeFirst(safeCount)
        }
        return false
    }

    /// Locate the nearest channel terminator: `<|end|>` (analysis / inter-
    /// message), `<|call|>` (function-call commentary), or `<|return|>`
    /// (final answer). All three are legal terminators in Harmony; the
    /// channel state decides what to do with the buffered content when one
    /// of them appears.
    private func firstTerminator() -> Range<String.Index>? {
        let candidates = ["<|end|>", "<|call|>", "<|return|>"]
            .compactMap { buffer.range(of: $0) }
        return candidates.min { $0.lowerBound < $1.lowerBound }
    }

    /// Drop buffer content except a small tail that could be a partial
    /// `<|...|>` marker we'd lose information about by truncating.
    private func shrinkBufferToSafetyMargin() {
        let drop = buffer.count - Self.markerSafetyMargin
        if drop > 0 {
            buffer.removeFirst(drop)
        }
    }

    /// Parse the tool name out of a commentary header. Header looks like
    /// `commentary to=functions.NAME <|constrain|>json` or just
    /// `commentary to=functions.NAME`. Returns nil if no recipient is set.
    private static func toolName(in header: String) -> String? {
        guard let r = header.range(of: "to=functions.") else { return nil }
        let after = header[r.upperBound...]
        // Take until whitespace or `<` (start of the next directive marker).
        let endIdx = after.firstIndex { $0 == " " || $0 == "<" || $0 == "\t" } ?? after.endIndex
        let name = after[after.startIndex..<endIdx]
        return name.isEmpty ? nil : String(name)
    }
}
