import Foundation

/// Retry wrapper for network operations. Transient errors get exponential
/// backoff; non-transient errors surface immediately so callers see real
/// failures fast (404 / disk full / auth).
///
/// We rely on swift-huggingface preserving partial bytes in
/// `<etag>.incomplete` between attempts — each retry calls `downloadFile`
/// again and the library issues an HTTP `Range:` request from the existing
/// offset automatically.
struct RetryPolicy: Sendable {
    /// Backoff schedule in seconds. The Nth retry waits `schedule[N-1]`. After
    /// running out of entries, retries continue at the last value (cap).
    var schedule: [Double]
    /// Maximum total attempts including the initial one.
    var maxAttempts: Int

    static let `default` = RetryPolicy(
        schedule: [0.5, 1.0, 2.0, 4.0, 8.0],
        maxAttempts: 5)

    /// Never retries — useful in tests.
    static let none = RetryPolicy(schedule: [], maxAttempts: 1)

    /// Invokes `body` repeatedly until success or non-transient failure.
    /// `onAttempt` is called before each attempt (1-indexed); use it to surface
    /// "retrying" state to the UI.
    func run<T: Sendable>(
        onAttempt: (@Sendable (_ attempt: Int, _ previousError: Error?) -> Void)? = nil,
        _ body: @Sendable () async throws -> T
    ) async throws -> T {
        var lastError: Error?
        for attempt in 1...maxAttempts {
            try Task.checkCancellation()
            onAttempt?(attempt, lastError)
            do {
                return try await body()
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
                if !Self.isTransient(error) || attempt == maxAttempts {
                    throw error
                }
                let delaySeconds =
                    schedule.indices.contains(attempt - 1)
                    ? schedule[attempt - 1]
                    : (schedule.last ?? 1.0)
                try await Task.sleep(nanoseconds: UInt64(delaySeconds * 1_000_000_000))
            }
        }
        // Unreachable — the loop either returns or throws.
        throw lastError ?? CancellationError()
    }

    /// Classify whether an error should be retried.
    /// Conservative on purpose — anything we don't explicitly recognise is
    /// treated as terminal, so misclassification can't silently loop forever.
    static func isTransient(_ error: Error) -> Bool {
        if let urlErr = error as? URLError {
            switch urlErr.code {
            case .timedOut,
                .networkConnectionLost,
                .notConnectedToInternet,
                .dnsLookupFailed,
                .cannotConnectToHost,
                .cannotFindHost,
                .resourceUnavailable,
                .secureConnectionFailed,
                .dataNotAllowed,
                .internationalRoamingOff:
                return true
            default:
                return false
            }
        }
        // POSIX errors from the BSD socket layer
        if let posix = error as? POSIXError {
            switch posix.code {
            case .ECONNRESET, .ECONNREFUSED, .EPIPE, .ETIMEDOUT, .ENETDOWN, .ENETUNREACH:
                return true
            default:
                return false
            }
        }
        // HTTP layer errors from swift-huggingface — string match since the
        // package doesn't export its error type publicly in a way that lets us
        // pattern-match on status codes cleanly.
        let description = String(describing: error)
        if description.contains("status code: 5") || description.contains("HTTP 5") {
            return true
        }
        if description.contains("429") {  // rate limit
            return true
        }
        return false
    }
}
