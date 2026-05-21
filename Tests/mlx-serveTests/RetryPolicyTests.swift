import Testing
import Foundation
@testable import mlx_serve

@Suite("RetryPolicy")
struct RetryPolicyTests {

    actor AttemptLog {
        var attempts: [(Int, String?)] = []
        func record(_ n: Int, _ err: Error?) {
            attempts.append((n, err.map { String(describing: type(of: $0)) }))
        }
    }

    @Test func classifiesTransientURLErrors() {
        #expect(RetryPolicy.isTransient(URLError(.timedOut)))
        #expect(RetryPolicy.isTransient(URLError(.networkConnectionLost)))
        #expect(RetryPolicy.isTransient(URLError(.notConnectedToInternet)))
        #expect(RetryPolicy.isTransient(URLError(.cannotConnectToHost)))
    }

    @Test func classifiesNonTransientURLErrors() {
        #expect(!RetryPolicy.isTransient(URLError(.badServerResponse)))
        #expect(!RetryPolicy.isTransient(URLError(.userCancelledAuthentication)))
        #expect(!RetryPolicy.isTransient(URLError(.unsupportedURL)))
    }

    @Test func succeedsOnFirstAttempt() async throws {
        let log = AttemptLog()
        let result: Int = try await RetryPolicy.default.run(
            onAttempt: { n, err in Task { await log.record(n, err) } },
            { 42 })
        #expect(result == 42)
        // We don't assert on the log because it's async-recorded — but if
        // run() were broken we'd see a non-42 return.
    }

    @Test func retriesTransientErrorsThenSucceeds() async throws {
        // Custom policy with near-zero delays to keep the test fast.
        let policy = RetryPolicy(schedule: [0.001, 0.001, 0.001], maxAttempts: 4)

        // Box around the call counter — sendable closure capture rules.
        actor Counter { var n = 0; func tick() -> Int { n += 1; return n } }
        let counter = Counter()

        let result = try await policy.run { () -> String in
            let attempt = await counter.tick()
            if attempt < 3 {
                throw URLError(.timedOut)
            }
            return "ok-on-\(attempt)"
        }
        #expect(result == "ok-on-3")
    }

    @Test func nonTransientErrorAbortsImmediately() async {
        let policy = RetryPolicy(schedule: [0.001, 0.001], maxAttempts: 3)
        actor Counter { var n = 0; func tick() -> Int { n += 1; return n } }
        let counter = Counter()

        do {
            _ = try await policy.run { () -> Int in
                _ = await counter.tick()
                throw URLError(.badURL)  // not transient
            }
            Issue.record("expected throw")
        } catch is URLError {
            // ok
        } catch {
            Issue.record("wrong error type: \(error)")
        }
        let calls = await counter.n
        #expect(calls == 1, "should not retry non-transient errors (got \(calls) attempts)")
    }

    @Test func exhaustsAttemptsAndThrows() async {
        let policy = RetryPolicy(schedule: [0.001, 0.001], maxAttempts: 3)
        actor Counter { var n = 0; func tick() -> Int { n += 1; return n } }
        let counter = Counter()

        do {
            _ = try await policy.run { () -> Int in
                _ = await counter.tick()
                throw URLError(.timedOut)
            }
            Issue.record("expected throw")
        } catch {
            // ok
        }
        let calls = await counter.n
        #expect(calls == 3, "should attempt exactly maxAttempts times (got \(calls))")
    }
}
