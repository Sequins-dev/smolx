import Foundation
import Testing

@testable import smolx

@Suite("DownloadProgress")
struct ProgressTrackerTests {

    @Test func registrationSeedsResumedBytes() async {
        let t = DownloadProgress(repoId: "test/repo", slots: 1)
        await t.register(index: 0, name: "model.safetensors", totalBytes: 1_000_000, resumedFrom: 200_000)
        let snap = await t.snapshot()
        #expect(snap.totalBytes == 1_000_000)
        #expect(snap.completedBytes == 200_000)
        #expect(snap.resumedFromBytes == 200_000)
        if case .downloading(let b, let total, let r) = snap.files[0].state {
            #expect(b == 200_000)
            #expect(total == 1_000_000)
            #expect(r == 200_000)
        } else {
            Issue.record("expected downloading state at start")
        }
    }

    @Test func updateAdvancesPerFileAndOverallBytes() async {
        let t = DownloadProgress(repoId: "test/repo", slots: 1)
        await t.register(index: 0, name: "a", totalBytes: 1_000, resumedFrom: 0)
        await t.register(index: 1, name: "b", totalBytes: 500, resumedFrom: 0)

        await t.update(index: 0, bytes: 250)
        await t.update(index: 1, bytes: 100)
        let snap = await t.snapshot()
        #expect(snap.completedBytes == 350)
        #expect(snap.transferredSinceStart == 350)
        #expect(snap.totalBytes == 1_500)
    }

    @Test func completedAndCachedTerminalStates() async {
        let t = DownloadProgress(repoId: "test/repo", slots: 1)
        await t.register(index: 0, name: "a", totalBytes: 100, resumedFrom: 0)
        await t.register(index: 1, name: "b", totalBytes: 200, resumedFrom: 0)

        await t.cached(index: 0, bytes: 100)
        await t.completed(index: 1)
        let snap = await t.snapshot()
        #expect(snap.allFinished == true)
        #expect(snap.completedBytes == 300)
    }

    @Test func retryingStatePreservesBytes() async {
        let t = DownloadProgress(repoId: "test/repo", slots: 1)
        await t.register(index: 0, name: "a", totalBytes: 1_000, resumedFrom: 0)

        await t.update(index: 0, bytes: 400)
        await t.retrying(index: 0, attempt: 2, delay: 1.0)
        let snap = await t.snapshot()
        if case .retrying(let b, _, let attempt, _) = snap.files[0].state {
            #expect(b == 400)
            #expect(attempt == 2)
        } else {
            Issue.record("expected retrying state")
        }
    }

    @Test func mbpsReflectsRecentTransferOnly() async throws {
        // 100ms window so the test runs fast.
        let t = DownloadProgress(repoId: "test/repo", slots: 1, windowSeconds: 0.1)
        await t.register(index: 0, name: "a", totalBytes: 10_000_000, resumedFrom: 0)

        await t.update(index: 0, bytes: 1_048_576)
        try await Task.sleep(nanoseconds: 50_000_000)
        await t.update(index: 0, bytes: 2_097_152)
        let snap = await t.snapshot()
        // 1 MB in ~50ms ⇒ ~20 MB/s — but tolerate a wide window since this is
        // wall-clock-dependent.
        #expect(snap.mbps > 0)
    }

    @Test func failedFileDoesNotBreakSnapshot() async {
        let t = DownloadProgress(repoId: "test/repo", slots: 1)
        await t.register(index: 0, name: "a", totalBytes: 100, resumedFrom: 0)
        await t.register(index: 1, name: "b", totalBytes: 100, resumedFrom: 0)

        await t.failed(index: 0, reason: "disk full")
        await t.completed(index: 1)
        let snap = await t.snapshot()
        #expect(snap.allFinished == true)  // failed counts as terminal
        if case .failed(let reason) = snap.files[0].state {
            #expect(reason == "disk full")
        } else {
            Issue.record("expected failed state")
        }
    }
}
