import Foundation
import Testing

@testable import smolx

@Suite("ModelManager")
struct ModelManagerTests {

    /// Stub provider that records load/unload events and reports a fixed size.
    actor StubProvider: ModelProvider {
        nonisolated let descriptor: ModelDescriptor
        private(set) var residentBytes: Int64
        private(set) var lastUsedAt: Date
        let counter: UnloadCounter

        init(descriptor: ModelDescriptor, counter: UnloadCounter) {
            self.descriptor = descriptor
            self.residentBytes = descriptor.diskSizeBytes
            self.lastUsedAt = Date()
            self.counter = counter
        }

        nonisolated func generate(
            messages: [ChatMessage],
            tools: [ToolDefinition],
            toolChoice: ToolChoice?,
            params: GenerationParams
        ) -> AsyncThrowingStream<StreamEvent, Error> {
            AsyncThrowingStream { continuation in
                continuation.yield(.textDelta("ok"))
                continuation.yield(.done(finishReason: .stop, usage: nil))
                continuation.finish()
            }
        }

        // Awaiting the counter directly removes the race that used to exist
        // when this dispatched a detached Task — the manager's
        // `await provider.unload()` now blocks until the increment is visible.
        func unload() async { await counter.tick() }
    }

    struct StubFactory: ProviderFactory {
        let unloaded: UnloadCounter

        func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider {
            StubProvider(descriptor: descriptor, counter: unloaded)
        }
    }

    actor UnloadCounter {
        private(set) var count = 0
        func tick() { count += 1 }
    }

    /// Lock-protected `MemoryStats` stub. Tests start it at `plenty()` (a
    /// huge constant) when they don't care about the floor, and explicitly
    /// pin a low value when they want to drive eviction.
    final class StubStats: MemoryStats, @unchecked Sendable {
        private let lock = NSLock()
        private var _bytes: Int64
        init(_ b: Int64) { _bytes = b }
        var availableBytes: Int64 {
            lock.lock()
            defer { lock.unlock() }
            return _bytes
        }
        func set(_ b: Int64) {
            lock.lock()
            defer { lock.unlock() }
            _bytes = b
        }
    }

    /// Stats high enough that the floor will never trip during a test.
    static func plenty() -> StubStats { StubStats(1024 * 1_073_741_824) }

    private func descriptor(_ name: String, sizeGB: Int) -> ModelDescriptor {
        ModelDescriptor(
            name: name, repoId: "test/\(name)",
            localPath: "/tmp/\(name)", capability: .text,
            diskSizeBytes: Int64(sizeGB) * 1_073_741_824,
            addedAt: Date())
    }

    private func registryWith(_ descs: [ModelDescriptor]) throws -> ModelRegistry {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("smolx-test-\(UUID().uuidString).json")
        let reg = ModelRegistry(url: tmp)
        try reg.save(descs)
        return reg
    }

    @Test func loadGateEvictsWhenAvailableBelowFloor() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([
            descriptor("a", sizeGB: 8),
            descriptor("b", sizeGB: 8),
        ])
        // Floor 8 GB, available 0 — makeRoom must evict the loaded entry
        // before proceeding with the new load. With stub stats locked at 0,
        // the loop drains `loaded` to empty (its fallback exit) and then
        // the new model loads.
        let stats = StubStats(0)
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 8 * 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil,
                stats: stats))

        _ = try await mgr.acquire("a")
        _ = try await mgr.acquire("b")
        // `a` was evicted during the makeRoom step when loading `b`.
        let loaded = await mgr.currentlyLoaded()
        #expect(loaded == ["b"])
        #expect(await counter.count == 1)
    }

    @Test func maxConcurrentCap() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([
            descriptor("a", sizeGB: 1),
            descriptor("b", sizeGB: 1),
            descriptor("c", sizeGB: 1),
        ])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: 3600, maxConcurrent: 2,
                stats: Self.plenty()))

        _ = try await mgr.acquire("a")
        _ = try await mgr.acquire("b")
        _ = try await mgr.acquire("c")
        #expect((await mgr.currentlyLoaded()).count == 2)
        #expect(await counter.count == 1)
    }

    @Test func unknownModelThrows() async throws {
        let reg = try registryWith([])
        let mgr = ModelManager(
            registry: reg, factory: StubFactory(unloaded: UnloadCounter()))
        await #expect(throws: ProviderError.self) {
            _ = try await mgr.acquire("ghost")
        }
    }

    // MARK: - Ref-counted idle-unload behavior

    /// Releasing the last lease starts the idle timer; once it elapses, the
    /// model is unloaded.
    @Test func idleUnloadAfterRelease() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: 0.05, maxConcurrent: nil,
                stats: Self.plenty()))

        let lease = try await mgr.acquire("a")
        await lease.release()
        try await Task.sleep(for: .seconds(0.3))
        #expect(await mgr.currentlyLoaded() == [])
        #expect(await counter.count == 1)
    }

    /// A model with at least one active lease must not be unloaded even after
    /// the idle timeout has elapsed.
    @Test func activeRequestSurvivesIdleTimeout() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: 0.05, maxConcurrent: nil,
                stats: Self.plenty()))

        let lease1 = try await mgr.acquire("a")
        let lease2 = try await mgr.acquire("a")
        try await Task.sleep(for: .seconds(0.2))
        #expect(await mgr.currentlyLoaded() == ["a"])
        #expect(await counter.count == 0)

        await lease1.release()
        try await Task.sleep(for: .seconds(0.2))
        // Still one active lease — should remain loaded.
        #expect(await mgr.currentlyLoaded() == ["a"])
        #expect(await counter.count == 0)

        await lease2.release()
        try await Task.sleep(for: .seconds(0.2))
        #expect(await mgr.currentlyLoaded() == [])
        #expect(await counter.count == 1)
    }

    /// Re-acquiring a model after release but before the idle timer fires
    /// cancels the pending unload, keeping the model warm.
    @Test func reacquireCancelsIdleTimer() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: 0.1, maxConcurrent: nil,
                stats: Self.plenty()))

        let lease1 = try await mgr.acquire("a")
        await lease1.release()
        // Re-acquire well before the 100ms timer would fire.
        let lease2 = try await mgr.acquire("a")
        try await Task.sleep(for: .seconds(0.25))
        #expect(await mgr.currentlyLoaded() == ["a"])
        #expect(await counter.count == 0)
        await lease2.release()
    }

    /// When the floor watcher fires, it must prefer the idle entry over the
    /// one that still has an active lease.
    @Test func floorEvictionPrefersIdleOverActive() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([
            descriptor("a", sizeGB: 8),
            descriptor("b", sizeGB: 8),
        ])
        let stats = Self.plenty()
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 8 * 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil,
                stats: stats))

        // `a` is held by an active lease for the entire test.
        let leaseA = try await mgr.acquire("a")
        // `b` is acquired then released — idle.
        let leaseB = try await mgr.acquire("b")
        await leaseB.release()

        // Drop available memory below the floor and ask the manager to
        // re-check. The idle `b` must be chosen over the active `a`.
        stats.set(0)
        await mgr.evictIfBelowFloor()
        let loaded = await mgr.currentlyLoaded()
        #expect(loaded == ["a"])
        #expect(await counter.count == 1)

        await leaseA.release()
    }

    /// `evictIfBelowFloor` is a no-op when available memory is above the
    /// configured floor — the watcher must not churn through models while
    /// the system has room.
    @Test func floorWatcherDoesNothingAboveFloor() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil,
                stats: Self.plenty()))

        _ = try await mgr.acquire("a")
        await mgr.evictIfBelowFloor()
        #expect(await mgr.currentlyLoaded() == ["a"])
        #expect(await counter.count == 0)
    }

    /// Two simultaneous `acquire` calls for the same not-yet-loaded model
    /// must trigger exactly one `factory.make(...)`. Without dedup, both
    /// callers race past the `loaded[name] == nil` check and the manager
    /// allocates two copies of the model in GPU memory — the bug the user
    /// hit with opencode firing concurrent streaming + non-streaming calls.
    @Test func concurrentAcquireDedupsLoad() async throws {
        actor SlowMakeCounter {
            private(set) var made = 0
            func tick() { made += 1 }
        }
        let madeCounter = SlowMakeCounter()
        let unloadCounter = UnloadCounter()

        struct SlowFactory: ProviderFactory {
            let madeCounter: SlowMakeCounter
            let unloadCounter: UnloadCounter

            func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider {
                await madeCounter.tick()
                // Sleep so the test reliably overlaps two `acquire` calls.
                try await Task.sleep(for: .milliseconds(80))
                return StubProvider(descriptor: descriptor, counter: unloadCounter)
            }
        }

        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let mgr = ModelManager(
            registry: reg,
            factory: SlowFactory(madeCounter: madeCounter, unloadCounter: unloadCounter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil,
                stats: Self.plenty()))

        async let l1 = mgr.acquire("a")
        async let l2 = mgr.acquire("a")
        let lease1 = try await l1
        let lease2 = try await l2

        let calls = await madeCounter.made
        #expect(calls == 1, "factory.make called \(calls) times — expected 1 (dedup failed)")
        // Both leases must independently drive the ref count; releasing one
        // does NOT unload because the other is still active.
        await lease1.release()
        try await Task.sleep(for: .milliseconds(50))
        #expect(await mgr.currentlyLoaded() == ["a"])
        await lease2.release()
    }

    // MARK: - Lazy mode (idleTimeout = nil)

    /// With no idle timeout configured, a released model stays resident — no
    /// per-release timer is scheduled. Only budget / max-concurrent eviction
    /// can unload it.
    @Test func noIdleTimeoutKeepsModelResident() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 1_073_741_824, idleTimeout: nil, maxConcurrent: nil,
                stats: Self.plenty()))

        let lease = try await mgr.acquire("a")
        await lease.release()
        try await Task.sleep(for: .seconds(0.3))
        #expect(await mgr.currentlyLoaded() == ["a"])
        #expect(await counter.count == 0)
    }

    /// Lazy mode (no idle timer) must still honour the keep-free floor — the
    /// background watcher evicts the LRU when available memory drops, even
    /// without an idle timer running.
    @Test func noIdleTimeoutStillEvictsOnFloor() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([descriptor("a", sizeGB: 1)])
        let stats = Self.plenty()
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(
                keepFreeBytes: 8 * 1_073_741_824, idleTimeout: nil, maxConcurrent: nil,
                stats: stats))

        let lease = try await mgr.acquire("a")
        await lease.release()
        stats.set(0)
        await mgr.evictIfBelowFloor()
        #expect(await mgr.currentlyLoaded() == [])
        #expect(await counter.count == 1)
    }
}
