import Testing
import Foundation
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

    @Test func budgetEvictionWhenLoadingExceedsBudget() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([
            descriptor("a", sizeGB: 8),
            descriptor("b", sizeGB: 8),
            descriptor("c", sizeGB: 8),
        ])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(memoryBudget: 16 * 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil))

        _ = try await mgr.acquire("a")
        _ = try await mgr.acquire("b")
        #expect(await mgr.currentlyLoaded() == ["a", "b"])
        // Loading c should evict the LRU (a).
        _ = try await mgr.acquire("c")
        let loaded = await mgr.currentlyLoaded()
        #expect(loaded.contains("c"))
        #expect(!loaded.contains("a"))
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
            settings: .init(memoryBudget: 1024 * 1_073_741_824, idleTimeout: 3600, maxConcurrent: 2))

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
            settings: .init(memoryBudget: 1024 * 1_073_741_824, idleTimeout: 0.05, maxConcurrent: nil))

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
            settings: .init(memoryBudget: 1024 * 1_073_741_824, idleTimeout: 0.05, maxConcurrent: nil))

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
            settings: .init(memoryBudget: 1024 * 1_073_741_824, idleTimeout: 0.1, maxConcurrent: nil))

        let lease1 = try await mgr.acquire("a")
        await lease1.release()
        // Re-acquire well before the 100ms timer would fire.
        let lease2 = try await mgr.acquire("a")
        try await Task.sleep(for: .seconds(0.25))
        #expect(await mgr.currentlyLoaded() == ["a"])
        #expect(await counter.count == 0)
        await lease2.release()
    }

    /// When budget eviction must drop a model, prefer the idle one over a
    /// model that still has an active lease.
    @Test func budgetEvictionPrefersIdleOverActive() async throws {
        let counter = UnloadCounter()
        let reg = try registryWith([
            descriptor("a", sizeGB: 8),
            descriptor("b", sizeGB: 8),
            descriptor("c", sizeGB: 8),
        ])
        let mgr = ModelManager(
            registry: reg,
            factory: StubFactory(unloaded: counter),
            settings: .init(memoryBudget: 16 * 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil))

        // `a` is held by an active lease for the entire test.
        let leaseA = try await mgr.acquire("a")
        // `b` is acquired then released — idle, with a 3600s timer that won't
        // fire during the test.
        let leaseB = try await mgr.acquire("b")
        await leaseB.release()

        // Loading `c` forces an eviction; the idle `b` must be chosen over `a`.
        let leaseC = try await mgr.acquire("c")
        let loaded = await mgr.currentlyLoaded()
        #expect(loaded.contains("a"))
        #expect(loaded.contains("c"))
        #expect(!loaded.contains("b"))
        #expect(await counter.count == 1)

        await leaseA.release()
        await leaseC.release()
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
            settings: .init(memoryBudget: 1024 * 1_073_741_824, idleTimeout: 3600, maxConcurrent: nil))

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
}
