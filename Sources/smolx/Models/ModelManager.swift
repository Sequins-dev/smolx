import Foundation
import Logging

/// Builds new provider instances on demand. Keeps `ModelManager` agnostic to
/// the concrete provider type, which makes the manager unit-testable with a
/// stub provider and lets us add new backends without touching the manager.
protocol ProviderFactory: Sendable {
    func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider
}

/// Source of "currently available system memory" readings. Pulled into a
/// protocol so tests can stub the value rather than depending on whatever
/// the host happens to have free at test time.
protocol MemoryStats: Sendable {
    var availableBytes: Int64 { get }
}

struct SystemMemoryStats: MemoryStats {
    var availableBytes: Int64 { SystemMemory.availableBytes }
}

/// Handle returned by `ModelManager.acquire`. Holds the loaded provider and a
/// one-shot `release` closure the caller must invoke when done. Release
/// decrements the entry's ref count; when the count hits zero the manager
/// schedules a per-entry idle-unload task that fires after `idleTimeout`.
struct ModelLease: Sendable {
    let provider: any ModelProvider
    let release: @Sendable () async -> Void
}

/// Owns the set of currently-loaded providers and enforces a memory budget +
/// idle-eviction policy. Connection liveness is tracked via per-entry ref
/// counting: every `acquire` increments the count, every `release` decrements
/// it. On count→0 the entry schedules an idle-unload task; that task is
/// cancelled if a new `acquire` re-claims the entry before it fires.
actor ModelManager {
    struct Settings: Sendable {
        /// Minimum bytes of OS-available memory to maintain. The manager
        /// evicts loaded models whenever loading more or background drift
        /// would push available memory below this floor — replaces the
        /// older static "budget" with a dynamic check against whatever the
        /// system actually has free right now.
        var keepFreeBytes: Int64
        /// When non-nil, the manager schedules a per-model unload task this
        /// many seconds after the last lease is released. When nil, models
        /// stay resident until the keep-free floor forces eviction.
        var idleTimeout: TimeInterval?
        var maxConcurrent: Int?
        /// Source of "available memory" readings. Defaults to live Mach VM
        /// stats; tests inject a stub.
        var stats: any MemoryStats = SystemMemoryStats()

        static let `default` = Settings(
            keepFreeBytes: 1_073_741_824,  // 1 GB
            idleTimeout: nil,
            maxConcurrent: nil)
    }

    /// Manager-owned bookkeeping per loaded model. `activeRequests` is the ref
    /// count; `lastReleasedAt` is set whenever the count drops to zero (the
    /// moment the idle clock starts). `idleUnloadTask` is the per-entry
    /// scheduled unload, cancelled whenever the count goes back above zero.
    private struct Entry {
        var provider: any ModelProvider
        var sizeHint: Int64
        var lastReleasedAt: Date
        var activeRequests: Int
        var idleUnloadTask: Task<Void, Never>?
    }

    private let registry: ModelRegistry
    private let factory: any ProviderFactory
    private let logger: Logger
    private var loaded: [String: Entry] = [:]
    /// In-flight model loads, keyed by model name. When a second `acquire`
    /// arrives for a name that's currently loading, we await this task
    /// instead of kicking off a parallel `factory.make(...)` — model loads
    /// pin gigabytes of GPU memory, so duplicate loads are catastrophic.
    /// The task is cleared once it completes (success or failure).
    private var inflightLoads: [String: Task<Void, Error>] = [:]
    private var settings: Settings
    private var memoryMonitor: MemoryMonitor?
    /// Background poller that evicts an LRU model when the system's
    /// available memory falls below `settings.keepFreeBytes`. macOS's
    /// dispatch pressure events fire too late (compressor is already
    /// running by `.warning`); this catches drift between load events.
    private var floorWatcher: Task<Void, Never>?

    init(
        registry: ModelRegistry,
        factory: any ProviderFactory,
        settings: Settings = .default,
        logger: Logger = Logger(label: "smolx.manager")
    ) {
        self.registry = registry
        self.factory = factory
        self.settings = settings
        self.logger = logger
    }

    // MARK: - Lifecycle

    func start() {
        let monitor = MemoryMonitor { [weak self] level in
            Task { await self?.handlePressure(level) }
        }
        monitor.start()
        memoryMonitor = monitor

        floorWatcher = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                if Task.isCancelled { return }
                await self?.evictIfBelowFloor()
            }
        }
    }

    func stop() async {
        memoryMonitor?.stop()
        memoryMonitor = nil
        floorWatcher?.cancel()
        floorWatcher = nil
        for (_, entry) in loaded {
            entry.idleUnloadTask?.cancel()
            await entry.provider.unload()
        }
        loaded.removeAll()
    }

    /// Single eviction step when available memory drops below the floor.
    /// Called from the periodic watcher; returning quickly is more important
    /// than recovering exactly to the floor in one tick — the next tick
    /// will evict again if we're still under. `internal` rather than
    /// `private` so tests can drive it deterministically.
    func evictIfBelowFloor() async {
        guard !loaded.isEmpty else { return }
        let available = settings.stats.availableBytes
        if available < settings.keepFreeBytes {
            await evictOneLRU(reason: "keep-free-poll")
        }
    }

    // MARK: - Acquisition

    /// Returns a `ModelLease` for the named model, loading + evicting LRUs as
    /// necessary. The caller must invoke `lease.release()` exactly once when
    /// done; release decrements the ref count and starts the idle-unload
    /// timer when the count hits zero.
    ///
    /// Concurrent `acquire` calls for the same not-yet-loaded model are
    /// deduplicated: the first call kicks off a load Task stored in
    /// `inflightLoads`, subsequent calls await the same Task. Without this,
    /// two simultaneous requests (e.g. opencode firing a streaming + a
    /// non-streaming completion at once) each call `factory.make(...)`,
    /// allocating two copies of the same model in GPU memory.
    func acquire(_ name: String) async throws -> ModelLease {
        if let entry = loaded[name] {
            bumpRefCount(name)
            return makeLease(name: name, provider: entry.provider)
        }
        // Join an in-flight load if there is one. When the task resolves we
        // re-enter the actor; loaded[name] will have been populated by the
        // initiator's body (performLoad runs on the actor's executor).
        if let inflight = inflightLoads[name] {
            try await inflight.value
        } else {
            let task = Task<Void, Error> { [self] in
                try await self.performLoad(name: name)
            }
            inflightLoads[name] = task
            do {
                try await task.value
            } catch {
                inflightLoads[name] = nil
                throw error
            }
            inflightLoads[name] = nil
        }
        guard let entry = loaded[name] else {
            // Load reported success but bookkeeping missing — defensive only.
            throw ProviderError.loadFailed("load completed but entry was not registered for \(name)")
        }
        bumpRefCount(name)
        return makeLease(name: name, provider: entry.provider)
    }

    /// The actual load body, factored out so the initiator's Task runs it on
    /// the actor's executor (and loaded[name] is set before any joiner wakes).
    private func performLoad(name: String) async throws {
        guard let descriptor = try registry.find(name) else {
            throw ProviderError.modelNotFound(name)
        }
        await makeRoom(for: descriptor.diskSizeBytes)
        if let cap = settings.maxConcurrent, loaded.count >= cap {
            await evictOneLRU(reason: "max-concurrent")
        }
        logger.info("Loading model \(descriptor.name)")
        let provider = try await factory.make(descriptor)
        // Read the provider's actual resident bytes after load — MLXProvider
        // updates it to observed `MLX.Memory.activeMemory` delta, which is
        // closer to truth than the on-disk quantized weight size.
        let actualBytes = await provider.residentBytes
        loaded[name] = Entry(
            provider: provider,
            sizeHint: actualBytes,
            lastReleasedAt: Date(),
            activeRequests: 0,
            idleUnloadTask: nil)
    }

    func currentlyLoaded() -> [String] {
        Array(loaded.keys).sorted()
    }

    func currentResidentBytes() -> Int64 {
        loaded.values.reduce(0) { $0 + $1.sizeHint }
    }

    func updateSettings(_ new: Settings) {
        settings = new
    }

    // MARK: - Ref counting

    private func bumpRefCount(_ name: String) {
        guard var entry = loaded[name] else { return }
        entry.idleUnloadTask?.cancel()
        entry.idleUnloadTask = nil
        entry.activeRequests += 1
        loaded[name] = entry
    }

    private func makeLease(name: String, provider: any ModelProvider) -> ModelLease {
        ModelLease(provider: provider) { [weak self] in
            await self?.releaseLease(name: name)
        }
    }

    private func releaseLease(name: String) {
        guard var entry = loaded[name] else { return }
        assert(entry.activeRequests > 0, "release without matching acquire for \(name)")
        entry.activeRequests = max(0, entry.activeRequests - 1)
        if entry.activeRequests == 0 {
            // `lastReleasedAt` updates whether or not the idle timer is
            // enabled — `lruKey()` reads it to pick eviction victims under
            // memory pressure even in lazy mode. Only the timer is gated.
            entry.lastReleasedAt = Date()
            entry.idleUnloadTask?.cancel()
            if let timeout = settings.idleTimeout {
                entry.idleUnloadTask = scheduleIdleUnload(name: name, timeout: timeout)
            } else {
                entry.idleUnloadTask = nil
            }
        }
        loaded[name] = entry
    }

    private func scheduleIdleUnload(name: String, timeout: TimeInterval) -> Task<Void, Never> {
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(timeout))
            if Task.isCancelled { return }
            await self?.idleUnloadIfStillIdle(name: name)
        }
    }

    private func idleUnloadIfStillIdle(name: String) async {
        guard let entry = loaded[name], entry.activeRequests == 0 else { return }
        loaded.removeValue(forKey: name)
        logger.info("Idle-evicting \(name)")
        await entry.provider.unload()
    }

    // MARK: - Eviction

    private func makeRoom(for needed: Int64) async {
        // Evict while loading `needed` more bytes would push OS-available
        // memory below the configured floor. `availableBytes` already
        // accounts for every loaded model — they're in the in-use total —
        // so the check is just `available - needed < keepFreeBytes`. After
        // each eviction we re-read `availableBytes` because the unloaded
        // model's GPU buffers return to the OS (via MLX.Memory.clearCache in
        // MLXProvider.unload).
        while settings.stats.availableBytes - needed < settings.keepFreeBytes,
            !loaded.isEmpty
        {
            await evictOneLRU(reason: "keep-free")
        }
    }

    private func evictOneLRU(reason: String) async {
        guard let victim = lruKey() else { return }
        guard let entry = loaded.removeValue(forKey: victim) else { return }
        if entry.activeRequests > 0 {
            // Last-resort eviction: every loaded model is in use but we need
            // memory for a new load. The provider's `unload()` drops the
            // container; in-flight `generate()` calls observe the loss via
            // their existing cancellation plumbing and error out cleanly.
            logger.warning(
                "Evicting active model \(victim) (reason: \(reason)) — in-flight requests will be cancelled"
            )
        } else {
            logger.info("Evicting \(victim) (reason: \(reason))")
        }
        entry.idleUnloadTask?.cancel()
        await entry.provider.unload()
    }

    /// Pick the LRU eviction candidate, preferring entries with no active
    /// requests so we don't interrupt streams unless we have no other choice.
    private func lruKey() -> String? {
        var idleKey: String?
        var idleOldest: Date = .distantFuture
        var activeKey: String?
        var activeOldest: Date = .distantFuture
        for (name, entry) in loaded {
            if entry.activeRequests == 0 {
                if entry.lastReleasedAt < idleOldest {
                    idleOldest = entry.lastReleasedAt
                    idleKey = name
                }
            } else {
                if entry.lastReleasedAt < activeOldest {
                    activeOldest = entry.lastReleasedAt
                    activeKey = name
                }
            }
        }
        return idleKey ?? activeKey
    }

    private func handlePressure(_ level: MemoryMonitor.Level) async {
        switch level {
        case .normal: return
        case .warning:
            logger.warning("Memory pressure warning — evicting LRU")
            await evictOneLRU(reason: "pressure-warning")
        case .critical:
            logger.error("Memory pressure critical — evicting all")
            let entries = Array(loaded.values)
            loaded.removeAll()
            for entry in entries {
                entry.idleUnloadTask?.cancel()
                await entry.provider.unload()
            }
        }
    }
}
