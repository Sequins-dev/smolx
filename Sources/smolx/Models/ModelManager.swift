import Foundation
import Logging

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
            idleTimeout: 30,
            maxConcurrent: nil)
    }

    private let registry: ModelRegistry
    private let factory: any ProviderFactory
    private let logger: Logger
    private var loaded: [String: LoadedModelEntry] = [:]
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
        loaded[name] = LoadedModelEntry(
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
        let active = entry.acquire()
        loaded[name] = entry
        logger.debug("Acquired \(name) — active=\(active)")
    }

    private func makeLease(name: String, provider: any ModelProvider) -> ModelLease {
        ModelLease(provider: provider) { [weak self] in
            await self?.releaseLease(name: name)
        }
    }

    private func releaseLease(name: String) {
        guard var entry = loaded[name] else { return }
        let idleUnload = settings.idleTimeout.map { timeout in
            { self.scheduleIdleUnload(name: name, timeout: timeout) }
        }
        let active = entry.release(scheduleIdleUnload: idleUnload)
        loaded[name] = entry
        logger.debug("Released \(name) — active=\(active)")
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
        LoadedModelEntry.lruKey(in: loaded)
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
