import Foundation
import Logging

/// Builds new provider instances on demand. Keeps `ModelManager` agnostic to
/// the concrete provider type, which makes the manager unit-testable with a
/// stub provider and lets us add new backends without touching the manager.
protocol ProviderFactory: Sendable {
    func make(_ descriptor: ModelDescriptor) async throws -> any ModelProvider
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
        var memoryBudget: Int64
        var idleTimeout: TimeInterval
        var maxConcurrent: Int?

        /// Default budget = total physical RAM. The manager won't kick out
        /// models proactively until that ceiling is hit; callers who want
        /// guaranteed headroom for the OS should override via `--memory-budget`.
        /// The idle timeout is short because the clock only starts counting
        /// after the last active request releases — a 2-minute grace window
        /// comfortably covers conversational pauses between user turns
        /// without holding GPU memory for stale sessions.
        static let `default` = Settings(
            memoryBudget: SystemMemory.physicalBytes,
            idleTimeout: 120,
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
    }

    func stop() async {
        memoryMonitor?.stop()
        memoryMonitor = nil
        for (_, entry) in loaded {
            entry.idleUnloadTask?.cancel()
            await entry.provider.unload()
        }
        loaded.removeAll()
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
        // activeRequests starts at 0; both the initiator and any joiners
        // will bumpRefCount on resume.
        loaded[name] = Entry(
            provider: provider,
            sizeHint: descriptor.diskSizeBytes,
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
            entry.lastReleasedAt = Date()
            entry.idleUnloadTask?.cancel()
            entry.idleUnloadTask = scheduleIdleUnload(name: name)
        }
        loaded[name] = entry
    }

    private func scheduleIdleUnload(name: String) -> Task<Void, Never> {
        let timeout = settings.idleTimeout
        return Task { [weak self] in
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
        while currentResidentBytes() + needed > settings.memoryBudget,
            !loaded.isEmpty
        {
            await evictOneLRU(reason: "budget")
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
