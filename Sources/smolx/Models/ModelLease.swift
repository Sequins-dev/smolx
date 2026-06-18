/// Handle returned by `ModelManager.acquire`. Holds the loaded provider and a
/// one-shot `release` closure the caller must invoke when done. Release
/// decrements the entry's ref count; when the count hits zero the manager
/// schedules a per-entry idle-unload task that fires after `idleTimeout`.
struct ModelLease: Sendable {
    let provider: any ModelProvider
    let release: @Sendable () async -> Void
}
