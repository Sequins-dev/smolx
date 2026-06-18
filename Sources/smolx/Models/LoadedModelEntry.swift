import Foundation

/// Manager-owned bookkeeping per loaded model.
struct LoadedModelEntry {
    var provider: any ModelProvider
    var sizeHint: Int64
    var lastReleasedAt: Date
    var activeRequests: Int
    var idleUnloadTask: Task<Void, Never>?

    var isIdle: Bool { activeRequests == 0 }

    mutating func acquire() -> Int {
        cancelIdleUnload()
        activeRequests += 1
        return activeRequests
    }

    mutating func release(scheduleIdleUnload: (() -> Task<Void, Never>)?) -> Int {
        assert(activeRequests > 0, "release without matching acquire")
        activeRequests = max(0, activeRequests - 1)
        if activeRequests == 0 {
            lastReleasedAt = Date()
            cancelIdleUnload()
            idleUnloadTask = scheduleIdleUnload?()
        }
        return activeRequests
    }

    mutating func cancelIdleUnload() {
        idleUnloadTask?.cancel()
        idleUnloadTask = nil
    }

    static func lruKey(in entries: [String: LoadedModelEntry]) -> String? {
        var idleKey: String?
        var idleOldest: Date = .distantFuture
        var activeKey: String?
        var activeOldest: Date = .distantFuture

        for (name, entry) in entries {
            if entry.isIdle {
                if entry.lastReleasedAt < idleOldest {
                    idleOldest = entry.lastReleasedAt
                    idleKey = name
                }
            } else if entry.lastReleasedAt < activeOldest {
                activeOldest = entry.lastReleasedAt
                activeKey = name
            }
        }
        return idleKey ?? activeKey
    }
}
