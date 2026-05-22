import Foundation

/// Per-file lifecycle reported by the downloader. The tracker stores one of
/// these per file; the renderer reads via `snapshot()` and decides what to
/// draw.
enum FileProgressState: Sendable, Equatable {
    /// In the queue, not yet picked up by a pool slot.
    case waiting
    /// Pool slot has begun this file but no bytes have been transferred yet —
    /// typically the preflight HEAD request or initial TCP/TLS handshake. We
    /// surface this so the renderer can distinguish "haven't started" from
    /// "started but the network is slow to ack the first byte."
    case preparing
    /// Currently being downloaded. `resumedFrom` is non-nil if the download
    /// started from a pre-existing `.incomplete` blob.
    case downloading(bytes: Int64, total: Int64, resumedFrom: Int64?)
    /// Already in cache on entry — no transfer needed.
    case cached(bytes: Int64)
    /// Successfully completed (either after a transfer or from cache).
    case completed(bytes: Int64, resumedFrom: Int64?)
    /// Currently waiting through a retry backoff. `attempt` is 1-indexed and
    /// counts the next attempt the policy will make.
    case retrying(bytes: Int64, total: Int64, attempt: Int, delay: TimeInterval)
    case failed(reason: String)
}

/// Snapshot returned to the renderer. Sendable + value type so it can be
/// shipped across actor / Task boundaries without locking.
struct ProgressSnapshot: Sendable {
    var files: [FileEntry]
    var startedAt: Date
    var totalBytes: Int64
    var completedBytes: Int64
    var transferredSinceStart: Int64
    var mbps: Double
    var resumedFromBytes: Int64
    var allFinished: Bool

    struct FileEntry: Sendable, Equatable {
        var index: Int
        var path: String
        var state: FileProgressState
    }
}

/// Shared state between downloader tasks (writers) and the renderer (reader).
/// Actor isolation serialises updates and gives the renderer a consistent
/// snapshot per frame.
actor ProgressTracker {
    private var files: [Int: ProgressSnapshot.FileEntry] = [:]
    private var totals: [Int: Int64] = [:]
    private var startedAt: Date = .distantPast
    /// Rolling window of (timestamp, totalBytesTransferred-since-start) samples.
    /// Used to compute MB/s over the last few seconds rather than the whole
    /// run, so the figure reflects current bandwidth, not average since pull
    /// began.
    private var samples: [(Date, Int64)] = []
    private let windowSeconds: TimeInterval
    /// Sum of bytes that were already on disk at start across all files. We
    /// subtract this from "transferred so far" when computing MB/s so the
    /// rate isn't inflated by resumed bytes that didn't actually move over
    /// the network.
    private var resumedBaseline: Int64 = 0
    /// Bytes transferred from the network since the tracker started. Distinct
    /// from completedBytes (which counts cached + resumed bytes the user
    /// cares about for "how much of the model do I have").
    private var transferredSinceStart: Int64 = 0

    init(windowSeconds: TimeInterval = 3.0) {
        self.windowSeconds = windowSeconds
    }

    // MARK: - Setup

    /// Register a file before the pool starts. `resumedFrom` is the size of
    /// any pre-existing `.incomplete` blob on disk so the bar opens at the
    /// resumed offset instead of zero.
    func register(index: Int, path: String, total: Int64, resumedFrom: Int64) {
        totals[index] = total
        resumedBaseline += resumedFrom
        let initial: FileProgressState
        if resumedFrom > 0 {
            initial = .downloading(bytes: resumedFrom, total: total, resumedFrom: resumedFrom)
        } else {
            initial = .waiting
        }
        files[index] = .init(index: index, path: path, state: initial)
    }

    func start() {
        startedAt = Date()
        samples = [(startedAt, 0)]
    }

    // MARK: - Update API used by downloader tasks

    /// Mark a file as picked up by a pool slot but not yet receiving bytes.
    /// Lets the UI distinguish "stuck before first byte" from "queued".
    func preparing(index: Int) {
        guard let entry = files[index] else { return }
        // Don't transition out of downloading/cached/completed/failed back to
        // preparing — that would happen if `preparing` is called after a
        // successful retry-after-progress.
        switch entry.state {
        case .waiting, .retrying:
            files[index] = .init(index: index, path: entry.path, state: .preparing)
        default:
            break
        }
    }

    func update(index: Int, bytes: Int64) {
        guard let entry = files[index], let total = totals[index] else { return }
        let resumedFrom: Int64? = {
            if case .downloading(_, _, let r) = entry.state { return r }
            return nil
        }()
        files[index] = .init(
            index: index, path: entry.path,
            state: .downloading(bytes: bytes, total: total, resumedFrom: resumedFrom))
        recordTransferSample(forFile: index, newBytes: bytes)
    }

    /// Mark a file as currently waiting on a retry backoff so the renderer can
    /// surface "retrying in 1s" instead of silently stalling.
    func retrying(index: Int, attempt: Int, delay: TimeInterval) {
        guard let entry = files[index], let total = totals[index] else { return }
        let bytes = entry.state.currentBytes
        files[index] = .init(
            index: index, path: entry.path,
            state: .retrying(bytes: bytes, total: total, attempt: attempt, delay: delay))
    }

    func completed(index: Int, finalBytes: Int64) {
        guard let entry = files[index] else { return }
        let resumed: Int64? = {
            switch entry.state {
            case .downloading(_, _, let r): return r
            case .retrying: return nil
            default: return nil
            }
        }()
        files[index] = .init(
            index: index, path: entry.path,
            state: .completed(bytes: finalBytes, resumedFrom: resumed))
    }

    func cached(index: Int, bytes: Int64) {
        guard let entry = files[index] else { return }
        files[index] = .init(
            index: index, path: entry.path,
            state: .cached(bytes: bytes))
    }

    func failed(index: Int, reason: String) {
        guard let entry = files[index] else { return }
        files[index] = .init(
            index: index, path: entry.path, state: .failed(reason: reason))
    }

    // MARK: - Read API used by the renderer

    func snapshot() -> ProgressSnapshot {
        let entries = files.values.sorted { $0.index < $1.index }
        let totalBytes = totals.values.reduce(0, +)
        let completedBytes = entries.reduce(Int64(0)) { $0 + $1.state.currentBytes }
        let allFinished = !files.isEmpty && entries.allSatisfy { $0.state.isTerminal }
        return ProgressSnapshot(
            files: entries,
            startedAt: startedAt,
            totalBytes: totalBytes,
            completedBytes: completedBytes,
            transferredSinceStart: transferredSinceStart,
            mbps: currentMBps(),
            resumedFromBytes: resumedBaseline,
            allFinished: allFinished)
    }

    // MARK: - Internals

    /// Tracks how many bytes a particular file has reported so we can compute
    /// the delta into `transferredSinceStart`. Without this, every update for
    /// a single file would double-count its absolute byte count.
    private var lastReportedBytes: [Int: Int64] = [:]

    private func recordTransferSample(forFile index: Int, newBytes: Int64) {
        let previous = lastReportedBytes[index] ?? 0
        let delta = max(0, newBytes - previous)
        lastReportedBytes[index] = newBytes
        transferredSinceStart += delta

        let now = Date()
        samples.append((now, transferredSinceStart))
        // Drop samples older than the window.
        let cutoff = now.addingTimeInterval(-windowSeconds)
        while samples.count > 1, samples[0].0 < cutoff {
            samples.removeFirst()
        }
    }

    private func currentMBps() -> Double {
        guard let first = samples.first, let last = samples.last, samples.count > 1 else {
            return 0
        }
        let elapsed = last.0.timeIntervalSince(first.0)
        guard elapsed > 0.05 else { return 0 }
        let bytes = Double(last.1 - first.1)
        return bytes / elapsed / 1_048_576
    }
}

extension FileProgressState {
    /// "How many bytes of this file are on disk now," used both for overall
    /// progress and for the per-file bar.
    var currentBytes: Int64 {
        switch self {
        case .waiting, .preparing: return 0
        case .downloading(let b, _, _): return b
        case .retrying(let b, _, _, _): return b
        case .cached(let b): return b
        case .completed(let b, _): return b
        case .failed: return 0
        }
    }

    var isTerminal: Bool {
        switch self {
        case .cached, .completed, .failed: return true
        default: return false
        }
    }
}
