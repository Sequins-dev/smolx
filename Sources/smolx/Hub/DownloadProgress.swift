import Darwin
import Foundation

// MARK: - Protocol

/// Write-side interface the downloader uses to report progress. `DownloadProgress`
/// is the concrete implementation; the protocol lets tests substitute a lightweight
/// sink without spinning up a renderer.
protocol DownloadProgressSink: Sendable {
    func register(index: Int, name: String, totalBytes: Int64, resumedFrom: Int64) async
    func start() async
    func preparing(index: Int) async
    func update(index: Int, bytes: Int64) async  // absolute: resumeFrom + received so far
    func retrying(index: Int, attempt: Int, delay: TimeInterval) async
    func completed(index: Int) async
    func cached(index: Int, bytes: Int64) async
    func failed(index: Int, reason: String) async
}

// MARK: - FileProgressState

enum FileProgressState: Sendable, Equatable {
    case waiting
    case preparing
    case downloading(bytes: Int64, total: Int64, resumedFrom: Int64?)
    case cached(bytes: Int64)
    case completed(bytes: Int64, resumedFrom: Int64?)
    case retrying(bytes: Int64, total: Int64, attempt: Int, delay: TimeInterval)
    case failed(reason: String)
}

extension FileProgressState {
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

    fileprivate var priority: Int {
        switch self {
        case .downloading: return 0
        case .retrying: return 1
        case .preparing: return 2
        case .waiting: return 3
        case .completed, .cached, .failed: return 99
        }
    }
}

// MARK: - ProgressSnapshot

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

// MARK: - RateSampler

private struct RateSampler {
    var samples: [(Date, Int64)] = []
    var transferred: Int64 = 0
    let windowSeconds: TimeInterval

    init(windowSeconds: TimeInterval = 3.0) {
        self.windowSeconds = windowSeconds
    }

    mutating func reset(at date: Date) {
        samples = [(date, 0)]
        transferred = 0
    }

    mutating func record(delta: Int64) {
        transferred += delta
        let now = Date()
        samples.append((now, transferred))
        let cutoff = now.addingTimeInterval(-windowSeconds)
        while samples.count > 1, samples[0].0 < cutoff { samples.removeFirst() }
    }

    func currentMBps() -> Double {
        guard let first = samples.first, let last = samples.last, samples.count > 1 else {
            return 0
        }
        let elapsed = last.0.timeIntervalSince(first.0)
        guard elapsed > 0.05 else { return 0 }
        return Double(last.1 - first.1) / elapsed / 1_048_576
    }
}

// MARK: - DownloadProgress

/// Unified progress tracker and TTY renderer. Owns the per-file state machine,
/// the MB/s sampler, and the 10 Hz render loop. Callers see one object instead
/// of the old ProgressTracker + ProgressRenderer pair.
///
/// Lifecycle:
///   1. `Downloader.download` calls `register` for each file, then `start`.
///   2. Downloader pool tasks call `preparing/update/completed/cached/failed`.
///   3. PullCommand calls `stop(success:)` after download returns or throws.
actor DownloadProgress: DownloadProgressSink {

    // MARK: Tracking state

    private var files: [Int: ProgressSnapshot.FileEntry] = [:]
    private var totals: [Int: Int64] = [:]
    private var resumedFromMap: [Int: Int64] = [:]
    private var lastReportedBytes: [Int: Int64] = [:]
    private var startedAt: Date = .distantPast
    private var sampler: RateSampler
    private var resumedBaseline: Int64 = 0
    private var transferredSinceStart: Int64 = 0

    // MARK: Render state

    private let repoId: String
    private let slotCount: Int
    private let isTTY: Bool
    private var renderTask: Task<Void, Never>?
    private var loggedTerminalFiles: Set<Int> = []
    private var stoppedAt: Date?

    init(repoId: String, slots: Int, windowSeconds: TimeInterval = 3.0) {
        self.repoId = repoId
        self.slotCount = max(1, slots)
        self.isTTY = isatty(fileno(stderr)) != 0
        self.sampler = RateSampler(windowSeconds: windowSeconds)
    }

    // MARK: - DownloadProgressSink

    func register(index: Int, name: String, totalBytes: Int64, resumedFrom: Int64) {
        totals[index] = totalBytes
        resumedFromMap[index] = resumedFrom
        resumedBaseline += resumedFrom
        let initial: FileProgressState =
            resumedFrom > 0
            ? .downloading(bytes: resumedFrom, total: totalBytes, resumedFrom: resumedFrom)
            : .waiting
        files[index] = .init(index: index, path: name, state: initial)
    }

    func start() {
        startedAt = Date()
        sampler.reset(at: startedAt)

        if isTTY {
            Terminal.writeStderr("\(repoIntroLine)\n")
            for _ in 0..<(slotCount + 1) { Terminal.writeStderr("\n") }
            Terminal.writeStderr(Terminal.hideCursor)
        } else {
            Terminal.writeStderr("Downloading \(repoId)\n")
        }

        renderTask = Task {
            while !Task.isCancelled {
                await self.tick()
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    func preparing(index: Int) {
        guard let entry = files[index] else { return }
        switch entry.state {
        case .waiting, .retrying:
            files[index] = .init(index: index, path: entry.path, state: .preparing)
        default:
            break
        }
    }

    func update(index: Int, bytes: Int64) {
        guard let entry = files[index], let total = totals[index] else { return }
        let previous = lastReportedBytes[index] ?? 0
        let delta = max(0, bytes - previous)
        lastReportedBytes[index] = bytes
        let resumedFrom = resumedFromMap[index].flatMap { $0 > 0 ? $0 : nil }
        files[index] = .init(
            index: index, path: entry.path,
            state: .downloading(bytes: bytes, total: total, resumedFrom: resumedFrom))
        transferredSinceStart += delta
        sampler.record(delta: delta)
    }

    func retrying(index: Int, attempt: Int, delay: TimeInterval) {
        guard let entry = files[index], let total = totals[index] else { return }
        let bytes = entry.state.currentBytes
        files[index] = .init(
            index: index, path: entry.path,
            state: .retrying(bytes: bytes, total: total, attempt: attempt, delay: delay))
    }

    func completed(index: Int) {
        guard let entry = files[index] else { return }
        let bytes = totals[index] ?? entry.state.currentBytes
        let resumed = resumedFromMap[index].flatMap { $0 > 0 ? $0 : nil }
        files[index] = .init(
            index: index, path: entry.path,
            state: .completed(bytes: bytes, resumedFrom: resumed))
    }

    func cached(index: Int, bytes: Int64) {
        guard let entry = files[index] else { return }
        files[index] = .init(index: index, path: entry.path, state: .cached(bytes: bytes))
    }

    func failed(index: Int, reason: String) {
        guard let entry = files[index] else { return }
        files[index] = .init(index: index, path: entry.path, state: .failed(reason: reason))
    }

    // MARK: - Public

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
            mbps: sampler.currentMBps(),
            resumedFromBytes: resumedBaseline,
            allFinished: allFinished)
    }

    func stop(success: Bool) {
        renderTask?.cancel()
        renderTask = nil
        stoppedAt = Date()
        tick()
        if isTTY {
            Terminal.writeStderr(Terminal.showCursor)
        }
        emitFinalSummary(success: success)
    }

    // MARK: - Render loop

    private func tick() {
        let snap = snapshot()
        if isTTY {
            renderTTY(snap)
        } else {
            renderLines(snap)
        }
    }

    // MARK: - TTY

    private func renderTTY(_ snap: ProgressSnapshot) {
        let cols = Terminal.columns(default: 100)
        var out = Terminal.moveUp(slotCount + 1) + Terminal.toColumn(1)
        out += Terminal.eraseLine + clamp(summaryLine(snap, cols: cols), to: cols) + "\n"

        let active = snap.files.filter { !$0.state.isTerminal }
            .sorted { lhs, rhs in
                let lp = lhs.state.priority, rp = rhs.state.priority
                if lp != rp { return lp < rp }
                return lhs.index < rhs.index
            }
        let toShow = Array(active.prefix(slotCount))

        for i in 0..<slotCount {
            out += Terminal.eraseLine
            let line: String
            if i < toShow.count {
                line = slotLine(toShow[i], cols: cols)
            } else {
                let cached = snap.files.filter {
                    if case .cached = $0.state { return true } else { return false }
                }.count
                let done = snap.files.filter {
                    if case .completed = $0.state { return true } else { return false }
                }.count
                let total = snap.files.count
                line = "  · " + ((cached + done == total && total > 0) ? "all done" : "(idle)")
            }
            out += clamp(line, to: cols) + "\n"
        }
        Terminal.writeStderr(out)
    }

    private func clamp(_ s: String, to width: Int) -> String {
        let limit = max(1, width - 1)
        if s.count <= limit { return s }
        return String(s.prefix(limit))
    }

    private var repoIntroLine: String { "Downloading \(repoId)" }

    private func summaryLine(_ snap: ProgressSnapshot, cols: Int) -> String {
        let pct =
            snap.totalBytes > 0
            ? Int(Double(snap.completedBytes) / Double(snap.totalBytes) * 100)
            : 0
        let mbps = String(format: "%.1f MB/s", snap.mbps)
        let eta = etaString(snap)
        let resume =
            snap.resumedFromBytes > 0
            ? " · resumed from \(Bytes.format(snap.resumedFromBytes))"
            : ""
        return "  [\(bar(percent: pct, width: 20))] \(pct)% · "
            + "\(Bytes.format(snap.completedBytes)) / \(Bytes.format(snap.totalBytes))"
            + " · \(mbps) · ETA \(eta)\(resume)"
    }

    private func slotLine(_ entry: ProgressSnapshot.FileEntry, cols: Int) -> String {
        let icon: String
        let trailing: String
        switch entry.state {
        case .waiting:
            icon = "  · "
            trailing = "queued"
        case .preparing:
            icon = "  ◦ "
            trailing = "preparing…"
        case .downloading(let bytes, let total, let resumed):
            icon = (resumed ?? 0) > 0 ? "  ↻ " : "  ▸ "
            let pct = total > 0 ? Int(Double(bytes) / Double(total) * 100) : 0
            trailing =
                "[\(bar(percent: pct, width: 10))] \(pct)%  "
                + "\(Bytes.format(bytes)) / \(Bytes.format(total))"
        case .retrying(_, _, let attempt, let delay):
            icon = "  ⟳ "
            trailing = String(format: "retry %d in %.1fs", attempt, delay)
        case .cached:
            icon = "  ✓ "
            trailing = "cached"
        case .completed(let bytes, _):
            icon = "  ✓ "
            trailing = Bytes.format(bytes)
        case .failed(let reason):
            icon = "  ✗ "
            trailing = reason
        }
        let nameWidth = max(20, cols - trailing.count - icon.count - 4)
        let name = ellipsize(entry.path, width: nameWidth)
        let padded = name.padding(toLength: nameWidth, withPad: " ", startingAt: 0)
        return "\(icon)\(padded)  \(trailing)"
    }

    // MARK: - Non-TTY

    private func renderLines(_ snap: ProgressSnapshot) {
        for entry in snap.files where entry.state.isTerminal {
            guard !loggedTerminalFiles.contains(entry.index) else { continue }
            loggedTerminalFiles.insert(entry.index)
            Terminal.writeStderr(nonTTYLine(entry, totalFiles: snap.files.count) + "\n")
        }
    }

    private func nonTTYLine(_ entry: ProgressSnapshot.FileEntry, totalFiles: Int) -> String {
        let prefix = "[\(entry.index + 1)/\(totalFiles)] \(entry.path)"
        switch entry.state {
        case .cached(let bytes):
            return "\(prefix)  \(Bytes.format(bytes))  cached"
        case .completed(let bytes, let resumed):
            if let r = resumed, r > 0 {
                return "\(prefix)  \(Bytes.format(bytes))  done (resumed from \(Bytes.format(r)))"
            }
            return "\(prefix)  \(Bytes.format(bytes))  done"
        case .failed(let reason):
            return "\(prefix)  failed: \(reason)"
        default:
            return prefix
        }
    }

    private func emitFinalSummary(success: Bool) {
        let snap = snapshot()
        let elapsed = stoppedAt?.timeIntervalSince(snap.startedAt) ?? 0
        let elapsedString = formatDuration(elapsed)
        if success {
            Terminal.writeStderr(
                "Downloaded \(snap.files.count) files (\(Bytes.format(snap.totalBytes))) in \(elapsedString)\n"
            )
        } else {
            let failedFiles = snap.files.filter {
                if case .failed = $0.state { return true } else { return false }
            }
            if failedFiles.isEmpty {
                Terminal.writeStderr("Download interrupted after \(elapsedString)\n")
            } else {
                Terminal.writeStderr(
                    "Download failed after \(elapsedString) — \(failedFiles.count) file(s) errored\n"
                )
            }
        }
    }

    // MARK: - Formatting helpers

    private func bar(percent: Int, width: Int) -> String {
        let filled = max(0, min(width, percent * width / 100))
        return String(repeating: "█", count: filled)
            + String(repeating: "░", count: width - filled)
    }

    private func etaString(_ snap: ProgressSnapshot) -> String {
        let remaining = max(0, snap.totalBytes - snap.completedBytes)
        guard snap.mbps > 0.01 else { return "—" }
        let seconds = Double(remaining) / (snap.mbps * 1_048_576)
        return formatDuration(seconds)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "—" }
        let s = Int(seconds.rounded())
        if s >= 3600 { return String(format: "%dh%02dm", s / 3600, (s % 3600) / 60) }
        if s >= 60 { return String(format: "%dm%02ds", s / 60, s % 60) }
        return "\(s)s"
    }

    private func ellipsize(_ s: String, width: Int) -> String {
        if s.count <= width { return s }
        if width <= 1 { return String(s.prefix(width)) }
        return "…" + String(s.suffix(width - 1))
    }
}
