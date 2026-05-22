import Darwin
import Foundation

/// Renders `ProgressTracker` snapshots to stderr. In a TTY: a docker-pull-style
/// multi-line bar that animates at ~10 Hz. Outside a TTY (CI, piped output):
/// one line per file completion, no escape codes.
///
/// Caller owns lifecycle:
///   let renderer = ProgressRenderer(tracker: tracker, repoId: id, slots: 4)
///   renderer.start()
///   try await downloader.download(...)
///   await renderer.stop(success: true)
final class ProgressRenderer: @unchecked Sendable {

    private let tracker: ProgressTracker
    private let repoId: String
    private let slotCount: Int
    private let isTTY: Bool
    private var renderTask: Task<Void, Never>?
    private var loggedTerminalFiles: Set<Int> = []
    private var stoppedAt: Date?

    init(tracker: ProgressTracker, repoId: String, slots: Int) {
        self.tracker = tracker
        self.repoId = repoId
        self.slotCount = max(1, slots)
        self.isTTY = isatty(fileno(stderr)) != 0
    }

    func start() {
        if isTTY {
            // Reserve room (one summary + N slots) and hide the cursor so the
            // redraw doesn't flicker. We undo both in stop().
            writeStderr("\(repoIntroLine)\n")
            for _ in 0..<(slotCount + 1) { writeStderr("\n") }
            writeStderr(Ansi.hideCursor)
        } else {
            writeStderr("Downloading \(repoId)\n")
        }
        renderTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.tick()
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    func stop(success: Bool) async {
        renderTask?.cancel()
        renderTask = nil
        await tick()  // final paint
        stoppedAt = Date()

        if isTTY {
            writeStderr(Ansi.showCursor)
        }
        await emitFinalSummary(success: success)
    }

    // MARK: - Render loop

    private func tick() async {
        let snap = await tracker.snapshot()
        if isTTY {
            renderTTY(snap)
        } else {
            renderLines(snap)
        }
    }

    // MARK: - TTY

    private func renderTTY(_ snap: ProgressSnapshot) {
        let cols = TerminalWidth.current(default: 100)

        // Move up to the summary line. We printed (slotCount + 1) blank lines
        // plus the intro — moving up exactly (slotCount + 1) puts us on the
        // first reserved line (the summary). Critically: each frame must emit
        // EXACTLY (slotCount + 1) lines of output, no more, no less, or the
        // cursor drifts and old frames stick around. Every line we write is
        // clamped to `cols - 1` characters so the terminal never wraps a line
        // onto a second row.
        var out = Ansi.moveUp(slotCount + 1) + Ansi.toColumn(1)

        out += Ansi.eraseLine + clamp(summaryLine(snap, cols: cols), to: cols) + "\n"

        let active = snap.files.filter { !$0.state.isTerminal }
            .sorted { lhs, rhs in
                let lp = lhs.state.priority
                let rp = rhs.state.priority
                if lp != rp { return lp < rp }
                return lhs.index < rhs.index
            }
        let toShow = Array(active.prefix(slotCount))

        for i in 0..<slotCount {
            out += Ansi.eraseLine
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

        writeStderr(out)
    }

    /// Clamps `s` to at most `width - 1` characters so the terminal can't
    /// auto-wrap it onto a second visual line. Counts in grapheme clusters
    /// which approximates monospace column count well enough for the ASCII
    /// + simple-Unicode (·, ▸, ✓, ░, █) content we emit. We leave one column
    /// of slack to avoid the corner case where some terminals treat writing
    /// to the last column as wrapping.
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
            ? " · resumed from \(formatBytes(snap.resumedFromBytes))"
            : ""
        return "  [\(bar(percent: pct, width: 20))] \(pct)% · "
            + "\(formatBytes(snap.completedBytes)) / \(formatBytes(snap.totalBytes))"
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
                + "\(formatBytes(bytes)) / \(formatBytes(total))"
        case .retrying(_, _, let attempt, let delay):
            icon = "  ⟳ "
            trailing = String(format: "retry %d in %.1fs", attempt, delay)
        case .cached:
            icon = "  ✓ "
            trailing = "cached"
        case .completed(let bytes, _):
            icon = "  ✓ "
            trailing = formatBytes(bytes)
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
            writeStderr(nonTTYLine(entry, totalFiles: snap.files.count) + "\n")
        }
    }

    private func nonTTYLine(_ entry: ProgressSnapshot.FileEntry, totalFiles: Int) -> String {
        let prefix = "[\(entry.index + 1)/\(totalFiles)] \(entry.path)"
        switch entry.state {
        case .cached(let bytes):
            return "\(prefix)  \(formatBytes(bytes))  cached"
        case .completed(let bytes, let resumed):
            if let r = resumed, r > 0 {
                return "\(prefix)  \(formatBytes(bytes))  done (resumed from \(formatBytes(r)))"
            }
            return "\(prefix)  \(formatBytes(bytes))  done"
        case .failed(let reason):
            return "\(prefix)  failed: \(reason)"
        default:
            return prefix
        }
    }

    private func emitFinalSummary(success: Bool) async {
        let snap = await tracker.snapshot()
        let elapsed = stoppedAt?.timeIntervalSince(snap.startedAt) ?? 0
        let elapsedString = formatDuration(elapsed)
        if success {
            writeStderr(
                "Downloaded \(snap.files.count) files (\(formatBytes(snap.totalBytes))) in \(elapsedString)\n"
            )
        } else {
            let failed = snap.files.filter {
                if case .failed = $0.state { return true } else { return false }
            }
            if failed.isEmpty {
                writeStderr("Download interrupted after \(elapsedString)\n")
            } else {
                writeStderr("Download failed after \(elapsedString) — \(failed.count) file(s) errored\n")
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

    private func formatBytes(_ bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var unit = 0
        while value >= 1024 && unit < units.count - 1 {
            value /= 1024
            unit += 1
        }
        if unit == 0 { return "\(Int(value)) \(units[unit])" }
        return String(format: "%.2f %@", value, units[unit])
    }

    private func ellipsize(_ s: String, width: Int) -> String {
        if s.count <= width { return s }
        if width <= 1 { return String(s.prefix(width)) }
        return "…" + String(s.suffix(width - 1))
    }

    private func writeStderr(_ s: String) {
        FileHandle.standardError.write(Data(s.utf8))
    }
}

// MARK: - ANSI helpers

private enum Ansi {
    static let hideCursor = "\u{1B}[?25l"
    static let showCursor = "\u{1B}[?25h"
    static let eraseLine = "\u{1B}[2K"
    static func moveUp(_ n: Int) -> String { n > 0 ? "\u{1B}[\(n)A" : "" }
    static func toColumn(_ n: Int) -> String { "\u{1B}[\(n)G" }
}

private enum TerminalWidth {
    static func current(default fallback: Int) -> Int {
        var w = winsize()
        if ioctl(fileno(stderr), TIOCGWINSZ, &w) == 0, w.ws_col > 20 {
            return Int(w.ws_col)
        }
        return fallback
    }
}

extension FileProgressState {
    /// Lower value = higher render priority (more interesting state).
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
