import Darwin
import Foundation

/// Arrow-key driven terminal picker. Caller passes a list of `Row` and
/// optionally a `fetchMore` closure (for infinite-scroll pagination);
/// gets back a `Selection` carrying the chosen row, or `nil` if aborted.
///
/// The picker puts the terminal into partial raw mode (canonical line
/// buffering + local echo off, plus `VMIN=0/VTIME=2` so `read()` returns
/// after ~200ms with whatever bytes arrived). The 200ms timeout doubles
/// as the marquee animation tick — there's only one loop, no separate
/// render task, no locks.
///
/// Layout invariant: the picker draws exactly one title line, a fixed
/// 8-line content block (4 rows × 2 lines in `.expanded`, 8 rows × 1
/// line in `.compact`), and one footer line. The 10-line height never
/// changes across redraws so the cursor-up math used to overwrite the
/// previous frame stays correct.
enum InteractivePicker {

    struct Row: Sendable, Equatable {
        /// The headline label — repo id in our use case. Marquees on
        /// the selected row when too wide; ellipsis-trims on others.
        let primary: String
        /// Metadata text shown under the primary in `.expanded` style.
        /// Ignored in `.compact`. Pre-formatted by the caller (we don't
        /// want a column-padding contract leaking into the picker).
        let secondary: String
    }

    struct Selection: Sendable {
        let index: Int
        let row: Row
    }

    /// Compact = one line per row (name only). Expanded = two lines per
    /// row (name + indented metadata). Total visible content stays at
    /// 8 lines either way; the viewport just holds fewer rows in
    /// expanded mode.
    enum Style: Sendable {
        case compact
        case expanded

        fileprivate var linesPerRow: Int {
            switch self {
            case .compact: return 1
            case .expanded: return 2
            }
        }

        fileprivate var viewportRows: Int { 8 / linesPerRow }
    }

    enum PickerError: Error, CustomStringConvertible {
        case notATTY
        var description: String {
            switch self {
            case .notATTY: return "Interactive picker requires a TTY on stdin"
            }
        }
    }

    // MARK: - Pure helpers (no termios, no I/O — unit-testable in isolation)

    /// Single-codepoint-ellipsis truncation: `"long-name-here"` →
    /// `"long-na…"` when `width=8`. Returns `s` untouched if it already
    /// fits. A width of 0 or 1 returns the empty string / just the
    /// ellipsis respectively — the picker never asks for sub-2 widths
    /// in practice but the helper is defensive.
    static func truncate(_ s: String, to width: Int) -> String {
        if width <= 0 { return "" }
        if s.count <= width { return s }
        if width == 1 { return "…" }
        return String(s.prefix(width - 1)) + "…"
    }

    /// One frame of the continuous-scroll marquee. When `s.count` is
    /// less than or equal to `width`, the string is returned padded to
    /// `width` with trailing spaces (so the row doesn't shrink and
    /// reveal stale background bytes). When it overflows, we build the
    /// scroll buffer `s + "   •   " + s` and return a `width`-wide
    /// substring starting at `offset` modulo the buffer's period.
    /// `offset` increments by one per render tick (~200ms), so the
    /// text drifts left at ~5 columns/sec.
    static func marqueeFrame(_ s: String, offset: Int, width: Int) -> String {
        if width <= 0 { return "" }
        if s.count <= width {
            // Pad to keep the row visually stable.
            return s.padding(toLength: width, withPad: " ", startingAt: 0)
        }
        let separator = "   •   "
        let period = s.count + separator.count
        let normalized = ((offset % period) + period) % period  // safe mod
        // Concatenate s+sep+s+sep so any window of `width` is in range.
        let doubled = s + separator + s + separator
        let chars = Array(doubled)
        let lower = normalized
        let upper = normalized + width
        // `doubled` is guaranteed long enough because `width < period`
        // (otherwise we would have taken the short branch above).
        return String(chars[lower..<upper])
    }

    /// Returns the new viewport top so `selected` stays in the
    /// half-open interval `[top, top + size)`. The result is also
    /// clamped so we don't scroll past the end (`top` will be at most
    /// `max(0, total - size)`). Pure — no mutation of input.
    static func viewport(selected: Int, total: Int, top: Int, size: Int) -> Int {
        if total == 0 || size <= 0 { return 0 }
        var newTop = top
        if selected < newTop { newTop = selected }
        if selected >= newTop + size { newTop = selected - size + 1 }
        let maxTop = max(0, total - size)
        return min(max(0, newTop), maxTop)
    }

    /// Best-effort terminal column count via `ioctl(TIOCGWINSZ)`. Falls
    /// back to 80 when stdout isn't a TTY or the syscall fails. Called
    /// once per render so window resizes are picked up naturally on the
    /// next frame — no SIGWINCH plumbing needed.
    static func terminalCols() -> Int {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0, ws.ws_col > 0 {
            return Int(ws.ws_col)
        }
        return 80
    }

    // MARK: - Public entry

    /// Show the picker. If `fetchMore` is non-nil it's called when the
    /// selection reaches the last loaded row; returning `nil` from
    /// `fetchMore` signals exhaustion. The closure runs on a detached
    /// `Task` so input keeps flowing during the network round-trip.
    static func pick(
        title: String,
        initialRows: [Row],
        style: Style = .compact,
        fetchMore: (@Sendable () async throws -> [Row]?)? = nil
    ) async throws -> Selection? {
        guard isatty(STDIN_FILENO) != 0 else { throw PickerError.notATTY }
        if initialRows.isEmpty { return nil }

        // Save cooked-mode termios so we can restore it on every exit
        // path (success, abort, throw, signal).
        var savedAttrs = termios()
        guard tcgetattr(STDIN_FILENO, &savedAttrs) == 0 else {
            throw PickerError.notATTY
        }
        PickerCleanup.savedAttrs = savedAttrs
        let prevSigint = signal(SIGINT, pickerSigintHandler)

        // Configure raw mode: drop ICANON+ECHO, set VMIN=0/VTIME=2 so
        // `read()` returns up to every 200ms even with no input. That
        // cadence drives the marquee.
        var raw = savedAttrs
        raw.c_lflag &= ~(tcflag_t(ICANON) | tcflag_t(ECHO))
        setVMINVTIME(&raw, vmin: 0, vtime: 2)
        guard tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0 else {
            _ = signal(SIGINT, prevSigint)
            throw PickerError.notATTY
        }

        defer {
            var saved = savedAttrs
            _ = tcsetattr(STDIN_FILENO, TCSANOW, &saved)
            writeStderr("\u{001B}[?25h")  // show cursor
            _ = signal(SIGINT, prevSigint)
            PickerCleanup.savedAttrs = nil
        }

        writeStderr("\u{001B}[?25l")  // hide cursor for the duration

        var rows = initialRows
        var selected = 0
        var top = 0
        var marqueeOffset = 0
        var loading = false
        var exhausted = (fetchMore == nil)
        var fetchTask: Task<Void, Never>?
        let box = FetchBox()

        let frameHeight = 1 /* title */ + 8 /* content */ + 1 /* footer */
        var firstDraw = true

        while true {
            // 1) Read up to 8 bytes with a 200ms timeout. Empty return
            //    is fine — that's the tick.
            var buf = [UInt8](repeating: 0, count: 8)
            let n = read(STDIN_FILENO, &buf, 8)

            // 2) Process whatever bytes arrived. Returns the new state
            //    plus a possible terminal action (confirm / abort).
            let prevSelected = selected
            var done: Selection? = nil
            var aborted = false
            var i = 0
            while i < n {
                let b = buf[i]
                switch b {
                case 0x03:  // Ctrl-C (defensive; signal handler also covers)
                    aborted = true
                    i = n  // stop processing buffer
                case 0x0a, 0x0d:  // Enter
                    done = Selection(index: selected, row: rows[selected])
                    i = n
                case UInt8(ascii: "q"), UInt8(ascii: "Q"):
                    aborted = true
                    i = n
                case UInt8(ascii: "j"):
                    if selected < rows.count - 1 { selected += 1 }
                    i += 1
                case UInt8(ascii: "k"):
                    if selected > 0 { selected -= 1 }
                    i += 1
                case 0x1b:
                    // CSI introducer. If `[<X>` follows in this same
                    // buffer, parse the arrow; otherwise it might be a
                    // standalone Esc or a delayed CSI — do a short
                    // follow-up read to disambiguate.
                    if i + 2 < n, buf[i + 1] == UInt8(ascii: "[") {
                        switch buf[i + 2] {
                        case UInt8(ascii: "A"):
                            if selected > 0 { selected -= 1 }
                        case UInt8(ascii: "B"):
                            if selected < rows.count - 1 { selected += 1 }
                        default: break
                        }
                        i += 3
                    } else {
                        // Try one more short read.
                        var more = [UInt8](repeating: 0, count: 2)
                        let m = read(STDIN_FILENO, &more, 2)
                        if m == 2 && more[0] == UInt8(ascii: "[") {
                            switch more[1] {
                            case UInt8(ascii: "A"):
                                if selected > 0 { selected -= 1 }
                            case UInt8(ascii: "B"):
                                if selected < rows.count - 1 { selected += 1 }
                            default: break
                            }
                        } else {
                            // Standalone Esc.
                            aborted = true
                        }
                        i = n
                    }
                default:
                    i += 1  // ignore everything else
                }
            }

            if let done {
                fetchTask?.cancel()
                return done
            }
            if aborted {
                fetchTask?.cancel()
                return nil
            }

            // 3) Reset the marquee whenever the selection changes; the
            //    user expects the newly-selected row to start scrolling
            //    from its leading edge.
            if selected != prevSelected { marqueeOffset = 0 }

            // 4) Drain any completed fetch result.
            if let result = await box.take() {
                switch result {
                case .some(let newRows):
                    rows.append(contentsOf: newRows)
                    if newRows.isEmpty { exhausted = true }
                case .none:
                    exhausted = true
                }
                loading = false
                fetchTask = nil
            }

            // 5) Trigger a fetch when the selection is at the last
            //    loaded row (or past it, defensively). We deliberately
            //    don't prefetch earlier — `pageSize=20` means a single
            //    fetch covers the user's next ~20 down-arrows.
            if !loading, !exhausted, let fetcher = fetchMore,
                selected >= rows.count - 1
            {
                loading = true
                let box = box
                fetchTask = Task {
                    do {
                        let next = try await fetcher()
                        await box.set(.some(next ?? []))
                        if next == nil || next!.isEmpty {
                            await box.set(.none)
                        }
                    } catch {
                        // Surfacing errors mid-picker is awkward; we
                        // treat any fetch error as "no more results"
                        // and quietly stop trying. The user can re-run
                        // pull with a more specific query.
                        await box.set(.none)
                    }
                }
            }

            // 6) Tick the marquee. Selection changes reset to 0 above,
            //    so this only advances when selection is stable.
            marqueeOffset += 1

            // 7) Adjust viewport top to keep `selected` visible.
            top = Self.viewport(
                selected: selected,
                total: rows.count,
                top: top,
                size: style.viewportRows)

            // 8) Render.
            if !firstDraw {
                writeStderr("\u{001B}[\(frameHeight)F\u{001B}[J")
            }
            firstDraw = false
            renderFrame(
                title: title,
                rows: rows,
                selected: selected,
                viewportTop: top,
                marqueeOffset: marqueeOffset,
                style: style,
                loading: loading,
                exhausted: exhausted,
                cols: Self.terminalCols())
        }
    }

    // MARK: - Render

    private static func renderFrame(
        title: String,
        rows: [Row],
        selected: Int,
        viewportTop: Int,
        marqueeOffset: Int,
        style: Style,
        loading: Bool,
        exhausted: Bool,
        cols: Int
    ) {
        writeStderr("\(title)\n")

        // Primary column width — leave room for a 2-char marker prefix
        // ("▸ " or "  ") plus a small right-edge gutter. The marker is
        // a single grapheme; the U+25B8 takes 1 column.
        let primaryWidth = max(10, cols - 2 - 2)

        for slot in 0..<style.viewportRows {
            let idx = viewportTop + slot
            if idx >= rows.count {
                writeStderr("\n")
                if style == .expanded { writeStderr("\n") }
                continue
            }
            let row = rows[idx]
            let isSelected = idx == selected
            let marker =
                isSelected
                ? "\u{001B}[1;36m▸\u{001B}[0m "
                : "  "
            let primaryText: String
            if isSelected {
                primaryText = Self.marqueeFrame(
                    row.primary, offset: marqueeOffset, width: primaryWidth)
            } else {
                primaryText = Self.truncate(row.primary, to: primaryWidth)
            }
            let line1: String
            if isSelected {
                line1 = "\(marker)\u{001B}[1m\(primaryText)\u{001B}[0m"
            } else {
                line1 = "\(marker)\(primaryText)"
            }
            writeStderr("\(line1)\n")

            if style == .expanded {
                // Metadata line under the name. Indented to align with
                // the primary text (2 spaces past the marker column).
                let metaWidth = max(10, cols - 4)
                let meta = Self.truncate(row.secondary, to: metaWidth)
                writeStderr("    \u{001B}[2m\(meta)\u{001B}[0m\n")
            }
        }

        // Footer.
        let footer = footerText(
            loading: loading, exhausted: exhausted,
            viewportTop: viewportTop, selected: selected, total: rows.count,
            viewportRows: style.viewportRows)
        writeStderr("\(footer)\n")
    }

    private static func footerText(
        loading: Bool, exhausted: Bool,
        viewportTop: Int, selected: Int, total: Int,
        viewportRows: Int
    ) -> String {
        if loading { return "\u{001B}[2m↓ more (loading…)\u{001B}[0m" }
        if exhausted && selected == total - 1 {
            return "\u{001B}[2m(end of results)\u{001B}[0m"
        }
        if !exhausted && selected >= total - 1 {
            // Just about to fetch — show the hint anyway.
            return "\u{001B}[2m↓ more\u{001B}[0m"
        }
        if viewportTop + viewportRows < total {
            return "\u{001B}[2m↓ more\u{001B}[0m"
        }
        if viewportTop > 0 {
            return "\u{001B}[2m↑ scroll up\u{001B}[0m"
        }
        return ""
    }

    // MARK: - Internal plumbing

    private static func writeStderr(_ s: String) {
        FileHandle.standardError.write(Data(s.utf8))
    }

    /// Patches `VMIN`/`VTIME` into a `termios` value. Swift surfaces
    /// `c_cc` as a 20-element homogenous tuple; we reinterpret it as a
    /// `cc_t` buffer so we can index by the POSIX `VMIN`/`VTIME`
    /// constants. Single-purpose helper — not worth a public API.
    private static func setVMINVTIME(
        _ t: UnsafeMutablePointer<termios>, vmin: cc_t, vtime: cc_t
    ) {
        withUnsafeMutablePointer(to: &t.pointee.c_cc) { ptr in
            let buf = UnsafeMutableRawPointer(ptr)
                .assumingMemoryBound(to: cc_t.self)
            buf[Int(VMIN)] = vmin
            buf[Int(VTIME)] = vtime
        }
    }
}

/// A one-slot mailbox for the background fetch task's result.
/// `.some(rows)` means new rows arrived (possibly empty); `.none` means
/// the source is exhausted. The main loop calls `take()` once per
/// iteration; if the slot was unfilled it returns nil.
private actor FetchBox {
    enum Result: Sendable {
        case some([InteractivePicker.Row])
        case none
    }
    private var result: Result?

    func set(_ r: Result) { result = r }
    func take() -> Result? {
        defer { result = nil }
        return result
    }
}

/// Holds the saved termios for the SIGINT handler to restore. Mutated
/// only by `InteractivePicker.pick` on a single Task; read by the
/// signal handler. Race-free in practice — the OS delivers signals
/// synchronously into the same process.
enum PickerCleanup {
    nonisolated(unsafe) static var savedAttrs: termios?
}

private func pickerSigintHandler(_ signal: Int32) {
    if var saved = PickerCleanup.savedAttrs {
        _ = tcsetattr(STDIN_FILENO, TCSANOW, &saved)
    }
    let showCursor = "\u{001B}[?25h"
    _ = showCursor.withCString { write(STDERR_FILENO, $0, strlen($0)) }
    Darwin.signal(SIGINT, SIG_DFL)
    raise(SIGINT)
}
