import Darwin
import Foundation

enum Terminal {
    static let hideCursor = "\u{1B}[?25l"
    static let showCursor = "\u{1B}[?25h"
    static let eraseLine = "\u{1B}[2K"

    static func moveUp(_ n: Int) -> String { n > 0 ? "\u{1B}[\(n)A" : "" }
    static func toColumn(_ n: Int) -> String { "\u{1B}[\(n)G" }
    static func moveUpAndClear(_ n: Int) -> String { "\u{1B}[\(n)F\u{1B}[J" }

    static func columns(default fallback: Int = 80) -> Int {
        var w = winsize()
        if ioctl(fileno(stderr), TIOCGWINSZ, &w) == 0, w.ws_col > 20 {
            return Int(w.ws_col)
        }
        return fallback
    }

    static func writeStderr(_ s: String) {
        FileHandle.standardError.write(Data(s.utf8))
    }
}
