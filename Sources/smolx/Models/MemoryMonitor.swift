import Dispatch
import Foundation

/// Listens for system-wide memory-pressure events and forwards them to a
/// handler. Lives for the lifetime of the server.
final class MemoryMonitor: @unchecked Sendable {
    enum Level: Sendable { case normal, warning, critical }

    private let source: DispatchSourceMemoryPressure
    private let queue = DispatchQueue(label: "smolx.memory-monitor")

    init(_ onLevel: @escaping @Sendable (Level) -> Void) {
        source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical], queue: queue)
        source.setEventHandler { [source] in
            let raw = source.data
            let level: Level
            if raw.contains(.critical) {
                level = .critical
            } else if raw.contains(.warning) {
                level = .warning
            } else {
                level = .normal
            }
            onLevel(level)
        }
    }

    func start() { source.resume() }
    func stop() { source.cancel() }
}

enum SystemMemory {
    static var physicalBytes: Int64 {
        Int64(ProcessInfo.processInfo.physicalMemory)
    }

    /// Parses values like "32GB", "512MB", "2147483648". Falls back to nil if
    /// the input is malformed.
    static func parse(_ s: String) -> Int64? {
        let trimmed = s.trimmingCharacters(in: .whitespaces).uppercased()
        let suffixes: [(String, Int64)] = [
            ("GB", 1_073_741_824), ("G", 1_073_741_824),
            ("MB", 1_048_576), ("M", 1_048_576),
            ("KB", 1024), ("K", 1024),
            ("B", 1),
        ]
        for (suffix, mult) in suffixes where trimmed.hasSuffix(suffix) {
            let num = trimmed.dropLast(suffix.count)
            if let v = Double(num) { return Int64(v * Double(mult)) }
        }
        return Int64(trimmed)
    }

    /// Parses durations like "10m", "1h", "30s". Returns seconds.
    static func parseDuration(_ s: String) -> TimeInterval? {
        let t = s.trimmingCharacters(in: .whitespaces).lowercased()
        let scale: [(String, TimeInterval)] = [
            ("h", 3600), ("m", 60), ("s", 1),
        ]
        for (suffix, mult) in scale where t.hasSuffix(suffix) {
            if let v = Double(t.dropLast(suffix.count)) { return v * mult }
        }
        return Double(t)
    }
}
