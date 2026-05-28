import Darwin
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

    /// Bytes remaining before we hit the machine's physical RAM ceiling, from
    /// this process's perspective. Uses `task_vm_info.phys_footprint` rather
    /// than `vm_statistics64` page counts because on Apple Silicon, Metal/GPU
    /// allocations go through IOKit and are invisible to the VM page counters —
    /// causing the floor-watcher to never see real pressure from MLX. The Mach
    /// `phys_footprint` field is what Activity Monitor uses for the Memory column
    /// on M-series Macs, and it correctly includes unified-memory GPU buffers.
    /// Result is `physicalBytes − footprint`; ignoring other processes is an
    /// acceptable trade-off for a dedicated model server. Falls back to
    /// `physicalBytes / 4` if the Mach call fails.
    static var availableBytes: Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: natural_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return physicalBytes / 4 }
        return max(0, physicalBytes - Int64(info.phys_footprint))
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
