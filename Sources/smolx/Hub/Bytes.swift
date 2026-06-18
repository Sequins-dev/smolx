import Foundation

enum Bytes {
    /// Full precision: `1.23 GB`, `42 B`, `512.00 KB`.
    static func format(_ bytes: Int64) -> String {
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

    /// Compact picker/summary form: `1.2 GB`, `412 MB`, `9 KB`.
    /// Single fractional digit only when the leading value is < 10.
    static func formatShort(_ bytes: Int64) -> String {
        let units: [(suffix: String, divisor: Double)] = [
            ("TB", 1024 * 1024 * 1024 * 1024),
            ("GB", 1024 * 1024 * 1024),
            ("MB", 1024 * 1024),
            ("KB", 1024),
        ]
        let v = Double(bytes)
        for (suffix, div) in units where v >= div {
            let scaled = v / div
            if scaled >= 10 {
                return "\(Int(scaled.rounded())) \(suffix)"
            }
            return String(format: "%.1f %@", scaled, suffix)
        }
        return "\(bytes) B"
    }
}
