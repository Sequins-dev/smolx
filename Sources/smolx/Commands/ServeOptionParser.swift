import Foundation

enum ServeOptionParser {
    /// Parses values like "32GB", "512MB", "2147483648".
    static func parseBytes(_ s: String) -> Int64? {
        let trimmed = s.trimmingCharacters(in: .whitespaces).uppercased()
        let suffixes: [(String, Int64)] = [
            ("GB", 1_073_741_824), ("G", 1_073_741_824),
            ("MB", 1_048_576), ("M", 1_048_576),
            ("KB", 1024), ("K", 1024),
            ("B", 1),
        ]
        for (suffix, multiplier) in suffixes where trimmed.hasSuffix(suffix) {
            let number = trimmed.dropLast(suffix.count)
            if let value = Double(number) {
                return Int64(value * Double(multiplier))
            }
            return nil
        }
        return Int64(trimmed)
    }

    /// Parses durations like "10m", "1h", "30s". Returns seconds.
    static func parseDuration(_ s: String) -> TimeInterval? {
        let trimmed = s.trimmingCharacters(in: .whitespaces).lowercased()
        let suffixes: [(String, TimeInterval)] = [
            ("h", 3600), ("m", 60), ("s", 1),
        ]
        for (suffix, multiplier) in suffixes where trimmed.hasSuffix(suffix) {
            if let value = Double(trimmed.dropLast(suffix.count)) {
                return value * multiplier
            }
            return nil
        }
        return Double(trimmed)
    }
}
