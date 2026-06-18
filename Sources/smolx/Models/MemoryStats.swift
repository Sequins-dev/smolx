/// Source of "currently available system memory" readings. Pulled into a
/// protocol so tests can stub the value rather than depending on whatever
/// the host happens to have free at test time.
protocol MemoryStats: Sendable {
    var availableBytes: Int64 { get }
}

struct SystemMemoryStats: MemoryStats {
    var availableBytes: Int64 { SystemMemory.availableBytes }
}
