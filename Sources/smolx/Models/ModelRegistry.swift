import Foundation

/// On-disk index of installed models. Backed by `~/.smolx/registry.json`.
/// Not an actor — operations are synchronous file I/O and are always called
/// from the main CLI flow or a `Task` that owns the I/O.
struct ModelRegistry: Sendable {
    let url: URL

    init(url: URL = Paths.registryFile) {
        self.url = url
    }

    func load() throws -> [ModelDescriptor] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder.iso8601.decode([ModelDescriptor].self, from: data)
    }

    func save(_ models: [ModelDescriptor]) throws {
        try Paths.ensureAppRoot()
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        enc.dateEncodingStrategy = .iso8601
        try enc.encode(models).write(to: url, options: .atomic)
    }

    @discardableResult
    func upsert(_ desc: ModelDescriptor) throws -> [ModelDescriptor] {
        var all = try load()
        if let idx = all.firstIndex(where: { $0.name == desc.name }) {
            all[idx] = desc
        } else {
            all.append(desc)
        }
        try save(all)
        return all
    }

    @discardableResult
    func remove(name: String) throws -> ModelDescriptor? {
        var all = try load()
        guard let idx = all.firstIndex(where: { $0.name == name || $0.repoId == name }) else {
            return nil
        }
        let removed = all.remove(at: idx)
        try save(all)
        return removed
    }

    func find(_ name: String) throws -> ModelDescriptor? {
        try load().first { $0.name == name || $0.repoId == name }
    }
}

extension JSONDecoder {
    fileprivate static let iso8601: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
