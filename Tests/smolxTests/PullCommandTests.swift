import Foundation
import HuggingFace
import Testing

@testable import smolx

@Suite("PullCommand")
struct PullCommandTests {

    // MARK: - Decision helper

    @Test func emptyResultsBecomeEmptyDecision() {
        let d = PullCommand.decide(input: "qwen3.6", results: [])
        #expect(d == .empty)
    }

    @Test func exactMatchWinsOverPickerEvenWithMultipleResults() {
        // When one of the results equals the input exactly, we should
        // pull it directly without bothering the user with a picker —
        // that's the "type the full id you already know" path.
        let d = PullCommand.decide(
            input: "mlx-community/Qwen3.6-27B-4bit",
            results: [
                "mlx-community/Qwen3.6-7B-4bit",
                "mlx-community/Qwen3.6-27B-4bit",
                "mlx-community/Qwen3.6-27B-mxfp4",
            ])
        #expect(d == .exact(repoId: "mlx-community/Qwen3.6-27B-4bit"))
    }

    @Test func exactMatchIsCaseInsensitive() {
        // Typing `mlx-community/qwen3.6-27b-4bit` against the canonical
        // mixed-case repo id should still auto-pull. HF repo URLs are
        // case-sensitive but mortal users are not.
        let d = PullCommand.decide(
            input: "mlx-community/qwen3.6-27b-4bit",
            results: ["mlx-community/Qwen3.6-27B-4bit"])
        #expect(d == .exact(repoId: "mlx-community/Qwen3.6-27B-4bit"))
    }

    @Test func partialQueryFallsThroughToPicker() {
        let results = [
            "mlx-community/Qwen3.6-27B-4bit",
            "mlx-community/Qwen3.6-7B-4bit",
        ]
        let d = PullCommand.decide(input: "Qwen3.6", results: results)
        #expect(d == .choose(results))
    }

    @Test func decisionPreservesHFRankingOrder() {
        // We trust HF's relevance sort; the picker must show rows in
        // exactly the order HF returned them, not alphabetically.
        let ordered = ["b/foo", "a/bar", "c/baz"]
        if case .choose(let xs) = PullCommand.decide(input: "x", results: ordered) {
            #expect(xs == ordered)
        } else {
            Issue.record("expected .choose, got something else")
        }
    }

    // MARK: - formatDownloads

    @Test func downloadsBelowAThousandShowAsRawInteger() {
        #expect(PullCommand.formatDownloads(0) == "0")
        #expect(PullCommand.formatDownloads(1) == "1")
        #expect(PullCommand.formatDownloads(999) == "999")
    }

    @Test func downloadsInThousandsShowOneDecimalUntil10k() {
        #expect(PullCommand.formatDownloads(1_000) == "1.0k")
        #expect(PullCommand.formatDownloads(1_234) == "1.2k")
        #expect(PullCommand.formatDownloads(9_999) == "10.0k")
        // 10k+ rounds to integer kilos — keeps the column narrow.
        #expect(PullCommand.formatDownloads(10_000) == "10k")
        #expect(PullCommand.formatDownloads(12_345) == "12k")
        #expect(PullCommand.formatDownloads(999_999) == "999k")
    }

    @Test func downloadsInMillionsShowOneDecimal() {
        #expect(PullCommand.formatDownloads(1_000_000) == "1.0M")
        #expect(PullCommand.formatDownloads(1_234_567) == "1.2M")
        #expect(PullCommand.formatDownloads(12_500_000) == "12.5M")
    }

    // MARK: - formatBytesShort

    @Test func bytesShortPicksCorrectUnit() {
        #expect(PullCommand.formatBytesShort(0) == "0 B")
        #expect(PullCommand.formatBytesShort(512) == "512 B")
        #expect(PullCommand.formatBytesShort(1024) == "1.0 KB")
        #expect(PullCommand.formatBytesShort(15 * 1024) == "15 KB")
        #expect(PullCommand.formatBytesShort(1024 * 1024) == "1.0 MB")
        #expect(PullCommand.formatBytesShort(412 * 1024 * 1024) == "412 MB")
        #expect(PullCommand.formatBytesShort(Int64(1.5 * 1024 * 1024 * 1024)) == "1.5 GB")
        #expect(PullCommand.formatBytesShort(16 * 1024 * 1024 * 1024) == "16 GB")
    }

    // MARK: - formatRelative

    /// Helper: an arbitrary fixed "now" so the relative-time tests are
    /// reproducible regardless of when the suite runs.
    private static let pinnedNow = Date(timeIntervalSince1970: 1_800_000_000)

    @Test func formatRelativeRendersDaysAgo() {
        let threeDaysAgo = Self.pinnedNow.addingTimeInterval(-3 * 86_400)
        let s = PullCommand.formatRelative(threeDaysAgo, now: Self.pinnedNow)
        // The exact rendering depends on Foundation's `.named` style.
        // We just need a recognisably-human result.
        #expect(s.contains("days"))
        #expect(s.contains("ago"))
    }

    @Test func formatRelativeRendersWeeksAgo() {
        let twoWeeksAgo = Self.pinnedNow.addingTimeInterval(-14 * 86_400)
        let s = PullCommand.formatRelative(twoWeeksAgo, now: Self.pinnedNow)
        // `unitsStyle = .full` lands on "weeks" once we're > a week out.
        #expect(s.contains("weeks") || s.contains("week"))
        #expect(s.contains("ago"))
    }

    @Test func formatRelativeNamedDateTimeStyleUsesYesterday() {
        // With `.named`, 1-day-ago renders as "yesterday" rather than
        // "1 day ago". That's the whole reason we set `.named` — the
        // numeric form looks awkward in the metadata row.
        let oneDayAgo = Self.pinnedNow.addingTimeInterval(-24 * 3600)
        let s = PullCommand.formatRelative(oneDayAgo, now: Self.pinnedNow)
        #expect(s.lowercased().contains("yesterday"))
    }

    // MARK: - formatRow size-slot collapsing

    @Test func formatRowOmitsSizeWhenNil() {
        // Compact mode passes sizeBytes: nil so the column should
        // collapse — we want " · "-separator joining of only the
        // present parts, not "60k downloads · — · yesterday" with a
        // gap dash.
        let now = Self.pinnedNow
        let lastModified = now.addingTimeInterval(-24 * 3600)
        let model = makeModel(
            id: "mlx-community/foo", downloads: 60_000,
            usedStorage: nil, lastModified: lastModified)
        let s = PullCommand.formatRow(for: model, sizeBytes: nil, now: now)
        // Two parts, one separator.
        #expect(s == "60k downloads · yesterday")
    }

    @Test func formatRowIncludesSizeWhenProvided() {
        let now = Self.pinnedNow
        let lastModified = now.addingTimeInterval(-3 * 86_400)
        let model = makeModel(
            id: "mlx-community/foo", downloads: 60_000,
            usedStorage: 16 * 1024 * 1024 * 1024,
            lastModified: lastModified)
        let s = PullCommand.formatRow(
            for: model, sizeBytes: 16 * 1024 * 1024 * 1024, now: now)
        // Three parts, two separators. Order: size · downloads · date.
        // Size leads — that's what users filter on when picking a quant.
        #expect(s.hasPrefix("16 GB · 60k downloads · "))
        #expect(s.contains("days ago"))
    }

    /// Construct a `HuggingFace.Model` by decoding a hand-rolled JSON
    /// fixture. We don't want a public initializer leaking out of the
    /// swift-huggingface library just for our tests, so we round-trip
    /// through the same Codable path the network response uses.
    /// `lastModified` is encoded as a Unix epoch seconds *Double* —
    /// that matches Swift's default `.deferredToDate` strategy, which
    /// is what plain `JSONDecoder()` uses. (The real network path uses
    /// a custom ISO8601 strategy on the library's own decoder; we don't
    /// need to mimic that since this fixture only exercises our
    /// `formatRow` helper, not the wire format.)
    private func makeModel(
        id: String, downloads: Int?, usedStorage: Int?, lastModified: Date
    ) -> Model {
        var fields: [String] = []
        fields.append("\"id\":\"\(id)\"")
        if let downloads { fields.append("\"downloads\":\(downloads)") }
        if let usedStorage { fields.append("\"usedStorage\":\(usedStorage)") }
        fields.append("\"lastModified\":\(lastModified.timeIntervalSinceReferenceDate)")
        let json = "{\(fields.joined(separator: ","))}"
        return try! JSONDecoder().decode(Model.self, from: Data(json.utf8))
    }
}
