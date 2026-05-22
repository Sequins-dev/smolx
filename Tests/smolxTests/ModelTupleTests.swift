import Testing
import Foundation
@testable import smolx

@Suite("ModelTuple")
struct ModelTupleTests {

    @Test func nothingConfiguredFallsBackToFirstInstalled() {
        let t = ModelTuple.resolve(
            config: UserConfig(),
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "X", fast: "X", small: "X"))
    }

    @Test func smartOnlyCascadesDownToBothLowerTiers() {
        // The classic "I just want one model" path.
        let t = ModelTuple.resolve(
            config: UserConfig(smart: "L"),
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "L", fast: "L", small: "L"))
    }

    @Test func smartAndFastSet_smallInheritsFast() {
        let t = ModelTuple.resolve(
            config: UserConfig(smart: "L", fast: "M"),
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "L", fast: "M", small: "M"))
    }

    @Test func allThreeSetUsedVerbatim() {
        let t = ModelTuple.resolve(
            config: UserConfig(smart: "L", fast: "M", small: "S"),
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "L", fast: "M", small: "S"))
    }

    @Test func onlySmallSet_upperTiersFallBackToFirstInstalled() {
        // The "I have a tiny model just for background tasks, default
        // everywhere else" case. Cascade only goes downward — so missing
        // smart falls back to first-installed, NOT to small.
        let t = ModelTuple.resolve(
            config: UserConfig(small: "S"),
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "X", fast: "X", small: "S"))
    }

    @Test func onlyFastSet_smartFallsBackToFirstInstalled() {
        let t = ModelTuple.resolve(
            config: UserConfig(fast: "M"),
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "X", fast: "M", small: "M"))
    }

    @Test func cliOverridesWinOverPersistedConfig() {
        // User has config: smart=L, fast=M, small=S
        // CLI overrides:   --smart=A         (fast/small unchanged)
        let t = ModelTuple.resolve(
            config: UserConfig(smart: "L", fast: "M", small: "S"),
            smartOverride: "A",
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "A", fast: "M", small: "S"))
    }

    @Test func modelSugarFillsNilTiersButLosesToExplicitOverrides() {
        // --model X is sugar — it only fills slots that nothing else
        // (config or per-tier override) already covers.
        let t = ModelTuple.resolve(
            config: UserConfig(smart: "L"),
            fastOverride: "F",
            modelSugar: "SUGAR",
            firstInstalled: "X")
        // smart from config (L), fast from override (F), small from sugar
        // (SUGAR — not yet filled by anything else after step 3).
        #expect(t == ModelTuple(smart: "L", fast: "F", small: "SUGAR"))
    }

    @Test func modelSugarOnlyAppliesWhenSlotsAreNil() {
        // All three slots are explicitly configured — --model X should be
        // a no-op for all of them.
        let t = ModelTuple.resolve(
            config: UserConfig(smart: "L", fast: "M", small: "S"),
            modelSugar: "SUGAR",
            firstInstalled: "X")
        #expect(t == ModelTuple(smart: "L", fast: "M", small: "S"))
    }

    @Test func noConfigAndNoFirstInstalledReturnsNil() {
        // Nothing to work with anywhere → resolve fails. Callers turn this
        // into a "no model available" error at the CLI boundary.
        let t = ModelTuple.resolve(config: UserConfig())
        #expect(t == nil)
    }

    @Test func singleConvenienceInitFillsAllThree() {
        let t = ModelTuple(single: "only")
        #expect(t.smart == "only")
        #expect(t.fast == "only")
        #expect(t.small == "only")
    }
}
