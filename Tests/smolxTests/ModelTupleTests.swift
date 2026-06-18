import Foundation
import Testing

@testable import smolx

@Suite("UserConfig.resolve")
struct ModelTupleTests {

    @Test func nothingConfiguredFallsBackToFirstInstalled() {
        let t = UserConfig().resolve(firstInstalled: "X")
        #expect(t == UserConfig(smart: "X", fast: "X", small: "X"))
    }

    @Test func smartOnlyCascadesDownToBothLowerTiers() {
        // The classic "I just want one model" path.
        let t = UserConfig(smart: "L").resolve(firstInstalled: "X")
        #expect(t == UserConfig(smart: "L", fast: "L", small: "L"))
    }

    @Test func smartAndFastSet_smallInheritsFast() {
        let t = UserConfig(smart: "L", fast: "M").resolve(firstInstalled: "X")
        #expect(t == UserConfig(smart: "L", fast: "M", small: "M"))
    }

    @Test func allThreeSetUsedVerbatim() {
        let t = UserConfig(smart: "L", fast: "M", small: "S").resolve(firstInstalled: "X")
        #expect(t == UserConfig(smart: "L", fast: "M", small: "S"))
    }

    @Test func onlySmallSet_upperTiersFallBackToFirstInstalled() {
        // The "I have a tiny model just for background tasks, default
        // everywhere else" case. Cascade only goes downward — so missing
        // smart falls back to first-installed, NOT to small.
        let t = UserConfig(small: "S").resolve(firstInstalled: "X")
        #expect(t == UserConfig(smart: "X", fast: "X", small: "S"))
    }

    @Test func onlyFastSet_smartFallsBackToFirstInstalled() {
        let t = UserConfig(fast: "M").resolve(firstInstalled: "X")
        #expect(t == UserConfig(smart: "X", fast: "M", small: "M"))
    }

    @Test func cliOverridesWinOverPersistedConfig() {
        // User has config: smart=L, fast=M, small=S
        // CLI overrides:   --smart=A         (fast/small unchanged)
        let t = UserConfig(smart: "L", fast: "M", small: "S").resolve(
            smartOverride: "A", firstInstalled: "X")
        #expect(t == UserConfig(smart: "A", fast: "M", small: "S"))
    }

    @Test func modelSugarOverridesPersistedConfigButLosesToExplicitOverrides() {
        // --model X is invocation-local sugar for all tiers. Per-tier CLI
        // overrides still win when both are present.
        let t = UserConfig(smart: "L").resolve(
            fastOverride: "F", modelSugar: "SUGAR", firstInstalled: "X")
        #expect(t == UserConfig(smart: "SUGAR", fast: "F", small: "SUGAR"))
    }

    @Test func modelSugarReplacesAllPersistedTiers() {
        let t = UserConfig(smart: "L", fast: "M", small: "S").resolve(
            modelSugar: "SUGAR", firstInstalled: "X")
        #expect(t == UserConfig(smart: "SUGAR", fast: "SUGAR", small: "SUGAR"))
    }

    @Test func noConfigAndNoFirstInstalledReturnsNil() {
        // Nothing to work with anywhere → resolve fails. Callers turn this
        // into a "no model available" error at the CLI boundary.
        let t = UserConfig().resolve()
        #expect(t == nil)
    }
}
