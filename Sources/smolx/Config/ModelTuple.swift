import Foundation

/// Three model aliases — one per tier — for the agent runners that
/// support a split. The wrappers in `AgentEnvironment` thread each tier
/// onto the right native config slot per agent (see the plan file's
/// per-agent mapping table). All three are non-optional after `resolve`
/// runs — missing user input gets filled in by the cascade.
struct ModelTuple: Sendable, Equatable {
    let smart: String
    let fast: String
    let small: String

    /// "I only have one model" convenience — back-compat for `--model X`
    /// and the first-installed-model fallback. All three tiers get `X`.
    init(single: String) {
        self.smart = single
        self.fast = single
        self.small = single
    }

    init(smart: String, fast: String, small: String) {
        self.smart = smart
        self.fast = fast
        self.small = small
    }

    /// Resolve the effective `(smart, fast, small)` tuple from the user
    /// inputs and the installed-model fallback. All four arguments may
    /// be nil; the rules below describe the cascade.
    ///
    /// Resolution order (highest precedence first):
    ///   1. Persisted config (`base.smart` / `.fast` / `.small`).
    ///   2. CLI per-tier overrides (`smartOverride` / `fastOverride` /
    ///      `smallOverride`) replace whatever was at step 1 for that tier.
    ///   3. `modelSugar` (i.e. `--model X`) fills any tier that's still nil.
    ///   4. Cascade downward: a still-nil `fast` becomes the resolved
    ///      `smart`; a still-nil `small` becomes the resolved `fast`.
    ///   5. If `smart` is still nil, fill it from `firstInstalled` and
    ///      re-cascade so `fast` and `small` inherit it.
    ///
    /// Returns `nil` only when `smart` cannot be filled by any source —
    /// i.e. nothing configured, no `--model`, and no installed models.
    /// Callers turn that into a "no model available" error at the CLI
    /// boundary.
    static func resolve(
        config base: UserConfig,
        smartOverride: String? = nil,
        fastOverride: String? = nil,
        smallOverride: String? = nil,
        modelSugar: String? = nil,
        firstInstalled: String? = nil
    ) -> ModelTuple? {
        // 1) start from persisted
        var s = base.smart
        var f = base.fast
        var t = base.small
        // 2) per-tier CLI overrides
        if let smartOverride { s = smartOverride }
        if let fastOverride { f = fastOverride }
        if let smallOverride { t = smallOverride }
        // 3) --model sugar fills still-nil tiers
        if let modelSugar {
            if s == nil { s = modelSugar }
            if f == nil { f = modelSugar }
            if t == nil { t = modelSugar }
        }
        // 4) cascade downward
        if f == nil { f = s }
        if t == nil { t = f }
        // 5) last-resort: first installed model fills `smart` and re-cascades
        if s == nil, let firstInstalled {
            s = firstInstalled
            if f == nil { f = firstInstalled }
            if t == nil { t = firstInstalled }
        }
        guard let smart = s, let fast = f, let small = t else { return nil }
        return ModelTuple(smart: smart, fast: fast, small: small)
    }
}
