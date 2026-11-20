# Study-level conclusions for the semi-synthetic 2×2 (n = 20 participants drawn from the 40, 2000 studies).
using CSV, DataFrames, Statistics, HypothesisTests, Distributions, StableRNGs
root = joinpath(@__DIR__, "..", ".."); df = CSV.read(joinpath(root, "results", "semisynthetic_2x2.csv"), DataFrame)
methods = ["reference: accumulator only, no noise", "epoch", "deconv", "residual"]; S = 2000; rng = StableRNG(3)
conclude(v) = (p = pvalue(OneSampleTTest(v)); p >= 0.05 ? "none" : mean(v) > 0 ? "high" : "low")
out = []
for (λ, r) in ((0.0, "accumulator"), (0.5, "components"), (1.0, "components")), c in ("identical", "differs"), m in methods
    pool = df[(df.lambda .== λ) .& (df.row .== r) .& (df.col .== c) .& (df.method .== m), :slope_diff]
    cc = [conclude(rand(rng, pool, 20)) for _ in 1:S]
    push!(out, (; lambda = λ, col = c, method = m, p_high = mean(cc .== "high"), p_none = mean(cc .== "none"), p_low = mean(cc .== "low")))
end
det = DataFrame(out); CSV.write(joinpath(root, "results", "semisynthetic_detection.csv"), det)
clip(p) = clamp(p, 1 / (2S), 1 - 1 / (2S))
dp = combine(groupby(det, [:lambda, :method])) do g
    H = only(g[g.col .== "differs", :p_high]); FA = 1 - only(g[g.col .== "identical", :p_none])
    (; hit = H, false_alarm = FA, fa_high = only(g[g.col .== "identical", :p_high]), fa_low = only(g[g.col .== "identical", :p_low]),
       dprime = round(quantile(Normal(), clip(H)) - quantile(Normal(), clip(FA)); digits = 2))
end
CSV.write(joinpath(root, "results", "semisynthetic_dprime.csv"), dp); show(stdout, dp; allrows = true); println()
