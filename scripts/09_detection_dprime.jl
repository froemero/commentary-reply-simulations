# Detection analysis for the 2×2 diagnostic. Each simulated dataset = one participant.
# A "study" = n participants drawn (with replacement) from the 100 simulated per cell; the group-level
# build-up-rate difference (high − low drift) is tested with a two-sided one-sample t-test, α = .05.
# Conclusion: "high > low" (significant positive), "low > high" (significant negative) or "no effect".
# Consistent with truth: drift-differs cells → "high > low"; signal-identical cells → "no effect".
# d′ per row and method = z(hit rate in drift-differs cell) − z(false-alarm rate in signal-identical cell),
# false alarm = any significant difference; rates clipped to [1/(2S), 1 − 1/(2S)].
using CSV, DataFrames, Statistics, HypothesisTests, Distributions, StableRNGs, CairoMakie
root = joinpath(@__DIR__, "..")
df = CSV.read(joinpath(root, "results", "diagnostic_2x2.csv"), DataFrame)
methods = ["reference: accumulator only, no noise", "epoch", "deconv", "residual"]
mlab = ["noise-free\naccumulator", "epoch", "deconv", "residual"]
S = 2000; α = 0.05
rng = StableRNG(7)
conclude(v) = (p = pvalue(OneSampleTTest(v)); p >= α ? "no effect" : mean(v) > 0 ? "high > low" : "low > high")
rows = []
for n in (10, 20), r in ("accumulator", "components"), c in ("identical", "differs"), m in methods
    pool = df[(df.row .== r) .& (df.col .== c) .& (df.method .== m), :slope_diff]
    concl = [conclude(rand(rng, pool, n)) for _ in 1:S]
    push!(rows, (; n, row = r, col = c, method = m,
                 p_high = mean(concl .== "high > low"), p_low = mean(concl .== "low > high"), p_none = mean(concl .== "no effect"),
                 consistent = c == "differs" ? mean(concl .== "high > low") : mean(concl .== "no effect")))
end
det = DataFrame(rows)
clip(p) = clamp(p, 1 / (2S), 1 - 1 / (2S))
dp = combine(groupby(det, [:n, :row, :method])) do g
    H = only(g[g.col .== "differs", :p_high]); FA = 1 - only(g[g.col .== "identical", :p_none])
    (; hit = H, false_alarm = FA, dprime = round(quantile(Normal(), clip(H)) - quantile(Normal(), clip(FA)); digits = 2))
end
CSV.write(joinpath(root, "results", "diagnostic_2x2_detection.csv"), det)
CSV.write(joinpath(root, "results", "diagnostic_2x2_dprime.csv"), dp)
show(stdout, det[det.n .== 20, :]; allrows = true); println(); show(stdout, dp; allrows = true); println()

# figure: 2×2, stacked bars of study conclusions per method (n = 20), true conclusion in green
cols = [("identical", "Signal identical (RT differs via encoding delay)\ntrue conclusion: no effect"),
        ("differs", "Signal differs (drift 1 vs 2)\ntrue conclusion: high > low")]
rowsL = [("accumulator", "Accumulator only"), ("components", "Accumulator + S/R components")]
cmap = Dict("high > low" => :firebrick, "no effect" => :gray70, "low > high" => :steelblue)
nstudy = 20
fig = Figure(size = (1050, 860))
for (i, (r, rl)) in enumerate(rowsL), (j, (c, cl)) in enumerate(cols)
    ax = Axis(fig[i, j]; xticks = (1:4, mlab), xticklabelsize = 11, ylabel = j == 1 ? "proportion of studies" : "",
              limits = (0.4, 4.6, 0, 1.12), yticks = 0:0.25:1)
    i == 1 && (ax.title = cl; ax.titlesize = 12)
    for (k, m) in enumerate(methods)
        g = only(eachrow(det[(det.n .== nstudy) .& (det.row .== r) .& (det.col .== c) .& (det.method .== m), :]))
        base = 0.0
        for (lab, p) in (("low > high", g.p_low), ("no effect", g.p_none), ("high > low", g.p_high))
            p > 0 && barplot!(ax, [k], [p]; offset = base, color = cmap[lab], width = 0.7,
                              strokecolor = (c == "differs" ? lab == "high > low" : lab == "no effect") ? :seagreen : :transparent,
                              strokewidth = 3)
            base += p
        end
        text!(ax, k, 1.03; text = "$(round(Int, 100g.consistent))% correct", align = (:center, :bottom), fontsize = 10)
    end
    j == 2 && Label(fig[i, 3], rl; rotation = -pi / 2, fontsize = 13, tellheight = false)
    hidexdecorations!(ax; ticklabels = i != 2, ticks = i != 2)
end
# d′ table under the panels
dtxt = join(["$(rl): " * join(["$(replace(ml, "\n" => " ")) $(only(dp[(dp.n .== nstudy) .& (dp.row .== r) .& (dp.method .== m), :dprime]))"
             for (m, ml) in zip(methods, mlab)], ",  ") for (r, rl) in rowsL], "\n")
Label(fig[3, 1:2], "d′ (hit = high > low when drift differs; false alarm = any significant difference when signal identical):\n" * dtxt; fontsize = 11)
Legend(fig[4, 1:2], [PolyElement(color = cmap[k]) for k in ("high > low", "no effect", "low > high")],
       ["study concludes high > low", "no significant difference", "study concludes low > high"];
       orientation = :horizontal, framevisible = false, labelsize = 11)
Label(fig[0, 1:2], "Conclusions of simulated studies ($(nstudy) participants each, two-sided t-test on build-up-rate difference, α = .05, $(S) studies); green outline = conclusion that matches the truth"; fontsize = 12)
save(joinpath(root, "figures", "diagnostic_2x2_detection.png"), fig, px_per_unit = 2)
println("ok")
