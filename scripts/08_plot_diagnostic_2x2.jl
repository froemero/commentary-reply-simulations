# Re-plot the 2×2 diagnostic (results/diagnostic_2x2.csv) laid out as the design:
# rows = accumulator only / accumulator + S/R components; columns = signal identical / signal differs.
# Shared y-axis; reference lines at 0 (no effect) and 1 (latent drift difference); the line that is
# the true effect in each cell is drawn in colour, the other in grey.
using CSV, DataFrames, CairoMakie
root = joinpath(@__DIR__, "..")
df = CSV.read(joinpath(root, "results", "diagnostic_2x2.csv"), DataFrame)
methods = ["reference: accumulator only, no noise", "epoch", "deconv", "residual"]
mlab = ["noise-free\naccumulator", "epoch", "deconv", "residual"]
rows = [("accumulator", "Accumulator only"), ("components", "Accumulator + S/R components")]
cols = [("identical", "Signal identical (RT differs via encoding delay)", 0.0), ("differs", "Signal differs (drift 1 vs 2)", 1.0)]
truecol = :seagreen
yl = (minimum(df.slope_diff) - 0.05, maximum(df.slope_diff) + 0.05)
yl = (min(yl[1], -1.0), max(yl[2], 2.0))

fig = Figure(size = (1000, 820))
for (i, (r, rl)) in enumerate(rows), (j, (c, cl, truth)) in enumerate(cols)
    ax = Axis(fig[i, j]; xticks = (1:4, mlab), xticklabelsize = 11,
              ylabel = j == 1 ? "build-up rate difference,\nhigh − low drift (a.u./s)" : "", ylabelsize = 11)
    ylims!(ax, yl...)
    for (v, isTrue) in ((0.0, truth == 0.0), (1.0, truth == 1.0))
        hlines!(ax, [v]; color = isTrue ? truecol : (:gray, 0.7), linewidth = isTrue ? 2.5 : 1.5,
                linestyle = isTrue ? :solid : :dash)
    end
    for (k, m) in enumerate(methods)
        v = df[(df.row .== r) .& (df.col .== c) .& (df.method .== m), :slope_diff]
        boxplot!(ax, fill(k, length(v)), v; width = 0.6, color = k == 1 ? (:gray, 0.25) : (:steelblue, 0.55),
                 whiskerwidth = 0.3, show_outliers = true)
    end
    i == 1 && (ax.title = cl; ax.titlesize = 13)
    j == 2 && Label(fig[i, 3], rl; rotation = -pi / 2, fontsize = 13, tellheight = false)
    hidexdecorations!(ax; ticklabels = i != 2, ticks = i != 2)
end
Legend(fig[3, 1:2],
       [LineElement(color = truecol, linewidth = 2.5), LineElement(color = (:gray, 0.7), linewidth = 1.5, linestyle = :dash)],
       ["true effect in this cell", "other reference (0 = no effect, 1 = drift effect)"];
       orientation = :horizontal, framevisible = false, labelsize = 11)
Label(fig[0, 1:2], "Build-up-rate difference (response-locked slope, −250 to −50 ms), 100 simulated participants per cell (each point = one participant)"; fontsize = 13)
save(joinpath(root, "figures", "diagnostic_2x2_slopes.png"), fig, px_per_unit = 2)
println("ok")
