# Figures: (1) reproduction of reply Supp. Fig. 2 layout; (2) realistic-range grid for the three
# implementations, with empirical ranges and the corrected positions of earlier simulations.
using CSV, DataFrames, Statistics, CairoMakie
root = joinpath(@__DIR__, ".."); R(x...) = joinpath(root, x...)
agg(d) = combine(groupby(d, [:dist, :urgency, :sens_sd, :motor_sd]),
    :deconv_norm => mean => :deconv, :residual_norm => mean => :residual, :rt_sd => mean => :rt_sd)
function panel!(ax, g, col, grid)
    M = [only(g[(g.sens_sd .== s) .& (g.motor_sd .== m), col]) for s in grid, m in grid]
    heatmap!(ax, 1:length(grid), 1:length(grid), M; colormap = :viridis, colorrange = (0, 1), highclip = :yellow)
    for i in eachindex(grid), j in eachindex(grid)
        text!(ax, i, j; text = string(round(M[i, j]; digits = 2)), align = (:center, :center), fontsize = 9,
              color = M[i, j] > 0.6 ? :black : :white)
    end
    ax.xticks = (1:length(grid), string.(grid)); ax.yticks = (1:length(grid), string.(grid))
end
box!(ax, grid, s, m; color = :red) = begin
    i = findfirst(≈(s), grid); j = findfirst(≈(m), grid)
    isnothing(i) || isnothing(j) || poly!(ax, Rect(i - 0.5, j - 0.5, 1, 1); color = :transparent, strokecolor = color, strokewidth = 3)
end
# grid positions on a continuous σ axis for the realistic plot
function marks!(ax, grid)
    pos(v) = begin  # piecewise-linear map from σ to cell index
        k = searchsortedlast(grid, v); k == length(grid) && return Float64(k)
        k + (v - grid[k]) / (grid[k+1] - grid[k])
    end
    # empirical range: sensory σ 0–0.03 (best ≈0.015), motor σ 0.017–0.046 (best ≈0.025)
    poly!(ax, Rect(pos(0.0) - 0.5, pos(0.017), pos(0.03) - pos(0.0) + 0.5, pos(0.046) - pos(0.017));
          color = :transparent, strokecolor = :white, strokewidth = 2.5, linestyle = :dash)
    scatter!(ax, [pos(0.015)], [pos(0.025)]; marker = :star5, color = :white, markersize = 16, strokecolor = :black, strokewidth = 1)
    # NHB 2024 Supp Fig 5 (uniform widths 0.1/0.2 → σ 0.029/0.058) and O'Connell 2025 (0.1/0.1 → 0.029)
    w = 0.1 / sqrt(12); w2 = 0.2 / sqrt(12)
    scatter!(ax, pos.([w, w2, w2]), pos.([w2, w2, w]); marker = :rect, color = :transparent, strokecolor = :red, strokewidth = 2.5, markersize = 14)
    scatter!(ax, [pos(w)], [pos(w)]; marker = :circle, color = :transparent, strokecolor = :orange, strokewidth = 2.5, markersize = 14)
end

# ---- Figure 1: Supp Fig 2 reproduction
d1 = agg(CSV.read(R("results", "suppfig2_grid.csv"), DataFrame)); d1.epoch .= 1.0
g1 = [0.0, 0.1, 0.2, 0.3]
f = Figure(size = (1050, 640))
for (r, urg) in enumerate((false, true)), (c, col) in enumerate((:deconv, :epoch, :residual))
    ax = Axis(f[r, c]; title = "$(col) — $(urg ? "with" : "no") urgency", xlabel = "sensor_encoding_delay_variability (σ, s)",
              ylabel = "motor_delay_variability (σ, s)", aspect = 1)
    panel!(ax, d1[d1.urgency .== urg, :], col, g1)
    for (s, m) in ((0.1, 0.2), (0.2, 0.2), (0.2, 0.1)); box!(ax, g1, s, m); end
    box!(ax, g1, 0.0, 0.0; color = :white)
end
Colorbar(f[:, 4]; colormap = :viridis, limits = (0, 1), label = "res_norm (relative to classic averaging)")
Label(f[0, :], "Reproduction of reply Supp. Fig. 2 (UnfoldSim KellyModel, Normal delays; mean of 3 seeds, N = 500 trials)"; fontsize = 14)
save(R("figures", "fig_suppfig2_reproduction.png"), f, px_per_unit = 2)

# ---- Figure 2: realistic range, three implementations
d2 = agg(CSV.read(R("results", "realistic_grid.csv"), DataFrame))
g2 = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
labs = Dict("normal" => "KellyModel, Normal delays", "uniform" => "KellyModel, uniform delays",
            "uniform_port" => "Kelly/O'Connell code (uniform)")
for col in (:deconv, :residual)
    f = Figure(size = (1150, 760))
    for (r, urg) in enumerate((false, true)), (c, dist) in enumerate(("normal", "uniform", "uniform_port"))
        ax = Axis(f[r, c]; title = "$(labs[dist]) — $(urg ? "with" : "no") urgency", titlesize = 12,
                  xlabel = "sensory-encoding delay σ (s)", ylabel = "motor delay σ (s)", aspect = 1)
        panel!(ax, d2[(d2.urgency .== urg) .& (d2.dist .== dist), :], col, g2)
        marks!(ax, g2)
    end
    Colorbar(f[:, 4]; colormap = :viridis, limits = (0, 1), label = "$(col) res_norm (relative to classic averaging)")
    Label(f[0, :], "$(col): fast-vs-slow difference relative to classic averaging, empirically plausible range\n" *
          "white dashed = empirical range (star = best estimate); red squares = NHB 2024 Supp. Fig. 5 settings; orange circle = O'Connell et al. 2025 settings (all as SD)"; fontsize = 13)
    save(R("figures", "fig_realistic_$(col).png"), f, px_per_unit = 2)
end
println("figures done")
