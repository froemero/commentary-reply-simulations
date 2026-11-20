# Grand averages for UCAP (CPz): stimulus- and response-locked averages by condition, deconvolved
# intercept-only stimulus and response kernels, and response-locked deconv/residual by condition.
using Unfold, CSV, Serialization, Statistics, CairoMakie, DataFrames
dir = get(ENV, "UCAP_DIR", "/home/claude/ucap"); root = joinpath(@__DIR__, "..", "..")
K = deserialize(joinpath(dir, "ucap_kernels.jls")); vps = sort(collect(keys(K)))
ga(f) = (t = f(K[vps[1]])[1]; (t, mean(hcat([f(K[v])[2] for v in vps]...), dims = 2)[:]))
fig = Figure(size = (1200, 700))
cols = (("blurr", :firebrick), ("normal", :steelblue))
for (j, (key, lab)) in enumerate(((:epochS, "classic average — stimulus-locked"), (:epochR, "classic average — response-locked"),
                                   (:deconvR, "deconv — response-locked"), (:residR, "residual — response-locked")))
    ax = Axis(fig[1 + (j > 2), 1 + (j - 1) % 2 ]; title = lab, xlabel = "time (s)", ylabel = "µV (CPz)")
    for (c, col) in cols
        t, y = ga(k -> getproperty(k, key)[c]); lines!(ax, t, y; color = col, label = c)
    end
    vlines!(ax, [0]; color = :gray, linestyle = :dash); j == 1 && axislegend(ax; position = :lt, framevisible = false)
end
ax = Axis(fig[1:2, 3]; title = "intercept-only deconvolved kernels", xlabel = "time (s)", ylabel = "µV (CPz)")
tS = K[vps[1]].S.time; tR = K[vps[1]].R.time
lines!(ax, tS, mean(hcat([K[v].S.estimate for v in vps]...), dims = 2)[:]; color = :black, label = "stimulus-locked")
lines!(ax, tR, mean(hcat([K[v].R.estimate for v in vps]...), dims = 2)[:]; color = :darkorange, label = "response-locked")
vlines!(ax, [0]; color = :gray, linestyle = :dash); axislegend(ax; position = :lt, framevisible = false)
Label(fig[0, :], "Frömer et al. 2018 (UCAP), CPz, correct trials, N = $(length(vps)); blurred vs normal"; fontsize = 13)
save(joinpath(root, "figures", "ucap_grand_averages.png"), fig, px_per_unit = 2); println("ok")
