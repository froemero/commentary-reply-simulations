# Figures and statistics for script 14.
# (1) Grand-average response-locked fast vs slow waveforms: rows = case, columns = truth + methods
#     (layout follows O'Connell et al.'s figure).
# (2) Study-level inference: studies of 20 participants drawn from 40 (500 studies); cluster-based sign-flip
#     permutation test (1000 permutations, cluster-forming t threshold |t| > 2.09, df = 19) on the fast − slow
#     difference, pre-response window −0.8 to 0 s. "Effect found" = at least one significant cluster.
#     For ramp cases, the truth has an effect; for components only, any cluster is a false alarm.
# (3) Recovery: correlation and RMSE between each method's grand-average difference and the truth.
using Unfold, CSV, Serialization, Statistics, StableRNGs, DataFrames, CairoMakie, Random
root = joinpath(@__DIR__, "..", ".."); R = deserialize(joinpath(root, "results", "oconnell_extension.jls"))
cases = [(:components, "Components only (truth: no effect)"), (:ramp, "Ramp only"), (:both, "Ramp + components")]
methods = ["classic", "Ŝ removed (S~1, R~1+RT)", "Ŝ removed (S~1+RT, R~1+RT)", "residual (S~1, R~1)"]
vps = sort(unique(last.(collect(keys(R)))))
t = R[(:ramp, vps[1])]["classic"][1]
win = (t .>= -0.8) .& (t .<= 0.0)
D(case, m) = hcat([(x = R[(case, v)][m]; x[2] .- x[3]) for v in vps]...)'     # participants × time
TR(case) = hcat([R[(case, v)]["truth"] for v in vps]...)'
GA(case, m, k) = vec(mean(hcat([R[(case, v)][m][k] for v in vps]...), dims = 2))

# (1) waveforms
fig = Figure(size = (1500, 900))
for (i, (c, cl)) in enumerate(cases)
    ax = Axis(fig[i, 1]; title = i == 1 ? "truth: noise-free ramp (fast − slow)" : "", ylabel = cl, ylabelsize = 11, titlesize = 11)
    lines!(ax, t, vec(mean(TR(c), dims = 1)); color = :seagreen, linewidth = 2); hlines!(ax, [0]; color = :gray); vlines!(ax, [0]; color = :gray, linestyle = :dash)
    for (j, m) in enumerate(methods)
        ax = Axis(fig[i, j + 1]; title = i == 1 ? m : "", titlesize = 11, xlabel = i == 3 ? "time from response (s)" : "")
        lines!(ax, t, GA(c, m, 2); color = :firebrick, label = "fast"); lines!(ax, t, GA(c, m, 3); color = (:firebrick4, 0.9), linestyle = :dash, label = "slow")
        vlines!(ax, [0]; color = :gray, linestyle = :dash)
        i == 1 && j == 1 && axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
    end
end
Label(fig[0, :], "Response-locked fast vs slow RT (µV, CPz): real UCAP noise, timing and kernels; N = $(length(vps))"; fontsize = 13)
save(joinpath(root, "figures", "oconnell_extension_waveforms.png"), fig, px_per_unit = 2)

# (2) cluster permutation on studies of n = 20
function clusters(tv, thr)
    cl = Tuple{Int,Int,Float64}[]; i = 1
    while i <= length(tv)
        if abs(tv[i]) > thr
            s = sign(tv[i]); j = i
            while j + 1 <= length(tv) && abs(tv[j + 1]) > thr && sign(tv[j + 1]) == s; j += 1; end
            push!(cl, (i, j, sum(tv[i:j]))); i = j + 1
        else
            i += 1
        end
    end
    cl
end
tstat(X) = vec(mean(X, dims = 1) ./ (std(X, dims = 1) ./ sqrt(size(X, 1))))
function cluster_test(X; nperm = 1000, thr = 2.09, rng)
    tv = tstat(X); obs = clusters(tv, thr); isempty(obs) && return (false, 0, 0)
    maxnull = zeros(nperm)
    for p in 1:nperm
        sgn = rand(rng, (-1.0, 1.0), size(X, 1)); c = clusters(tstat(X .* sgn), thr)
        maxnull[p] = isempty(c) ? 0.0 : maximum(abs.(last.(c)))
    end
    q = quantile(maxnull, 0.95)
    (any(abs(m) > q for (_, _, m) in obs), sum(m < -q for (_, _, m) in obs) > 0, sum(m > q for (_, _, m) in obs) > 0)
end
rng = StableRNG(11); nst = 500; rows = []
for (c, cl) in cases, m in methods
    X = D(c, m)[:, win]; found = 0; neg = 0; pos = 0
    for s in 1:nst
        idx = randperm(rng, size(X, 1))[1:20]   # without replacement
        a, b, cc = cluster_test(X[idx, :]; rng); found += a; neg += b; pos += cc
    end
    tru = vec(mean(TR(c)[:, win], dims = 1)); est = vec(mean(X, dims = 1))
    push!(rows, (; case = String(c), method = m, effect_found = found / nst, slow_gt_fast_cluster = neg / nst, fast_gt_slow_cluster = pos / nst,
                 truth_has_effect = c != :components,
                 r_with_truth = c == :components ? NaN : round(cor(est, tru); digits = 2),
                 rmse = round(sqrt(mean((est .- tru) .^ 2)); digits = 2),
                 mean_abs_diff = round(mean(abs.(est)); digits = 2), mean_abs_truth = round(mean(abs.(tru)); digits = 2)))
end
df = DataFrame(rows); CSV.write(joinpath(root, "results", "oconnell_extension_stats.csv"), df)
show(stdout, df; allrows = true, allcols = true); println()
