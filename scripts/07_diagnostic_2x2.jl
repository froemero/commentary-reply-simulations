# 2×2 diagnostic: (accumulator only | accumulator + separate stimulus/response components)
#                × (signal identical, only RTs differ | signal differs: low vs high drift).
# Model: Kelly & O'Connell trial loop with O'Connell et al. (2025) settings (d = 1 vs 2, U = 1, Z = 0.3, Sz = 0.4,
# Su = 0.17) and empirically realistic delay variability (σ encoding 15 ms, σ motor 25 ms).
# Contrast = experimental condition (low vs high drift), not an RT split. Pink noise added.
# Each method is scored against ground truth, not against classic averaging.
include(joinpath(@__DIR__, "..", "src", "ReplySims.jl")); using .ReplySims
using Unfold, StableRNGs, Random, Statistics, DataFrames, CSV, CairoMakie, FFTW
fs = 100; dt = 1 / fs; tv = collect(0:dt:(1.6 - dt)); L = length(tv)
τS = (-0.2, 1.2); τR = (-0.8, 0.4)
Nper = 300; nseeds = 50; noise_sd = 0.05
sd_enc, sd_mot = 0.015, 0.025
d_low, d_high = 1.0, 2.0
root = joinpath(@__DIR__, "..")

# separate components (used in the "components" row): identical in both conditions
hann(t, a, b) = (a <= t <= b) ? sin(pi * (t - a) / (b - a))^2 : 0.0
Sker = [0.25 * hann(t, 0.0, 0.6) for t in τS[1]:dt:τS[2]]        # stimulus-locked, P3-like, peak 0.3 s
Rker_t = collect(τR[1]:dt:τR[2])
Rker = [0.25 * hann(t, -0.35, 0.15) for t in Rker_t]              # response-locked, peak −0.1 s

function pink(rng, n)
    f = rfftfreq(n); a = [x == 0 ? 0.0 : 1 / sqrt(x) for x in f]
    x = irfft(a .* cis.(2pi .* rand(rng, length(f))), n); x ./ std(x)
end

# one dataset
function simulate(rng; row::Symbol, col::Symbol, Δenc = 0.0, Δmot = 0.0)
    conds = shuffle(rng, vcat(fill("low", Nper), fill("high", Nper)))
    onsets = Int[]; rts = Float64[]; traces = Vector{Vector{Float64}}()
    t = 50
    for c in conds
        d = col === :differs ? (c == "low" ? d_low : d_high) : (d_low + d_high) / 2
        extra_enc = (col === :identical && c == "low") ? Δenc : 0.0
        extra_mot = (col === :identical && c == "low") ? Δmot : 0.0
        rt, ch, cpp = kelly_port_trial(rng, tv, dt; d, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot),
                                       evonT = 0.2 + extra_enc, mT = 0.1 + extra_mot)
        push!(onsets, t); push!(rts, rt); push!(traces, cpp)
        t += round(Int, rand(rng, (1.5, 2.0, 2.5)) * fs)
    end
    n = t + 3fs
    acc = zeros(n)
    for i in eachindex(conds); acc[onsets[i]:(onsets[i] + L - 1)] .+= traces[i]; end
    comp = zeros(n)
    if row === :components
        s0 = round(Int, τS[1] * fs); r0 = round(Int, τR[1] * fs)
        for i in eachindex(conds)
            a = onsets[i] + s0; comp[a:(a + length(Sker) - 1)] .+= Sker
            b = onsets[i] + round(Int, rts[i] * fs) + r0; comp[b:(b + length(Rker) - 1)] .+= Rker
        end
    end
    eeg = acc .+ comp .+ noise_sd .* pink(rng, n)
    S = DataFrame(latency = onsets, event = "S", condition = conds, rt = rts)
    R = DataFrame(latency = onsets .+ round.(Int, rts .* fs), event = "R", condition = conds, rt = rts)
    evts = sort(vcat(S, R), :latency)
    return eeg, acc, evts
end

# measures on a response-locked waveform (t, y)
slope(t, y; w = (-0.25, -0.05)) = (ix = (t .>= w[1]) .& (t .<= w[2]); X = [ones(sum(ix)) t[ix]]; (X \ y[ix])[2])
ampl(t, y; w = (-0.18, -0.08)) = mean(y[(t .>= w[1]) .& (t .<= w[2])])

function avg(signal, evts, ev, τ)
    tbl = evts[evts.event .== ev, :]
    e, t = Unfold.epoch(data = signal, tbl = tbl, τ = τ, sfreq = fs)
    ok = .!vec(any(ismissing.(e), dims = (1, 2))); e = coalesce.(e[:, :, ok], 0.0); tb = tbl[ok, :]
    Dict(c => (collect(t), vec(mean(e[1, :, tb.condition .== c], dims = 2))) for c in ("low", "high"))
end

function estimate(eeg, acc, evts)
    out = Dict{String,Dict{String,Tuple{Vector{Float64},Vector{Float64}}}}()
    out["reference: accumulator only, no noise"] = avg(acc, evts, "R", τR)
    out["epoch"] = avg(eeg, evts, "R", τR)
    f = @formula(0 ~ 1 + condition)
    m = fit(UnfoldModel, ["S" => (f, firbasis(τ = τS, sfreq = fs)), "R" => (f, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    eff = effects(Dict(:condition => ["low", "high"]), m); eff = eff[eff.eventname .== "R", :]
    out["deconv"] = Dict(c => (eff[eff.condition .== c, :time], eff[eff.condition .== c, :yhat]) for c in ("low", "high"))
    f0 = @formula(0 ~ 1)
    m0 = fit(UnfoldModel, ["S" => (f0, firbasis(τ = τS, sfreq = fs)), "R" => (f0, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    out["residual"] = avg(vec(Unfold.residuals(m0, eeg)), evts, "R", τR)
    out
end

# pilot: RT difference produced by the drift contrast, to match it in the "identical" column
rng = StableRNG(999)
rt_low = [kelly_port_trial(rng, tv, dt; d = d_low, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot))[1] for _ in 1:4000]
rt_high = [kelly_port_trial(rng, tv, dt; d = d_high, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot))[1] for _ in 1:4000]
ΔRT = median(rt_low) - median(rt_high)
println("median RT low/high drift: ", round(median(rt_low); digits = 3), " / ", round(median(rt_high); digits = 3), "  Δ = ", round(ΔRT; digits = 3))

# ground truth (high − low): build-up rate of the latent accumulation = Δd (a.u./s); zero when signal identical.
truth = Dict(:identical => 0.0, :differs => d_high - d_low)

rows = []; examples = Dict()
for row in (:accumulator, :components), col in (:identical, :differs), seed in 1:nseeds
    eeg, acc, evts = simulate(StableRNG(seed); row, col, Δenc = ΔRT / 2, Δmot = ΔRT / 2)
    est = estimate(eeg, acc, evts)
    seed == 1 && (examples[(row, col)] = est)
    for (meth, d) in est
        push!(rows, (; row = String(row), col = String(col), seed, method = meth,
                     slope_diff = slope(d["high"]...) - slope(d["low"]...),
                     ampl_diff = ampl(d["high"]...) - ampl(d["low"]...), truth_slope = truth[col]))
    end
end
df = DataFrame(rows)
CSV.write(joinpath(root, "results", "diagnostic_2x2.csv"), df)
summ = combine(groupby(df, [:row, :col, :method]),
    :slope_diff => (x -> round(mean(x); digits = 3)) => :slope_mean, :slope_diff => (x -> round(std(x); digits = 3)) => :slope_sd,
    :ampl_diff => (x -> round(mean(x); digits = 3)) => :ampl_mean, :ampl_diff => (x -> round(std(x); digits = 3)) => :ampl_sd,
    :truth_slope => first => :truth_slope)
show(stdout, summ; allrows = true, allcols = true); println()
CSV.write(joinpath(root, "results", "diagnostic_2x2_summary.csv"), summ)

# figure 1: response-locked waveforms (seed 1)
methods = ["reference: accumulator only, no noise", "epoch", "deconv", "residual"]
fig = Figure(size = (1300, 900)); axs = Dict{Tuple{Int,Int},Axis}()
rlab = Dict(:accumulator => "accumulator only", :components => "accumulator + S/R components")
clab = Dict(:identical => "signal identical (only RT differs)", :differs => "signal differs (drift 1 vs 2)")
for (i, (row, col)) in enumerate([(r, c) for r in (:accumulator, :components) for c in (:identical, :differs)]), (j, meth) in enumerate(methods)
    ax = axs[(i, j)] = Axis(fig[i, j]; title = i == 1 ? meth : "", titlesize = 11, xlabel = i == 4 ? "time from response (s)" : "",
              ylabel = j == 1 ? "$(rlab[row])\n$(clab[col])" : "", ylabelsize = 10)
    for (c, colr) in (("low", :steelblue), ("high", :firebrick))
        t, y = examples[(row, col)][meth][c]; lines!(ax, t, y; color = colr, label = "$c drift")
    end
    vlines!(ax, [0]; color = :gray, linestyle = :dash); vspan!(ax, [-0.25], [-0.05]; color = (:gray, 0.12))
    i == 1 && j == 1 && axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
end
linkyaxes!([axs[(i, j)] for i in 1:4, j in 1:3]...); linkyaxes!([axs[(i, 4)] for i in 1:4]...)
for i in 1:4, j in 1:3; ylims!(axs[(i, j)], -0.05, 0.9); end
Label(fig[5, 1:4], "Columns 1–3 share one y-axis; residual (column 4) shares its own, smaller y-axis. Grey band = slope window."; fontsize = 11)
save(joinpath(root, "figures", "diagnostic_2x2_waveforms.png"), fig, px_per_unit = 2)

include(joinpath(@__DIR__, "08_plot_diagnostic_2x2.jl"))  # 2×2 slope figure
println("done")
