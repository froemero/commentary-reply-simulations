# Semi-synthetic 2×2 on real EEG (Frömer et al. 2018, UCAP, CPz):
#  - noise    = each participant's real residual EEG (data minus intercept-only stimulus/response kernels),
#  - timing   = that participant's real stimulus onsets (correct trials), conditions assigned at random,
#  - accumulator = Kelly & O'Connell model (O'Connell et al. 2025 settings, realistic delay variability,
#                  participant-specific drift), scaled to µV so its response-locked peak ≈ the real CPP peak,
#  - components row = that participant's real deconvolved stimulus- and response-locked kernels (scale λ),
#                     added at the real stimulus onsets and the simulated responses.
# Each real participant = one simulated participant per cell (N = 40). Measure and methods as in 07.
include(joinpath(@__DIR__, "..", "..", "src", "ReplySims.jl")); using .ReplySims
using Unfold, CSV, DataFrames, Statistics, Serialization, StableRNGs, Random
dir = get(ENV, "UCAP_DIR", "/home/claude/ucap"); root = joinpath(@__DIR__, "..", "..")
fs = 100; dt = 1 / fs; tv = collect(0:dt:(1.6 - dt)); L = length(tv)
τS = (-0.2, 1.2); τR = (-0.8, 0.4)
sd_enc, sd_mot = 0.015, 0.025; d_low, d_high = 1.0, 2.0; sd_d0, sd_Δ = 0.3, 0.5
λs = parse.(Float64, split(get(ENV, "LAMBDAS", "1.0"), ","))
K = deserialize(joinpath(dir, "ucap_kernels.jls")); vps = sort(collect(keys(K)))
slope(t, y; w = (-0.25, -0.05)) = (ix = (t .>= w[1]) .& (t .<= w[2]); X = [ones(sum(ix)) t[ix]]; (X \ y[ix])[2])
function avg(sig, evts, τ)
    tbl = evts[evts.event .== "R", :]
    e, t = Unfold.epoch(data = sig, tbl = tbl, τ = τ, sfreq = fs)
    ok = .!vec(any(ismissing.(e), dims = (1, 2))); e = coalesce.(e[:, :, ok], 0.0); tb = tbl[ok, :]
    Dict(c => (collect(t), vec(mean(e[1, :, tb.condition .== c], dims = 2))) for c in ("low", "high"))
end
# µV scaling: model CPP peak (response-locked average, mean drift) → median real response-locked CPP peak
rng0 = StableRNG(1)
tr = [kelly_port_trial(rng0, tv, dt; d = 1.5, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot)) for _ in 1:2000]
model_peak = maximum(mean(hcat([c[3] for c in tr]...), dims = 2))  # approx (stim-locked mean); refined below
real_peak = median([maximum(0.5 .* (K[v].epochR["blurr"][2] .+ K[v].epochR["normal"][2])) for v in vps])
scale = real_peak / model_peak
# ΔRT for the encoding-delay null
rtl = [kelly_port_trial(rng0, tv, dt; d = d_low, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot))[1] for _ in 1:3000]
rth = [kelly_port_trial(rng0, tv, dt; d = d_high, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot))[1] for _ in 1:3000]
ΔRT = median(rtl) - median(rth)
println("scale µV per model unit: ", round(scale; digits = 2), "  ΔRT ", round(1000ΔRT; digits = 1), " ms")

function one(vp, row, col, λ, seed)
    rng = StableRNG(seed)
    noise = Float64.(reinterpret(Float32, read(joinpath(dir, "$(vp)_resid.f32"))))
    k = K[vp]; onsets = round.(Int, k.stim) .+ 1
    n = length(onsets); conds = shuffle(rng, vcat(fill("low", n ÷ 2), fill("high", n - n ÷ 2)))
    d0 = (d_low + d_high) / 2 + sd_d0 * randn(rng); Δd = (d_high - d_low) + sd_Δ * randn(rng)
    Δenc = max(0.0, ΔRT + ΔRT / 2 * randn(rng))
    acc = zeros(length(noise)); comp = zeros(length(noise)); rts = zeros(n)
    Sk = k.S.estimate; Rk = k.R.estimate; s0 = round(Int, τS[1] * fs); r0 = round(Int, τR[1] * fs)
    for i in 1:n
        c = conds[i]
        d = col === :differs ? (c == "low" ? d0 - Δd / 2 : d0 + Δd / 2) : d0
        ee = (col === :identical && c == "low") ? Δenc : 0.0
        rt, ch, cpp = kelly_port_trial(rng, tv, dt; d, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot), evonT = 0.2 + ee)
        rts[i] = rt; o = onsets[i]
        o + L - 1 <= length(acc) && (acc[o:(o + L - 1)] .+= scale .* cpp)
        if row === :components
            a = o + s0; a >= 1 && a + length(Sk) - 1 <= length(comp) && (comp[a:(a + length(Sk) - 1)] .+= λ .* Sk)
            b = o + round(Int, rt * fs) + r0; b >= 1 && b + length(Rk) - 1 <= length(comp) && (comp[b:(b + length(Rk) - 1)] .+= λ .* Rk)
        end
    end
    eeg = noise .+ acc .+ comp
    S = DataFrame(latency = onsets, event = "S", condition = conds)
    R = DataFrame(latency = onsets .+ round.(Int, rts .* fs), event = "R", condition = conds)
    evts = sort(vcat(S, R), :latency); evts = evts[evts.latency .< length(eeg) - 200, :]
    out = Dict{String,Float64}()
    sd(dd) = slope(dd["high"]...) - slope(dd["low"]...)
    out["reference: accumulator only, no noise"] = sd(avg(acc, evts, τR))
    out["epoch"] = sd(avg(eeg, evts, τR))
    f = @formula(0 ~ 1 + condition)
    m = fit(UnfoldModel, ["S" => (f, firbasis(τ = τS, sfreq = fs)), "R" => (f, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    eff = effects(Dict(:condition => ["low", "high"]), m); eff = eff[eff.eventname .== "R", :]
    out["deconv"] = sd(Dict(c => (eff[eff.condition .== c, :time], eff[eff.condition .== c, :yhat]) for c in ("low", "high")))
    f0 = @formula(0 ~ 1)
    m0 = fit(UnfoldModel, ["S" => (f0, firbasis(τ = τS, sfreq = fs)), "R" => (f0, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    out["residual"] = sd(avg(vec(Unfold.residuals(m0, eeg)), evts, τR))
    out
end

rows = []
for λ in λs, row in (:accumulator, :components), col in (:identical, :differs), (j, vp) in enumerate(vps)
    row === :accumulator && λ != λs[1] && continue
    o = one(vp, row, col, λ, 1000j)
    for (meth, v) in o
        push!(rows, (; lambda = row === :accumulator ? 0.0 : λ, row = String(row), col = String(col), vp, method = meth,
                     slope_diff = v / scale, truth_slope = col === :differs ? 1.0 : 0.0))
    end
    println(λ, " ", row, " ", col, " ", vp); flush(stdout)
end
df = DataFrame(rows)
CSV.write(joinpath(root, "results", "semisynthetic_2x2.csv"), df)
show(stdout, combine(groupby(df, [:lambda, :row, :col, :method]), :slope_diff => mean => :mean, :slope_diff => std => :sd,
     :slope_diff => (x -> mean(x) / std(x)) => :dz); allrows = true); println()
