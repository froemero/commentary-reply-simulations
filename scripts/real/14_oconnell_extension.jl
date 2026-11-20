# Extension of O'Connell et al.'s S/R-components vs ramping-accumulation simulation, on real EEG (UCAP, CPz).
# Cases: components only | ramp only | ramp + components. Per UCAP participant:
#   noise = real residual EEG; trial timing = real stimulus onsets (correct trials);
#   RTs and ramp = Kelly & O'Connell model (O'Connell et al. 2025 settings, realistic delay variability,
#   participant-specific drift), ramp scaled so its response-locked average peaks at the real CPP peak (µV);
#   components = the participant's real deconvolved stimulus- and response-locked kernels (identical on every trial).
# Contrast: fast vs slow RT (median split within participant). Measure: response-locked fast − slow difference
# waveform. Methods: classic averaging; Ŝ removed after Unfold S ~ 1, R ~ 1 + rt (M1) and S ~ 1 + rt, R ~ 1 + rt (M2);
# residuals of an intercept-only model (S ~ 1, R ~ 1). Truth: fast − slow difference of the noise-free ramp,
# averaged from single trials aligned to their own response (zero for components only).
include(joinpath(@__DIR__, "..", "..", "src", "ReplySims.jl")); using .ReplySims
using Unfold, CSV, DataFrames, Statistics, Serialization, StableRNGs, Random, SparseArrays
dir = get(ENV, "UCAP_DIR", "/home/claude/ucap"); root = joinpath(@__DIR__, "..", "..")
fs = 100; dt = 1 / fs; tv = collect(0:dt:(1.6 - dt)); L = length(tv)
τS = (-0.2, 1.2); τR = (-1.0, 0.4); τE = (-1.0, 0.2)
sd_enc, sd_mot = 0.015, 0.025; d_mean, sd_d0 = 1.5, 0.3
K = deserialize(joinpath(dir, "ucap_kernels.jls")); vps = sort(collect(keys(K)))
real_peak = median([maximum(0.5 .* (K[v].epochR["blurr"][2] .+ K[v].epochR["normal"][2])) for v in vps])
trial(rng, d) = kelly_port_trial(rng, tv, dt; d, Terz = sd_to_width(sd_enc), st = sd_to_width(sd_mot))
# response-aligned single-trial average of noise-free ramps (used for scaling and truth)
function ralign(traces, rts, sel)
    lags = round(Int, τE[1] * fs):round(Int, τE[2] * fs); out = zeros(length(lags)); n = 0
    for i in findall(sel)
        r = round(Int, rts[i] * fs) + 1
        idx = r .+ lags                                   # samples before stimulus onset / after trace end = 0
        out .+= [1 <= j <= L ? traces[i][j] : 0.0 for j in idx]; n += 1
    end
    out ./ n
end
rng0 = StableRNG(1); tr0 = [trial(rng0, d_mean) for _ in 1:3000]
scale = real_peak / maximum(ralign(last.(tr0), first.(tr0), trues(3000)))
println("µV per model unit: ", round(scale; digits = 2))

function epoch_avg(sig, evts, sel)
    tbl = evts[sel, :]
    e, t = Unfold.epoch(data = sig, tbl = tbl, τ = τE, sfreq = fs)
    ok = .!vec(any(ismissing.(e), dims = (1, 2))); e = coalesce.(e[:, :, ok], 0.0)
    collect(t), vec(mean(e[1, :, :], dims = 2))
end
function s_removed(eeg, evts, fS, fR)
    m = fit(UnfoldModel, ["S" => (fS, firbasis(τ = τS, sfreq = fs)), "R" => (fR, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    X = modelmatrix(m); β = vec(coef(m)); nS = sum(coeftable(m).eventname .== "S")
    sp = X[:, 1:nS] * β[1:nS]; out = copy(eeg); out[1:length(sp)] .-= sp; out
end
function residual(eeg, evts)
    f0 = @formula(0 ~ 1)
    m = fit(UnfoldModel, ["S" => (f0, firbasis(τ = τS, sfreq = fs)), "R" => (f0, firbasis(τ = τR, sfreq = fs))], evts, eeg; eventcolumn = :event, show_progress = false)
    vec(Unfold.residuals(m, eeg))
end

function one(vp, case, seed)
    rng = StableRNG(seed); k = K[vp]
    noise = Float64.(reinterpret(Float32, read(joinpath(dir, "$(vp)_resid.f32"))))
    onsets = round.(Int, k.stim) .+ 1; n = length(onsets)
    d = d_mean + sd_d0 * randn(rng)
    tr = [trial(rng, d) for _ in 1:n]; rts = first.(tr); ramps = last.(tr)
    fast = rts .<= median(rts)
    acc = zeros(length(noise)); comp = zeros(length(noise))
    Sk = k.S.estimate; Rk = k.R.estimate; s0 = round(Int, τS[1] * fs); r0 = round(Int, -0.8 * fs)  # kernels span τ used in script 10
    for i in 1:n
        o = onsets[i]
        if case != :components
            o + L - 1 <= length(acc) && (acc[o:(o + L - 1)] .+= scale .* ramps[i])
        end
        if case != :ramp
            a = o + s0; a >= 1 && a + length(Sk) - 1 <= length(comp) && (comp[a:(a + length(Sk) - 1)] .+= Sk)
            b = o + round(Int, rts[i] * fs) + r0; b >= 1 && b + length(Rk) - 1 <= length(comp) && (comp[b:(b + length(Rk) - 1)] .+= Rk)
        end
    end
    eeg = noise .+ acc .+ comp
    grp = ifelse.(fast, "fast", "slow")
    S = DataFrame(latency = onsets, event = "S", rt = rts, grp = grp)
    R = DataFrame(latency = onsets .+ round.(Int, rts .* fs), event = "R", rt = rts, grp = grp)
    evts = sort(vcat(S, R), :latency); evts = evts[evts.latency .< length(eeg) - 200, :]
    isR = evts.event .== "R"; isF = isR .& (evts.grp .== "fast"); isSl = isR .& (evts.grp .== "slow")
    function diffw(sig)
        t, a = epoch_avg(sig, evts, isF); _, b = epoch_avg(sig, evts, isSl)
        (t, a, b)
    end
    out = Dict{String,Any}()
    truth = case == :components ? zeros(length(round(Int, τE[1] * fs):round(Int, τE[2] * fs))) :
            scale .* (ralign(ramps, rts, fast) .- ralign(ramps, rts, .!fast))
    out["truth"] = truth
    out["classic"] = diffw(eeg)
    out["Ŝ removed (S~1, R~1+RT)"] = diffw(s_removed(eeg, evts, @formula(0 ~ 1), @formula(0 ~ 1 + rt)))
    out["Ŝ removed (S~1+RT, R~1+RT)"] = diffw(s_removed(eeg, evts, @formula(0 ~ 1 + rt), @formula(0 ~ 1 + rt)))
    out["residual (S~1, R~1)"] = diffw(residual(eeg, evts))
    out
end

results = Dict()
for case in (:components, :ramp, :both), (j, vp) in enumerate(vps)
    results[(case, vp)] = one(vp, case, 1000j)
    println(case, " ", vp); flush(stdout)
end
serialize(joinpath(root, "results", "oconnell_extension.jls"), results)
println("done")
