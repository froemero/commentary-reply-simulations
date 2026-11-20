"""
ReplySims

Simulations for the reply to O'Connell et al. (2025): evidence-accumulation (CPP-like) signals with
trial-to-trial variability in the delay before accumulation (sensory encoding) and after it
(motor execution), analysed with classic averaging ("epoch"), deconvolution with conditions
("deconv"), and averaging of residuals from an intercept-only deconvolution ("residual").
"""
module ReplySims

using UnfoldSim, Unfold, Distributions, StableRNGs, Random, DataFrames, Statistics

export run_cell_port, kelly_model, kelly_port_trial, simulate_continuous, analyse, run_cell, sd_to_width, width_to_sd

const SQRT12 = sqrt(12)
"Full width of a uniform distribution with standard deviation `sd`."
sd_to_width(sd) = sd * SQRT12
"Standard deviation of a uniform distribution with full width `w`."
width_to_sd(w) = w / SQRT12

"""
    kelly_model(; sens_sd, motor_sd, urgency=true, dist=:normal, drift_rate=6.0, kwargs...)

UnfoldSim `KellyModel` (feature-sequentialSamplingModels branch) with the two delay
variabilities given as standard deviations in seconds. `dist=:normal` uses the package default
(Normal with σ = sd, as in the reply's Supplementary Figure 2); `dist=:uniform` uses a uniform
distribution with the same SD (width = sd·√12, as in the Kelly & O'Connell MATLAB code).
`urgency=false` sets the urgency slope to (numerically) zero for every trial.
"""
function kelly_model(; sens_sd, motor_sd, urgency::Bool = true, dist::Symbol = :normal,
                     drift_rate = 6.0, sensor_encoding_delay = 0.2, motor_delay = 0.4, kwargs...)
    onset = dist === :normal ? Normal(sensor_encoding_delay, sens_sd) :
            Uniform(sensor_encoding_delay - sd_to_width(sens_sd) / 2 - 1e-12,
                    sensor_encoding_delay + sd_to_width(sens_sd) / 2 + 1e-12)
    motor = dist === :normal ? Normal(motor_delay, motor_sd) :
            Uniform(motor_delay - sd_to_width(motor_sd) / 2 - 1e-12,
                    motor_delay + sd_to_width(motor_sd) / 2 + 1e-12)
    extra = urgency ? (;) : (; urgency = 0.0, urgency_variability = 0.0,
                             urgency_distribution = Uniform(0.0, 1e-12))
    return UnfoldSim.KellyModel(; drift_rate, sensor_encoding_delay, motor_delay,
                                sensor_encoding_delay_variability = sens_sd,
                                motor_delay_variability = motor_sd,
                                event_onset_distribution = onset,
                                motor_onset_distribution = motor, extra..., kwargs...)
end

"""
    kelly_port_trial(rng, timevec, dt; d=6, evonT=0.2, Terz=0.1, mT=0.1, st=0.1, Z=0.3, U=1.0,
                     Su=0.17, Sz=0.4, s=0.5, bound=1.0, postAccDurM=0.1, postAccDurS=0.2,
                     CPPrampdownDur=0.1)

One trial of the Kelly & O'Connell model as implemented in the MATLAB code of O'Connell et al.
(2025; NHB_NA_2024_DriftModelSimulations.m) and the Julia port in Frömer et al. (2024;
nb_2023-09-09_driftrateUnfold.jl): two racing decision variables with motor-level urgency and
start-point variability, uniform jitter (full widths `Terz`, `st`) on evidence onset and motor
time. Returns (rt in s, choice, CPP trace = |cumulative differential evidence|).
"""
function kelly_port_trial(rng, timevec, dt; d = 6.0, evonT = 0.2, Terz = 0.1, mT = 0.1, st = 0.1,
                          Z = 0.3, U = 1.0, Su = 0.17, Sz = 0.4, s = 0.5, bound = 1.0,
                          postAccDurM = 0.1, postAccDurS = 0.2, CPPrampdownDur = 0.1)
    evidence = zeros(length(timevec))
    evidence[timevec .>= (evonT + (rand(rng) - 0.5) * Terz)] .= 1
    startAccT = timevec[findfirst(==(1), evidence)]
    u1 = Z .+ (U + randn(rng) * Su) .* timevec .+ (rand(rng) - 0.5) * Sz
    u2 = Z .+ (U + randn(rng) * Su) .* timevec .+ (rand(rng) - 0.5) * Sz
    noise = vcat(zeros(sum(timevec .< startAccT)), randn(rng, sum(timevec .>= startAccT)) .* s .* sqrt(dt))
    cumdifev = cumsum(evidence .* d .* dt .+ noise)
    DV1 = u1 .+ max.(0, cumdifev)
    DV2 = u2 .+ max.(0, -cumdifev)
    t1 = findfirst(>(bound), DV1); t2 = findfirst(>(bound), DV2)
    if isnothing(t1) && isnothing(t2)
        return NaN, 2, abs.(cumdifev)
    end
    dti, ch = isnothing(t2) ? (t1, 1) : isnothing(t1) ? (t2, 0) : (min(t1, t2), t1 <= t2 ? 1 : 0)
    rt = timevec[dti] + mT + (rand(rng) - 0.5) * st
    accStop = dti + round(Int, (postAccDurM + postAccDurS * rand(rng)) / dt)
    cpp = abs.(cumdifev)
    if accStop < length(timevec)
        nT = length(timevec)
        cpp[(accStop + 1):end] .= max.(0, cpp[accStop] .- (1:(nT - accStop)) .* cpp[accStop] .* (dt / CPPrampdownDur))
    end
    return rt, ch, cpp
end

"""
    simulate_continuous(rng, trial; N=500, fs=100, trial_len=2.0, iti=(0.0, 0.5))

Concatenate `N` single-trial traces into a continuous signal. `trial(rng)` must return
(rt in s, trace sampled at `fs` of length trial_len·fs). Trials start every trial_len + U(iti)
seconds; only the simulated accumulation trace is present (no separate stimulus- or
response-locked components). Returns (eeg, events) with :S (stimulus) and :R (response) events.
"""
function simulate_continuous(rng, trial; N = 500, fs = 100, trial_len = 2.0, iti = (0.0, 0.5))
    L = round(Int, trial_len * fs)
    onsets = Int[]; rts = Float64[]; traces = Vector{Vector{Float64}}()
    t = 1
    for n in 1:N
        rt, tr = trial(rng)
        push!(onsets, t); push!(rts, rt); push!(traces, tr[1:L])
        t += L + round(Int, rand(rng, Uniform(iti...)) * fs)
    end
    eeg = zeros(t + L)
    for n in 1:N
        eeg[onsets[n]:(onsets[n] + L - 1)] .+= traces[n]
    end
    keep = .!isnan.(rts) .& (rts .< trial_len)
    grp = fill("slow", N); grp[rts .<= median(rts[keep])] .= "fast"
    S = DataFrame(latency = onsets, event = "S", rt = rts, rtgroup = grp)
    R = DataFrame(latency = onsets .+ round.(Int, replace(rts, NaN => 0.0) .* fs), event = "R", rt = rts, rtgroup = grp)
    evts = sort(vcat(S[keep, :], R[keep, :]), :latency)
    return eeg, evts
end

absdiff(a, b) = sum(abs.(a .- b))

"""
    analyse(eeg, evts, fs; τS=(-0.2, 1.6), τR=(-1.6, 0.4))

Fast-vs-slow difference (sum over time of |fast − slow|, summed over the stimulus- and
response-locked windows) for three approaches:
- `epoch`: classic stimulus- and response-locked averages;
- `deconv`: Unfold deconvolution with `0 ~ 1 + rtgroup` for both events (effects for fast/slow);
- `residual`: Unfold deconvolution with intercept-only models, residuals then epoched and averaged.
"""
function analyse(eeg, evts, fs; τS = (-0.2, 1.6), τR = (-1.6, 0.4))
    out = Dict{String,Float64}()
    sumdiff(epochs, tbl) = begin
        f = tbl.rtgroup .== "fast"; s = tbl.rtgroup .== "slow"
        absdiff(vec(mean(epochs[1, :, f], dims = 2)), vec(mean(epochs[1, :, s], dims = 2)))
    end
    function epoched(signal)
        tot = 0.0
        for (ev, τ) in (("S", τS), ("R", τR))
            tbl = evts[evts.event .== ev, :]
            ep, _ = Unfold.epoch(data = signal, tbl = tbl, τ = τ, sfreq = fs)
            ok = .!vec(any(ismissing.(ep), dims = (1, 2)))
            tot += sumdiff(coalesce.(ep[:, :, ok], 0.0), tbl[ok, :])
        end
        tot
    end
    out["epoch"] = epoched(eeg)
    # deconvolution with condition
    f = @formula(0 ~ 1 + rtgroup)
    des = ["S" => (f, firbasis(τ = τS, sfreq = fs)), "R" => (f, firbasis(τ = τR, sfreq = fs))]
    m = fit(UnfoldModel, des, evts, eeg; eventcolumn = :event, show_progress = false)
    eff = effects(Dict(:rtgroup => ["fast", "slow"]), m)
    tot = 0.0
    for ev in unique(eff.eventname)
        e = eff[eff.eventname .== ev, :]
        tot += absdiff(e[e.rtgroup .== "fast", :yhat], e[e.rtgroup .== "slow", :yhat])
    end
    out["deconv"] = tot
    # intercept-only deconvolution, then residuals
    f0 = @formula(0 ~ 1)
    m0 = fit(UnfoldModel, ["S" => (f0, firbasis(τ = τS, sfreq = fs)), "R" => (f0, firbasis(τ = τR, sfreq = fs))],
             evts, eeg; eventcolumn = :event, show_progress = false)
    res = vec(Unfold.residuals(m0, eeg))
    out["residual"] = epoched(res)
    return out
end

"""
    run_cell(seed; sens_sd, motor_sd, urgency, dist=:normal, N=500, fs=100)

Simulate one grid cell with the UnfoldSim KellyModel and return a NamedTuple with the three
raw metrics, their values normalised to `epoch` (res_norm) and RT summaries.
"""
function run_cell(seed; sens_sd, motor_sd, urgency, dist = :normal, N = 500, fs = 100, trial_len = 2.0, kwargs...)
    model = kelly_model(; sens_sd, motor_sd, urgency, dist, kwargs...)
    L = round(Int, trial_len * fs)
    trial(rng) = begin
        rt, tr = UnfoldSim.SSM_Simulate(rng, model, fs, L)
        (rt / fs, tr)
    end
    eeg, evts = simulate_continuous(StableRNG(seed), trial; N, fs, trial_len)
    m = analyse(eeg, evts, fs)
    rts = evts[evts.event .== "S", :rt]
    return (; seed, sens_sd, motor_sd, urgency, dist = String(dist), n = length(rts),
            rt_mean = mean(rts), rt_sd = std(rts),
            epoch = m["epoch"], deconv = m["deconv"], residual = m["residual"],
            deconv_norm = m["deconv"] / m["epoch"], residual_norm = m["residual"] / m["epoch"])
end

"""
    run_cell_port(seed; sens_sd, motor_sd, urgency, d=6.0, N=500, fs=100, kwargs...)

Same analysis as `run_cell`, but trials come from `kelly_port_trial` (the Kelly & O'Connell
two-accumulator implementation). Delay variabilities are given as SDs and converted to the
uniform full widths that model uses (width = SD·√12). `urgency=false` sets U = Su = 0.
"""
function run_cell_port(seed; sens_sd, motor_sd, urgency, d = 6.0, N = 500, fs = 100, trial_len = 2.0, kwargs...)
    dt = 1 / fs
    timevec = collect(0:dt:(trial_len - dt))
    uk = urgency ? (;) : (; U = 0.0, Su = 0.0)
    trial(rng) = begin
        rt, ch, cpp = kelly_port_trial(rng, timevec, dt; d, Terz = sd_to_width(sens_sd), st = sd_to_width(motor_sd), uk..., kwargs...)
        (rt, cpp)
    end
    eeg, evts = simulate_continuous(StableRNG(seed), trial; N, fs, trial_len)
    m = analyse(eeg, evts, fs)
    rts = evts[evts.event .== "S", :rt]
    return (; seed, sens_sd, motor_sd, urgency, dist = "uniform_port", n = length(rts),
            rt_mean = mean(rts), rt_sd = std(rts),
            epoch = m["epoch"], deconv = m["deconv"], residual = m["residual"],
            deconv_norm = m["deconv"] / m["epoch"], residual_norm = m["residual"] / m["epoch"])
end

end # module
