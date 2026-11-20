# Real data (Frömer et al. 2018, "UCAP"): CPz at 100 Hz, correct trials, condition = blurred vs normal stimuli.
# Per participant: RT difference; response-locked build-up rate (−250 to −50 ms) for blurr vs normal with
# classic averaging (epoch), deconvolution (0 ~ 1 + n_b, stimulus [−0.2, 1.2] s, response [−0.8, 0.4] s) and
# residual averaging (intercept-only deconvolution); also the intercept-only stimulus/response kernels and the
# residual noise, used for the semi-synthetic simulations.
# Input: data_local/ucap_jl/<vp>_cpz.f32, <vp>_events.csv, <vp>_boundary.f64 (made by data_local/extract_ucap.py).
using Unfold, DataFrames, CSV, Statistics, HypothesisTests, Serialization
dir = get(ENV, "UCAP_DIR", "/home/claude/ucap"); fs = 100
τS = (-0.2, 1.2); τR = (-0.8, 0.4)
slope(t, y; w = (-0.25, -0.05)) = (ix = (t .>= w[1]) .& (t .<= w[2]); X = [ones(sum(ix)) t[ix]]; (X \ y[ix])[2])
function avg(sig, ev, which, τ)
    tbl = ev[ev.event .== which, :]
    e, t = Unfold.epoch(data = sig, tbl = tbl, τ = τ, sfreq = fs)
    ok = .!vec(any(ismissing.(e), dims = (1, 2))); e = coalesce.(e[:, :, ok], 0.0); tb = tbl[ok, :]
    Dict(c => (collect(t), vec(mean(e[1, :, tb.n_b .== c], dims = 2))) for c in ("blurr", "normal"))
end
vps = sort(unique(first.(split.(filter(f -> endswith(f, "_cpz.f32"), readdir(dir)), "_"))))
rows = []; kernels = Dict{String,Any}()
for vp in vps
    x = Float64.(reinterpret(Float32, read(joinpath(dir, "$(vp)_cpz.f32"))))
    x .-= mean(x)
    ev = CSV.read(joinpath(dir, "$(vp)_events.csv"), DataFrame; missingstring = ["nan", "NaN"])
    ev = ev[(ev.err .== "right") .& .!ismissing.(ev.resp), :]
    bnd = reinterpret(Float64, read(joinpath(dir, "$(vp)_boundary.f64")))
    keep = [all(abs.(s .- bnd) .> 300) for s in ev.stim]; ev = ev[keep, :]
    rt = (ev.resp .- ev.stim) ./ fs
    S = DataFrame(latency = ev.stim .+ 1, event = "S", n_b = ev.n_b, rt = rt)
    R = DataFrame(latency = ev.resp .+ 1, event = "R", n_b = ev.n_b, rt = rt)
    evts = sort(vcat(S, R), :latency)
    ep = avg(x, evts, "R", τR)
    f = @formula(0 ~ 1 + n_b)
    m = fit(UnfoldModel, ["S" => (f, firbasis(τ = τS, sfreq = fs)), "R" => (f, firbasis(τ = τR, sfreq = fs))], evts, x; eventcolumn = :event, show_progress = false)
    eff = effects(Dict(:n_b => ["blurr", "normal"]), m); effR = eff[eff.eventname .== "R", :]
    dc = Dict(c => (effR[effR.n_b .== c, :time], effR[effR.n_b .== c, :yhat]) for c in ("blurr", "normal"))
    f0 = @formula(0 ~ 1)
    m0 = fit(UnfoldModel, ["S" => (f0, firbasis(τ = τS, sfreq = fs)), "R" => (f0, firbasis(τ = τR, sfreq = fs))], evts, x; eventcolumn = :event, show_progress = false)
    res = vec(Unfold.residuals(m0, x))
    rs = avg(res, evts, "R", τR)
    c0 = coeftable(m0)
    kernels[vp] = (S = c0[c0.eventname .== "S", [:time, :estimate]], R = c0[c0.eventname .== "R", [:time, :estimate]],
                   epochR = ep, epochS = avg(x, evts, "S", τS), deconvR = dc, residR = rs,
                   stim = ev.stim, rt = rt, n_b = ev.n_b, resid_sd = std(res))
    open(joinpath(dir, "$(vp)_resid.f32"), "w") do io; write(io, Float32.(res)); end
    push!(rows, (; vp, n = nrow(ev), rt_blurr = median(rt[ev.n_b .== "blurr"]), rt_normal = median(rt[ev.n_b .== "normal"]),
                 epoch = slope(ep["blurr"]...) - slope(ep["normal"]...),
                 deconv = slope(dc["blurr"]...) - slope(dc["normal"]...),
                 residual = slope(rs["blurr"]...) - slope(rs["normal"]...),
                 cpp_peak_R = maximum(0.5 .* (ep["blurr"][2] .+ ep["normal"][2])), resid_sd = std(res)))
    println(vp, " done")
end
df = DataFrame(rows)
root = joinpath(@__DIR__, "..", "..")
CSV.write(joinpath(root, "results", "ucap_real_slopes.csv"), df)
serialize(joinpath(dir, "ucap_kernels.jls"), kernels)
println("ΔRT (blurr − normal) median across participants: ", round(1000 * median(df.rt_blurr .- df.rt_normal); digits = 1), " ms; mean ", round(1000 * mean(df.rt_blurr .- df.rt_normal); digits = 1))
for m in (:epoch, :deconv, :residual)
    v = df[!, m]; t = OneSampleTTest(v)
    println(rpad(m, 9), " mean ", round(mean(v); digits = 2), " µV/s  SD ", round(std(v); digits = 2), "  d_z ", round(mean(v) / std(v); digits = 2), "  p ", round(pvalue(t); digits = 4))
end
println("response-locked CPP peak (µV), median: ", round(median(df.cpp_peak_R); digits = 2), "; residual noise SD (µV), median: ", round(median(df.resid_sd); digits = 2))
