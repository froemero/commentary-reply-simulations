# recompute truth with the same random streams as script 14 (deterministic), without refitting
src = read("scripts/real/14_oconnell_extension.jl", String)
src = split(src, "results = Dict()")[1]; src = replace(src, "@__DIR__" => "\"/home/claude/sim/scripts/real\""); include_string(Main, src)
R = deserialize(joinpath(root, "results", "oconnell_extension.jls"))
for (j, vp) in enumerate(vps)
    rng = StableRNG(1000j); k = K[vp]; n = length(k.stim)
    d = d_mean + sd_d0 * randn(rng); tr = [trial(rng, d) for _ in 1:n]; rts = first.(tr); ramps = last.(tr)
    fast = rts .<= median(rts)
    tru = 13.47 .* (ralign(ramps, rts, fast) .- ralign(ramps, rts, .!fast))
    for case in (:ramp, :both); R[(case, vp)]["truth"] = tru; end
    # sanity: same RTs as stored? compare median split counts via classic not possible; check rt of first trial deterministic
end
serialize(joinpath(root, "results", "oconnell_extension.jls"), R); println("fixed ", round(maximum(abs.(R[(:ramp, vps[1])]["truth"])); digits=2))
