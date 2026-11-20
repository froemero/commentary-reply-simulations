# Validation: UnfoldSim `KellyModel` vs the Kelly & O'Connell implementation

Reference implementations:
- O'Connell et al. (2025) MATLAB: `NHB_NA_2024_DriftModelSimulations.m` (Elisabeth-Pares/NHB_MA_OConnell_etal_2024)
- Frömer et al. (2024) Julia port: `analyses/nb_2023-09-09_driftrateUnfold.jl` (froemero/common_neural_choice_signals_emerge_artifactually)
- UnfoldSim.jl `feature-sequentialSamplingModels` branch, `src/sequentialSamplingModels.jl` (`KellyModel`)

The two reference implementations have the same trial loop; they differ only in parameter values.
`src/ReplySims.jl::kelly_port_trial` reimplements that loop.

| Aspect | Kelly / O'Connell code | UnfoldSim `KellyModel` |
|---|---|---|
| Sensory-encoding (evidence-onset) jitter | uniform, **full width** `Terz` (O'C 2025: 0.1 → SD 29 ms; NHB 2024: 0.1–0.2 → SD 29–58 ms) | `Normal(μ, σ)`, `sensor_encoding_delay_variability` is **σ** (default 0.1 = 100 ms) |
| Motor-time jitter | uniform, full width `st`, mean `mT` = 0.1 s | `Normal(0.4, σ)`, default σ = 0.1 |
| Accumulators | race of two DVs (DV1 = u1 + max(0, x), DV2 = u2 + max(0, −x)); errors possible | single accumulator, clamped at 0; no error responses |
| Start-point variability | at motor level: Z = 0.3 ± uniform(Sz = 0.4)/2, separate for each DV; not visible in the CPP | added to the evidence, σ = 0.2 truncated at the bound; visible in the CPP (CPP starts > 0 before onset) |
| Effective bound | 1 − Z ≈ 0.7 at t = 0 | 1 at t = 0 |
| Urgency | U + N(0, Su) per DV (O'C 2025: U = 1; NHB: U = 0.5), Su = 0.17 | truncated Normal(1, 0.17), lower = 0; setting `urgency = 0` alone still gives a half-normal slope unless the variability is also 0 |
| Post-decision accumulation | 0.1 + U(0, 0.2) s | truncated Normal(0.1, 0.001) s |
| CPP | abs(cumulative differential evidence), linear ramp-down over 0.1 s | same |

Consequences:
1. Values in Supp. Fig. 2 axes are σ of a Normal; the earlier simulations (NHB 2024 Supp. Fig. 5; O'Connell 2025) used uniform widths. Converting: σ = width/√12, so NHB 0.1/0.2 widths are σ 0.029/0.058 and O'Connell's 0.1/0.1 is σ 0.029/0.029. The red boxes in the original Supp. Fig. 2 (at σ 0.1/0.2) therefore sit ~3.5× too far out.
2. Default `KellyModel` RTs are slower and more variable (median 0.72 s, SD 135 ms) than the Kelly/O'Connell settings (median 0.37–0.40 s, SD 46–83 ms); see `results/validation_behaviour.csv`, `figures/validation_cpp_traces.png`.
3. With means/spreads matched (motor mean 0.1, uniform widths, no start-point variability, post-acc U(0.1, 0.3)), `KellyModel` reproduces the reference's RT spread (SD 45 vs 46 ms) with ~40 ms longer RTs (bound 1 vs ≈0.7) and a slightly larger CPP peak.
