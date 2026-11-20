# commentary-reply-simulations

Simulations for the reply to O'Connell et al. (2025, *Imaging Neuroscience*): how much of a fast-vs-slow
(RT median split) difference in a CPP-like accumulation signal survives classic averaging,
deconvolution (Unfold), and residual averaging, as a function of trial-to-trial variability in the
sensory-encoding delay (before accumulation) and the motor delay (after the bound).

## Contents
| Path | What |
|---|---|
| `src/ReplySims.jl` | Model wrappers (UnfoldSim `KellyModel`; reimplementation of the Kelly & O'Connell trial loop), continuous-signal construction, the three analyses, `res_norm` |
| `scripts/01_reproduce_suppfig2.jl` | Grid of Supp. Fig. 2: σ ∈ {0, .1, .2, .3} s × urgency off/on (KellyModel, Normal delays) |
| `scripts/02_realistic_grid.jl` | Finer grid σ ∈ {0, .01, .02, .03, .05, .075, .1} s for KellyModel (Normal), KellyModel (uniform), Kelly/O'Connell code |
| `scripts/03_validate_kelly.jl` | Behaviour + mean CPP of KellyModel vs the Kelly/O'Connell implementation |
| `scripts/04_figures.jl` | Figures from `results/*.csv` |
| `docs/validation.md` | Parameter-by-parameter comparison of the implementations |
| `results/`, `figures/` | Outputs |

## Analysis (per grid cell)
- 500 trials, 100 Hz, 2 s trace per trial, trials concatenated with 0–0.5 s uniform gaps; events S (trace onset) and R (RT).
- RT median split → fast/slow.
- **epoch**: S-locked [−0.2, 1.6] s and R-locked [−1.6, 0.4] s averages; metric = Σ_t |fast − slow| summed over both windows.
- **deconv**: Unfold FIR, `0 ~ 1 + rtgroup` for S and R (same windows); metric from the fast/slow effects.
- **residual**: Unfold FIR intercept-only for S and R; residuals epoched and averaged as in *epoch*.
- `res_norm` = metric / epoch metric. Mean over 3 seeds.

Assumptions that may differ from the original Supp. Fig. 2 code (which was not available here): the windows, sampling rate, ITI, N, and seed count above; KellyModel defaults otherwise (drift 6, motor mean 0.4 s, start-point σ 0.2). "No urgency" = urgency slope fixed at 0 (urgency and its variability both 0).

## Run
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/01_reproduce_suppfig2.jl   # ~1 min
julia --project=. scripts/02_realistic_grid.jl       # ~5 min
julia --project=. scripts/03_validate_kelly.jl
julia --project=. scripts/04_figures.jl
```
UnfoldSim comes from the `feature-sequentialSamplingModels` branch (pinned in `Project.toml` `[sources]`, Julia ≥ 1.11).
`LocalPreferences.toml` disables MixedModels' precompile workload (only needed where its test-data artifact can't be downloaded).

## Empirical variability values used for the markers
- Sensory-encoding delay SD ≈ 15 ms (range 0–~30 ms): N200 peak-latency covariance method, Nunez et al. 2019 data.
- Motor time SD ≈ 25 ms (17–46 ms): EMG onset-to-response, Weindel et al. 2021 data.
- Uniform widths in earlier simulations converted to SD (width/√12): NHB 2024 Supp. Fig. 5 (σ 0.029/0.058), O'Connell et al. 2025 (σ 0.029/0.029).
