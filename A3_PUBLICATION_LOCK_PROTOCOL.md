# A3 Publication-Lock Protocol

Version: `A3-v1.0`
Status: frozen before the confirmatory archive was generated.

This protocol defines the confirmatory observer-memory experiment. It is an implementation-level specification for the toy model, not a claim about Lacanian theory.

## Model and fixed design

- Nine discrete states (`0`–`8`) and three agents with Real, Symbolic, and Imaginary components.
- Social observation sharpness: `sigma = 0.40`.
- Primary environment: `env_init = [5, 2, 6]`.
- Robustness environment: `env_init = [8, 8, 8]`.
- Trajectory length: `T = 200` rounds.
- Confirmatory seeds: `1000..1199` (200 seeds).
- Update schedules: `sync` and sequential `seq_ABC`.
- Observer conditions: `hard`, `smooth`, and recursive `filter` with `lambda ∈ {0.00, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00}`.
- Robustness grid: `hard`, `lambda=0.00`, `lambda=0.20`, and `lambda=1.00` in the robustness environment.

The observer update is

```text
d_t = (1 - lambda) q_t + lambda * uniform
```

where `q_t` is the posterior over the observed agent's current Symbolic position and `d_t` is the prior for the next observation. `lambda=0` is the static-memory limit; `lambda=1` is the memoryless limit. `lambda` is a forgetting/volatility parameter, not desire strength or clinical structure.

## Randomness and schedule control

Each seed creates one keyed common-random-number stream. The same social-observation uniforms and action uniforms are reused across observer conditions and schedules. The schedules differ only in their information set: `sync` uses a round-start snapshot, whereas `seq_ABC` updates agents in A→B→C order.

## Endpoints and confirmatory contrasts

For each round, alignment is the indicator that all three Symbolic observations are equal. The primary endpoint is mean alignment occupancy over rounds `100..199` (the second half). `final20_aligned` is a separate uninterrupted-lock endpoint for rounds `180..199`.

The prespecified primary contrast is the per-seed schedule average:

```text
Delta_memory = occupancy(lambda=0) - occupancy(lambda=1)
```

The principal schedule contrast is the paired hard-observer difference:

```text
Delta_schedule = occupancy(seq_ABC) - occupancy(sync)
```

Report effect estimates before p-values, with seed-cluster percentile-bootstrap 95% confidence intervals. The primary test is a paired sign-flip Monte Carlo test; secondary contrasts use the frozen correction procedure in `publication_lock_a3.py`. The robustness environment is reported separately from the primary inferential grid.

## Canonical archive and integrity

Run from the repository root with:

```bash
python3 publication_lock_a3.py pipeline
```

The canonical archive is:

- `publication_lock_a3_results.csv`;
- `publication_lock_a3_summary.csv`;
- `publication_lock_a3_timeseries.npz`;
- `publication_lock_a3_manifest.json`;
- `publication_lock_a3_statistics.md`;
- `publication_lock_a3_report.md`;
- `publication_lock_a3_regression.md`.

The manifest records SHA-256 hashes for the A3 script, its A2 and factorial dependencies, the merged results, the summary, the NPZ, and the derived reports. Batch shards and recovery outputs are working files and are not part of the release.

## Interpretation boundary

The experiment supports only model-internal statements about Symbolic alignment occupancy, schedule sensitivity, observer memory, and position-estimation error. It does not establish the Lacanian Big Other, desire, clinical efficacy, or any empirical claim about patients or human participants.
