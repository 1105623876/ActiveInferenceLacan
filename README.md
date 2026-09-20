# ActiveInferenceLacan: Reproduction and Mechanism Audit

This repository is a mechanism-audit continuation of the Lacan–FEP toy model. It is not a republication of the original paper. The upstream work is Li & Li (2025), *Formalizing Lacanian Psychoanalysis through the Free Energy Principle*, together with the original [DigitalTwinMind/ActiveInferenceLacan](https://github.com/DigitalTwinMind/ActiveInferenceLacan) repository.

The continuation keeps the model deliberately small and makes its coordination mechanisms auditable: shared versus isolated environments, Symbolic preference coupling, direct versus inferred access to another agent's Symbolic position, and observer memory under synchronous versus sequential updating.

## Scope and non-claims

The results are computational observations in a fixed nine-state, three-agent toy model. They do not prove or measure the Lacanian Big Other, desire, jouissance, or clinical validity. The observer estimates another agent's discrete Symbolic position; it does not infer that agent's desire.

Both update controls are present in the frozen A3 release: `sync` and `seq_ABC`. The earlier statement that a synchronous control was missing no longer applies.

## Frozen A3 release

The public, confirmatory evidence layer is the A3-v1.0 archive:

- `A3_PUBLICATION_LOCK_PROTOCOL.md` — frozen design, endpoint, schedules, observer family, and analysis plan;
- `publication_lock_a3.py` — reproducibility entry point;
- `publication_lock_a3_results.csv` — 5,200 trajectory-level records;
- `publication_lock_a3_summary.csv` — 26 frozen cell summaries;
- `publication_lock_a3_timeseries.npz` — merged per-round time series;
- `publication_lock_a3_statistics.md`, `publication_lock_a3_report.md`, and `publication_lock_a3_regression.md` — derived report and regression checks;
- `publication_lock_a3_manifest.json` — SHA-256 manifest for the canonical A3 code and outputs.

The A3 archive contains the two main effects used by the manuscript:

| Effect | Estimate | 95% CI |
|---|---:|---:|
| Memory contrast, `lambda=0 - lambda=1`, schedule-averaged second-half occupancy | 0.6599 | [0.6279, 0.6906] |
| Hard-observer schedule contrast, `seq_ABC - sync`, second-half occupancy | 0.7639 | [0.7000, 0.8239] |

## Reproduce A3

The archived run used Python 3.10.6 and NumPy 2.2.6. From the repository root, install those dependencies in the environment of your choice and run:

```bash
python3 publication_lock_a3.py pipeline
```

The command resumes from the committed canonical archive and regenerates the A3 analysis outputs. It should recover the two estimates and confidence intervals above. The A3 script imports `factorial_env_symbolic.py` and `opaque_other_observer_schedule.py`; both are included as frozen upstream-control dependencies. The complete grid is 5,200 trajectories × 200 rounds, so a clean rerun is substantially more expensive than checking the committed archive.

For a direct checksum check of the released data and reports:

```bash
shasum -a 256 \
  publication_lock_a3.py \
  publication_lock_a3_results.csv \
  publication_lock_a3_summary.csv \
  publication_lock_a3_timeseries.npz \
  publication_lock_a3_statistics.md \
  publication_lock_a3_report.md \
  publication_lock_a3_regression.md
```

Data availability: the canonical archive is identified by the GitHub tag `rc-0.2` (and the commit pointed to by that tag). Batch shards, smoke outputs, recovery logs, caches, and local TeX build products are intentionally excluded.

## Exploratory and precursor material

The root-level `factorial_*`, `opaque_other_*`, `coupling_*`, `delayed_initialization_*`, `env_init_*`, and `deep_experiments_v2.py` files document precursor controls and exploratory analyses. They are not substitutes for the frozen A3 protocol. The A3 files listed above are the only files used for the confirmatory memory and schedule contrasts.

## Local manuscript package

The complete RC 0.2 manuscript, supplementary material, figures, table generators, and LaTeX build are kept locally under `paper/`. `paper/` is listed in `.gitignore` by design and is not part of the GitHub upload. Its local release notes retain the status `ANALYZED`, not `VERIFIED`, until an independent clean-environment full rerun is completed.

## Attribution

Li, L. and Li, C. (2025). *Formalizing Lacanian Psychoanalysis Through the Free Energy Principle*. Frontiers in Psychology, 16, 1574650. [DOI](https://doi.org/10.3389/fpsyg.2025.1574650).

The upstream prototype remains in `agent.py`, `cofig.py`, and `simulations.py`; this repository preserves that attribution and labels the continuation's controls and A3 archive separately.
