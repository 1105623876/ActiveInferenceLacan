# ActiveInferenceLacan: Reproduction and Mechanism Audit

This repository is a mechanism-audit continuation of the Lacan–FEP toy model. It is not a republication of the original paper. The upstream work is Li & Li (2025), *Formalizing Lacanian Psychoanalysis through the Free Energy Principle*, together with the original [DigitalTwinMind/ActiveInferenceLacan](https://github.com/DigitalTwinMind/ActiveInferenceLacan) repository.

The continuation keeps the model deliberately small and makes its coordination mechanisms auditable: shared versus isolated environments, Symbolic preference coupling, direct versus inferred access to another agent's Symbolic position, and observer memory under synchronous versus sequential updating.

## Repository map

The repository root is intentionally kept short:

| Directory | Contents |
|---|---|
| `src/` | Core model, exploratory controls, observer audits, and the frozen A3 entry point |
| `outputs/a3/` | Canonical A3 CSV/NPZ archive, manifest, and A3 reports |
| `outputs/*/` | Experiment-specific data, figures, and reports; each experiment has its own folder |
| `outputs/a3/shards/` | Local A3 batch/recovery shards; ignored by Git and not part of the release |
| `docs/protocol/` | Frozen A3 protocol |
| `docs/roadmap/` | Future opaque-other coupling plan |
| `docs/analysis/` | Analysis notes, claims matrices, and final synthesis |
| `docs/notes/` | Working research notes |
| `archive/v1/` | Historical v1 scripts; their paths now resolve through `src/` and `outputs/legacy/` |
| `paper/` | Local RC 0.2 manuscript package; ignored by Git and not uploaded |

The code and outputs are grouped by function rather than left as root-level files. The canonical public archive is under `outputs/a3/`; the earlier exploratory and control experiments remain available under their corresponding `outputs/` directories.

## Scope and non-claims

The results are computational observations in a fixed nine-state, three-agent toy model. They do not prove or measure the Lacanian Big Other, desire, jouissance, or clinical validity. The observer estimates another agent's discrete Symbolic position; it does not infer that agent's desire.

Both update controls are present in the frozen A3 release: `sync` and `seq_ABC`. The earlier statement that a synchronous control was missing no longer applies.

## Frozen A3 release

The public confirmatory evidence layer is the A3-v1.0 archive in `outputs/a3/`:

- `src/publication_lock_a3.py` — reproducibility entry point;
- `outputs/a3/publication_lock_a3_results.csv` — 5,200 trajectory-level records;
- `outputs/a3/publication_lock_a3_summary.csv` — 26 frozen cell summaries;
- `outputs/a3/publication_lock_a3_timeseries.npz` — merged per-round time series;
- `outputs/a3/publication_lock_a3_statistics.md`, `publication_lock_a3_report.md`, and `publication_lock_a3_regression.md` — derived report and regression checks;
- `outputs/a3/publication_lock_a3_manifest.json` — SHA-256 manifest for the canonical A3 code and outputs.

The A3 archive contains the two main effects used by the manuscript:

| Effect | Estimate | 95% CI |
|---|---:|---:|
| Memory contrast, `lambda=0 - lambda=1`, schedule-averaged second-half occupancy | 0.6599 | [0.6279, 0.6906] |
| Hard-observer schedule contrast, `seq_ABC - sync`, second-half occupancy | 0.7639 | [0.7000, 0.8239] |

## Reproduce or verify A3

The archived run used Python 3.10.6 and NumPy 2.2.6. From the repository root, the full entry point is:

```bash
python3 src/publication_lock_a3.py pipeline
```

This command reads or resumes the A3 archive and regenerates the analysis outputs under `outputs/a3/`. The complete grid is 5,200 trajectories × 200 rounds, so a clean rerun is substantially more expensive than checking the committed archive. Do not use the batch driver or `pipeline` merely to inspect the release.

For a checksum-only inspection of the committed A3 archive:

```bash
shasum -a 256 \
  src/publication_lock_a3.py \
  outputs/a3/publication_lock_a3_results.csv \
  outputs/a3/publication_lock_a3_summary.csv \
  outputs/a3/publication_lock_a3_timeseries.npz \
  outputs/a3/publication_lock_a3_statistics.md \
  outputs/a3/publication_lock_a3_report.md \
  outputs/a3/publication_lock_a3_regression.md
```

Data availability: the canonical archive is identified by the GitHub tag `rc-0.2` and the commit pointed to by that tag. Batch shards, smoke outputs, recovery logs, caches, and local TeX build products are intentionally excluded from GitHub.

## Exploratory and precursor material

The scripts in `src/` other than `publication_lock_a3.py` document precursor controls and exploratory analyses. Their generated data, figures, and reports are grouped under `outputs/` by experiment. They are not substitutes for the frozen A3 protocol. The A3 files listed above are the only files used for the confirmatory memory and schedule contrasts.

## Local manuscript package

The complete RC 0.2 manuscript, supplementary material, figures, table generators, and LaTeX build are kept locally under `paper/`. `paper/` is listed in `.gitignore` by design and is not part of the GitHub upload. Its local release notes retain the status `ANALYZED`, not `VERIFIED`, until an independent clean-environment full rerun is completed.

## Attribution

Li, L. and Li, C. (2025). *Formalizing Lacanian Psychoanalysis Through the Free Energy Principle*. Frontiers in Psychology, 16, 1574650. [DOI](https://doi.org/10.3389/fpsyg.2025.1574650).

The upstream prototype is in `src/agent.py`, `src/cofig.py`, and `src/simulations.py`; this repository preserves that attribution and labels the continuation's controls and A3 archive separately.
