# An Active-Inference Model of Lacanian Psychoanalysis

This repository contains a small computational prototype inspired by Lacanian psychoanalysis and the free energy principle (FEP). It should be read as a conceptual toy model rather than as a validated neural, clinical, or canonical FEP implementation.

The original model represents Lacan's three registers as three coupled active-inference units:

- **Real (R):** bodily and affective dynamics;
- **Symbolic (S):** language-like and socially mediated states;
- **Imaginary (I):** perceptual and self-image dynamics.

At the individual level, prediction-error coupling between R, S, and I is used as a computational analogue of their Borromean interdependence. At the dyadic level, an agent's Symbolic preference is set by the other agent's Symbolic state, providing a toy implementation of desire as generalized synchronization. At the triadic level, a cyclic desire structure is used to study collective Symbolic dynamics as a possible computational analogue of the Lacanian Other.

<p align="center">
  <img src="RGM.png" width="600" alt="Recurrent generative model">
</p>

<p align="center">
  <img src="WFG.png" width="400" alt="Free-energy coupling diagram">
</p>

## Reference

The original model is based on:

> Li, L. and Li, C. (2025). *Formalizing Lacanian Psychoanalysis through the Free Energy Principle*. Frontiers in Psychology, 16, 1574650.

- [Paper](https://doi.org/10.3389/fpsyg.2025.1574650)
- [Original repository](https://github.com/DigitalTwinMind/ActiveInferenceLacan)

```bibtex
@article{li2025formalizing,
  title   = {Formalizing Lacanian Psychoanalysis Through the Free Energy Principle},
  author  = {Li, Lingyu and Li, Chunbo},
  journal = {Frontiers in Psychology},
  volume  = {16},
  pages   = {1574650},
  year    = {2025},
  doi     = {10.3389/fpsyg.2025.1574650}
}
```

## This continuation

The extension in this repository explores what follows from the original toy model rather than claiming to prove Lacanian theory. The current synthesis is the Chinese report `FINAL_REPORT.md`; allowed and disallowed claims are listed in `paper_claims_matrix_v2.md`.

After correcting the original implementation, later experiments unconfound shared environment from inter-agent preference coupling, compare several coupling rules, and replace direct copying with a filtered belief about the other's Symbolic **position** (not their desire):

```text
S_j  →  noisy social observation  →  q(s_j)  →  C_i
```

The strongest claims supported by the current experiments are:

1. Inter-agent Symbolic preference coupling is the primary driver of Symbolic consensus in this model; hard copying is not necessary.
2. A shared physical environment is only a weak alternative pathway.
3. When the other is visible only through a noisy channel, a filtered belief still produces consensus.
4. Under high opacity, agents can share a Symbolic coordinate while still misidentifying one another's position after they have aligned.

These are operational computational analogues. They are not a clinical model, a proof of Lacanian psychoanalysis, or a canonical FEP implementation. Surplus-jouissance proxies (preference mismatch, expected-free-energy floor, inter-subjective Symbolic gap) remain hypotheses and are **not** identified with plus-de-jouir.

## Reproduction

From the repository root (Windows or WSL):

```bash
python deep_experiments_v2.py          # corrected v2 baseline (no pymdp)
python factorial_env_symbolic.py       # shared/isolated × coupling on/off
python coupling_mechanism_control.py   # off / hard / soft / delayed / noisy
python opaque_other_coupling.py phase1 # inferred-other position coupling
```

`deep_experiments_v2.py` is self-contained and writes the v2 figures. The Chinese log `实验记录与观察.md` records the v1/v2 corrections.

## Limitations and next steps

- Discrete 9-state spaces and hand-designed A/B matrices.
- Coupling rules are model-specific, not derived from canonical active inference.
- Updates are sequential (A→B→C); a synchronous control is still missing.
- Inferred-other coupling tracks **where** the other is, not **what** the other wants.
- Symbolic states have no linguistic content.
- Surplus-jouissance measures are not a formalization of plus-de-jouir.

## Original files

`agent.py`, `cofig.py`, and `simulations.py` contain the original prototype. v1 scripts are archived for comparison. Read `FINAL_REPORT.md` for the current argument.
