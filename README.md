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

The extension in this repository explores what follows from the original toy model rather than claiming to prove Lacanian theory.

### Surplus jouissance as an operational hypothesis

The original paper does not formalize surplus jouissance (plus-de-jouir). The exploratory extension introduces three measurable quantities:

- **Preference mismatch:** the KL divergence between the current posterior and the current preference;
- **Expected-free-energy floor:** the minimum expected free energy over the finite policy set;
- **Inter-subjective Symbolic gap:** the KL divergence between two agents' Symbolic posteriors.

These are operational proxies and should not be interpreted as identities between information-theoretic quantities and Lacanian concepts.

### Corrected v2 experiments

`deep_experiments_v2.py` contains the current corrected experiments:

1. **Symbolic-coupling ablation:** examines how changing R/S/I coupling gains affects preference mismatch and trajectory stability.
2. **Long-range dyadic synchronization:** evaluates Symbolic synchronization over 100 steps and 20 random seeds with isolated environments.
3. **Expected-free-energy decomposition:** separates the observation-entropy term from the preference-divergence term while varying likelihood sharpness.
4. **Triadic collective dynamics:** tests whether Symbolic coordination persists when every agent has its own R/S/I environments.

The main operational observations are:

- removing Symbolic coupling increases mismatch variability and trajectory length under the current parameters;
- dyadic Symbolic synchronization is a stable attractor in the current model configuration;
- observation noise produces a positive expected-free-energy floor in the finite-state model, although this is not evidence for surplus jouissance by itself;
- a cyclic triadic desire structure can produce Symbolic-level collective convergence while Real and Imaginary states remain heterogeneous.

The strongest claim supported by the current experiments is therefore:

> Under the specified active-inference dynamics, state-dependent Symbolic coupling can generate a stable collective Symbolic attractor in isolated environments.

This is a computational analogue compatible with an interpretation of the Other as emergent Symbolic coordination. It is not a clinical model, a proof of Lacanian psychoanalysis, or a standard implementation of all aspects of FEP.

## Reproduction

The experiments are currently organized for WSL. If the repository is available at `/mnt/d/vibecoding/ActiveInferenceLacan`, run:

```bash
python deep_experiments_v2.py
```

The script is self-contained and does not require `pymdp`. It prints experiment summaries and writes the v2 figures:

- `plot_v2_exp1_ablation.png`
- `plot_v2_exp2_multiseed.png`
- `plot_v2_exp3_decomposed.png`
- `plot_v2_exp4_triadic.png`

The Chinese experiment log `实验记录与观察.md` contains the v1/v2 comparison and the rationale for the corrections.

## Limitations and next steps

- The model uses small discrete state spaces and hand-designed likelihood and transition matrices.
- The coupling update is a model-specific operational rule, not a complete derivation from canonical active inference.
- Dyadic and triadic interactions are currently sequential/asynchronous; a synchronous-update control is still needed.
- Directly setting an agent's Symbolic preference to another agent's state makes synchronization an explicit property of the controller. No-coupling, one-way-coupling, delayed-coupling, and heterogeneous-agent controls are needed before making stronger claims about emergence.
- The surplus-jouissance measures remain theoretical hypotheses requiring further formal and empirical development.

## Original files

`agent.py`, `cofig.py`, and `simulations.py` contain the original prototype implementation. The v1 exploratory scripts are retained for comparison, but `deep_experiments_v2.py` is the recommended entry point for the corrected experiments.
