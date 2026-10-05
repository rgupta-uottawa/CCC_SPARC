# CCC vs SPARC rotation curves

Analysis pipeline for **"Testing Covarying Coupling Constants (CCC) against the
full SPARC rotation-curve sample: a like-for-like comparison with MOND and NFW"**
(Gupta et al., MNRAS).

This repository contains the complete pipeline used in the paper: spline
reconstruction of the observed rotation curve, the CCC density-turn-off model
(smooth and sharp), Monte-Carlo error propagation, the χ² minimisation, the MOND
and NFW comparison fits, and all figure- and table-generation scripts.

All fitting is done in the **inverse mode**: each model predicts the baryonic
rotation curve `V_b` from the observed curve `V_obs`, and the prediction is
compared against the SPARC baryonic curve.

## Installation

```bash
git clone https://github.com/<username>/ccc-sparc.git
cd ccc-sparc
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.10+ is required.

## Data

The SPARC data files are **not** redistributed here. Download them from
<http://astroweb.case.edu/SPARC/>:

- `SPARC_Lelli2016c_mrt.txt` (Table 1 — global galaxy properties)
- `SPARC_MassModels_Lelli2016c_mrt.txt` (Table 2 — rotation curves & mass models)

Place them in a `data/` directory at the repository root, or edit the two paths at
the top of `ccc_sparc/sparc_io.py`.

## Layout

| path | contents |
|------|----------|
| `ccc_sparc/` | Core modules and the analysis pipeline (import these). |
| `figures/`   | One script per paper figure. |
| `scripts/`   | Validation, robustness, and exploratory analyses. |

Key modules in `ccc_sparc/`:

| module | purpose |
|--------|---------|
| `sparc_io.py` | Load the two SPARC tables. |
| `ccc_core.py` | Density reconstruction: spline smoothing, differentiation to density, enclosed-mass integration, fixed-slope inner extrapolation. |
| `smooth_ccc.py` | The smooth CCC turn-off (`standard`/`simple`/`sharp`). |
| `regen_all.py` | **Master script** — fits all models to all 165 galaxies and writes the CSVs behind Table 3. |
| `btfr.py`, `morph_test.py`, `residuals_all.py` | BTFR, morphology, and residual analyses. |
| `density_relation.py` | Forward density-space RAR analogue (§5.1). |

## Reproducing the paper

```bash
# 1. fit all models, write the core CSVs
python ccc_sparc/regen_all.py

# 2. downstream analyses
python ccc_sparc/btfr.py
python ccc_sparc/morph_test.py
python ccc_sparc/residuals_all.py

# 3. figures
python figures/make_diag.py         # Fig. 1
python figures/regen_density.py     # Fig. 2
python figures/regen_paper7.py      # Fig. 3
python figures/chi2_cdf.py          # Fig. 4
python figures/regen_residuals.py   # Fig. 5
python figures/at_profile.py        # Fig. 6
python figures/plot_params.py       # Fig. 7
python figures/plot_btfr.py         # Fig. 8

# 4. Li et al. (2018) validation (Appendix B)
python scripts/litest.py
```

Output paths are set near the top of each script; adjust them for your environment.

## Method notes

- **Fixed inner slope**: the power-law inner-extrapolation slope is measured once
  per galaxy and held fixed during Monte-Carlo error propagation, so that it
  propagates observational noise rather than amplifying it.
- **Common error model**: MC-propagated reconstruction scatter + 5% `V_b` + 1 km/s
  floor, applied identically to every model.
- **Statistics**: means for the skewed χ²_ν; Spearman ρ with Benjamini–Hochberg
  FDR control; Theil–Sen for robust slopes.

## Citation

If you use this code, please cite the paper (reference / DOI to be added on
acceptance).

## License

MIT — see `LICENSE`.
