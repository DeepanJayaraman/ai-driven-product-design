# From Intuition to Intelligence: AI-Driven Product Design

Reproducibility package for the paper:

> Jayaraman, D., Kuruva, R., & Ramachandran, R. **From Intuition to Intelligence: A
> Simulation-Based Framework for AI-Driven Product Design in Early-Stage Ventures.**
> (Under review, *Journal of Business Venturing Insights*.)

**Repository:** https://github.com/DeepanJayaraman/ai-driven-product-design

This repository contains everything needed to reproduce **every number, table, and figure**
in the paper from scratch. The pipeline couples predictive machine learning, explainable AI,
and multi-objective optimization, demonstrated on a calibrated simulation of 500
product-development scenarios.

## What the code does

`simulation_code.py` runs the full pipeline end to end:

1. **Data generation** — 500 synthetic product records drawn from a utility-based acceptance
   process, with parameter ranges calibrated to published evidence (System Usability Scale
   norms; the meta-analytic price–quality correlation, r ≈ 0.24; positively skewed platform
   ratings; and a ~42% new-product failure rate).
2. **Predictive models** — product success (Random Forest, XGBoost), revenue (Random Forest
   on log-revenue), and customer rating (clipped Random Forest), on a stratified 70/30 split.
3. **Explainability** — permutation importance and partial-dependence plots.
4. **Multi-objective optimization** — Pareto screening over (price, feature complexity,
   development time).
5. **Benchmark** — Pareto-recommended configurations vs. the average sampled configuration.

Everything is deterministic with `SEED = 42`.

## Requirements

- Python 3.10+ (tested on 3.12)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Run

```bash
python simulation_code.py
```

## Expected outputs

Printed to stdout (held-out test set):

| Result | Value |
|---|---|
| Success base rate | 0.576 |
| Majority-class baseline | accuracy 0.576, F1 0.731 |
| Random Forest classifier | accuracy 0.660, F1 0.724, ROC-AUC 0.753 |
| XGBoost classifier | accuracy 0.607, F1 0.663, ROC-AUC 0.725 |
| Revenue regression | R² 0.367, MAE ≈ 410,600 |
| Rating regression | R² 0.257, MAE 0.379 |
| Permutation importance | price and development time dominant; quality near-zero (correlated-attribute artifact) |
| Pareto benchmark | +35 pts success, −61% development time, ≈ +2.6% revenue |

Files written to the working directory:

- `synthetic_dataset.csv` — the generated 500-record dataset (also included here for convenience)
- `fig_permutation_importance.png` — Figure 2
- `fig_partial_dependence.png` — Figure 3
- `fig_pareto.png` — Figure 4

(The conceptual-framework figure, Figure 1, is a schematic and is not produced by this script.)

> Note: minor last-digit differences across machines are possible due to BLAS/library builds,
> but the qualitative results and reported figures are stable.

## Files

- `simulation_code.py` — the full pipeline (single file, ~250 lines)
- `synthetic_dataset.csv` — precomputed dataset (regenerable from the script)
- `requirements.txt` — pinned dependencies
- `LICENSE` — MIT
- `CITATION.cff` — machine-readable citation metadata

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). Once the archived release
has a DOI, cite that DOI as well.

## License

MIT — see `LICENSE`.
