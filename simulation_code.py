"""
AI-Driven Product Design for Entrepreneurial Decision-Making
Reconstruction of the full simulation + modeling + optimization pipeline.

Pipeline (matches the manuscript):
  1. Generate N=500 synthetic product records from a utility-based acceptance process
  2. Predictive models: success (RF, XGBoost), revenue (RF on log), rating (RF, clipped)
  3. Explainability: permutation importance, partial dependence
  4. Multi-objective optimization: Pareto frontier over (price, complexity, dev time)
  5. Benchmark: Pareto-recommended configurations vs. average sampled configuration

Reproducible with SEED = 42.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
N = 500
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# 1. DATA GENERATION — ranges calibrated to published evidence
#
# Quality  [0.30, 0.95]: products that reach market are pre-screened; a
#   normalized objective-quality score near zero would not be launched.
#   Skewed toward decent quality (beta(5,3)).
# Usability [0.40, 0.90]: anchored to System Usability Scale norms —
#   mean ~68-70/100, SD ~12 (Bangor, Kortum & Miller, 2008; Sauro, 2011).
#   Sampled N(0.68, 0.125), clipped.
# Price coupled to quality with target correlation r ~ 0.27, the mean
#   objective price-quality correlation in Tellis & Wernerfelt (1987).
# Dev time 2-9 months, mode ~5: cycle times for low-complexity products
#   (Griffin, 1997). Triangular.
# Sentiment [-0.7, 0.7], positively skewed (mean ~ +0.15): online product
#   review sentiment skews positive (Hu, Pavlou & Zhang, 2009).
# Marketing spend 50-249k USD: assumed seed-stage range (no strong anchor;
#   stated as a modeling assumption).
# ----------------------------------------------------------------------
complexity = rng.choice(["Low", "Medium", "High"], size=N, p=[0.35, 0.4, 0.25])
quality = 0.30 + 0.70 * rng.beta(2.5, 2.0, N)            # [0.30, 1.00], mean ~0.69
q_norm = (quality - 0.30) / 0.70
price = 700 + (2583 - 700) * np.clip(0.24 * q_norm + 0.76 * rng.uniform(0, 1, N), 0, 1)
dev_time = rng.integers(2, 10, N).astype(float)          # 2-9 months (Griffin, 1997)
usability = np.clip(rng.normal(0.68, 0.15, N), 0.35, 0.95)
mktg = rng.uniform(50, 249, N)                           # k USD
sentiment = -0.7 + 1.4 * rng.beta(6, 4, N)

# normalizations used by the utility function
p_n = (price - 700) / (2583 - 700)
t_n = (dev_time - 2) / (9 - 2)
m_n = (mktg - 50) / (249 - 50)
delta_c = np.select([complexity == "Low", complexity == "Medium"], [0.15, 0.0], -0.2)

# latent utility and logistic acceptance; threshold calibrated so the
# simulated success rate (~60%) matches evidence that roughly 40% of
# launched new products fail (Castellion & Markham, 2013)
U = 1.2 * quality + 1.0 * usability + 0.6 * sentiment + 0.5 * m_n \
    - 1.0 * p_n - 0.8 * t_n + delta_c
p_accept = 1.0 / (1.0 + np.exp(-3.0 * (U - 0.8)))
success = rng.binomial(1, p_accept)

# revenue: price x demand, demand driven by acceptance & marketing, lognormal noise
units = 800 * p_accept * (1 + 0.6 * m_n) * rng.lognormal(0, 0.45, N)
revenue = price * units

# customer rating: quality/usability/sentiment plus idiosyncratic noise;
# intercept calibrated so mean lands near ~4, consistent with positively
# skewed online rating distributions (Hu, Pavlou & Zhang, 2009)
rating_core = 0.45 * quality + 0.35 * usability + 0.20 * (sentiment + 0.7) / 1.4
rating = np.clip(1.8 + 3.6 * rating_core + rng.normal(0, 0.50, N), 1, 5)

df = pd.DataFrame({
    "Complexity": complexity, "Price": price, "Dev_Time": dev_time,
    "Quality": quality, "Usability": usability, "Mktg_Spend": mktg,
    "Sentiment": sentiment, "p_accept": p_accept, "Success": success,
    "Rating": rating, "Revenue": revenue,
})

print("=== DATA AUDIT ===")
print(df.describe().round(3).to_string())
print(f"\nSuccess base rate: {success.mean():.3f}")
print(f"Majority-class accuracy (predict all success): {max(success.mean(), 1-success.mean()):.3f}")
print(f"Majority-class F1: {f1_score(success, np.ones(N)):.3f}")

# ----------------------------------------------------------------------
# 2. PREDICTIVE MODELS  (70/30 train-test split, stratified for classification)
# ----------------------------------------------------------------------
X = pd.get_dummies(df[["Complexity", "Price", "Dev_Time", "Quality",
                       "Usability", "Mktg_Spend", "Sentiment"]],
                   columns=["Complexity"])
Xtr, Xte, ytr, yte = train_test_split(X, success, test_size=0.3,
                                      random_state=SEED, stratify=success)

rf_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=SEED)
rf_clf.fit(Xtr, ytr)
xgb_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                        random_state=SEED, eval_metric="logloss")
xgb_clf.fit(Xtr, ytr)

print("\n=== SUCCESS CLASSIFICATION (held-out 30%) ===")
for name, m in [("RandomForest", rf_clf), ("XGBoost", xgb_clf)]:
    pred, proba = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    print(f"{name:12s}  Acc={accuracy_score(yte, pred):.3f}  "
          f"F1={f1_score(yte, pred):.3f}  AUC={roc_auc_score(yte, proba):.3f}")

# revenue (log-transformed target)
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, revenue, test_size=0.3, random_state=SEED)
rf_rev = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED)
rf_rev.fit(Xtr_r, np.log1p(ytr_r))
rev_pred = np.expm1(rf_rev.predict(Xte_r))
print("\n=== REVENUE REGRESSION ===")
print(f"R2={r2_score(yte_r, rev_pred):.3f}  MAE={mean_absolute_error(yte_r, rev_pred):,.0f}"
      f"  (mean revenue={revenue.mean():,.0f})")

# rating (regression baseline, clipped)
Xtr_g, Xte_g, ytr_g, yte_g = train_test_split(X, rating, test_size=0.3, random_state=SEED)
rf_rat = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED)
rf_rat.fit(Xtr_g, ytr_g)
rat_pred = np.clip(rf_rat.predict(Xte_g), 1, 5)
print("\n=== RATING REGRESSION ===")
print(f"R2={r2_score(yte_g, rat_pred):.3f}  MAE={mean_absolute_error(yte_g, rat_pred):.3f}")

# ----------------------------------------------------------------------
# 3. EXPLAINABILITY
# ----------------------------------------------------------------------
pi = permutation_importance(rf_clf, Xte, yte, n_repeats=30, random_state=SEED)
order = pi.importances_mean.argsort()[::-1]
print("\n=== PERMUTATION IMPORTANCE (success model) ===")
for i in order:
    print(f"{X.columns[i]:26s} {pi.importances_mean[i]:+.4f} ± {pi.importances_std[i]:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh([X.columns[i] for i in order[::-1]], pi.importances_mean[order[::-1]], color="#4878a8")
ax.set_xlabel("Mean importance (decrease in accuracy)")
ax.set_title("Permutation Importance - Product Success")
plt.tight_layout(); plt.savefig("fig_permutation_importance.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
PartialDependenceDisplay.from_estimator(rf_clf, Xte, ["Quality", "Usability", "Sentiment"], ax=ax)
plt.tight_layout(); plt.savefig("fig_partial_dependence.png", dpi=150); plt.close()

# ----------------------------------------------------------------------
# 4. MULTI-OBJECTIVE OPTIMIZATION (grid sampling + Pareto screening)
# ----------------------------------------------------------------------
grid = []
for cx in ["Low", "Medium", "High"]:
    for pr in np.linspace(700, 2583, 25):
        for dt in range(2, 10):
            grid.append((cx, pr, dt))
cand = pd.DataFrame(grid, columns=["Complexity", "Price", "Dev_Time"])
# non-decision attributes held at favorable-but-feasible levels (75th percentile)
for col, val in [("Quality", np.quantile(quality, 0.75)),
                 ("Usability", np.quantile(usability, 0.75)),
                 ("Mktg_Spend", np.quantile(mktg, 0.75)),
                 ("Sentiment", 0.0)]:
    cand[col] = val
Xc = pd.get_dummies(cand, columns=["Complexity"]).reindex(columns=X.columns, fill_value=0)
cand["p_hat"] = rf_clf.predict_proba(Xc)[:, 1]
cand["rev_hat"] = np.expm1(rf_rev.predict(Xc))
cand["cost_time"] = ((cand.Price - 700) / (2583 - 700) + (cand.Dev_Time - 2) / 7) / 2

def pareto_mask(cost, benefit):
    idx = np.argsort(cost.values)
    best, mask = -np.inf, np.zeros(len(cost), bool)
    for i in idx:
        if benefit.values[i] > best:
            mask[i], best = True, benefit.values[i]
    return mask

cand["pareto"] = pareto_mask(cand.cost_time, cand.p_hat)
front = cand[cand.pareto].sort_values("cost_time")
print("\n=== PARETO FRONTIER (recommended configurations) ===")
print(front[["Complexity", "Price", "Dev_Time", "p_hat", "rev_hat"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(cand.cost_time, cand.p_hat, s=8, alpha=0.35, label="All candidates")
ax.scatter(front.cost_time, front.p_hat, s=30, color="crimson", label="Pareto frontier")
ax.set_xlabel("Normalized cost+time (lower is better)")
ax.set_ylabel("Predicted success probability")
ax.set_title("Pareto Frontier: Cost/Time vs Acceptance"); ax.legend()
plt.tight_layout(); plt.savefig("fig_pareto.png", dpi=150); plt.close()

# ----------------------------------------------------------------------
# 5. BENCHMARK: recommended vs average sampled configuration
# ----------------------------------------------------------------------
knee = front.iloc[(front.p_hat - 0.9).abs().argsort()[:3]]   # high-acceptance, low-cost region
print("\n=== BENCHMARK: AI-recommended vs average configuration ===")
print(f"Average config : dev_time={df.Dev_Time.mean():.2f} mo, "
      f"success rate={df.Success.mean():.3f}, revenue={df.Revenue.mean():,.0f}")
print(f"Recommended    : dev_time={knee.Dev_Time.mean():.2f} mo, "
      f"pred. success={knee.p_hat.mean():.3f}, pred. revenue={knee.rev_hat.mean():,.0f}")
print(f"Delta          : dev_time {100*(1-knee.Dev_Time.mean()/df.Dev_Time.mean()):.1f}% shorter, "
      f"success +{100*(knee.p_hat.mean()-df.Success.mean()):.1f} pp, "
      f"revenue {100*(knee.rev_hat.mean()/df.Revenue.mean()-1):+.1f}%")

df.round(4).to_csv("synthetic_dataset.csv", index=False)
print("\nSaved: synthetic_dataset.csv, fig_permutation_importance.png, "
      "fig_partial_dependence.png, fig_pareto.png")
