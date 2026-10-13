"""Compare per-query AP of two methods (paired Wilcoxon signed-rank test).

Usage:
    python src/compare_ap.py results/figures/A_ap.csv results/figures/B_ap.csv
"""
import sys

import pandas as pd
from scipy.stats import wilcoxon

a_path, b_path = sys.argv[1], sys.argv[2]
a = pd.read_csv(a_path).dropna(subset=["ap"])
b = pd.read_csv(b_path).dropna(subset=["ap"])
df = a.merge(b, on=["songid", "tunefamily"], suffixes=("_a", "_b"))
diff = df["ap_b"] - df["ap_a"]

print(f"A: {a_path}\nB: {b_path}")
print(f"Paired queries: {len(df)}")
print(f"MAP A = {df['ap_a'].mean():.4f} | MAP B = {df['ap_b'].mean():.4f} | diff = {diff.mean():+.4f}")
print(f"B better: {(diff > 0).sum()} | B worse: {(diff < 0).sum()} | tied: {(diff == 0).sum()}")
print(f"Wilcoxon p-value: {wilcoxon(df['ap_a'], df['ap_b']).pvalue:.4g}")

# Families that gain or lose the most
fam = df.assign(diff=diff).groupby("tunefamily")["diff"].mean().sort_values()
print("\nMean AP change per family (worst 3 / best 3):")
print(fam.head(3).round(3).to_string())
print(fam.tail(3).round(3).to_string())