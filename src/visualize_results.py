"""Visualize how well a similarity matrix groups melodies by tune family.

Works for any method that follows the shared format (results/<method>.npz
with `sim` and `ids`), e.g. CLaMP 2 or interval_edit.

Outputs (in --out_dir):
    <method>_heatmap.png    similarity matrix sorted by tune family
    <method>_hist.png       within- vs between-family similarity distributions
    <method>_ap_family.png  average precision per query, grouped by family
    <method>_ap.csv         average precision per query (for later analysis)

With --exclude_same_recording, results that come from the same recording as
the query (other strophes, same NLB number) are removed from each ranking,
and output files get the suffix "_norec" (e.g. <method>_norec_ap.csv).

Usage:
    python src/visualize_results.py --sim results/clamp2_cosine.npz --mtc_root data/MTC-ANN-2.0.1
    python src/visualize_results.py --sim results/clamp2_cosine.npz --exclude_same_recording
"""
from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt


def load_families(mtc_root):
    """Return {songid: tunefamily} from the MTC-ANN tune-family labels table."""
    # Exact name: a glob would also match MTC-ANN-tune-family-labels-fieldnames.csv
    path = Path(mtc_root, "metadata", "MTC-ANN-tune-family-labels.csv")
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    print(f"Reading family labels from {path}")
    families = {}
    with open(path, encoding="utf-8-sig") as f:  # -sig strips a BOM if present
        for row in csv.reader(f):
            # Skip empty lines and a header row, if present
            if len(row) < 2 or row[0].strip().lower() == "songid":
                continue
            families[row[0].strip()] = row[1].strip()
    return families


def recording_id(songid):
    """Recording number without the strophe suffix: NLB072355_12 -> NLB072355."""
    return songid.rsplit("_", 1)[0]


def average_precision(sim, labels, groups=None):
    """AP per query: rank all other melodies, relevant = same tune family.

    If groups is given (one recording id per melody), candidates from the
    query's own recording are removed from its ranking.
    """
    n = len(labels)
    ap = np.zeros(n)
    for i in range(n):
        order = np.argsort(-sim[i], kind="stable")
        keep = order != i  # the query itself is not a result
        if groups is not None:
            keep &= groups[order] != groups[i]  # nor are its sibling strophes
        order = order[keep]
        relevant = labels[order] == labels[i]
        if relevant.sum() == 0:
            ap[i] = np.nan
            continue
        hits = np.cumsum(relevant)
        precision_at_k = hits / np.arange(1, len(order) + 1)
        ap[i] = precision_at_k[relevant].mean()
    return ap


def report_same_recording(ids, labels, groups):
    """Print how many melodies share their recording with another melody."""
    same_rec = groups[:, None] == groups[None, :]
    np.fill_diagonal(same_rec, False)
    n_queries = same_rec.any(axis=1).sum()
    n_pairs = same_rec.sum() // 2
    same_fam = (same_rec & (labels[:, None] == labels[None, :])).sum() // 2
    print(f"Queries with sibling strophes: {n_queries} | "
          f"sibling pairs: {n_pairs} (same family: {same_fam})")


def plot_heatmap(sim, labels, method, out_path):
    """Similarity matrix with rows/cols grouped by family: good methods show bright diagonal blocks."""
    order = np.argsort(labels, kind="stable")
    S, L = sim[np.ix_(order, order)], labels[order]

    # Family boundaries and centers for lines and tick labels
    change = np.flatnonzero(L[1:] != L[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(L)]])
    centers = (starts + ends) / 2 - 0.5

    off_diag = S[~np.eye(len(S), dtype=bool)]
    vmin, vmax = np.percentile(off_diag, [2, 98])  # robust color range

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(S, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    for b in change:
        ax.axhline(b - 0.5, color="white", lw=0.5)
        ax.axvline(b - 0.5, color="white", lw=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels(L[starts], rotation=90, fontsize=6)
    ax.set_yticks(centers)
    ax.set_yticklabels(L[starts], fontsize=6)
    ax.set_title(f"{method}: similarity matrix sorted by tune family")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histogram(sim, labels, method, out_path):
    """Within-family vs between-family similarity: less overlap = better method."""
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    within, between = sim[same & off_diag], sim[~same]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(within.min(), between.min()), max(within.max(), between.max()), 60)
    ax.hist(between, bins=bins, density=True, alpha=0.6, label=f"between families (mean {between.mean():.3f})")
    ax.hist(within, bins=bins, density=True, alpha=0.6, label=f"within family (mean {within.mean():.3f})")
    ax.set_xlabel("similarity")
    ax.set_ylabel("density")
    ax.set_title(f"{method}: similarity distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ap_per_family(ap, labels, method, out_path):
    """Boxplot of AP per family, sorted by median: shows easy vs hard families."""
    fams = np.unique(labels)
    data = [ap[labels == f][~np.isnan(ap[labels == f])] for f in fams]
    order = np.argsort([np.median(d) for d in data])
    fams, data = fams[order], [data[k] for k in order]

    fig, ax = plt.subplots(figsize=(8, 9))
    try:
        ax.boxplot(data, orientation="horizontal", widths=0.6)  # matplotlib >= 3.10
    except TypeError:
        ax.boxplot(data, vert=False, widths=0.6)  # older matplotlib
    ax.set_yticks(range(1, len(fams) + 1))
    ax.set_yticklabels(fams, fontsize=7)
    map_score = np.nanmean(ap)
    ax.axvline(map_score, color="red", ls="--", label=f"MAP = {map_score:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("average precision per query")
    ax.set_title(f"{method}: AP per tune family")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", required=True, help="results/<method>.npz")
    parser.add_argument("--mtc_root", default="data/MTC-ANN-2.0.1")
    parser.add_argument("--out_dir", default="results/figures")
    parser.add_argument("--exclude_same_recording", action="store_true",
                        help="Ignore results from the query's own recording (same NLB number)")
    args = parser.parse_args()

    method = Path(args.sim).stem
    d = np.load(args.sim)
    sim, ids = d["sim"], [str(i) for i in d["ids"]]

    families = load_families(args.mtc_root)
    missing = [i for i in ids if i not in families]
    if missing:
        print(f"Warning: {len(missing)} ids without family label, e.g. {missing[:3]}")
    keep = np.array([i in families for i in ids])
    if keep.sum() < 2:
        raise SystemExit(
            f"Only {keep.sum()} of {len(ids)} ids have a family label. "
            f"Example ids in matrix: {ids[:3]} | in labels: {list(families)[:3]}"
        )
    sim = sim[np.ix_(keep, keep)]
    ids = [i for i, k in zip(ids, keep) if k]
    labels = np.array([families[i] for i in ids])

    groups = None
    if args.exclude_same_recording:
        groups = np.array([recording_id(i) for i in ids])
        report_same_recording(ids, labels, groups)
        method += "_norec"

    ap = average_precision(sim, labels, groups)
    print(f"{method}: {len(ids)} melodies, {len(np.unique(labels))} families")
    print(f"MAP = {np.nanmean(ap):.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(sim, labels, method, out_dir / f"{method}_heatmap.png")
    if np.all(np.isnan(ap)):
        # Happens with tiny test sets where no two melodies share a family
        print("No two melodies share a family: skipping histogram and AP plots")
    else:
        plot_histogram(sim, labels, method, out_dir / f"{method}_hist.png")
        plot_ap_per_family(ap, labels, method, out_dir / f"{method}_ap_family.png")

    with open(out_dir / f"{method}_ap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["songid", "tunefamily", "ap"])
        w.writerows(zip(ids, labels, np.round(ap, 4)))

    print(f"Figures and AP table saved to {out_dir}")


if __name__ == "__main__":
    main()
