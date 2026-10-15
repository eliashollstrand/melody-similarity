"""Step 2a: final-note diagnosis before transposing (no re-extraction needed).

For every MTC-ANN melody, take the pitch class of its final note (folk-song
tonic heuristic) and check:
  1. how often melodies of the same tune family end on different pitch classes
  2. (with --sim) whether the model scores pairs lower when their final notes
     differ -> sign of absolute-pitch dependence
  3. (with --ap) whether queries in a "minority" key of their family get lower AP

Usage (from the repo root):
    python src/final_note_diagnosis.py
    python src/final_note_diagnosis.py --sim results/clamp2_cosine_centered.npz --ap results/figures/clamp2_cosine_centered_ap.csv
"""
from pathlib import Path
from collections import Counter
import argparse
import csv

import numpy as np
import mido
from scipy.stats import mannwhitneyu

from visualize_results import load_families

PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def final_note(midi_path):
    """MIDI number of the last note onset (melodies are monophonic)."""
    last = None
    # merge_tracks yields messages in time order, so the last note_on wins
    for msg in mido.merge_tracks(mido.MidiFile(midi_path).tracks):
        if msg.type == "note_on" and msg.velocity > 0:
            last = msg.note
    return last


def compare(name, a, b):
    """Print means of two groups and a Mann-Whitney U test (a vs b)."""
    if len(a) == 0 or len(b) == 0:
        print(f"  {name}: not enough data")
        return
    p = mannwhitneyu(a, b).pvalue
    print(f"  {name}: {a.mean():.4f} (n={len(a)}) vs {b.mean():.4f} (n={len(b)}) "
          f"| diff {a.mean() - b.mean():+.4f} | Mann-Whitney p = {p:.2g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtc_root", default="data/MTC-ANN-2.0.1")
    parser.add_argument("--sim", default=None, help="Optional: results/<method>.npz")
    parser.add_argument("--ap", default=None, help="Optional: results/figures/<method>_ap.csv")
    parser.add_argument("--out", default="results/final_notes.csv")
    args = parser.parse_args()

    # --- Final note of every labelled melody ---
    families = load_families(args.mtc_root)
    finals = {}
    for p in sorted(Path(args.mtc_root, "mid").glob("*.mid")):
        if p.stem in families:
            n = final_note(p)
            if n is not None:
                finals[p.stem] = n
    print(f"Final note found for {len(finals)} of {len(families)} labelled melodies")

    ids = sorted(finals)
    labels = np.array([families[i] for i in ids])
    pcs = np.array([finals[i] % 12 for i in ids])

    print("\nFinal pitch class over the whole corpus:")
    for pc, c in Counter(pcs.tolist()).most_common():
        print(f"  {PC_NAMES[pc]:>2}: {c}")

    # --- 1. Same-family pairs with different final pitch class ---
    iu = np.triu_indices(len(ids), k=1)  # each unordered pair once
    same_fam = (labels[:, None] == labels[None, :])[iu]
    same_pc = (pcs[:, None] == pcs[None, :])[iu]
    print(f"\nSame-family pairs: {same_fam.sum()} | "
          f"different final pitch class: {1 - same_pc[same_fam].mean():.1%}")
    print(f"Different-family pairs (reference): "
          f"different final pitch class: {1 - same_pc[~same_fam].mean():.1%}")

    # Per-family view: majority final and how many distinct finals
    majority = {}
    rows = []
    for fam in np.unique(labels):
        fam_pcs = pcs[labels == fam]
        c = Counter(fam_pcs.tolist())
        pc, n = c.most_common(1)[0]  # ties: first one found
        majority[fam] = pc
        rows.append((fam, len(fam_pcs), PC_NAMES[pc], n / len(fam_pcs), len(c)))
    rows.sort(key=lambda r: r[3])
    print("\nPer family (sorted by share of the majority final):")
    print(f"  {'family':<45} {'n':>3} {'maj':>4} {'share':>6} {'#finals':>8}")
    for fam, n, pc, share, k in rows:
        print(f"  {fam:<45} {n:>3} {pc:>4} {share:>6.0%} {k:>8}")

    in_majority = np.array([pcs[k] == majority[labels[k]] for k in range(len(ids))])
    n_finals = {fam: k for fam, _, _, _, k in rows}
    mixed = np.array([n_finals[f] > 1 for f in labels])  # families with >1 final

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["songid", "tunefamily", "final_midi", "final_pc", "in_majority"])
        for k, i in enumerate(ids):
            w.writerow([i, labels[k], finals[i], PC_NAMES[pcs[k]], int(in_majority[k])])
    print(f"\nPer-melody table saved to {args.out}")

    # --- 2. Does the model care about the final pitch class? ---
    # Note: pairs are not independent, so p-values are only indicative
    if args.sim:
        d = np.load(args.sim)
        pos = {str(s): k for k, s in enumerate(d["ids"])}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} ids missing from {args.sim}, e.g. {missing[:3]}")
        idx = np.array([pos[i] for i in ids])
        S = d["sim"][np.ix_(idx, idx)][iu]
        print(f"\nSimilarity, same final vs different final ({Path(args.sim).stem}):")
        compare("within family ", S[same_fam & same_pc], S[same_fam & ~same_pc])
        compare("between family", S[~same_fam & same_pc], S[~same_fam & ~same_pc])

    # --- 3. Do minority-key queries get lower AP? ---
    if args.ap:
        ap = {}
        with open(args.ap, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["ap"] not in ("", "nan"):
                    ap[r["songid"]] = float(r["ap"])
        has_ap = np.array([i in ap for i in ids])
        ap_arr = np.array([ap.get(i, np.nan) for i in ids])
        sel = mixed & has_ap  # only families where a minority exists
        print(f"\nAP in families with >1 final ({Path(args.ap).stem}):")
        compare("majority vs minority final", ap_arr[sel & in_majority], ap_arr[sel & ~in_majority])


if __name__ == "__main__":
    main()
