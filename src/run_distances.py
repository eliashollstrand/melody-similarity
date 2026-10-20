"""Compute similarity matrices for all representation x distance measure combinations.

Usage (from the repo root):
    python src/run_distances.py
Writes results/<representation>_<measure>.npz. Existing files are skipped,
so the script can be stopped and restarted.
"""

from functools import partial
from pathlib import Path

from data_loading import load_melodies
from representations import build_representations
from distances import edit_similarity, dtw_similarity, ngram_similarity, similarity_matrix, save_matrix

MTC_ROOT = Path("data/MTC-ANN-2.0.1")
OUT_DIR = Path("results")

MEASURES = {
    "ngram3": partial(ngram_similarity, n=3),
    "edit": edit_similarity,
    "dtw": dtw_similarity,
}

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    melodies = load_melodies(MTC_ROOT)
    reps = build_representations(melodies)
    print(f"{len(melodies)} melodies, {len(reps)} representations, {len(MEASURES)} measures")

    for measure_name, sim_fn in MEASURES.items():
        for rep_name, sequences in reps.items():
            path = OUT_DIR / f"{rep_name}_{measure_name}.npz"
            
            if path.exists():
                print(f"skip  {path.name} (already exists)")
                continue
            
            S, ids = similarity_matrix(sequences, sim_fn)
            save_matrix(path, S, ids)
            print(f"Saved {path.name}")