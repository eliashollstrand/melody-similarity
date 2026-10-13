"""Build the CLaMP 2 cosine similarity matrix from extracted features.

Expects one .npy file per melody in --features, named <songid>.npy,
as produced by: clamp2/code/extract_clamp2.py <mtf_dir> <out_dir> --normalize

Output: results/clamp2_cosine.npz with
    sim: N x N array, higher = more similar
    ids: songids in the same row/column order
"""
from html import parser
from pathlib import Path
import argparse

import numpy as np

# Old version, new version below with optional centering of embeddings for
# better similarity results. Keeping this here for reference.

# def load_embeddings(features_dir):
#     """Return (ids, E) where E has shape (N, D), rows in sorted songid order."""
#     files = sorted(Path(features_dir).rglob("*.npy"))
#     if not files:
#         raise FileNotFoundError(f"No .npy files found in {features_dir}")

#     ids, vecs = [], []
#     for f in files:
#         v = np.load(f).astype(np.float64)
#         # Features may be (D,), (1, D) or (n_segments, D): mean-pool to one vector
#         v = v.reshape(-1, v.shape[-1]).mean(axis=0)
#         ids.append(f.stem)
#         vecs.append(v)

#     E = np.stack(vecs)
#     # Re-normalize (pooling breaks unit length), so dot product = cosine
#     E /= np.linalg.norm(E, axis=1, keepdims=True)
#     return ids, E

def load_embeddings(features_dir, center=False):
    """Return (ids, E) where E has shape (N, D), rows in sorted songid order."""
    files = sorted(Path(features_dir).rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {features_dir}")

    ids, vecs = [], []
    for f in files:
        v = np.load(f).astype(np.float64)
        # Features may be (D,), (1, D) or (n_segments, D): mean-pool to one vector
        v = v.reshape(-1, v.shape[-1]).mean(axis=0)
        ids.append(f.stem)
        vecs.append(v)

    E = np.stack(vecs)
    # Re-normalize (pooling breaks unit length), so dot product = cosine
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    if center:
        # Remove the direction shared by all melodies, then re-normalize
        E -= E.mean(axis=0, keepdims=True)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    return ids, E


def sanity_check(sim, ids, families=None):
    """Print basic checks; families is {songid: tunefamily} or None."""
    print(f"Matrix shape: {sim.shape}")
    print(f"Diagonal min/max: {sim.diagonal().min():.4f} / {sim.diagonal().max():.4f}")
    print(f"Symmetric: {np.allclose(sim, sim.T)}")

    if families is None:
        return

    missing = set(families) - set(ids)
    extra = set(ids) - set(families)
    print(f"Missing ids: {len(missing)} | Unexpected ids: {len(extra)}")

    # Within-family vs between-family mean similarity (excluding self-pairs)
    labels = np.array([families.get(i) for i in ids])
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(ids), dtype=bool)
    print(f"Mean sim within families:  {sim[same & off_diag].mean():.4f}")
    print(f"Mean sim between families: {sim[~same].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Dir with <songid>.npy files")
    parser.add_argument("--out", default="results/clamp2_cosine.npz")
    parser.add_argument("--mtc_root", default=None, help="Optional: check ids/families")
    parser.add_argument("--center", action="store_true",
                    help="Subtract the mean embedding before cosine similarity")
    args = parser.parse_args()

    ids, E = load_embeddings(args.features, center=args.center)
    sim = E @ E.T #cosine similarity matrix, since E is normalized

    families = None
    if args.mtc_root:
        try:
            from data_loading import load_tune_families
            families = load_tune_families(args.mtc_root)
        except ImportError:
            print("data_loading.py not found, skipping family checks")

    sanity_check(sim, ids, families)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, sim=sim, ids=np.array(ids))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
  