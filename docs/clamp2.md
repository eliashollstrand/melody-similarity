# CLaMP 2 part: setup, pipeline and first results

Owner: Lucas. Branch: `clamp2`.

This document explains what has been done for the CLaMP 2 part, how to reproduce it,
and a shared evaluation script that every method can use.

## Summary

- CLaMP 2 embeddings computed for all 360 MTC-ANN melodies.
- Cosine similarity matrix saved in the shared format: `results/clamp2_cosine.npz`.
- `src/visualize_results.py` computes **MAP** and figures for **any** method's `.npz`
  (covers the "Evaluation" part, currently TBD in the README).
- First result: **MAP = 0.378** (random ranking: 0.053).

## Pipeline

    MTC-ANN MIDI (data/MTC-ANN-2.0.1/mid/)
        │  clamp2/process_data/batch_midi2mtf.py
        ▼
    MTF text files (data/MTC-ANN-2.0.1/mid_mtf/)
        │  clamp2/code/extract_clamp2.py --normalize
        ▼
    One 768-d embedding per melody (data/clamp2_feats/<songid>.npy)
        │  src/clamp_embeddings.py
        ▼
    Cosine similarity matrix (results/clamp2_cosine.npz)
        │  src/visualize_results.py
        ▼
    MAP + figures (results/figures/)

## Setup

Tested on Windows (PowerShell) with Miniconda. Use **Python 3.10**: CLaMP 2 needs
`numpy<2`, which has no builds for Python 3.13.

```bash
conda create -n melody python=3.10 -c conda-forge -y
conda activate melody
pip install "numpy<2" pandas pretty_midi librosa matplotlib
pip install torch transformers accelerate mido samplings==0.1.7 unidecode
```

Do **not** use CLaMP 2's `environment.yml` (Windows + CUDA only) or its
`requirements.txt` (includes training-only packages such as `horovod`). No GPU needed.

CLaMP 2 code and weights (≈4.5 GB, ignored by git):

```bash
git clone https://github.com/sanderwood/clamp2.git clamp2
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sander-wood/clamp2', 'weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth', local_dir='clamp2/code')"
```

The M3 weights are **not** needed: the CLaMP 2 checkpoint already contains the music encoder.
The first extraction run also downloads XLM-RoBERTa (≈1 GB) automatically.

## Running it

**1. MIDI → MTF.** The script has no arguments: edit its first line to the absolute path
of the MIDI folder (keep `m3_compatible = True`). Output goes to a sibling `mid_mtf/` folder.

```python
input_dir = r"C:\...\melody-similarity\data\MTC-ANN-2.0.1\mid"
```

```bash
python clamp2/process_data/batch_midi2mtf.py
```

**2. Embeddings.** Must be run from `clamp2/code/` (config and weights use relative paths).
Already-processed files are skipped, so it can be re-run safely.

```bash
cd clamp2/code
python extract_clamp2.py ../../data/MTC-ANN-2.0.1/mid_mtf ../../data/clamp2_feats --normalize
cd ../..
```

**3. Similarity matrix.**

```bash
python src/clamp_embeddings.py --features data/clamp2_feats --out results/clamp2_cosine.npz
```

Prints sanity checks (shape, diagonal ≈ 1, symmetry).

## Evaluation script (usable by everyone)

```bash
python src/visualize_results.py --sim results/<method>.npz
```

Works with any `.npz` in the shared format (`sim`, `ids`). Reads labels from
`data/MTC-ANN-2.0.1/metadata/MTC-ANN-tune-family-labels.csv`.

| Output | What it shows |
|---|---|
| MAP (console) | Main score, 0–1, higher is better |
| `<method>_heatmap.png` | Matrix sorted by family: good methods show bright diagonal blocks |
| `<method>_hist.png` | Within- vs between-family similarities: less overlap is better |
| `<method>_ap_family.png` | AP per query grouped by family: easy vs hard families |
| `<method>_ap.csv` | AP per melody, for further analysis |

AP is computed per query by ranking all other 359 melodies; relevant = same tune family.

## First results (CLaMP 2, cosine)

- **MAP = 0.378** vs **0.053** for a random ranking.
- Large differences between families: `Ik_kwam_laatst_eens_in_de_stad` (median AP ≈ 0.86)
  vs `Zolang_de_boom_zal_bloeien_1` (≈ 0.12). 26 % of queries have AP < 0.2.
- Similarities are compressed into a narrow range (mostly 0.8–1.0): within-family mean
  0.913, between-family mean 0.857, with strong overlap.
- Some families are confused with each other (e.g. `Er_was_een_koopman…`,
  `Er_woonde_een_vrouwtje…`, `Femmes_voulez_vous_eprouver`).

## Notes on the data

- IDs like `NLB072587_01`: `NLB…` is the record number in the Dutch Song Database,
  `_01`/`_02` is the strophe. It says nothing about the tune family.
- 40 of the 360 melodies are not `_01`; some recordings have several strophes in the corpus.
  These pairs are near-duplicates and may inflate MAP for all methods.

## Open ideas

1. **Centering** embeddings (subtract the mean vector) to counter the narrow range; may change rankings.
2. **Transposition**: MTF uses absolute pitch, so variants in different keys may look dissimilar.
   Transposing all melodies to a common tonic would test this (counterpart of `interval`).
3. **Same-recording strophes**: recompute MAP ignoring results that share the query's NLB number.
4. **Comparison** with the hand-crafted methods once their `.npz` files are ready.