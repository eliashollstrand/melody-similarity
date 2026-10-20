# CLaMP 2 part: setup, pipeline and results

Owner: Lucas. Branch: `clamp2`.

This document explains what has been done for the CLaMP 2 part, how to reproduce it,
and the shared evaluation scripts that every method can use.

## Summary

- CLaMP 2 embeddings computed for all 360 MTC-ANN melodies.
- Cosine similarity matrices saved in the shared format:
  `results/clamp2_cosine.npz` (raw) and `results/clamp2_cosine_centered.npz` (**main version**).
- `src/visualize_results.py` computes **MAP** and figures for **any** method's `.npz`
  (covers the "Evaluation" part, currently TBD in the README). Optional
  `--exclude_same_recording` removes near-duplicate strophes from the rankings.
- `src/compare_ap.py` compares two methods query by query (paired Wilcoxon test).
- Results: **MAP = 0.378** (raw) → **0.399** (centered), random ranking: 0.053.
  Excluding same-recording strophes barely changes it (0.397).

## Pipeline

    MTC-ANN MIDI (data/MTC-ANN-2.0.1/mid/)
        │  clamp2/process_data/batch_midi2mtf.py
        ▼
    MTF text files (data/MTC-ANN-2.0.1/mid_mtf/)
        │  clamp2/code/extract_clamp2.py --normalize
        ▼
    One 768-d embedding per melody (data/clamp2_feats/<songid>.npy)
        │  src/clamp_embeddings.py [--center]
        ▼
    Cosine similarity matrix (results/clamp2_cosine[_centered].npz)
        │  src/visualize_results.py [--exclude_same_recording]
        ▼
    MAP + figures (results/figures/)
        │  src/compare_ap.py
        ▼
    Paired comparison between two methods

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

Note: `--normalize` does **not** L2-normalize; it pools the segments and applies the
projection to the shared embedding space. L2 normalization is done in `clamp_embeddings.py`.

**3. Similarity matrices.**

```bash
python src/clamp_embeddings.py --features data/clamp2_feats --out results/clamp2_cosine.npz
python src/clamp_embeddings.py --features data/clamp2_feats --center --out results/clamp2_cosine_centered.npz
```

`--center` subtracts the mean embedding of all melodies before the cosine similarity.
Both commands print sanity checks (shape, diagonal ≈ 1, symmetry).

## Evaluation scripts (usable by everyone)

### `visualize_results.py`

```bash
python src/visualize_results.py --sim results/<method>.npz
python src/visualize_results.py --sim results/<method>.npz --exclude_same_recording
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

AP is computed per query by ranking all other melodies; relevant = same tune family.

With `--exclude_same_recording`, other strophes of the query's own recording
(same `NLB…` number, e.g. `NLB072355_01` vs `_12`) are removed from its ranking,
and outputs get the suffix `_norec` (e.g. `<method>_norec_ap.csv`).

### `compare_ap.py`

```bash
python src/compare_ap.py results/figures/<A>_ap.csv results/figures/<B>_ap.csv
```

Prints MAP of both methods, how many queries improve or get worse, the p-value of a
paired Wilcoxon signed-rank test on per-query AP, and the families that change most.

## Results (CLaMP 2, cosine)

| Version | MAP | Notes |
|---|---|---|
| Random ranking | 0.053 | simulated baseline |
| `clamp2_cosine` | 0.378 | raw embeddings |
| `clamp2_cosine_centered` | **0.399** | mean embedding subtracted |
| `clamp2_cosine_centered`, same recording excluded | 0.397 | `--exclude_same_recording` (robustness check) |

**Raw embeddings**
- Large differences between families: `Ik_kwam_laatst_eens_in_de_stad` (median AP ≈ 0.86)
  vs `Zolang_de_boom_zal_bloeien_1` (≈ 0.12). 26 % of queries have AP < 0.2.
- Similarities are compressed into a narrow range (mostly 0.8–1.0): within-family mean
  0.913, between-family mean 0.857, with strong overlap.
- Some families are confused with each other (e.g. `Er_was_een_koopman…`,
  `Er_woonde_een_vrouwtje…`, `Femmes_voulez_vous_eprouver`).

**Centering (raw vs centered, paired over 360 queries)**
- MAP +0.021 (0.378 → 0.399); 207 queries improve, 153 get worse.
- Wilcoxon signed-rank p = 2.1 · 10⁻⁵: the improvement is significant.
- Largest gains: `Er_was_een_herderinnetje_1` (+0.158), `Er_was_een_meisje_van_zestien_jaren_1`
  (+0.121), `Kom_laat_ons_nu_zo_stil_niet_zijn_1` (+0.091).
- Largest losses: `Het_was_op_een_driekoningenavond_1` (−0.082), `Soldaat_kwam_uit_de_oorlog`
  (−0.041), `Wat_zag_ik_daar_van_verre_1` (−0.027).
- Centered is used as the main CLaMP 2 version from now on.

**Same-recording strophes (centered, with vs without `--exclude_same_recording`)**
- MAP 0.399 → 0.397 (−0.003); only 25 of 360 queries change (21 worse, 4 better, 335 identical).
- Wilcoxon p = 1.0 · 10⁻⁴, but the effect size is negligible: removing a relevant
  near-duplicate that ranks high can only lower AP, so the direction is expected.
- Conclusion: near-duplicate strophes do **not** meaningfully inflate MAP. The main
  figures are reported without the flag; the flag is kept as a robustness check.

## Findings for the report

**Hubness hypothesis (to verify).** All raw embeddings share a large common component
(similarities 0.85–0.92). Melodies close to the mean embedding are moderately similar to
everything, so they act as "hubs" that appear near the top of many rankings and push
genuinely related but atypical melodies down. Centering moves the mean to the origin and
removes that advantage. Supporting evidence: the family that gains most from centering,
`Er_was_een_herderinnetje_1`, was one of the dark bands in the raw heatmap (melodies the
model saw as dissimilar to everything). Hubness is a known effect in embedding-based
retrieval. A direct check would be the k-occurrence distribution (how often each melody
appears in the top-k of other queries) before and after centering: hubs show up as a long
right tail that should shrink.

## Notes on the data

- IDs like `NLB072587_01`: `NLB…` is the record number in the Dutch Song Database,
  `_01`/`_02` is the strophe. It says nothing about the tune family.
- 40 of the 360 melodies are not `_01`; some recordings have several strophes in the corpus.
  These pairs are near-duplicates, but excluding them changes CLaMP 2's MAP by only
  −0.003 (`--exclude_same_recording`). Worth re-checking for the hand-crafted methods.

## Open ideas

1. ~~Centering~~ done (+0.021 MAP, significant).
2. **Transposition**: MTF uses absolute pitch, so variants in different keys may look dissimilar.
   Transposing all melodies to a common tonic (e.g. the final note) would test this
   (counterpart of `interval`).
3. ~~Same-recording strophes~~ done: negligible effect (−0.003 MAP).
4. **Hubness check**: k-occurrence distribution, raw vs centered.
5. **Comparison** with the hand-crafted methods once their `.npz` files are ready
   (`compare_ap.py` against `clamp2_cosine_centered`).
