# Comparing melodic similarity methods

DT2470 Music Informatics, KTH, group project (autumn 2026).

## What we do

We compare methods for measuring how similar two melodies are, a core problem in MIR tasks such as plagiarism detection. We test:

- **Melody representations:** pitch (MIDI note numbers), intervals (semitone steps between notes) and contour (up / down / same), each with and without note durations
- **Distance measures:** edit distance, dynamic time warping (DTW) and n-gram overlap
- **A pretrained symbolic music embedding model:** CLaMP 2, compared with cosine similarity

This gives 3 representations × 2 (with/without durations) × 3 distance measures = 18 hand-crafted methods, plus CLaMP 2.

**Dataset:** MTC-ANN (Meertens Tune Collections, annotated corpus): 360 Dutch folk song melodies grouped by experts into 26 tune families.

**Evaluation:** each melody is used as a query and all other melodies are ranked by similarity to it. We measure how well melodies from the same tune family end up at the top with mean average precision (MAP).

**Optional, if time allows:** apply the methods to melody transcriptions of real copyright cases with known verdicts.

## Repository structure

```
melody-similarity/
├── README.md
├── .gitignore
├── data/        # local only, not committed: MTC-ANN and other downloads
├── src/         # project code
├── notebooks/   # exploration, figures
└── results/     # local only, not committed: similarity matrices, result tables
```

## Who does what

| Part | Owner | Contents |
|---|---|---|
| Data loading | Fabian | Load MTC-ANN MIDI files and tune-family labels |
| Representations | Elias | Pitch / interval / contour sequences, with and without durations |
| Distance measures | Elias | Edit distance, DTW, n-gram overlap and similarity matrices for all combinations |
| CLaMP 2 | Lucas | Embeddings for all melodies, cosine similarity matrix |
| Evaluation | TBD | MAP, results table and figures |

## Shared formats

Everyone codes against these, so the parts can be built in parallel.

**Melody IDs:** the MTC-ANN `songid`, i.e. the MIDI file name without extension, e.g. `NLB072638_01`.

**Tune-family labels:** a dictionary `{songid: tunefamily}`.

**Representations:** one dictionary per representation, `{songid: sequence}`:

| Name | Example element | Example sequence |
|---|---|---|
| `pitch` | MIDI note number | `[60, 60, 67, 67, 69, 69, 67]` |
| `interval` | semitone step | `[0, 7, 0, 2, 0, -2]` |
| `contour` | -1 / 0 / +1 | `[0, 1, 0, 1, 0, -1]` |
| `pitch+dur`, `interval+dur`, `contour+dur` | `(value, duration)` | `[(0, 1.0), (7, 1.0), ...]` |

Durations are in quarter notes, measured from one note's start to the next note's start.

**Similarity matrices:** an N × N NumPy array saved as `.npy` (higher = more similar), together with the list of melody IDs in the same row/column order. Save to `results/`, named after the method, e.g. `interval_edit.npy`, `clamp2_cosine.npy`.

## Setup

### Python

Python 3.10 or newer. Main packages: `numpy`, `pandas`, `pretty_midi`, `librosa` (for DTW), `matplotlib`.

### MTC-ANN

1. Download **MTC-ANN 2.0.1** via the download form at http://www.liederenbank.nl/mtc/download.php
2. Unpack it into `data/`, so that you get `data/MTC-ANN-2.0.1/mid/` (MIDI files) and `data/MTC-ANN-2.0.1/metadata/` (including the tune-family labels, columns `songid` and `tunefamily`)

### CLaMP 2

1. Clone https://github.com/sanderwood/clamp2 into `clamp2/` in the repo root (ignored by git)
2. Download the CLaMP 2 and M3 weights from https://huggingface.co/sander-wood/clamp2 into `clamp2/code/`
3. Convert MIDI to MTF with `clamp2/process_data/batch_midi2mtf.py`, then extract features with `clamp2/code/extract_clamp2.py <mtf_dir> <output_dir> --normalize`

## Workflow

- Work on your own branch (e.g. `representations`, `data-loading`, `clamp2`) and merge into `main` when a piece works.
- Never commit data, model weights or large result files (see `.gitignore`).
- Keep this README up to date if anything changes.
