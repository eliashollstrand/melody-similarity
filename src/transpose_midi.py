"""Step 2b: transpose MTC-ANN MIDI files with the expert pitch_shift annotation.

Reads metadata/MTC-ANN-pitch-duration-normalization.csv (songid, time_stretch,
pitch_shift) and writes every MIDI file of mid/ to --out_dir:
  - pitch_shift != 0: note numbers of note_on / note_off are shifted; all other
    messages (tempo, meter, timing) are left untouched
  - pitch_shift == 0: the file is copied byte for byte

Note: this is an "oracle" normalization, made by experts who knew the tune
families. It gives an upper bound, not a blind method.

With --feats_src / --feats_dst, the existing CLaMP 2 features of the unchanged
melodies are copied to the new feature folder. extract_clamp2.py skips .npy
files that already exist, so only the shifted melodies are re-embedded.

Usage (from the repo root):
    python src/transpose_midi.py --feats_src data/clamp2_feats --feats_dst data/clamp2_feats_transposed
"""
from pathlib import Path
import argparse
import csv
import shutil

import mido


def load_shifts(mtc_root):
    """Return {songid: pitch_shift in semitones}."""
    path = Path(mtc_root, "metadata", "MTC-ANN-pitch-duration-normalization.csv")
    shifts = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            # Skip empty lines and a header row, if present
            if len(row) < 3 or row[0].strip().lower() == "songid":
                continue
            shifts[row[0].strip()] = int(row[2])
    return shifts


def transpose_file(src, dst, shift):
    """Shift note numbers of note_on/note_off by `shift`, keep everything else.

    Events with note 0 (C-1, inaudible) are placeholders in some MTC files,
    not melody notes: they are left untouched. Returns how many were skipped.
    """
    mid = mido.MidiFile(src)
    skipped = 0
    for track in mid.tracks:
        for k, msg in enumerate(track):
            if msg.type in ("note_on", "note_off"):
                if msg.note == 0:
                    skipped += 1
                    continue
                new = msg.note + shift
                if not 0 <= new <= 127:
                    raise ValueError(f"{src.name}: note {msg.note} + {shift} out of MIDI range")
                track[k] = msg.copy(note=new)
    mid.save(dst)
    return skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtc_root", default="data/MTC-ANN-2.0.1")
    parser.add_argument("--out_dir", default=None, help="Default: <mtc_root>/mid_transposed")
    parser.add_argument("--feats_src", default=None, help="Existing CLaMP 2 features (optional)")
    parser.add_argument("--feats_dst", default=None, help="New feature folder (optional)")
    args = parser.parse_args()

    shifts = load_shifts(args.mtc_root)
    src_dir = Path(args.mtc_root, "mid")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.mtc_root, "mid_transposed")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.mid"))
    if not files:
        raise SystemExit(f"No .mid files found in {src_dir}")

    changed, unchanged, missing = [], [], []
    with_note0 = {}  # songid -> number of note-0 events left untouched
    for p in files:
        if p.stem not in shifts:
            missing.append(p.stem)
        shift = shifts.get(p.stem, 0)
        if shift == 0:
            shutil.copy2(p, out_dir / p.name)
            unchanged.append(p.stem)
        else:
            skipped = transpose_file(p, out_dir / p.name, shift)
            if skipped:
                with_note0[p.stem] = skipped
            changed.append(p.stem)

    print(f"{len(files)} MIDI files -> {out_dir}")
    print(f"Transposed: {len(changed)} | copied unchanged: {len(unchanged)}")
    if missing:
        print(f"Warning: {len(missing)} files without pitch_shift (copied), e.g. {missing[:3]}")
    if with_note0:
        print(f"Note-0 events left untouched in {len(with_note0)} transposed files: {with_note0}")

    # Reuse features of unchanged melodies so only the shifted ones are re-extracted
    if args.feats_src and args.feats_dst:
        feats_src, feats_dst = Path(args.feats_src), Path(args.feats_dst)
        feats_dst.mkdir(parents=True, exist_ok=True)
        stale = [s for s in changed if (feats_dst / f"{s}.npy").exists()]
        if stale:
            raise SystemExit(f"{len(stale)} shifted melodies already have features in {feats_dst} "
                             f"(would be skipped by extract_clamp2.py), e.g. {stale[:3]}. Delete them first.")
        n_copied = 0
        for s in unchanged:
            f = feats_src / f"{s}.npy"
            if f.exists():
                shutil.copy2(f, feats_dst / f.name)
                n_copied += 1
        print(f"Copied {n_copied} unchanged feature files to {feats_dst}; "
              f"{len(changed)} melodies left to extract")


if __name__ == "__main__":
    main()
