import pandas as pd
import pretty_midi
from pathlib import Path

def clean_number(value):
    """Round a beat value and return integers as floats without noise."""
    return round(float(value), 6)


def load_melodies(mtc_root):
    """Load all MTC-ANN MIDI files. Takes the path to the MTC-ANN-2.0.1 dataset and 
    returns melodies, a dict with keys "song ID" and its values are a lists (notes) of 
    touples containing (onset_in_quarter_notes, duration_in_quarter_notes, midi_pitch)."""

    mtc_root = Path(mtc_root)
    midi_dir = mtc_root / "mid"

    if not midi_dir.is_dir():
        raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")

    midi_files = sorted(
        path
        for path in midi_dir.iterdir()
    )

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in: {midi_dir}")

    melodies = {} 

    for midi_path in midi_files:
        midi = pretty_midi.PrettyMIDI(str(midi_path)) # parses the MIDI
        resolution = midi.resolution # records the MIDIs ticks per quarter note
        notes = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
     
            for note in instrument.notes:
                start_tick = midi.time_to_tick(note.start)
                end_tick = midi.time_to_tick(note.end)

                onset = clean_number(start_tick / resolution)
                duration = clean_number((end_tick - start_tick) / resolution)

                if duration > 0:
                    notes.append((onset, duration, int(note.pitch)))

        notes.sort(key=lambda item: (item[0], item[2]))
        melodies[midi_path.stem] = notes

    return melodies

def load_tune_families(mtc_root):
    """Load the expert-assigned tune family for every melody. It takes the path to 
    the MTC-ANN-2.0.1 dataset and returns a dict with keys "song ID" and its values 
    are the tune family for that melody."""

    mtc_root = Path(mtc_root)
    labels_path = (mtc_root/"metadata"/"MTC-ANN-tune-family-labels.csv")

    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Tune-family labels file not found: {labels_path}"
        )

    table = pd.read_csv(
        labels_path,
        header=None,
        names=["songid", "tunefamily"],
    )

    table = table.dropna(subset=["songid", "tunefamily"])
    table["songid"] = table["songid"].astype(str).str.strip()
    table["tunefamily"] = table["tunefamily"].astype(str).str.strip()

    table = table[
        (table["songid"] != "")
        & (table["tunefamily"] != "")
    ]

    duplicated = table.loc[
        table["songid"].duplicated(),
        "songid",
    ].tolist()

    if duplicated:
        raise ValueError(
            f"Duplicate song IDs in tune-family metadata: {duplicated[:5]}"
        )

    return dict(zip(table["songid"], table["tunefamily"]))


if __name__ == "__main__":
    root = Path("data/MTC-ANN-2.0.1")

    melodies = load_melodies(root)
    families = load_tune_families(root)

    melody_ids = set(melodies)
    family_ids = set(families)

    print(f"Loaded melodies: {len(melodies)}")
    print(f"Loaded labels:   {len(families)}")
    print(f"Tune families:   {len(set(families.values()))}")
    print(f"Missing labels:  {sorted(melody_ids - family_ids)}")
    print(f"Missing MIDIs:   {sorted(family_ids - melody_ids)}")

    if melodies:
        first_id = sorted(melodies)[0]
        print(f"\nExample: {first_id}")
        print(melodies[first_id][:5])
        print(f"Family: {families.get(first_id)}")