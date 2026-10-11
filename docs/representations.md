# Melody representations

Before two melodies can be compared they have to be turned into sequences. The choice of representation decides what the distance measures can even *see*: a measure working on intervals cannot tell a melody from its transposition, and a measure working on contour cannot tell a small step from a leap. So the representation is a choice about which differences between two melodies should count, and which should be ignored.

Code: `src/representations.py`

## Input

The loader gives every melody as a list of notes `(onset, duration, pitch)`, with onset and duration in quarter notes and pitch as a MIDI note number. The representations only use the pitch and duration of each note, in order.

## The three representations

We use the opening of *Twinkle Twinkle Little Star* (C C G G A A G) as a running example:

| Representation | What each element is | Twinkle |
|---|---|---|
| `pitch` | MIDI note number | `[60, 60, 67, 67, 69, 69, 67]` |
| `interval` | semitone step to the next note | `[0, 7, 0, 2, 0, -2]` |
| `contour` | direction of the step: −1 down, 0 same, +1 up | `[0, 1, 0, 1, 0, -1]` |

Each representation throws away more information than the one before:

- **Pitch** keeps everything, including the key. The same tune sung a tone higher gives a completely different sequence (every value +2).
- **Interval** is **transposition invariant**: only the steps between notes are kept, so the melody in any key gives the same sequence. This matters for MTC-ANN, where variants of the same tune are often written down in different keys.
- **Contour** also forgets the *size* of each step and keeps only its direction. It is robust to a singer who hits a slightly different note, but it is very coarse: with only three symbols, many unrelated melodies share long stretches of the same contour.

Intervals and contour are one element shorter than pitch, because they describe the steps *between* notes.

## With and without durations

Each of the three representations also exists with durations (`pitch+dur`, `interval+dur`, `contour+dur`). Every element then becomes a tuple `(value, duration)`:

```
interval+dur:  [(0, 1), (7, 1), (0, 1), (2, 1), (0, 1), (-2, 2)]
```

- For **pitch**, each note gets its own duration.
- For **interval** and **contour**, each step is paired with the duration of the note it *arrives on*. The first note's duration is dropped so that values and durations line up. We lose the length of the very first note, which we consider a small price, since it is a single note in a melody of typically 30–60 notes.

Durations are in quarter notes, as given by the score, so the tempo of the MIDI file plays no role.

This gives **3 representations × 2 = 6 sequence types**, which combined with the 3 distance measures gives the 18 methods we evaluate.

## What we expect

There is a trade-off between **invariance** (ignoring differences that don't matter) and **discrimination** (keeping differences that do):

- Pitch should do worst, since every transposed variant looks completely different to it.
- Intervals should do best, since they ignore the key but keep the shape of the melody precisely.
- Contour may do worse than intervals, because it throws away so much that unrelated melodies start to look alike.
- Durations could help (rhythm is part of a tune's identity) or hurt (variants of the same tune often differ in rhythm, e.g. an extra syllable in the lyrics splits a note in two). Which of these effects is larger is one of the things the results will show.

## Limitations

- **No transposition normalisation for pitch.** Pitch is used as an "absolute" baseline on purpose. An alternative would be to shift each melody, e.g. by its median pitch, which we leave as possible future work.
- **Durations are absolute.** A variant notated with all note values halved (e.g. 2/4 instead of 4/4) gets completely different durations, even though it sounds the same.
- **Rests are not represented separately.** How rests are counted depends on how the loader defines duration.
- **Monophony is assumed.** MTC-ANN melodies are monophonic, so this is not a problem here, but the representations cannot handle chords.

## Tests

The script tests itself with `python src/representations.py`. Besides checking the twinkle sequences above, it checks the central property directly: transposing twinkle up two semitones changes `pitch` but leaves `interval` and `contour` unchanged.
