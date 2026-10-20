"""Melody representations: pitch, interval and contour, with and without durations.

Input:  {songid: [(onset, duration, pitch), ...]}
Output: {representation_name: {songid: sequence}}
"""

def pitch(notes):
    """
    MIDI note number of each note.
    [(0, 1, 60), (1, 1, 67)] -> [60, 67]
    """
    return [p for _, _, p in notes]

def interval(notes):
    """Semitone step between consecutive notes (one element shorter than pitch).
    [60, 60, 67] -> [0, 7]
    """
    p = pitch(notes)
    return [p[i + 1] - p[i] for i in range(len(p) - 1)]

def contour(notes):
    """Direction of each step: -1 down, 0 same, +1 up.
    [0, 7, -2] -> [0, 1, -1]
    """
    return [(i > 0) - (i < 0) for i in interval(notes)]

def add_durations(values, notes):
    """Pair each value with a duration: [(value, duration), ...].

    For pitch: each note's own duration.
    For interval/contour: values is one shorter than notes, so drop the 
    first duration.
    """
    durations = [d for _, d, _ in notes]
    durations = durations[len(durations) - len(values):]
    return list(zip(values, durations))

def build_representations(melodies):
    """All six representations for every melody.
    
    Returns {"pitch": {songid: seq}, "pitch+dur": {...}, "interval": {...}, ...}
    """
    return {
        "pitch":        {songid: pitch(notes) for songid, notes in melodies.items()},
        "pitch+dur":    {songid: add_durations(pitch(notes), notes) for songid, notes in melodies.items()},
        "interval":     {songid: interval(notes) for songid, notes in melodies.items()},
        "interval+dur": {songid: add_durations(interval(notes), notes) for songid, notes in melodies.items()},
        "contour":      {songid: contour(notes) for songid, notes in melodies.items()},
        "contour+dur":  {songid: add_durations(contour(notes), notes) for songid, notes in melodies.items()}
    }

if __name__ == "__main__":
    twinkle = [(0, 1, 60), (1, 1, 60), (2, 1, 67), (3, 1, 67), (4, 1, 69), (5, 1, 69), (6, 2, 67)]
    twinkle_in_d = [(o, d, p + 2) for o, d, p in twinkle] # Transposed up 2 semitones

    assert pitch(twinkle) == [60, 60, 67, 67, 69, 69, 67]
    assert interval(twinkle) == [0, 7, 0, 2, 0, -2]
    assert contour(twinkle) == [0, 1, 0, 1, 0, -1]

    # Transposition: pitch changes, intervals and contour don't
    assert pitch(twinkle_in_d) != pitch(twinkle)
    assert interval(twinkle_in_d) == interval(twinkle)
    assert contour(twinkle_in_d) == contour(twinkle)

    # Durations: same length as the values
    assert len(add_durations(interval(twinkle), twinkle)) == len(interval(twinkle))

    # The full output has all six representations
    reps = build_representations({"twinkle": twinkle})
    assert set(reps) == {"pitch", "interval", "contour", "pitch+dur", "interval+dur", "contour+dur"}

    print("All tests passed")