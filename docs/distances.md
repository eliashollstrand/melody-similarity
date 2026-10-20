# Distance measures

Given two melodies as sequences (from `representations.py`), a distance measure gives a single number for how different they are. We implement three classical measures that look at very different things: **edit distance** counts the changes needed, **DTW** measures how far apart the notes are after the best possible stretching, and **n-gram overlap** counts shared short fragments, regardless of where they occur.

Code: `src/distances.py`

All three are turned into **similarities** (higher = more similar), so that every method produces the same kind of matrix. For the evaluation (MAP) only the *ranking* of melodies matters, so the exact scale of each similarity is not important.

Running example (intervals):

```
A = [0, 7, 0, 2, 0, -2]       twinkle
B = [0, 7, 0, 2, 0, -4]       last step changed
C = [0, 7, 0, 0, 2, 0, -2]    one extra repeated note
```

## Edit distance

The edit distance is the smallest number of **insertions, deletions and substitutions** needed to turn one sequence into the other. It is computed with dynamic programming: a table where cell `DP[i][j]` holds the distance between the first `i` elements of `a` and the first `j` elements of `b`. Each cell is the cheapest of three moves: delete (from the cell above, +1), insert (from the cell to the left, +1), or match/substitute (diagonally, +0 if the elements are equal, +1 otherwise). The answer is in the bottom-right cell.

A → B needs one substitution (−2 → −4), and A → C one insertion (the extra 0), so both have distance 1.

To make melodies of different lengths comparable, the distance is normalised by the length of the longer sequence:

$$\text{edit\_similarity}(a, b) = 1 - \frac{\text{edit}(a, b)}{\max(|a|, |b|)}$$

which is 1 for identical sequences and 0 when nothing matches.

**Character:** edit distance only knows whether two elements are *equal or not*. A wrong note one semitone off costs the same as a wrong note an octave off.

## Dynamic time warping (DTW)

DTW finds the best **alignment** of the two sequences, where one note may be paired with several notes in the other sequence. It is the same algorithm we used on audio features in the labs, here applied to note sequences.

1. **Local cost:** `C[n, m] = |a[n] − b[m]|`, how far apart two elements are. With durations, the differences in value and duration are added together.
2. **Accumulated cost:** `D[n, m] = C[n, m] + min(D[n−1, m−1], D[n−1, m], D[n, m−1])`, the cheapest total cost of aligning the beginnings of the two sequences up to these positions. The first row and column are running sums.
3. **Traceback:** from the bottom-right cell, walk back to the start, always to the cheapest neighbour. This gives the optimal path, i.e. the list of aligned note pairs.
4. **Distance:** the total cost divided by the number of pairs on the path, so that long melodies are not penalised just for being long.

The distance has no upper bound, so it is turned into a similarity with

$$\text{dtw\_similarity}(a, b) = \frac{1}{1 + \text{dtw}(a, b)}$$

**Character:** unlike edit distance, DTW uses the *size* of the difference: a step of −4 instead of −2 costs 2, not 1. And it can stretch: in A vs C, A's third element (0) is paired with *both* zeros in C, so the extra repeated note costs nothing and the distance is 0. This fits folk song variants well, where a note is often split in two for an extra syllable. On the other hand, DTW cannot skip a note: every element must be paired with something, and the first and last notes of the two melodies are always paired with each other. An extra phrase in one variant must therefore be squeezed onto notes of the other melody, where it adds cost.

## N-gram overlap

An n-gram is a run of `n` consecutive elements. We take all overlapping 3-grams of each melody as a **set** and compare the sets with the Jaccard index:

$$\text{ngram\_similarity}(a, b) = \frac{|A \cap B|}{|A \cup B|}$$

For A and B, three of the five distinct 3-grams are shared, giving 3/5 = 0.6.

**Character:** n-grams ignore *where* in the melody a fragment occurs, so reordered phrases or a variant with an extra verse still share most of their fragments. Within a fragment, however, everything must match exactly: one changed note breaks up to three 3-grams. Using a set means that repeated fragments are only counted once. We chose `n = 3` as a compromise: shorter fragments (n = 1, 2) are shared by almost all melodies, while longer ones are too rarely repeated exactly between variants.

If both melodies are shorter than `n`, they have no n-grams; they are then counted as equal only if they are identical.

## How the measures differ

The same pairs give quite different scores:

| | edit | DTW | n-gram |
|---|---|---|---|
| A vs B (one note changed) | 0.83 | 0.75 | 0.60 |
| A vs C (one note repeated) | 0.86 | **1.00** | 0.50 |

| | Edit distance | DTW | N-gram overlap |
|---|---|---|---|
| Compares | equal / not equal | size of difference | equal / not equal |
| Order of notes | matters | matters | only within 3 notes |
| Extra notes | cost 1 each | free if a repetition | break the surrounding n-grams |

## Durations

With durations, each element is a tuple `(value, duration)`:

- **Edit distance and n-grams** compare tuples for exact equality, so durations make matching **stricter**: two notes now only match if both the value and the duration are the same.
- **DTW** adds the two differences: `|Δvalue| + |Δduration|`. This weights one semitone the same as one quarter note, which is an arbitrary choice. For contour, whose values only lie between −1 and +1, the durations get relatively more weight than for pitch.

## Similarity matrices

`similarity_matrix(sequences, sim_fn)` computes the similarity between every pair of the 360 melodies, giving a 360 × 360 matrix. Since all three measures are symmetric, only the upper half is computed and copied to the lower half (64,620 pairs instead of 129,600). The melody IDs are sorted, so that row `i` in every matrix refers to the same melody. Each matrix is saved as `results/<representation>_<measure>.npz` together with the list of IDs.

Edit distance and DTW take time proportional to the product of the two sequence lengths; n-grams are close to linear. With melodies of a few dozen notes, all 18 matrices can be computed in pure Python in a few minutes.

## Limitations

- **Hand-picked parameters:** n = 3 for n-grams and equal weighting of value and duration in DTW were chosen, not tuned. Both could be varied to see how sensitive the results are.
- **Global alignment:** edit distance and DTW compare whole melodies. Variants with an extra or missing phrase would be better served by comparing the best-matching parts.
- **No musical weighting:** every note counts the same, whereas stressed notes on the beat are often more important for recognising a tune.

## Tests

`python src/distances.py` runs the examples above: one substitution and one insertion give edit distance 1, the extra repeated note gives DTW distance 0, the 3-gram overlap of A and B is 3/5, and similarity matrices are symmetric with ones on the diagonal.