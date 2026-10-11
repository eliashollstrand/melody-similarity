"""Distance / similarity measures between melody sequences.

Input:  sequences from representations.py, e.g. [0, 7, 0, 2] or [(0, 1.0), (7, 1.0), ...]
Output: similarity matrices, higher = more similar
"""
import numpy as np

def edit_distance(a, b):
    """Find minimum number of insertions, deletions and substitutions turning a into b using
    DP array.
    """
    n = len(a)
    m = len(b)
    DP = [[0] * (m + 1) for _ in range(n + 1)] # (n+1) x (m+1) table

    for i in range(n + 1):
        DP[i][0] = i # Turning first i of a into an empty sequence: i deletions
    for j in range(m + 1):
        DP[0][j] = j # Turning an empty sequence into first j of b: j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            DP[i][j] =  min(DP[i-1][j] + 1,
                           DP[i][j-1] + 1,
                           DP[i-1][j-1] + sub)

    return DP[n][m] # The edit distance is the bottom right element


def edit_similarity(a, b):
    """Turn the edit distance into a score between 0 and 1."""
    if len(a) == 0 and len(b) == 0:
        return 1.0
    
    return 1 - (edit_distance(a, b) / max(len(a), len(b)))


def cost(x, y):
    """Cost between two sequence elements."""
    if isinstance(x, tuple):
        # If there are durations, add together the value and duration differences
        return abs(x[0] - y[0]) + abs(x[1] - y[1])
    return abs(x - y)


def cost_matrix(a, b):
    """C[n, m] = cost of pairing note n of a with note m of b."""
    C = np.zeros((len(a), len(b)))
    for n in range(len(a)):
        for m in range(len(b)):
            C[n, m] = cost(a[n], b[m])
    return C


def accumulated_cost(C):
    """D[n, m] = cheapest total cost of aligning a[:n+1] with b[:m+1]."""
    N, M = C.shape
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n - 1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m - 1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n - 1, m - 1], 
                                    D[n - 1, m], 
                                    D[n, m - 1])
    return D


def optimal_path(D):
    """Trace back from the end to the start. Returns [(n, m), ...]."""
    N, M = D.shape
    n, m = N - 1, M - 1
    P = [(n, m)]
    while (n, m) != (0, 0):
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            next_moves = [
                ((n - 1, m - 1), D[n - 1][m - 1]),
                ((n - 1, m), D[n - 1][m]),
                ((n, m - 1), D[n][m - 1]),
            ]
            min_cost = min(move[1] for move in next_moves)

            # If multiple next steps give same minimal cost, take lexicographically smallest cell
            next_coords = [coord for coord, c in next_moves if c == min_cost]
            n, m = min(next_coords)
        
        P.append((n, m))

    return P[::-1] # Reverse the path in case it is needed in order


def dtw_distance(a, b):
    """Total cost of the best alignment divided by the number of aligned pairs."""
    D = accumulated_cost(cost_matrix(a, b))
    P = optimal_path(D)
    return D[-1, -1] / len(P)


def dtw_similarity(a, b):
    """Turn DTW distance into a similarity between 1 and 0."""
    return 1 / (1 + dtw_distance(a, b))


def ngrams(seq, n=3):
    """Set of all overlapping chunks of length n, as tuples."""
    return {tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)}


def ngram_similarity(a, b, n=3):
    """Jaccard index: shared n-grams / all distinct n-grams."""
    A, B = ngrams(a, n), ngrams(b, n)
    if len(A) == 0 and len(B) == 0:
        # If both melodies are shorter than n we only call them equal if they are identical
        return 1.0 if list(a) == list(b) else 0.0

    # Jaccard index is |A ∩ B| / |A ∪ B|
    return len(A.intersection(B)) / len(A.union(B))


def similarity_matrix(sequences, sim_fn):
    """N x N matrix of sim_fn for all pairs in {songid: sequence}.
    Returns (S, ids) with ids in the same order as the rows/columns.
    """
    ids = sorted(sequences)
    N = len(ids)
    S = np.zeros((N, N))
    for i in range(N):
        S[i, i] = sim_fn(sequences[ids[i]], sequences[ids[i]])
        for j in range(i + 1, N): # Since it is symmetric we only compute the upper half
            s = sim_fn(sequences[ids[i]], sequences[ids[j]])
            S[i, j] = S[j, i] = s # Copy to lower half
    return S, ids


def save_matrix(path, S, ids):
    np.savez(path, sim=S, ids=np.array(ids))


# Testing
if __name__ == "__main__":
    A = [0, 7, 0, 2, 0, -2]
    B = [0, 7, 0, 2, 0, -4] # Last step changed
    C = [0, 7, 0, 0, 2, 0, -2] # One extra repeated note

    # Edit distance
    assert edit_distance(A, A) == 0
    assert edit_distance(A, B) == 1 # One substitution
    assert edit_distance(A, C) == 1 # One insertion
    assert edit_similarity(A, A) == 1

    # DTW
    assert dtw_distance(A, A) == 0
    assert dtw_distance([0, 7, 2], [0, 7, 7, 3]) == 1 / 4 # Total cost 1, path length 4
    assert dtw_distance(A, C) == 0 # The extra 0 is absorbed by stretching

    # N-grams
    assert ngrams([1, 2, 3, 4], n=3) == {(1, 2, 3), (2, 3, 4)}
    assert ngram_similarity(A, B, n=3) == 3 / 5

    # With durations work too
    w_dur = [(0, 1.0), (7, 1.0), (0, 1.0)]
    assert edit_distance(w_dur, w_dur) == 0
    assert dtw_distance(w_dur, w_dur) == 0

    # Creating similarity matrix
    S, ids = similarity_matrix({"a": A, "b": B, "c": C}, edit_similarity)
    assert S.shape == (3, 3)
    assert np.allclose(S, S.T) # Verify symmetric
    assert np.allclose(np.diag(S), 1) # Verify that every melody is identical to itself

    print("All tests passed")