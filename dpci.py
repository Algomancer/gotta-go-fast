import numpy as np

_index = np.array([10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11])


def get_dpci(a, b):
    """
    Computes the directed pitch class interval vector for
    the chord transition a -> b.
    Args:
        a, b - lists of pitch values
    Returns:
        A 12-vector, with the frequency counts of each directed pitch
        interval for the notes in a and the notes in b.
        Indices are arranged [0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5, 6]
    """
    dist = np.expand_dims(b, axis=1) - a
    dist %= 12
    dist[dist > 6] -= 12
    dist = dist.reshape(-1)
    indices = _index[dist+5]
    return np.bincount(indices, minlength=12)


def get_dpci_matrix(transitions):
    """
    Given a binary pitch transition matrix (128, seq_len)
    :param transitions:
    :return:
    """
    notes = (np.where(transitions[:, i] > 0)[0]
             for i in range(transitions.shape[1]))
    notes = [n for n in notes if len(n) > 0]
    out = [get_dpci(notes[i], notes[i+1]) for i in range(len(notes)-1)]

    return np.stack(out, axis=1)


input = np.load('matrix.npy')
output = get_dpci_matrix(input)
print(output)
