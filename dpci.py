import numpy as np


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

    # define the mapping from directed interval --> vector index
    index = {
        '0': 0,
        '1': 1,
        '-1': 2,
        '2': 3,
        '-2': 4,
        '3': 5,
        '-3': 6,
        '4': 7,
        '-4': 8,
        '5': 9,
        '-5': 10,
        '6': 11
    }

    # initialize counts to zero
    dpci = [0] * 12

    # tally the intervals
    for x in a:
        for y in b:
            # get the absolute distance
            dist = y - x

            # normalise to a pitch interval
            interval = dist % 12

            # convert to directed interval
            # (i.e. distance to closest key in this pitch)
            if interval > 6:
                interval = interval - 12

            i = index[str(interval)]
            dpci[i] += 1

    return np.array(dpci)


def get_dpci_matrix(transitions):
    """
    Given a binary pitch transition matrix (128, seq_len)
    :param transitions: 
    :return: 
    """

    out = []
    prev_notes = np.where(transitions[:, 0] > 0)[0]
    for i in range(1, transitions.shape[1]):
        notes = np.where(transitions[:, i] > 0)[0]

        # skip over empty columns
        if len(notes) == 0:
            continue
        else:
            out.append(get_dpci(prev_notes, notes))
            prev_notes = notes

    return np.array(out).T

input = np.load('matrix.npy')
output = get_dpci_matrix(input)
print(output)
