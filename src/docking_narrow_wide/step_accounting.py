"""Count real transitions under Gymnasium's NEXT_STEP vector autoreset."""

import numpy as np


def real_transitions(previous_autoreset):

    return int(np.count_nonzero(~np.asarray(previous_autoreset, dtype=bool)))


def accrue_updates(credit, transitions, utd_ratio):
    credit += transitions * utd_ratio
    updates = int(credit + 1e-12)
    return updates, credit - updates
