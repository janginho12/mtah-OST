import numpy as np

def missing_runs(ball):
    missing = ~np.isfinite(ball).all(axis=1)
    edges = np.diff(np.r_[False, missing, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def fixed_distance_weights(baseline, anchor):
    distances = np.linalg.norm(baseline-anchor, axis=1)
    if not np.isfinite(distances).all() or not len(distances):
        raise ValueError("Distance weights require finite selected positions")
    weights = 1/(distances+1.)
    return weights/weights.sum()


def reconstruct_gap(players, ball, start, end, pitch, distance_weighted=False):
    """Return fixed slot indices, estimates through reappearance, counts, status.

    end is exclusive for missing frames. Its estimate, when available, is only
    an endpoint diagnostic and must not overwrite the reobserved ball.
    """
    if not 0 <= start < end <= len(ball) or len(players) != len(ball):
        raise ValueError("Invalid gap bounds or mismatched tracking lengths")
    if np.isfinite(ball[start:end]).all(axis=1).any():
        raise ValueError("Requested gap contains an observed ball")
    stop = min(end + 1, len(ball))
    estimates = np.full((stop - start, 2), np.nan)
    counts = np.zeros(stop - start, dtype=int)
    empty = np.array([], dtype=int)
    if start == 0 or not np.isfinite(ball[start - 1]).all():
        return empty, estimates, counts, "no_observed_anchor"
    anchor = ball[start - 1]
    if ((anchor < 0) | (anchor > np.asarray(pitch))).any():
        return empty, estimates, counts, "anchor_outside_pitch"
    baseline = players[start - 1]
    available = np.flatnonzero(np.isfinite(baseline).all(axis=1))
    if len(available) < 3:
        return empty, estimates, counts, "fewer_than_three_initial_players"
    distances = np.linalg.norm(baseline[available] - anchor, axis=1)
    selected = available[np.argsort(distances, kind="stable")[:3]]
    displacement = players[start:stop, selected] - baseline[selected]
    valid = np.isfinite(displacement).all(axis=2)
    counts = valid.sum(axis=1)
    usable = counts > 0
    weights = (fixed_distance_weights(baseline[selected], anchor) if distance_weighted
               else np.ones(3))
    available_weights = valid*weights
    sums = (np.where(valid[:, :, None], displacement, 0)*weights[None, :, None]).sum(axis=1)
    estimates[usable] = anchor + sums[usable] / available_weights.sum(axis=1)[usable, None]
    return selected, estimates, counts, "attempted"
