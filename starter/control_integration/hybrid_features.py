import numpy as np

def _runs(values):
    """All equal-value contiguous runs, including false/zero; exclusive ends."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Runs require a one-dimensional array")
    if not len(values):
        return []
    cuts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return list(zip(cuts[:-1], cuts[1:]))


def _quality(points, blocks, fps, max_speed, pitch, player=True):
    points = np.asarray(points)
    blocks = np.asarray(blocks)
    if (points.ndim not in (2, 3) or points.shape[-1] != 2
            or blocks.shape != (len(points),) or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid position shape, blocks or FPS")
    expanded = blocks.reshape((-1,) + (1,) * (points.ndim - 2))
    valid = np.isfinite(points).all(axis=-1) & (expanded > 0)
    if player:
        valid &= ((points >= 0) & (points <= np.asarray(pitch))).all(axis=-1)
    same = (expanded[1:] == expanded[:-1]) & (expanded[1:] > 0)
    bad = (valid[1:] & valid[:-1] & same
           & (np.linalg.norm(np.diff(points, axis=0), axis=-1) * fps > max_speed))
    jumps = np.zeros_like(valid)
    jumps[1:] |= bad
    jumps[:-1] |= bad
    valid &= ~jumps
    edges = np.zeros_like(valid)
    edges[1:] = valid[1:] & valid[:-1] & same
    return valid, edges, jumps


def _velocity(points, edges, fps, radius):
    points = np.asarray(points)
    edges = np.asarray(edges, bool)
    if (points.shape[:-1] != edges.shape or points.shape[-1] != 2
            or radius < 1 or int(radius) != radius or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid derivative inputs")
    radius = int(radius)
    n = len(points)
    result = np.full(points.shape, np.nan, np.float32)
    if n <= 2 * radius:
        return result
    # Edges [t-r+1, t+r] must all be valid.
    invalid_prefix = np.concatenate((np.zeros((1,) + edges.shape[1:], np.int32),
                                     np.cumsum(~edges, axis=0, dtype=np.int32)))
    t = np.arange(radius, n - radius)
    ok = (invalid_prefix[t + radius + 1] - invalid_prefix[t - radius + 1]) == 0
    velocity = (points[t + radius] - points[t - radius]) * (fps / (2 * radius))
    result[t] = np.where(ok[..., None], velocity, np.nan)
    return result


def _mean(values, radius, blocks, min_fraction=.75):
    values = np.asarray(values, float)
    blocks = np.asarray(blocks)
    if (blocks.shape != (len(values),) or radius < 0 or int(radius) != radius
            or not 0 < min_fraction <= 1):
        raise ValueError("Invalid rolling mean inputs")
    radius = int(radius)
    result = np.full(values.shape, np.nan, np.float32)
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        part = values[lo:hi]
        finite = np.isfinite(part)
        zero = np.zeros((1,) + part.shape[1:])
        sums = np.concatenate((zero, np.cumsum(np.where(finite, part, 0), axis=0)))
        counts = np.concatenate((zero, np.cumsum(finite, axis=0)))
        t = np.arange(hi-lo)
        left, right = np.maximum(0, t-radius), np.minimum(hi-lo, t+radius+1)
        count = counts[right] - counts[left]
        width = (right-left).reshape((-1,) + (1,) * (part.ndim-1))
        ok = (count > 0) & (count >= min_fraction*width)
        result[lo:hi] = np.divide(sums[right]-sums[left], count,
                                 out=np.full(part.shape, np.nan), where=ok)
    return result
