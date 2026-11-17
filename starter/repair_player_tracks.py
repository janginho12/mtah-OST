from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from starter.detect_dead_ball import runs

@dataclass(frozen=True)
class Config:
    pitch_tolerance_m: float = 2.
    max_speed_mps: float = 12.
    max_fill_seconds: float = 3.
    short_fill_seconds: float = 1.
    longer_gap_velocity_tolerance_mps: float = 1.
    longer_gap_prediction_tolerance_m: float = 1.
    motion_context_seconds: float = .4
    endpoint_velocity_tolerance_mps: float = 3.
    crossing_distance_m: float = 3.
    fill_ambiguity_distance_m: float = 1.5
    association_improvement_m: float = 1.
    max_association_residual_m: float = 1.5
    measurement_std_m: float = .3
    acceleration_std_mps2: float = 5.


def cleanup(raw, fps, pitch, config):
    p = raw.copy()
    observed = np.isfinite(raw).all(axis=2)
    status = np.where(observed, 0, 1).astype(np.uint8)
    bad = observed & ((raw < -config.pitch_tolerance_m) | (raw > pitch+config.pitch_tolerance_m)).any(axis=2)
    p[bad] = np.nan
    status[bad] = 2
    speed = np.linalg.norm(np.diff(p, axis=0), axis=2)*fps
    fast = speed > config.max_speed_mps
    bad = np.zeros(observed.shape, bool)
    bad[:-1] |= fast
    bad[1:] |= fast
    p[bad] = np.nan
    status[bad] = 3
    return p, status


def kalman_weights(n, dt, config):
    """Linear weights of a scalar CV Kalman filter fitted to n observations.

    All candidate contexts are complete, so shared covariance/weights are exact;
    the weights give [position, velocity] for each axis without refiltering slots.
    """
    f = np.array([[1., dt], [0., 1.]])
    g = np.array([dt*dt/2, dt])
    q = np.outer(g, g)*config.acceleration_std_mps2**2
    r = config.measurement_std_m**2
    p = np.diag([r, 25.])
    weights = np.zeros((2, n))
    weights[0, 0] = 1.
    for i in range(1, n):
        weights = f @ weights
        p = f @ p @ f.T+q
        gain = p[:, 0]/(p[0, 0]+r)
        weights -= gain[:, None]*weights[0].copy()[None, :]
        weights[:, i] += gain
        residual = np.eye(2)-np.outer(gain, [1., 0.])
        p = residual @ p @ residual.T+np.outer(gain, gain)*r
    return weights


def stitch_candidates(p, teams, keepers, fps, config, half, first, pitch):
    """One-to-one candidate joins of distinct, non-overlapping provider slots.

    Births/deaths near pitch boundaries are excluded to reduce substitution and
    sideline confusion. This is local tracklet continuity, not verified identity.
    """
    n = max(3, round(config.motion_context_seconds*fps))
    weights = kalman_weights(n, 1/fps, config)
    tracklets = []
    for slot in range(p.shape[1]):
        for a, b in runs(np.isfinite(p[:, slot]).all(axis=1)):
            if b-a < 2*n:
                continue
            pre, post = p[a:a+n, slot], p[b-n:b, slot]
            if max(np.linalg.norm(np.diff(pre, axis=0), axis=1).max(),
                   np.linalg.norm(np.diff(post, axis=0), axis=1).max())*fps > config.max_speed_mps:
                continue
            tracklets.append({"slot": slot, "a": a, "b": b, "start": pre[0], "end": post[-1],
                              "v_start": weights[1] @ pre, "v_end": weights[1] @ post})
    records = []
    for team in (0, 1):
        for keeper in (False, True):
            candidates = [row for row in tracklets if teams[row["slot"]] == team and keepers[row["slot"]] == keeper]
            ends = [row for row in candidates if row["b"] < len(p) and (row["end"] > 3).all() and (row["end"] < pitch-3).all()]
            starts = [row for row in candidates if row["a"] > 0 and (row["start"] > 3).all() and (row["start"] < pitch-3).all()]
            if not ends or not starts:
                continue
            cost = np.full((len(ends), len(starts)+len(ends)), 2.5)
            cost[:, :len(starts)] = 1e6
            details = {}
            for i, end in enumerate(ends):
                for j, begin in enumerate(starts):
                    delta = begin["a"]-end["b"]+1
                    if begin["slot"] == end["slot"] or delta <= 0 or delta > config.max_fill_seconds*fps:
                        continue
                    # Reject a join if the original slot is already visible again,
                    # or the new slot existed alongside the old end context.
                    if np.isfinite(p[begin["a"]:begin["a"]+n, end["slot"]]).all(axis=1).any():
                        continue
                    if np.isfinite(p[end["b"]-n:end["b"], begin["slot"]]).all(axis=1).any():
                        continue
                    dt = delta/fps
                    forward = float(np.linalg.norm(end["end"]+end["v_end"]*dt-begin["start"]))
                    backward = float(np.linalg.norm(begin["start"]-begin["v_start"]*dt-end["end"]))
                    dv = float(np.linalg.norm(end["v_end"]-begin["v_start"]))
                    if max(forward, backward) > config.max_association_residual_m or dv > config.endpoint_velocity_tolerance_mps:
                        continue
                    cost[i, j] = (forward+backward)/2+.2*dv
                    details[i, j] = (forward, backward, dv)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if (i, j) not in details:
                    continue
                end, begin = ends[i], starts[j]
                forward, backward, dv = details[i, j]
                records.append({"half": half, "source_slot_1based": end["slot"]+1,
                                "destination_slot_1based": begin["slot"]+1,
                                "end_frame_exclusive": first+end["b"], "start_frame": first+begin["a"],
                                "gap_seconds": (begin["a"]-end["b"])/fps,
                                "forward_residual_m": forward, "backward_residual_m": backward,
                                "velocity_difference_mps": dv,
                                "meaning": "one-to-one tracklet continuity candidate; not applied to IDs"})
    return records


def identity_candidates(p, teams, keepers, fps, config, half, first):
    n = max(3, round(config.motion_context_seconds*fps))
    weights = kalman_weights(n, 1/fps, config)
    dt = np.arange(1, n+1)/fps
    records, last_pair = [], {}
    uncertain = np.zeros(p.shape[:2], bool)
    for t in range(n, len(p)-n, 5):
        before, after = p[t-n:t], p[t:t+n]
        complete = np.isfinite(before).all(axis=(0, 2)) & np.isfinite(after).all(axis=(0, 2))
        complete &= (np.linalg.norm(np.diff(before, axis=0), axis=2)*fps <= config.max_speed_mps).all(axis=0)
        complete &= (np.linalg.norm(np.diff(after, axis=0), axis=2)*fps <= config.max_speed_mps).all(axis=0)
        for side in (0, 1):
            for keeper in (False, True):
                slots = np.flatnonzero(complete & (teams == side) & (keepers == keeper))
                if len(slots) < 2:
                    continue
                state = np.einsum('wn,nsc->swc', weights, before[:, slots])
                predicted = state[:, 0, None, :]+state[:, 1, None, :]*dt[None, :, None]
                future = after[:, slots].transpose(1, 0, 2)
                cost = np.median(np.linalg.norm(predicted[:, None]-future[None], axis=3), axis=2)
                _, columns = linear_sum_assignment(cost)
                for i, j in enumerate(columns):
                    if i >= j or columns[j] != i:
                        continue  # Audit reciprocal pair swaps only; no multi-player permutation.
                    pair = tuple(sorted((int(slots[i]), int(slots[j]))))
                    keep_cost = float(cost[i, i]+cost[j, j])
                    swap_cost = float(cost[i, j]+cost[j, i])
                    gain = keep_cost-swap_cost
                    close = np.linalg.norm(before[-1, slots[i]]-before[-1, slots[j]]) <= config.crossing_distance_m
                    individually_better = min(cost[i, i]-cost[i, j], cost[j, j]-cost[j, i]) >= config.association_improvement_m
                    if not close or not individually_better or max(cost[i, j], cost[j, i]) > config.max_association_residual_m:
                        continue
                    uncertain[t-n:t+n, list(pair)] = True
                    if t-last_pair.get(pair, -fps*2) < fps:
                        continue
                    last_pair[pair] = t
                    records.append({"half": half, "frame": first+t, "slot_a_1based": pair[0]+1,
                                    "slot_b_1based": pair[1]+1, "team": "HA"[side],
                                    "keep_cost_m": keep_cost, "swap_cost_m": swap_cost,
                                    "improvement_m": gain,
                                    "meaning": "motion-supported switch candidate; real turns may look the same; IDs unchanged"})
    return records, uncertain


def mark_stitch_uncertainty(mask, stitches, first, fps, config):
    result = mask.copy()
    context = max(3, round(config.motion_context_seconds*fps))
    for row in stitches:
        a = max(0, int(row["end_frame_exclusive"])-first-context)
        b = min(len(result), int(row["start_frame"])-first+context)
        slots = [int(row["source_slot_1based"])-1, int(row["destination_slot_1based"])-1]
        result[a:b, slots] = True
    return result
