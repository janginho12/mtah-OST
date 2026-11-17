from dataclasses import asdict, dataclass
import numpy as np
from starter.detect_dead_ball import runs, samples
from starter.motion_candidates import Settings

@dataclass(frozen=True)
class StabilityConfig:
    max_speed_rate_mps2: float = .5
    max_speed_range_mps: float = .4

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        return self


def stable_support(values, rate, config, stability):
    """Union of full 3-second low, low-slope, narrow-range windows.

    A rate refers to the edge between adjacent samples, not the transition into
    the window. All samples and all internal edges must pass; NaNs break support.
    """
    n = samples(config.min_low_seconds, rate)
    result = np.zeros(len(values), dtype=bool)
    if len(values) < n:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, n)
    ok = (np.isfinite(windows).all(axis=1)
          & (windows <= config.low_team_speed_mps).all(axis=1)
          & (np.ptp(windows, axis=1) <= stability.max_speed_range_mps))
    if n > 1:
        ok &= (np.abs(np.diff(windows, axis=1))*rate <= stability.max_speed_rate_mps2).all(axis=1)
    edges = np.zeros(len(values)+1, dtype=int)
    starts = np.flatnonzero(ok)
    np.add.at(edges, starts, 1)
    np.add.at(edges, starts+n, -1)
    return np.cumsum(edges[:-1]) > 0


def pooled_events(table, half_bounds, fps, config, stability, require_stability):
    first, stop = half_bounds
    bin_frames = Settings().bin_frames
    rate = fps/bin_frames
    context = samples(config.pre_context_seconds, rate)
    min_low = samples(config.min_low_seconds, rate)
    values = table.all_smooth_mps.to_numpy()
    frames = table.frame.to_numpy()
    support = (stable_support(values, rate, config, stability) if require_stability
               else np.isfinite(values) & (values <= config.low_team_speed_mps))
    records, audit = [], []
    for a, b in runs(support):
        row = {"half": int(table.half.iloc[0]), "support_start_frame": int(frames[a])-bin_frames//2,
               "support_end_frame_exclusive": min(stop, int(frames[b-1])+bin_frames//2+1),
               "accepted": False, "reason": "short_support_or_missing_pre_context"}
        audit.append(row)
        if b-a < min_low or a < context:
            continue
        pre = values[a-context:a]
        valid = np.isfinite(pre)
        if valid.sum() < int(np.ceil(context*config.min_context_fraction)):
            row["reason"] = "insufficient_valid_pre_context"
            continue
        before, low = float(np.median(pre[valid])), float(np.median(values[a:b]))
        drop = before-low
        row.update(all_before_mps=before, all_low_mps=low, all_drop_mps=drop)
        if drop < config.min_team_drop_mps:
            row["reason"] = "insufficient_pooled_speed_drop"
            continue
        onset = a
        threshold = before-config.min_team_drop_mps*config.onset_drop_fraction
        while onset > a-context:
            previous = values[onset-1]
            if not np.isfinite(previous) or previous > threshold:
                break
            onset -= 1
        end_basis = ("half_end" if b == len(values) else
                     "motion_observation_lost" if not np.isfinite(values[b]) else
                     "low_stable_activity_ended" if require_stability else "low_activity_ended")
        row.update(accepted=True, reason="pooled_decline_then_stable_low" if require_stability else "pooled_decline_then_low")
        records.append({"half": int(table.half.iloc[0]), "method": "collective_slowing",
                        "start_frame": max(first, int(frames[onset])-bin_frames//2),
                        "end_frame_exclusive": row["support_end_frame_exclusive"],
                        "start_basis": "estimated_pooled_decline_onset", "end_basis": end_basis,
                        "start_resolution_frames": bin_frames, "low_start_frame": row["support_start_frame"],
                        "all_before_mps": before, "all_low_mps": low, "all_drop_mps": drop,
                        "stable_required": require_stability})
    return records, audit, support
