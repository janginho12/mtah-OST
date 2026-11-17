from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from starter.motion_candidates import Settings, bin_positions, kinematics

@dataclass(frozen=True)
class Config:
    # Engineering defaults, not selected against this game's public GT.
    ball_history_seconds: float = .2
    ball_max_step_speed_mps: float = 50.
    line_margin_m: float = 2.
    min_outward_speed_mps: float = .5
    max_time_to_line_seconds: float = .75
    min_missing_seconds: float = .4
    min_observed_outside_seconds: float = .2
    motion_smoothing_seconds: float = .6
    low_team_speed_mps: float = 1.5
    min_low_seconds: float = 3.
    pre_context_seconds: float = 3.
    min_team_drop_mps: float = .5
    onset_drop_fraction: float = .5
    min_context_fraction: float = .8

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("{} must be numeric".format(name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(name))
        if self.onset_drop_fraction > 1 or self.min_context_fraction > 1:
            raise ValueError("Fractions must be in (0, 1]")
        return self


EVENT_COLUMNS = [
    "event_id", "half", "method", "start_frame", "end_frame_exclusive",
    "start_video_seconds", "duration_seconds", "start_basis", "end_basis",
    "start_resolution_frames", "low_start_frame", "H_before_mps", "A_before_mps",
    "H_low_mps", "A_low_mps", "H_drop_mps", "A_drop_mps", "line",
    "last_ball_margin_m", "outward_speed_mps", "projected_time_to_line_seconds",
]


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def samples(seconds, fps):
    return max(1, int(np.ceil(seconds * fps)))


def read_tracking(path, n_frames, n_slots):
    """One pass, using the delivered ATD's 1-based video-row frame index."""
    players = np.empty((n_frames, n_slots, 2))
    ball = np.empty((n_frames, 2))
    count = 0
    with open(path) as handle:
        for i, line in enumerate(handle):
            frame, raw_players, raw_ball = line.strip().split(":")
            if i >= n_frames or int(frame) != i + 1:
                raise ValueError("Unexpected ATD frame sequence")
            p = np.fromstring(raw_players.replace(";", ","), sep=",")
            b = np.fromstring(raw_ball, sep=",")
            if p.size != n_slots * 3 or b.size != 3:
                raise ValueError("Unexpected ATD channel count at frame {}".format(i))
            players[i] = p.reshape(n_slots, 3)[:, :2]
            ball[i] = b[:2]
            count += 1
    if count != n_frames:
        raise ValueError("ATD row count differs from tactical video")
    return players, ball


def motion_features(players, start, half, teams, keepers, fps, pitch, config):
    """5 FPS, mean individual speed; no GK, interpolation or missing-as-zero."""
    guards = Settings()
    points, frames, quality = bin_positions(players, start, fps, pitch, guards)
    velocity, _, _ = kinematics(points, guards.bin_frames / fps, guards)
    speed = np.linalg.norm(velocity, axis=2)
    table = pd.DataFrame({"frame": frames, "half": half,
                          "video_seconds": frames / fps})
    width = samples(config.motion_smoothing_seconds, fps / guards.bin_frames)
    # Odd centered window so its timestamp is unambiguous.
    width += 1 - width % 2
    for team, side in enumerate(("H", "A")):
        selected = (teams == team) & ~keepers
        npos = np.isfinite(points[:, selected]).all(axis=2).sum(axis=1)
        values = speed[:, selected]
        count = np.isfinite(values).sum(axis=1)
        good = ((npos >= guards.min_players) & (npos <= guards.max_players)
                & (count >= guards.min_players))
        means = np.full(len(frames), np.nan)
        np.divide(np.nansum(values, axis=1), count, out=means, where=good)
        table[side + "_position_count"] = npos
        table[side + "_velocity_count"] = count
        table[side + "_speed_mps"] = means
        # Complete window required: smoothing cannot bridge lost observations.
        table[side + "_smooth_mps"] = pd.Series(means).rolling(
            width, center=True, min_periods=width).mean().to_numpy()
    table["both_reliable"] = table[["H_smooth_mps", "A_smooth_mps"]].notna().all(axis=1)
    table["both_low"] = (table.both_reliable
                         & (table.H_smooth_mps <= config.low_team_speed_mps)
                         & (table.A_smooth_mps <= config.low_team_speed_mps))
    # Pooled diagnostic/experimental signal: equal weight per observed outfielder,
    # not equal weight per team. Its validity does not require both team means.
    selected = ~keepers
    npos = np.isfinite(points[:, selected]).all(axis=2).sum(axis=1)
    values = speed[:, selected]
    count = np.isfinite(values).sum(axis=1)
    good = ((npos >= 2*guards.min_players) & (npos <= 2*guards.max_players)
            & (count >= 2*guards.min_players))
    pooled = np.full(len(frames), np.nan)
    np.divide(np.nansum(values, axis=1), count, out=pooled, where=good)
    table["all_position_count"] = npos
    table["all_velocity_count"] = count
    table["all_speed_mps"] = pooled
    table["all_smooth_mps"] = pd.Series(pooled).rolling(
        width, center=True, min_periods=width).mean().to_numpy()
    table["all_speed_rate_mps2"] = table.all_smooth_mps.diff()/(guards.bin_frames/fps)
    return table, quality


def ball_history_valid(points, fps, config):
    return (len(points) >= 2 and np.isfinite(points).all()
            and (np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
                 <= config.ball_max_step_speed_mps).all())


def line_evidence(history, fps, pitch, config):
    """Project a short *observed* pre-gap velocity to the nearest plausible line."""
    p = history[-1]
    velocity = (p - history[0]) / ((len(history)-1) / fps)
    margins = np.array([p[0], pitch[0]-p[0], p[1], pitch[1]-p[1]])
    outward = np.array([-velocity[0], velocity[0], -velocity[1], velocity[1]])
    times = np.full(4, np.inf)
    eligible = ((np.abs(margins) <= config.line_margin_m)
                & (outward >= config.min_outward_speed_mps))
    times[eligible] = np.maximum(margins[eligible], 0) / outward[eligible]
    index = int(np.argmin(times))
    if not np.isfinite(times[index]):
        return None
    return {"line": ("x_min", "x_max", "y_min", "y_max")[index],
            "last_ball_margin_m": float(margins[index]),
            "outward_speed_mps": float(outward[index]),
            "projected_time_to_line_seconds": float(times[index])}


def line_events(ball, half, start, fps, pitch, config):
    """Audit every missing run and also retain reliable observed line crossings.

    ball is in metres and restricted to one half. No reconstructed coordinates.
    """
    observed = np.isfinite(ball).all(axis=1)
    history_size = samples(config.ball_history_seconds, fps)
    records, audit = [], []
    for a, b in runs(~observed):
        row = {"half": half, "start_frame": start + a,
               "end_frame_exclusive": start + b, "duration_seconds": (b-a)/fps,
               "accepted": False}
        if b-a < samples(config.min_missing_seconds, fps):
            row["reason"] = "gap_too_short"
        elif a < history_size+1:
            row["reason"] = "no_pre_gap_history_in_half"
        else:
            history = ball[a-history_size-1:a]
            if not ball_history_valid(history, fps, config):
                row["reason"] = "invalid_or_discontinuous_pre_gap_history"
            else:
                evidence = line_evidence(history, fps, pitch, config)
                if evidence is None:
                    row["reason"] = "no_near_line_outward_motion"
                else:
                    row.update(evidence)
                    if evidence["projected_time_to_line_seconds"] > min(
                            config.max_time_to_line_seconds, (b-a)/fps):
                        row["reason"] = "projected_exit_too_late"
                    else:
                        row.update(accepted=True, reason="near_line_outward_dropout")
                        records.append({
                            "half": half, "method": "line_dropout",
                            "start_frame": start+a, "end_frame_exclusive": start+b,
                            "start_basis": "first_missing_frame_not_verified_line_crossing",
                            "end_basis": "ball_reappeared" if b < len(ball) else "half_end",
                            "start_resolution_frames": 1, **evidence,
                        })
        audit.append(row)

    inside = observed & (ball >= 0).all(axis=1) & (ball <= pitch).all(axis=1)
    outside = observed & ~inside
    for a, b in runs(outside):
        if b-a < samples(config.min_observed_outside_seconds, fps) or a < history_size:
            continue
        # Require a continuous observed crossing, not reacquisition outside the pitch.
        if not inside[a-1] or not ball_history_valid(ball[a-history_size:b], fps, config):
            continue
        end_basis = ("half_end" if b == len(ball) else
                     "ball_observation_lost" if not observed[b] else "ball_returned_inside")
        records.append({
            "half": half, "method": "observed_line_exit", "start_frame": start+a,
            "end_frame_exclusive": start+b, "start_basis": "first_observed_outside_frame",
            "end_basis": end_basis, "start_resolution_frames": 1,
        })
    return records, audit


def event_table(records, fps):
    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    table = pd.DataFrame(records).sort_values(
        ["half", "start_frame", "method"], kind="stable").reset_index(drop=True)
    table["event_id"] = np.arange(1, len(table)+1)
    table["start_video_seconds"] = table.start_frame/fps
    table["duration_seconds"] = (table.end_frame_exclusive-table.start_frame)/fps
    return table.reindex(columns=EVENT_COLUMNS)


def merge_intervals(events, fps):
    """Union overlapping/adjacent support only, never bridge a gap or half."""
    merged = []
    for row in events.to_dict("records"):
        a, b, half = int(row["start_frame"]), int(row["end_frame_exclusive"]), int(row["half"])
        if b <= a:
            raise ValueError("Empty/reversed candidate interval")
        if merged and merged[-1]["half"] == half and a <= merged[-1]["end_frame_exclusive"]:
            previous = merged[-1]
            previous["end_frame_exclusive"] = max(previous["end_frame_exclusive"], b)
            previous["methods"].add(row["method"])
            previous["event_ids"].append(str(row["event_id"]))
        else:
            merged.append({"half": half, "start_frame": a, "end_frame_exclusive": b,
                           "methods": {row["method"]}, "event_ids": [str(row["event_id"])]})
    for i, row in enumerate(merged, 1):
        row.update(interval_id=i, duration_seconds=(row["end_frame_exclusive"]-row["start_frame"])/fps,
                   methods=";".join(sorted(row["methods"])), event_ids=";".join(row["event_ids"]))
    return pd.DataFrame(merged, columns=["interval_id", "half", "start_frame", "end_frame_exclusive",
                                         "duration_seconds", "methods", "event_ids"])


def frame_table(n_frames, blocks, events):
    frame = np.arange(n_frames)
    half_id = np.zeros(n_frames, dtype=int)
    for half, start, end in blocks:
        half_id[start:end+1] = half
    data = {"frame": frame, "half": half_id}
    for method in ("line_dropout", "observed_line_exit", "collective_slowing"):
        mask = np.zeros(n_frames, dtype=bool)
        for row in events[events.method.eq(method)].itertuples():
            mask[row.start_frame:row.end_frame_exclusive] = True
        data[method] = mask
    data["is_d_candidate"] = np.logical_or.reduce([data[m] for m in
        ("line_dropout", "observed_line_exit", "collective_slowing")])
    data["status"] = np.where(half_id == 0, "outside_scope",
                              np.where(data["is_d_candidate"], "D_candidate", "unclassified"))
    return pd.DataFrame(data)
