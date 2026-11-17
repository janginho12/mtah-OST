from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from starter.detect_dead_ball import runs
from starter.refine_dead_ball_restarts import classify_zone, defending_team, kick_contact_evidence, player_evidence, valid_ball_window

@dataclass(frozen=True)
class SetPieceConfig:
    min_wait_seconds: float = 3.
    max_extension_seconds: float = 5.
    prior_motion_seconds: float = 3.
    post_motion_seconds: float = 2.
    valid_motion_fraction: float = .8
    stable_motion_fraction: float = .6
    low_speed_mps: float = 1.5
    reactivation_increase_mps: float = .5
    minimum_raw_before_fraction: float = .5
    max_interpolated_run_seconds: float = .2
    penalty_spot_tolerance_m: float = 1.5
    center_spot_tolerance_m: float = 2.

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(name+" must be positive and finite")
            if "fraction" in name and value > 1:
                raise ValueError(name+" must be <=1")
        return self


def restart_kind(origin, pitch, half, directions, config, restart):
    for line, x in (("x_min", 11.), ("x_max", pitch[0]-11.)):
        if np.linalg.norm(origin-np.array([x, pitch[1]/2])) <= config.penalty_spot_tolerance_m:
            defender = defending_team(line, half, directions)
            return "penalty_like", ("A" if defender == "H" else "H"), line
    if np.linalg.norm(origin-np.asarray(pitch)/2) <= config.center_spot_tolerance_m:
        return "kickoff_like", "X", ""
    if classify_zone(origin, pitch, restart) is not None:
        return "line_restart_region", "X", ""
    return "free_kick_like", "X", ""


def launch_candidate(t, raw, mixed, players, teams, fps, pitch, half, start, directions, config, restart):
    pre = max(2, round(restart.stationary_seconds*fps))
    post = max(2, round(restart.post_seconds*fps))
    if t < pre or t+post >= len(raw):
        return None
    before, after = mixed[t-pre:t], raw[t:t+post+1]
    observed_before = np.isfinite(raw[t-pre:t]).all(axis=1)
    if observed_before.mean() < config.minimum_raw_before_fraction:
        return None
    if any(b-a > config.max_interpolated_run_seconds*fps for a, b in runs(~observed_before)):
        return None
    if not valid_ball_window(np.concatenate((before, after)), fps, restart):
        return None
    origin = np.median(before, axis=0)
    if not ((origin >= 0).all() and (origin <= pitch).all()):
        return None
    if np.max(np.linalg.norm(before-origin, axis=1)) > restart.stationary_radius_m:
        return None
    if np.median(np.linalg.norm(np.diff(before, axis=0), axis=1)*fps) > restart.stationary_speed_mps:
        return None
    displacement = after[-1]-origin
    distance = float(np.linalg.norm(displacement))
    if distance < restart.departure_displacement_m or distance/(post/fps) < restart.departure_speed_mps:
        return None
    if not ((after[-1] >= 0).all() and (after[-1] <= pitch).all()):
        return None
    moved = np.flatnonzero(np.linalg.norm(after-origin, axis=1) >= restart.visible_movement_m)
    launch = t+int(moved[0])
    kind, expected, goal_line = restart_kind(origin, pitch, half, directions, config, restart)
    if kind == "line_restart_region":
        return None
    team, _ = player_evidence(players, origin, launch, teams, fps, pitch, restart)
    distances, frames, slots = kick_contact_evidence(raw, players, launch, teams, fps, pitch, restart)
    status = "supported"
    contact_frame, slot, contact_distance = None, None, None
    if team == "X":
        status = "ambiguous_or_missing_player"
    else:
        side = "HA".index(team)
        contact_distance = distances[side] if np.isfinite(distances[side]) else None
        if distances[side] > restart.max_kick_contact_distance_m:
            status = "no_raw_close_contact"
        else:
            contact_frame, slot = start+frames[side], slots[side]
        if expected != "X" and team != expected:
            status = "wrong_penalty_attacking_team"
    if kind == "penalty_like" and displacement[0]*(1 if goal_line == "x_max" else -1) <= 0:
        status = "penalty_not_forward"
    return {"half": half, "frame": start+launch, "restart_type": kind, "restart_team": team,
            "expected_team": expected, "origin_x_m": float(origin[0]), "origin_y_m": float(origin[1]),
            "ball_evidence_status": status, "status": status, "contact_distance_m": contact_distance,
            "contact_frame": contact_frame, "contact_player_slot": slot,
            "pre_ball_interpolated_frames": int((~observed_before).sum()),
            "departure_distance_m": distance}


def motion_confirmation(row, features, fps, config):
    t = row["frame"]
    pre = features[(features.frame >= t-config.prior_motion_seconds*fps) & (features.frame < t)]
    post = features[(features.frame >= t) & (features.frame < t+config.post_motion_seconds*fps)]
    rate = fps/5
    enough_pre = pre.all_smooth_mps.notna().sum() >= np.ceil(config.prior_motion_seconds*rate*config.valid_motion_fraction)
    enough_post = post.all_smooth_mps.notna().sum() >= np.ceil(config.post_motion_seconds*rate*config.valid_motion_fraction)
    before = float(pre.all_smooth_mps.median()) if enough_pre else np.nan
    after = float(post.all_smooth_mps.median()) if enough_post else np.nan
    stable_fraction = float(pre.pooled_stable_support.mean()) if len(pre) else 0.
    low = enough_pre and before <= config.low_speed_mps and stable_fraction >= config.stable_motion_fraction
    reactivated = enough_post and after >= config.low_speed_mps and after-before >= config.reactivation_increase_mps
    return {"pre_mean_speed_mps": before, "post_mean_speed_mps": after,
            "pre_stable_fraction": stable_fraction, "prior_collective_stop": bool(low),
            "post_collective_reactivation": bool(reactivated)}


def refine_setpieces(events, intervals, features, raw, mixed, players, teams, directions, blocks, fps, pitch, config, restart):
    revised = events.copy(deep=True)
    revised["end_basis"] = revised.end_basis.astype(object)
    for column, default in (("restart_type", "unknown"), ("restart_team", "X")):
        if column not in revised:
            revised[column] = default
    proposals, audit = [], []
    revised["setpiece_proposal_id"] = np.nan
    for group in intervals.to_dict("records"):
        half = int(group["half"])
        ids = [int(value) for value in str(group["event_ids"]).split(";")]
        members = events[events.event_id.isin(ids)]
        original_end = int(group["end_frame_exclusive"])
        row = {"interval_id": int(group["interval_id"]), "half": half, "start_frame": int(group["start_frame"]),
               "old_end_frame_exclusive": original_end, "new_end_frame_exclusive": original_end,
               "status": "no_supported_setpiece_restart", "proposal_id": None, "restart_type": "unknown",
               "restart_team": "X", "source_event_ids": group["event_ids"]}
        audit.append(row)
        if members.method.isin(["line_dropout", "observed_line_exit"]).any():
            row["status"] = "existing_line_evidence_kept"
            continue
        _, start, last = next(block for block in blocks if block[0] == half)
        following = intervals[(intervals.half == half) & (intervals.start_frame >= original_end)]
        next_start = int(following.start_frame.min()) if len(following) else last+1
        stop = min(last+1, next_start, original_end+round(config.max_extension_seconds*fps))
        first = max(start, int(group["start_frame"])+round(config.min_wait_seconds*fps))
        row["search_start_frame"], row["search_end_exclusive"] = first, stop
        half_features = features[features.half == half]
        local_rows = []
        for frame in range(first, stop):
            candidate = launch_candidate(frame-start, raw[start:last+1], mixed[start:last+1],
                                         players[start:last+1], teams, fps, pitch, half, start, directions, config, restart)
            if candidate is None or candidate["frame"] >= stop:
                continue
            candidate.update(motion_confirmation(candidate, half_features, fps, config))
            if candidate["status"] == "supported":
                if not candidate["prior_collective_stop"]:
                    candidate["status"] = "no_prior_collective_stop"
                elif candidate["restart_type"] != "penalty_like" and not candidate["post_collective_reactivation"]:
                    candidate["status"] = "no_collective_reactivation"
            if any(p["status"] == candidate["status"] and abs(p["frame"]-candidate["frame"]) < fps
                   for p in local_rows[-20:]):
                continue
            candidate["interval_id"] = row["interval_id"]
            candidate["proposal_id"] = len(proposals)+1
            proposals.append(candidate)
            local_rows.append(candidate)
        supported = sorted((p for p in local_rows if p["status"] == "supported"), key=lambda p: p["frame"])
        row["geometric_proposals"] = len(local_rows)
        row["supported_proposals"] = len(supported)
        if not supported:
            continue
        chosen = supported[0]
        end = int(chosen["frame"])
        mask = revised.event_id.isin(ids)
        seed = int(members.sort_values("start_frame").iloc[0].event_id)
        revised.loc[revised.event_id.eq(seed), "end_frame_exclusive"] = end
        revised.loc[mask, "end_frame_exclusive"] = np.minimum(revised.loc[mask, "end_frame_exclusive"], end)
        revised.loc[mask, "setpiece_proposal_id"] = chosen["proposal_id"]
        revised.loc[mask, "restart_type"] = chosen["restart_type"]
        revised.loc[mask, "restart_team"] = chosen["restart_team"]
        revised.loc[mask & revised.end_frame_exclusive.eq(end), "end_basis"] = "stationary_raw_kick_and_collective_restart_hypothesis"
        row.update(status="refined", new_end_frame_exclusive=end, proposal_id=chosen["proposal_id"],
                   restart_type=chosen["restart_type"], restart_team=chosen["restart_team"])
    revised = revised[revised.end_frame_exclusive > revised.start_frame].copy()
    revised["duration_seconds"] = (revised.end_frame_exclusive-revised.start_frame)/fps
    return revised, pd.DataFrame(proposals), pd.DataFrame(audit)
