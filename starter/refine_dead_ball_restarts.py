from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RestartConfig:
    zone_tolerance_m: float = 2.
    stationary_seconds: float = .4
    stationary_radius_m: float = .6
    stationary_speed_mps: float = 1.
    post_seconds: float = .4
    departure_displacement_m: float = 1.
    departure_speed_mps: float = 2.
    visible_movement_m: float = .3
    max_ball_step_speed_mps: float = 50.
    player_context_seconds: float = .2
    max_player_distance_m: float = 3.
    max_kick_contact_distance_m: float = 1.
    kick_contact_lookback_seconds: float = .08
    team_distance_margin_m: float = .75
    max_player_step_speed_mps: float = 12.
    player_pitch_tolerance_m: float = 3.
    proposal_separation_seconds: float = 1.
    max_restart_wait_seconds: float = 60.
    min_restart_wait_seconds: float = .2
    throw_location_tolerance_m: float = 10.

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(name))
        if self.min_restart_wait_seconds >= self.max_restart_wait_seconds:
            raise ValueError("Restart wait bounds are reversed")
        return self


PROPOSAL_COLUMNS = ["proposal_id", "half", "frame", "restart_type", "goal_line", "touchline",
                    "origin_x_m", "origin_y_m", "restart_team", "expected_team", "team_status",
                    "H_distance_m", "A_distance_m", "H_contact_distance_m", "A_contact_distance_m",
                    "contact_frame", "contact_player_slot", "departure_distance_m", "boundary_basis"]


def classify_zone(point, pitch, config):
    """Goal area is 5.5 m deep and 18.32 m wide; corner arc radius is 1 m.

    Tolerance accommodates tracking error; it is not a change to football rules.
    """
    x, y = point
    length, width = pitch
    tolerance = config.zone_tolerance_m
    for goal_line, goal_x in (("x_min", 0.), ("x_max", length)):
        for touchline, touch_y in (("y_min", 0.), ("y_max", width)):
            if np.hypot(x-goal_x, y-touch_y) <= 1.+tolerance:
                return "corner_kick", goal_line, touchline
        depth = x if goal_line == "x_min" else length-x
        if -tolerance <= depth <= 5.5+tolerance and abs(y-width/2) <= 7.32/2+5.5+tolerance:
            return "goal_kick", goal_line, ""
    return None


def defending_team(goal_line, half, first_directions):
    directions = np.asarray(first_directions) * (1 if half == 1 else -1)
    wanted = 1 if goal_line == "x_min" else -1
    indexes = np.flatnonzero(directions == wanted)
    if len(indexes) != 1:
        raise ValueError("Expected opposite team attack directions")
    return "HA"[int(indexes[0])]


def player_evidence(players, origin, launch, teams, fps, pitch, config):
    """Team nearest the release origin BEFORE launch, never the later receiver."""
    width = max(2, round(config.player_context_seconds*fps))
    if launch < width:
        return "X", [np.inf, np.inf]
    p = players[launch-width:launch]
    valid = (np.isfinite(p).all(axis=(0, 2))
             & (p >= -config.player_pitch_tolerance_m).all(axis=(0, 2))
             & (p <= np.asarray(pitch)+config.player_pitch_tolerance_m).all(axis=(0, 2)))
    speed = np.linalg.norm(np.diff(p, axis=0), axis=2)*fps
    valid &= (speed <= config.max_player_step_speed_mps).all(axis=0)
    distances = np.full(p.shape[1], np.inf)
    distances[valid] = np.median(np.linalg.norm(p[:, valid]-origin, axis=2), axis=0)
    by_team = [float(distances[(teams == side)].min()) if (teams == side).any() else np.inf
               for side in (0, 1)]
    winner = int(np.argmin(by_team))
    other = 1-winner
    if by_team[winner] <= config.max_player_distance_m and by_team[other]-by_team[winner] >= config.team_distance_margin_m:
        return "HA"[winner], by_team
    return "X", by_team


def valid_ball_window(points, fps, config):
    return (len(points) >= 2 and np.isfinite(points).all()
            and (np.linalg.norm(np.diff(points, axis=0), axis=1)*fps
                 <= config.max_ball_step_speed_mps).all())


def kick_contact_evidence(ball, players, launch, teams, fps, pitch, config):
    """Synchronous raw XY proximity just before/at departure; includes keepers.

    Validate each player's continuous context, but measure contact only in the
    final lookback window. Never count the later receiver or an imputed point.
    Returns per-team minimum distances, local frames and zero-based player slots.
    """
    lookback = max(1, round(config.kick_contact_lookback_seconds*fps))
    context = max(lookback, 2, round(config.player_context_seconds*fps))
    empty = ([np.inf, np.inf], [None, None], [None, None])
    if launch < context or launch >= len(ball):
        return empty
    p = players[launch-context:launch+1]
    valid = (np.isfinite(p).all(axis=(0, 2))
             & (p >= -config.player_pitch_tolerance_m).all(axis=(0, 2))
             & (p <= np.asarray(pitch)+config.player_pitch_tolerance_m).all(axis=(0, 2))
             & (np.linalg.norm(np.diff(p, axis=0), axis=2)*fps
                <= config.max_player_step_speed_mps).all(axis=0))
    observed = ball[launch-lookback:launch+1]
    if not valid_ball_window(observed, fps, config):
        return empty
    distances = np.linalg.norm(p[-len(observed):]-observed[:, None, :], axis=2)
    distances[:, ~valid] = np.inf
    by_team, frames, slots = [], [], []
    for side in (0, 1):
        indexes = np.flatnonzero((teams == side) & valid)
        if not len(indexes):
            by_team.append(np.inf)
            frames.append(None)
            slots.append(None)
            continue
        row, col = np.unravel_index(np.argmin(distances[:, indexes]), (len(observed), len(indexes)))
        by_team.append(float(distances[row, indexes[col]]))
        frames.append(launch-lookback+int(row))
        slots.append(int(indexes[col]))
    return by_team, frames, slots


def find_proposals(ball, players, teams, half, start, fps, pitch, first_directions, config):
    """Find observed stationary launches and touchline entry, within ONE half."""
    pre_n = max(2, round(config.stationary_seconds*fps))
    post_n = max(2, round(config.post_seconds*fps))
    throw_pre_n = max(2, round(config.player_context_seconds*fps))
    rows = []

    def record(local_frame, kind, goal_line, touchline, origin, distance, basis):
        near_team, distances = player_evidence(players, origin, local_frame, teams, fps, pitch, config)
        expected = "X"
        if kind in ("goal_kick", "corner_kick"):
            defender = defending_team(goal_line, half, first_directions)
            expected = defender if kind == "goal_kick" else ("A" if defender == "H" else "H")
        status = ("ambiguous_or_missing_player" if near_team == "X" else
                  "team_conflict" if expected != "X" and expected != near_team else "supported")
        contact_distances, contact_frames, contact_slots = [np.nan, np.nan], [None, None], [None, None]
        contact_frame, contact_slot = None, None
        if kind in ("goal_kick", "corner_kick"):
            contact_distances, contact_frames, contact_slots = kick_contact_evidence(
                ball, players, local_frame, teams, fps, pitch, config)
            if near_team != "X":
                side = "HA".index(near_team)
                if contact_distances[side] <= config.max_kick_contact_distance_m:
                    contact_frame = start+contact_frames[side]
                    contact_slot = contact_slots[side]
                elif status == "supported":
                    status = "no_close_kick_contact"
        rows.append({"half": half, "frame": start+local_frame, "restart_type": kind,
                     "goal_line": goal_line, "touchline": touchline,
                     "origin_x_m": float(origin[0]), "origin_y_m": float(origin[1]),
                     "restart_team": near_team if status == "supported" else "X",
                     "expected_team": expected, "team_status": status,
                     "H_distance_m": distances[0], "A_distance_m": distances[1],
                     "H_contact_distance_m": contact_distances[0],
                     "A_contact_distance_m": contact_distances[1],
                     "contact_frame": contact_frame, "contact_player_slot": contact_slot,
                     "departure_distance_m": float(distance), "boundary_basis": basis})

    for t in range(pre_n, len(ball)-post_n):
        if not np.isfinite(ball[t-1]).all() or classify_zone(ball[t-1], pitch, config) is None:
            continue
        before, after = ball[t-pre_n:t], ball[t:t+post_n+1]
        if not valid_ball_window(ball[t-pre_n:t+post_n+1], fps, config):
            continue
        origin = np.median(before, axis=0)
        zone = classify_zone(origin, pitch, config)
        if zone is None:
            continue
        if np.max(np.linalg.norm(before-origin, axis=1)) > config.stationary_radius_m:
            continue
        if np.median(np.linalg.norm(np.diff(before, axis=0), axis=1)*fps) > config.stationary_speed_mps:
            continue
        displacement = after[-1]-origin
        distance = np.linalg.norm(displacement)
        if distance < config.departure_displacement_m or distance/(post_n/fps) < config.departure_speed_mps:
            continue
        if not ((after[-1] >= 0).all() and (after[-1] <= pitch).all()):
            continue
        kind, goal_line, touchline = zone
        inward_x = displacement[0]*(1 if goal_line == "x_min" else -1)
        if kind == "goal_kick" and inward_x <= 0:
            continue
        if kind == "corner_kick":
            inward_y = displacement[1]*(1 if touchline == "y_min" else -1)
            if min(inward_x, inward_y) < -.2:
                continue
        moved = np.flatnonzero(np.linalg.norm(after-origin, axis=1) >= config.visible_movement_m)
        record(t+int(moved[0]), kind, goal_line, touchline, origin, distance,
               "first_observed_departure_after_stationary_origin")

    for t in range(throw_pre_n, len(ball)-post_n):
        p, q = ball[t-1], ball[t]
        if not np.isfinite([p, q]).all():
            continue
        touchline = ("y_min" if p[1] < 0 <= q[1] else
                     "y_max" if p[1] > pitch[1] >= q[1] else "")
        if not touchline or not valid_ball_window(ball[t-throw_pre_n:t+post_n+1], fps, config):
            continue
        if not (0 <= q[0] <= pitch[0]) or abs(p[1]-(0 if touchline == "y_min" else pitch[1])) > config.zone_tolerance_m:
            continue
        displacement = ball[t+post_n]-p
        inward = displacement[1]*(1 if touchline == "y_min" else -1)
        distance = np.linalg.norm(displacement)
        if inward < config.departure_displacement_m or distance/(post_n/fps) < config.departure_speed_mps:
            continue
        if not ((ball[t+post_n] >= 0).all() and (ball[t+post_n] <= pitch).all()):
            continue
        record(t, "throw_in", "", touchline, p, distance, "first_observed_touchline_entry")

    # Collapse repeated windows of the SAME launch, never choose by GT performance.
    chosen = []
    for row in sorted(rows, key=lambda r: (r["frame"], r["restart_type"])):
        key = (row["restart_type"], row["goal_line"], row["touchline"], row["team_status"])
        if any((r["restart_type"], r["goal_line"], r["touchline"], r["team_status"]) == key
               and abs(row["frame"]-r["frame"]) < config.proposal_separation_seconds*fps
               for r in chosen[-20:]):
            continue
        chosen.append(row)
    return chosen


def exit_geometry(event, ball, pitch):
    frame = int(event["start_frame"])
    point = ball[frame-1] if event["method"] == "line_dropout" else ball[frame]
    if not np.isfinite(point).all():
        return "", point
    line = event.get("line", "")
    if line not in ("x_min", "x_max", "y_min", "y_max"):
        margins = [point[0], pitch[0]-point[0], point[1], pitch[1]-point[1]]
        line = ("x_min", "x_max", "y_min", "y_max")[int(np.argmin(margins))]
    return line, point


def compatible(proposal, line, exit_point, pitch, config):
    if line.startswith("x"):
        if proposal["restart_type"] not in ("corner_kick", "goal_kick") or proposal["goal_line"] != line:
            return False
        if proposal["restart_type"] == "corner_kick":
            return proposal["touchline"] == ("y_min" if exit_point[1] <= pitch[1]/2 else "y_max")
        return True
    return (line.startswith("y") and proposal["restart_type"] == "throw_in"
            and proposal["touchline"] == line
            and abs(proposal["origin_x_m"]-exit_point[0]) <= config.throw_location_tolerance_m)


def refine(events, intervals, proposals, ball, blocks, fps, pitch, config):
    """Pair each line-containing baseline group with one future compatible launch.

    New independent line-out groups cap the search. Unsupported groups retain
    their ORIGINAL endpoints. Motion-only groups are unchanged.
    """
    groups = []
    for row in intervals.to_dict("records"):
        ids = [int(x) for x in str(row["event_ids"]).split(";")]
        members = events[events.event_id.isin(ids)]
        lines = members[members.method.isin(["line_dropout", "observed_line_exit"])]
        if lines.empty:
            continue
        seed = lines.sort_values(["start_frame", "event_id"]).iloc[0].to_dict()
        line, point = exit_geometry(seed, ball, pitch)
        groups.append({**row, "ids": ids, "seed": seed, "line": line, "point": point})
    revised = events.copy()
    revised["end_basis"] = revised.end_basis.astype(object)
    revised["source_event_id"] = revised.event_id
    revised["restart_proposal_id"] = np.nan
    revised["restart_type"] = "unknown"
    revised["restart_team"] = "X"
    used, audit = set(), []
    for index, group in enumerate(groups):
        seed, half = group["seed"], int(group["half"])
        a = int(seed["start_frame"])
        end_half = next(last+1 for h, _, last in blocks if h == half)
        limit = min(end_half, a+int(round(config.max_restart_wait_seconds*fps)))
        next_groups = [g for g in groups[index+1:] if g["half"] == half]
        if next_groups:
            limit = min(limit, int(next_groups[0]["seed"]["start_frame"]))
        pool = proposals[(proposals.half == half) & (proposals.frame >= a+config.min_restart_wait_seconds*fps)
                         & (proposals.frame < limit)]
        geometry = [r for r in pool.to_dict("records")
                    if compatible(r, group["line"], group["point"], pitch, config)]
        supported = [r for r in geometry if r["team_status"] == "supported" and r["proposal_id"] not in used]
        row = {"baseline_interval_id": int(group["interval_id"]), "half": half,
               "start_frame": int(group["start_frame"]), "exit_seed_frame": a,
               "exit_line": group["line"], "old_end_frame_exclusive": int(group["end_frame_exclusive"]),
               "new_end_frame_exclusive": int(group["end_frame_exclusive"]),
               "search_end_exclusive": limit, "geometric_proposals": len(geometry),
               "supported_proposals": len(supported), "status": "unchanged_no_supported_restart",
               "restart_type": "unknown", "restart_team": "X", "proposal_id": None,
               "source_event_ids": ";".join(map(str, group["ids"]))}
        if supported:
            proposal = supported[0]
            restart = int(proposal["frame"])
            used.add(proposal["proposal_id"])
            # The seed supplies continuous D support up to the proposed restart.
            seed_mask = revised.event_id.eq(int(seed["event_id"]))
            revised.loc[seed_mask, "end_frame_exclusive"] = restart
            members = revised.event_id.isin(group["ids"])
            # A matched restart also caps overlapping old motion evidence.
            revised.loc[members, "end_frame_exclusive"] = np.minimum(
                revised.loc[members, "end_frame_exclusive"], restart)
            revised.loc[members, "restart_proposal_id"] = proposal["proposal_id"]
            revised.loc[members, "restart_type"] = proposal["restart_type"]
            revised.loc[members, "restart_team"] = proposal["restart_team"]
            changed = members & revised.end_frame_exclusive.eq(restart)
            revised.loc[changed, "end_basis"] = "restart_geometry_and_departure_hypothesis"
            row.update(status="refined", new_end_frame_exclusive=restart,
                       restart_type=proposal["restart_type"], restart_team=proposal["restart_team"],
                       proposal_id=int(proposal["proposal_id"]))
        audit.append(row)
    revised = revised[revised.end_frame_exclusive > revised.start_frame].copy()
    revised["duration_seconds"] = (revised.end_frame_exclusive-revised.start_frame)/fps
    return revised, pd.DataFrame(audit)
