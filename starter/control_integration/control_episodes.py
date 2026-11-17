from __future__ import annotations
import numpy as np
from .hybrid_features import _quality, _runs, _velocity

MOTION_UNCERTAIN = 0


MOTION_CONTROLLED = 1


MOTION_FREE = 2


MOTION_AERIAL_SUSPECTED = 3


MOTION_CONTESTED = 4


MOTION_NAMES = {
    MOTION_UNCERTAIN: "missing_or_uncertain",
    MOTION_CONTROLLED: "controlled_or_co_moving",
    MOTION_FREE: "free_motion",
    MOTION_AERIAL_SUSPECTED: "aerial_or_projection_suspected",
    MOTION_CONTESTED: "contested",
}


DEFAULT_CONTROL = {
    "pitch_length_m": 105.3,
    "pitch_width_m": 68.0,
    "max_player_speed_mps": 12.0,
    "derivative_radius_seconds": 0.12,
    "distance_scale_m": 1.8,
    "relative_speed_scale_mps": 3.0,
    "control_distance_m": 2.0,
    "control_relative_speed_mps": 2.8,
    "control_step_relative_speed_mps": 8.0,
    "control_team_margin_m": 0.5,
    "control_score_min": 0.32,
    "ball_reliability_min": 0.65,
    "confirmation_seconds": 0.20,
    "strong_distance_m": 0.8,
    "strong_relative_speed_mps": 1.2,
    "strong_step_relative_speed_mps": 3.0,
    "strong_team_margin_m": 1.0,
    "aerial_confirmation_max": 0.5,
    "dead_reset_score": 0.80,
    "touch_min": 0.15,
    "interaction_distance_m": 2.5,
    "interaction_team_margin_m": 0.4,
    "short_action_lookahead_seconds": 2.0,
    "short_action_weight": 0.18,
    "memory_decay_seconds": 12.0,
    "memory_confidence_floor": 0.25,
}


def _configuration(config):
    cfg = {**DEFAULT_CONTROL, **(config or {})}
    if set(cfg) != set(DEFAULT_CONTROL):
        raise ValueError("Unknown control config key")
    if any(not np.isfinite(value) or value < 0 for value in cfg.values()):
        raise ValueError("Control configuration must be finite and nonnegative")
    for key in ("pitch_length_m", "pitch_width_m", "max_player_speed_mps",
                "distance_scale_m", "relative_speed_scale_mps", "memory_decay_seconds"):
        if cfg[key] <= 0:
            raise ValueError("Control scale must be positive: " + key)
    for key in ("control_score_min", "ball_reliability_min", "aerial_confirmation_max",
                "dead_reset_score", "touch_min", "short_action_weight", "memory_confidence_floor"):
        if cfg[key] > 1:
            raise ValueError("Control score outside [0,1]: " + key)
    return cfg


def _axis(features, name, n, shape=(), default=None):
    if name not in features:
        if default is None:
            raise ValueError("Missing control feature: " + name)
        value = np.full((n,) + shape, default, dtype=np.float32)
    else:
        value = np.asarray(features[name])
    if value.shape != (n,) + shape:
        raise ValueError("Misaligned control feature: " + name)
    return value


def build_control_evidence(features, players, teams, blocks, fps, config=None):
    """Return per-frame control, touch, motion and last-control memory evidence.

    ``features`` is the mapping from ``hybrid_features.extract_features``.
    ``players`` is (N, slots, 2), ``teams`` is (slots,) with 0=home/1=away,
    and ``blocks`` is (N,).  Slot continuity is used only within velocity
    windows whose every edge passes the jump test; this is not player Re-ID.

    A confirmed episode needs unique team proximity AND reliable relative
    motion.  A short strong episode may be confirmed without a duration floor.
    Short touch + later same-team control contributes only weak action evidence.
    Missing balls and ambiguous contacts retain the last confirmed team until
    new actual control or a confident dead-ball/half boundary.

    Returns only serializable metadata and NumPy arrays, including a local-only
    ``episodes`` list.  Do not publish that list for restricted matches.
    """
    cfg = _configuration(config)
    players = np.asarray(players, dtype=np.float32)
    teams = np.asarray(teams)
    blocks = np.asarray(blocks)
    if (players.ndim != 3 or players.shape[2] != 2 or len(players) == 0
            or teams.shape != (players.shape[1],) or blocks.shape != (len(players),)
            or not np.isin(teams, (0, 1)).all()
            or not all(np.any(teams == k) for k in (0, 1))
            or not np.isfinite(blocks).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid control shapes, teams, blocks or FPS")
    n = len(players)
    ball = _axis(features, "ball_position", n, (2,)).astype(np.float32, copy=False)
    ball_velocity = _axis(features, "ball_velocity", n, (2,)).astype(np.float32, copy=False)
    ball_reliability = _axis(features, "ball_reliability", n).astype(np.float32, copy=False)
    ball_valid = _axis(features, "ball_valid", n).astype(bool, copy=False)
    touch = _axis(features, "touch", n).astype(np.float32, copy=False)
    aerial = _axis(features, "aerial", n, default=0.0).astype(np.float32, copy=False)
    p_dead = _axis(features, "p_dead", n).astype(np.float32, copy=False)
    for name, value in (("ball_reliability", ball_reliability), ("touch", touch),
                        ("aerial", aerial), ("p_dead", p_dead)):
        if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError("Invalid control score: " + name)

    active = (blocks > 0) & (p_dead < cfg["dead_reset_score"])
    # Re-index contiguous active segments.  Derivatives, confirmations and
    # future action support cannot bridge even a single confident-D barrier.
    cuts = np.ones(n, dtype=bool)
    cuts[1:] = (blocks[1:] != blocks[:-1]) | (active[1:] != active[:-1])
    segment = np.where(active, np.cumsum(cuts, dtype=np.int32), 0)
    player_valid, player_edges, _ = _quality(
        players, segment, fps, cfg["max_player_speed_mps"],
        (cfg["pitch_length_m"], cfg["pitch_width_m"]), player=True)
    velocity_radius = max(1, round(cfg["derivative_radius_seconds"] * fps))
    player_velocity = _velocity(players, player_edges, fps, velocity_radius)

    distance = np.linalg.norm(players - ball[:, None], axis=2)
    distance[~player_valid | ~np.isfinite(distance)] = np.inf
    distance[~active] = np.inf
    relative_speed = np.linalg.norm(player_velocity - ball_velocity[:, None], axis=2)
    velocity_known = np.isfinite(relative_speed)
    co_motion = np.where(velocity_known,
                         np.exp(-np.nan_to_num(relative_speed, nan=0.0)
                                / cfg["relative_speed_scale_mps"]), 0.2)
    per_player = np.exp(-distance / cfg["distance_scale_m"]) * (0.15 + 0.85 * co_motion)
    per_player *= ball_reliability[:, None] * (1 - 0.65 * aerial[:, None])
    per_player[~active] = 0

    # Max rather than a sum: duplicate/fragmented slots must not manufacture
    # additional independent team evidence.
    team_scores = np.column_stack([per_player[:, teams == k].max(axis=1) for k in (0, 1)])
    team_distance = np.column_stack([distance[:, teams == k].min(axis=1) for k in (0, 1)])
    best_slot = np.argmax(per_player, axis=1)
    rows = np.arange(n)
    best_team = teams[best_slot].astype(np.int8)
    best_distance = distance[rows, best_slot]
    best_relative = relative_speed[rows, best_slot]
    best_score = per_player[rows, best_slot]
    best_velocity_known = velocity_known[rows, best_slot]
    # A centred velocity may be zero at a sharp rebound although the ball is
    # fast on BOTH sides.  Check one-sided relative displacements as well;
    # otherwise a single deflection can masquerade as very strong control.
    before_rows = np.maximum(rows - 1, 0)
    after_rows = np.minimum(rows + 1, n - 1)
    relative_before = ((ball - ball[before_rows])
                       - (players[rows, best_slot] - players[before_rows, best_slot])) * fps
    relative_after = ((ball[after_rows] - ball)
                      - (players[after_rows, best_slot] - players[rows, best_slot])) * fps
    relative_step_speed = np.maximum(np.linalg.norm(relative_before, axis=1),
                                     np.linalg.norm(relative_after, axis=1))
    step_supported = (ball_valid & ball_valid[before_rows] & ball_valid[after_rows]
                      & player_edges[rows, best_slot] & player_edges[after_rows, best_slot]
                      & (segment == segment[before_rows]) & (segment == segment[after_rows])
                      & (rows > 0) & (rows < n - 1))
    # An absent opposing team is not a large, reliable separation margin.
    both_teams_visible = np.isfinite(team_distance).all(axis=1)
    opponent_distance = team_distance[rows, 1 - best_team]
    team_margin = np.zeros(n, np.float32)
    comparable = both_teams_visible & np.isfinite(best_distance)
    team_margin[comparable] = opponent_distance[comparable] - best_distance[comparable]
    separation = np.clip(team_margin / 1.5, 0, 1)
    reliability = ball_reliability * (0.25 + 0.75 * best_velocity_known)
    reliability *= (0.25 + 0.75 * separation) * (1 - 0.65 * aerial)
    reliability *= active & np.isfinite(best_distance)

    eligible = (active & ball_valid & np.isfinite(ball).all(axis=1)
                & (ball_reliability >= cfg["ball_reliability_min"])
                & both_teams_visible & best_velocity_known
                & (best_distance <= cfg["control_distance_m"])
                & (best_relative <= cfg["control_relative_speed_mps"])
                & step_supported & (relative_step_speed <= cfg["control_step_relative_speed_mps"])
                & (team_margin >= cfg["control_team_margin_m"])
                & (best_score >= cfg["control_score_min"])
                & (aerial < cfg["aerial_confirmation_max"]))
    strong = (eligible & (best_distance <= cfg["strong_distance_m"])
              & (best_relative <= cfg["strong_relative_speed_mps"])
              & (relative_step_speed <= cfg["strong_step_relative_speed_mps"])
              & (team_margin >= cfg["strong_team_margin_m"]))
    # Identity changes that pass a speed gate can still be wrong.  Reliability
    # is evidence quality, never a promise that a slot is a persistent identity.
    provisional = np.where(eligible, best_team, -1).astype(np.int8)
    confirmed = np.full(n, -1, dtype=np.int8)
    onset = np.zeros(n, dtype=bool)
    episodes = []
    minimum_support = max(1, int(np.ceil(cfg["confirmation_seconds"] * fps)))
    episode_key = np.where(eligible, segment.astype(np.int64) * 3 + best_team, -1)
    for lo, hi in _runs(episode_key):
        if not eligible[lo]:
            continue
        sustained = hi - lo >= minimum_support
        exceptionally_clear = bool(strong[lo:hi].any())
        if not (sustained or exceptionally_clear):
            continue
        team = int(provisional[lo])
        confirmed[lo:hi] = team
        onset[lo] = True
        episodes.append({"start_frame": int(lo), "end_frame_exclusive": int(hi),
                         "team": team, "block": int(blocks[lo]),
                         "duration_seconds": float((hi - lo) / fps),
                         "reason": "strong_co_motion" if exceptionally_clear else "sustained_co_motion"})

    # Touch assignment means "plausible interacting team", NOT possession.
    # Use nearest team for interaction, and reject team-ambiguous near-ties.
    nearest_team = np.argmin(team_distance, axis=1).astype(np.int8)
    nearest_distance = team_distance[rows, nearest_team]
    touch_margin = np.zeros(n, np.float32)
    touch_margin[both_teams_visible] = np.abs(
        team_distance[both_teams_visible, 0] - team_distance[both_teams_visible, 1])
    touch_score = touch * ball_reliability * active * ball_valid
    interaction = np.full(n, -1, dtype=np.int8)
    interaction_valid = (active & ball_valid & both_teams_visible
                         & (touch_score >= cfg["touch_min"])
                         & (nearest_distance <= cfg["interaction_distance_m"])
                         & (touch_margin >= cfg["interaction_team_margin_m"]))
    interaction[interaction_valid] = nearest_team[interaction_valid]

    # Forward context is used only at an observed short-action candidate.  A
    # subsequent same-team controller is supportive, not proof of its control.
    short_action = np.full(n, -1, dtype=np.int8)
    lookahead = max(1, round(cfg["short_action_lookahead_seconds"] * fps))
    for lo, hi in _runs(segment):
        if segment[lo] <= 0:
            continue
        next_index = -1
        for t in range(hi - 1, lo - 1, -1):
            if confirmed[t] >= 0:
                next_index = t
            elif (interaction[t] >= 0 and next_index >= 0 and next_index - t <= lookahead
                  and interaction[t] == confirmed[next_index]
                  and aerial[t] < cfg["aerial_confirmation_max"]):
                short_action[t] = interaction[t]
    action_weight = cfg["short_action_weight"] * touch_score
    for team in (0, 1):
        candidate = short_action == team
        team_scores[candidate, team] = np.maximum(team_scores[candidate, team], action_weight[candidate])

    owner = np.full(n, -1, dtype=np.int8)
    age = np.full(n, np.inf, dtype=np.float32)
    for lo, hi in _runs(segment):
        if segment[lo] <= 0:
            continue
        last_team = -1
        last_frame = lo
        for t in range(lo, hi):
            if confirmed[t] >= 0:
                last_team = int(confirmed[t])
                last_frame = t
            owner[t] = last_team
            if last_team >= 0:
                age[t] = (t - last_frame) / fps

    # A heuristic evidence blend, not a likelihood/product of independent
    # observations. Unconfirmed proximity can only weakly disagree with memory.
    denominator = np.maximum(team_scores.max(axis=1), 0) + 0.12
    instantaneous_home = 0.5 + 0.45 * (team_scores[:, 0] - team_scores[:, 1]) / denominator
    home_evidence = 0.5 + reliability * (instantaneous_home - 0.5)
    memory_confidence = cfg["memory_confidence_floor"] + (1 - cfg["memory_confidence_floor"]) * np.exp(
        -age / cfg["memory_decay_seconds"])
    memory_home = 0.5 + np.where(owner == 0, 0.43, -0.43) * memory_confidence
    has_memory = owner >= 0
    direct_weight = np.where(confirmed >= 0, 0.85, 0.25) * reliability
    home_evidence[has_memory] = ((1 - direct_weight[has_memory]) * memory_home[has_memory]
                                 + direct_weight[has_memory] * instantaneous_home[has_memory])
    home_evidence[~active] = 0.5

    motion = np.full(n, MOTION_UNCERTAIN, dtype=np.int8)
    observed = active & ball_valid & (ball_reliability >= cfg["ball_reliability_min"])
    motion[observed] = MOTION_FREE
    contested = observed & both_teams_visible & (np.min(team_distance, axis=1) < cfg["control_distance_m"])
    contested &= np.abs(team_margin) < cfg["control_team_margin_m"]
    motion[contested] = MOTION_CONTESTED
    motion[observed & (aerial >= cfg["aerial_confirmation_max"])] = MOTION_AERIAL_SUSPECTED
    motion[confirmed >= 0] = MOTION_CONTROLLED

    result = {"control_home": team_scores[:, 0], "control_away": team_scores[:, 1],
              "home_evidence": home_evidence, "control_reliability": reliability,
              "touch_score": touch_score, "interaction_team": interaction,
              "confirmed_control_team": confirmed, "control_onset": onset,
              "motion_mode": motion, "owner_memory": owner,
              "memory_confidence": np.where(has_memory, memory_confidence, 0),
              "owner_reliability": np.where(has_memory, memory_confidence, 0),
              "short_action_team": short_action, "episodes": episodes,
              "metadata": {"config": cfg, "gt_used": False,
                           "probability_calibration": "hand-set scores; no calibration on possession GT",
                           "future_use": "confirmation only inside actual control condition; weak touch support",
                           "motion_modes": MOTION_NAMES,
                           "height_observed": False,
                           "slot_identity_assumed": False}}
    for name in ("control_home", "control_away", "home_evidence", "control_reliability",
                 "touch_score", "memory_confidence"):
        value = np.asarray(result[name], dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite control result: " + name)
        result[name] = np.clip(value, 0, 1)
    result["owner_reliability"] = result["memory_confidence"].copy()
    return result
