from __future__ import annotations
import numpy as np
from .hybrid_features import _mean, _runs
from .hybrid_outfield_activity import DEFAULT_ACTIVITY, _keeper_mask
from .hybrid_features import _velocity

DEFAULT_STOPPAGE = dict(
    max_player_speed_mps=12., player_derivative_seconds=.2,
    activity_smooth_seconds=.6, activity_stability_seconds=1.2,
    minimum_team_players=5, low_speed_mps=1.5, stable_scale_mps=.5,
    quiet_duration_seconds=4., ball_stationary_mps=.6,
    stationary_seconds=.6, outside_margin_m=.35,
    outside_confirmation_seconds=.24, boundary_context_m=2.,
    outward_speed_mps=1., outward_memory_seconds=6.,
    restart_speed_mps=1.5, restart_confirm_seconds=.32,
    restart_context_seconds=2., restart_player_distance_m=2.,
    restart_displacement_m=2., restart_activity_mps=1.6,
    memory_decay_seconds=35., onset_threshold=.68,
    restart_threshold=.65, reliability_threshold=.6,
    weak_only_support_cap=.25,
)


MECHANISMS = {
    1: 'persistent outside-pitch projection (height unobserved)',
    2: 'quiet outfield plus stationary ball',
    4: 'quiet outfield plus weak set-piece context',
    8: 'observed outward motion followed by missing ball',
    16: 'previous stoppage evidence retained',
    32: 'player-associated sustained departure and post-play support',
    64: 'stationary goalkeeper-held live-ball alternative',
}


def _field(features, name, n, default=0., dtype=float):
    value = np.asarray(features.get(name, np.full(n, default)), dtype=dtype)
    if value.shape != (n,):
        raise ValueError(f'{name} must have shape ({n},)')
    return value


def _duration_score(flag, blocks, fps, seconds):
    """Complete-run soft support, using future context within the same block."""
    score = np.zeros(len(flag), np.float32)
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        for a, b in _runs(flag[lo:hi]):
            if flag[lo + a]:
                score[lo+a:lo+b] = 1. - np.exp(-(b-a)/(fps*seconds))
    return score


def _activity(players, teams, blocks, fps, pitch, goalkeeper_mask, cfg):
    """Missing-aware pooled speed; slots are local motion tracks, never IDs."""
    n, p, _ = players.shape
    valid = np.isfinite(players).all(axis=2)
    valid &= ((players >= 0) & (players <= pitch)).all(axis=2)
    valid &= blocks[:, None] > 0
    speed_native = np.linalg.norm(np.diff(players, axis=0), axis=2)*fps
    same = (blocks[1:] == blocks[:-1]) & (blocks[1:] > 0)
    jump = np.zeros((n, p), bool)
    bad = valid[1:] & valid[:-1] & same[:, None] & (speed_native > cfg['max_player_speed_mps'])
    jump[1:] |= bad
    jump[:-1] |= bad
    valid &= ~jump
    edge = np.zeros((n, p), bool)
    edge[1:] = valid[1:] & valid[:-1] & same[:, None]
    velocity = _velocity(players, edge, fps, max(1, round(cfg['player_derivative_seconds']*fps)))
    speed = np.linalg.norm(velocity, axis=2)
    speed = _mean(speed, max(1, round(cfg['activity_smooth_seconds']*fps/2)), blocks, .75)
    speed[~valid] = np.nan
    if goalkeeper_mask is None:
        keeper, uncertain, _, roles = _keeper_mask(players, valid, teams, blocks, fps, pitch, DEFAULT_ACTIVITY)
        role_source = 'conservative per-block geometric role hypothesis'
    else:
        keeper = np.asarray(goalkeeper_mask, bool)
        if keeper.shape == (p,):
            keeper = np.broadcast_to(keeper, (n, p))
        if keeper.shape != (n, p):
            raise ValueError('goalkeeper_mask must have shape (players,) or (frames, players)')
        uncertain = np.zeros((n, 2), bool)
        roles = []
        role_source = 'caller-provided mask; identity correctness not guaranteed'
    available = np.isfinite(speed) & valid & ~keeper
    counts = np.column_stack([available[:, teams == k].sum(axis=1) for k in (0, 1)])
    total = available.sum(axis=1)
    pooled = np.divide(np.where(available, speed, 0.).sum(axis=1), total,
                       out=np.full(n, np.nan), where=total > 0)
    enough = (counts >= cfg['minimum_team_players']).all(axis=1) & (blocks > 0)
    pooled[~enough] = np.nan
    radius = max(1, round(cfg['activity_stability_seconds']*fps/2))
    mean = _mean(pooled, radius, blocks, .75)
    mean2 = _mean(pooled**2, radius, blocks, .75)
    stability = np.exp(-np.sqrt(np.maximum(mean2-mean**2, 0.))/cfg['stable_scale_mps'])
    low = np.clip((cfg['low_speed_mps']-pooled)/max(cfg['low_speed_mps']-.4, .1), 0., 1.)
    coverage = np.clip(counts.min(axis=1)/10., 0., 1.)
    overcount = np.minimum(1., 10./np.maximum(counts.max(axis=1), 1))
    reliability = coverage*overcount*np.where(uncertain.any(axis=1), .85, 1.)
    reliability[~enough] = 0.
    low = np.nan_to_num(low*stability, nan=0.)*enough
    return dict(speed=pooled, low=low, reliability=reliability,
                counts=counts, keeper=keeper, valid=valid,
                roles=roles, role_source=role_source, jumps=jump.sum(axis=1))


def build_stoppage_evidence(features, players, teams, blocks, fps=25.,
                            pitch=(105.3, 68.), goalkeeper_mask=None, config=None):
    """Return frame-aligned D/start/restart evidence with no labels or file IO.

    ``features`` follows ``hybrid_features.extract_features``. Required fields:
    raw_ball (or ball_position), ball_velocity, ball_speed, ball_reliability.
    Optional: ball_valid, aerial, set_piece, control. Existing p_dead/restart are
    deliberately ignored. Positive contiguous blocks are decoded independently.
    """
    cfg = {**DEFAULT_STOPPAGE, **(config or {})}
    if set(cfg) != set(DEFAULT_STOPPAGE):
        raise ValueError('Unknown stoppage config key')
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in cfg.values()):
        raise ValueError('Stoppage constants must be finite and positive')
    if any(cfg[k] > 1 for k in ('onset_threshold', 'restart_threshold', 'reliability_threshold')):
        raise ValueError('Score threshold exceeds one')
    if cfg['weak_only_support_cap'] >= (1./3.-.04)/.92:
        raise ValueError('Weak-only support must keep p_dead below one third')
    players = np.asarray(players, np.float32)
    teams = np.asarray(teams)
    blocks = np.asarray(blocks)
    pitch = np.asarray(pitch, float)
    n = len(players)
    if (players.ndim != 3 or players.shape[-1] != 2 or teams.shape != (players.shape[1],)
            or blocks.shape != (n,) or not np.isin(teams, (0, 1)).all()
            or any(np.sum(teams == k) < 2 for k in (0, 1)) or n == 0
            or not np.isfinite(fps) or fps <= 0 or pitch.shape != (2,)
            or not np.isfinite(pitch).all() or np.any(pitch <= 0)
            or not np.isfinite(blocks).all() or np.any(blocks < 0)):
        raise ValueError('Invalid shape, coordinate system, blocks or FPS')
    activity = _activity(players, teams, blocks, fps, pitch, goalkeeper_mask, cfg)
    ball = np.asarray(features.get('raw_ball', features.get('ball_position')), float)
    velocity = np.asarray(features.get('ball_velocity'), float)
    if ball.shape != (n, 2) or velocity.shape != (n, 2):
        raise ValueError('raw_ball/ball_position and ball_velocity need shape (frames, 2)')
    reliability = np.clip(np.nan_to_num(_field(features, 'ball_reliability', n)), 0., 1.)
    observed = np.isfinite(ball).all(axis=1)
    valid = observed & _field(features, 'ball_valid', n, True, bool) & (blocks > 0)
    reliable = valid & (reliability >= cfg['reliability_threshold'])
    ball_speed = _field(features, 'ball_speed', n, np.nan)
    aerial = np.clip(np.nan_to_num(_field(features, 'aerial', n)), 0., 1.)
    set_piece = np.clip(np.nan_to_num(_field(features, 'set_piece', n)), 0., 1.)
    control = np.asarray(features.get('control', np.zeros((n, 2))), float)
    if control.shape != (n, 2):
        raise ValueError('control must have shape (frames, 2)')
    control_max = np.nan_to_num(control).max(axis=1)
    dist = np.linalg.norm(players-ball[:, None], axis=2)
    dist[~activity['valid'] | ~np.isfinite(dist)] = np.inf
    nearest = np.argmin(dist, axis=1)
    nearest_distance = np.min(dist, axis=1)
    nearest_keeper = activity['keeper'][np.arange(n), nearest] & np.isfinite(nearest_distance)
    boundary = np.min(np.column_stack((ball[:, 0], pitch[0]-ball[:, 0], ball[:, 1], pitch[1]-ball[:, 1])), axis=1)
    inside = (boundary >= 0.) & reliable
    outside = (boundary < -cfg['outside_margin_m']) & reliable
    exit_score = np.zeros(n, np.float32)
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        for a, b in _runs(outside[lo:hi]):
            if outside[lo+a]:
                duration = min(1., (b-a)/(fps*cfg['outside_confirmation_seconds']))
                exit_score[lo+a:lo+b] = .95*duration*reliability[lo+a:lo+b]*(1.-.85*aerial[lo+a:lo+b])

    stationary = _mean(np.where(reliable & np.isfinite(ball_speed),
                                (ball_speed < cfg['ball_stationary_mps']).astype(float), np.nan),
                       max(1, round(cfg['stationary_seconds']*fps/2)), blocks, .7)
    stationary = np.nan_to_num(stationary, nan=0.)*reliable
    low = activity['low']
    quiet_duration = _duration_score(low > .25, blocks, fps, cfg['quiet_duration_seconds'])
    # Low collective speed supports a stoppage only in conjunction with a ball/
    # set-piece observation. Four seconds changes confidence, never label rules.
    quiet = low*(.55+.45*quiet_duration)*np.minimum(1., activity['reliability']/.6)
    near_boundary = valid & (np.abs(boundary) <= cfg['boundary_context_m'])
    held_by_keeper = nearest_keeper & (nearest_distance < 2.) & (set_piece < .3)
    quiet_static = .9*quiet*stationary
    # Controlled stationary live-ball pauses are ambiguous away from a boundary.
    live_pause = (control_max > .35) & ~near_boundary & (set_piece < .3)
    quiet_static[live_pause] *= .35
    quiet_static[held_by_keeper] *= .1
    # A broad tagging interval may continue into live play. Moving observed
    # balls therefore do not inherit a D decision from that interval alone.
    tag_ball_context = np.maximum(stationary, .85*(~observed))
    quiet_tag = .88*quiet*set_piece*tag_ball_context

    outward_gap = np.zeros(n, np.float32)
    normals = np.array(((-1., 0.), (1., 0.), (0., -1.), (0., 1.)))
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        for a, b in _runs(observed[lo:hi]):
            start, stop = lo+a, lo+b
            if observed[start] or start == lo:
                continue
            t = start-1
            if not reliable[t]:
                continue
            v = velocity[t]
            if not np.isfinite(v).all():
                back = t-max(1, round(.2*fps))
                if back < lo or not reliable[back:t+1].all():
                    continue
                v = (ball[t]-ball[back])*fps/(t-back)
            distances = np.array((ball[t, 0], pitch[0]-ball[t, 0], ball[t, 1], pitch[1]-ball[t, 1]))
            edge = int(np.argmin(distances))
            if distances[edge] > cfg['boundary_context_m'] or np.dot(v, normals[edge]) < cfg['outward_speed_mps']:
                continue
            frames = np.arange(start, stop)
            outward_gap[frames] = .82*reliability[t]*(1.-.85*aerial[t])*np.exp(-(frames-start)/(fps*cfg['outward_memory_seconds']))

    support = np.maximum.reduce((exit_score, quiet_static, quiet_tag, outward_gap))
    # Restart = association at launch + sustained in-pitch departure + subsequent
    # live evidence. A returned rolling ball alone meets neither association nor
    # post-play condition; a ball carried for placement usually has little
    # outfield response. Some returned-ball/live cases remain unidentifiable.
    restart = np.zeros(n, np.float32)
    moving = inside & (ball_speed >= cfg['restart_speed_mps'])
    moving_duration = _duration_score(moving, blocks, fps, cfg['restart_confirm_seconds'])
    context = max(1, round(cfg['restart_context_seconds']*fps))
    launch_radius = max(1, round(.36*fps))
    pulse = max(1, round(.16*fps))
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        for a, b in _runs(moving[lo:hi]):
            t, stop = lo+a, lo+b
            if not moving[t] or stop-t < max(2, round(cfg['restart_confirm_seconds']*fps)):
                continue
            left, launch_left = max(lo, t-context), max(lo, t-launch_radius)
            right = min(hi, t+context)
            preparation = max(float(np.max(stationary[left:t], initial=0.)),
                              float(np.max(exit_score[left:t], initial=0.)),
                              float(np.max(outward_gap[left:t], initial=0.)),
                              float(np.max(set_piece[left:t], initial=0.)))
            if preparation < .5:
                continue
            association = np.exp(-np.min(nearest_distance[launch_left:t+1])/cfg['restart_player_distance_m'])
            # The association threshold is deliberately generous for smoothed
            # launch coordinates, while entirely unassociated ball returns fail.
            association = min(1., association/np.exp(-1.))
            end = min(stop-1, t+context)
            displacement = np.linalg.norm(ball[end]-ball[t])
            departure = min(1., displacement/cfg['restart_displacement_m'])
            post_speed = activity['speed'][t:right]
            finite = np.isfinite(post_speed)
            post_active = float(np.mean(post_speed[finite] >= cfg['restart_activity_mps'])) if finite.any() else 0.
            future_control = float(np.max(control_max[t:right], initial=0.))
            different_receiver = bool(np.any((nearest[t:right] != nearest[t]) & (control_max[t:right] > .5)))
            transfer = future_control*float(different_receiver)*departure
            post_play = max(post_active, transfer)
            confidence = association*departure*post_play*float(np.mean(reliability[t:stop]))
            restart[t:min(stop, t+pulse)] = max(confidence, restart[t])

    dead_start = np.zeros(n, np.float32)
    memory = np.zeros(n, np.float32)
    p_dead = np.full(n, .04, np.float32)
    mechanisms = np.zeros(n, np.uint8)
    mechanisms[exit_score >= cfg['onset_threshold']] |= 1
    mechanisms[quiet_static >= cfg['onset_threshold']] |= 2
    mechanisms[quiet_tag >= cfg['onset_threshold']] |= 4
    mechanisms[outward_gap >= cfg['onset_threshold']] |= 8
    mechanisms[held_by_keeper] |= 64
    decay = np.exp(-1./(fps*cfg['memory_decay_seconds']))
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        current = 0.
        previous_support = False
        for t in range(lo, hi):
            confirmed_restart = restart[t] >= cfg['restart_threshold']
            strong = support[t] >= cfg['onset_threshold'] and not confirmed_restart
            if confirmed_restart:
                current *= .05
                mechanisms[t] |= 32
            elif strong:
                if not previous_support:
                    dead_start[t] = support[t]
                current = max(current, float(support[t]))
            else:
                current *= decay
                # A missed launch must not trap the decoder in D indefinitely.
                # This fallback needs sustained real movement AND active teams.
                if moving_duration[t] > .8 and np.isfinite(activity['speed'][t]) and activity['speed'][t] >= cfg['restart_activity_mps']:
                    current *= np.exp(-1./(2.*fps))
            memory[t] = current
            if current >= .2:
                mechanisms[t] |= 16
            # At p_home_given_alive=.5 a three-state decoder compares p_dead
            # with (1-p_dead)/2. Keep unconfirmed support below one third, so
            # uncertain H/A evidence alone cannot make weak D the MAP label.
            evidence = max(current, min(float(support[t]), cfg['weak_only_support_cap']))
            if confirmed_restart:
                evidence = current
            p_dead[t] = .04+.92*evidence
            previous_support = strong
    return dict(
        p_dead=p_dead, dead_start_score=dead_start, restart_score=restart,
        dead_memory=memory, dead_support=support.astype(np.float32),
        outfield_speed=activity['speed'].astype(np.float32),
        low_activity=low.astype(np.float32), activity_reliability=activity['reliability'].astype(np.float32),
        activity_valid_counts=activity['counts'], goalkeeper_mask=activity['keeper'],
        quiet_duration_score=quiet_duration, stationary=stationary.astype(np.float32),
        exit_score=exit_score, outward_gap_score=outward_gap,
        mechanism_code=mechanisms,
        metadata=dict(config=cfg, calibration='fixed score shaping; not learned/calibrated probability',
                      goalkeeper_source=activity['role_source'], goalkeeper_roles=activity['roles'],
                      mechanism_codes=MECHANISMS, height_observed=False, gt_used=False,
                      limitations=['2D pitch exit is projection evidence, not a geometric proof of line-out',
                                   'referee whistle and some ball-return/restart situations are unobserved',
                                   'no permanent slot identity assumption; undetectably small ID swaps remain possible']))
