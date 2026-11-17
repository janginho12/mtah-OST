import numpy as np
from . import control_episodes, control_model, stoppage_evidence
from .hybrid_features import _quality, _velocity

FEATURE_CONFIG = dict(raw_ball_reliability=.85, imputed_ball_reliability=.20,
                      max_ball_speed_mps=50., max_player_speed_mps=12.,
                      derivative_radius_seconds=.12, touch_distance_m=1.,
                      touch_velocity_change_mps=1.)


def extract_features(raw_ball, reconstructed_ball, interpolated, players, teams,
                     blocks, fps, pitch, config=None):
    """Preserve raw missingness and never confirm control from player-derived ball.

    Velocity uses raw observations exclusively. Reconstructed positions may give
    weak spatial support, but ball_valid remains false and reliability < .65.
    A change in ball velocity is a weak contact candidate only with a raw player
    within 1m. A candidate is not a detected kick or confirmed possession.
    """
    cfg = {**FEATURE_CONFIG, **(config or {})}
    if set(cfg) != set(FEATURE_CONFIG) or any(not np.isfinite(x) or x <= 0 for x in cfg.values()):
        raise ValueError("Invalid feature configuration")
    if not (0 < cfg['imputed_ball_reliability'] < .65 <= cfg['raw_ball_reliability'] <= 1):
        raise ValueError("Imputation must remain below control confirmation reliability")
    raw_ball = np.asarray(raw_ball, np.float32)
    reconstructed_ball = np.asarray(reconstructed_ball, np.float32)
    players = np.asarray(players, np.float32)
    teams = np.asarray(teams)
    blocks = np.asarray(blocks)
    interpolated = np.asarray(interpolated, bool)
    n = len(blocks)
    if (n == 0 or raw_ball.shape != (n, 2) or reconstructed_ball.shape != (n, 2)
            or interpolated.shape != (n,) or players.ndim != 3 or players.shape[0] != n
            or players.shape[2] != 2 or teams.shape != (players.shape[1],)
            or not np.isin(teams, (0, 1)).all()
            or not all(np.any(teams == k) for k in (0, 1))):
        raise ValueError("Misaligned feature inputs or missing team")
    raw_present = np.isfinite(raw_ball).all(axis=1)
    valid, edges, jumps = _quality(raw_ball, blocks, fps, cfg['max_ball_speed_mps'], pitch,
                                   player=False)
    # Outside-pitch raw balls remain usable for line-out evidence.
    radius = max(1, round(cfg['derivative_radius_seconds'] * fps))
    velocity = _velocity(raw_ball, edges, fps, radius)
    filled = interpolated & ~raw_present & np.isfinite(reconstructed_ball).all(axis=1) & (blocks > 0)
    ball = np.full((n, 2), np.nan, np.float32)
    ball[valid] = raw_ball[valid]
    ball[filled] = reconstructed_ball[filled]
    reliability = np.zeros(n, np.float32)
    reliability[valid] = cfg['raw_ball_reliability']
    reliability[filled] = cfg['imputed_ball_reliability']

    p_valid, p_edges, _ = _quality(players, blocks, fps, cfg['max_player_speed_mps'], pitch)
    p_velocity = _velocity(players, p_edges, fps, radius)
    distance = np.linalg.norm(players - raw_ball[:, None], axis=2)
    distance[~p_valid | ~np.isfinite(distance)] = np.inf
    nearest = distance.min(axis=1)
    relative = np.linalg.norm(p_velocity - velocity[:, None], axis=2)
    relative = np.nan_to_num(relative, nan=np.inf)
    geometric = np.exp(-distance/1.8) * np.exp(-relative/3.) * reliability[:, None]
    control = np.column_stack([geometric[:, teams == k].max(axis=1) for k in (0, 1)])

    touch = np.zeros(n, np.float32)
    if n > 2:
        change = np.linalg.norm((raw_ball[2:] - 2*raw_ball[1:-1] + raw_ball[:-2])*fps, axis=1)
        supported = (edges[1:-1] & edges[2:] & valid[1:-1]
                     & (nearest[1:-1] <= cfg['touch_distance_m']))
        touch[1:-1] = np.where(supported, np.clip(
            change/cfg['touch_velocity_change_mps'] - 1., 0., 1.), 0.)
    # Height, set pieces and tag ownership are unobserved in this adapter.
    # Zero scores disable those optional signals; zero aerial is not proof of a
    # ground ball. In particular no manual match-event annotations are inserted.
    zeros = np.zeros(n, np.float32)
    return dict(raw_ball=raw_ball, ball_position=ball, ball_velocity=velocity,
                ball_speed=np.linalg.norm(velocity, axis=1), ball_valid=valid,
                ball_reliability=reliability, ball_interpolated=filled,
                ball_jump_rejected=jumps, touch=touch, control=control,
                aerial=zeros.copy(), set_piece=zeros.copy(),
                tag_reliability=zeros.copy(), tag_home=np.full(n, .5, np.float32)), cfg


def infer_staged(features, players, teams, blocks, fps, pitch, dead_mask,
                 goalkeeper_mask, config=None):
    """Use all three attached modules while preserving D -> H/A ordering.

    D frames split the Viterbi paths and reset confirmed-control memory. The new
    stoppage module supplies audit evidence, never extra hard D boundaries.
    """
    config = config or {}
    blocks = np.asarray(blocks)
    dead_mask = np.asarray(dead_mask, bool)
    if dead_mask.shape != blocks.shape or np.any(dead_mask & (blocks <= 0)):
        raise ValueError("D candidates must be frame-aligned and inside match blocks")
    n = len(blocks)
    cfg = control_model.validate_config(config.get('decoder'))
    control_cfg = {**(config.get('control') or {}), 'pitch_length_m': float(pitch[0]),
                   'pitch_width_m': float(pitch[1]), 'interaction_distance_m': 1.}
    # The 1m rule applies to contact candidates. Continuous co-motion uses the
    # supplied control model's separate 2m neighbourhood and relative velocity.
    control = control_episodes.build_control_evidence(
        {**features, 'p_dead': dead_mask.astype(np.float32)},
        players, teams, blocks, fps, control_cfg)
    dead = stoppage_evidence.build_stoppage_evidence(
        features, players, teams, blocks, fps, pitch, goalkeeper_mask,
        config.get('stoppage'))
    clip = cfg['probability_clip']
    strength = np.maximum(control['control_reliability'], control['owner_reliability'])
    tag_weight = cfg['weak_tag_weight']*features['tag_reliability']*(1-strength)
    ph = np.clip((1-tag_weight)*control['home_evidence'] + tag_weight*features['tag_home'],
                 clip, 1-clip)
    # The supplied three-state decoder is reused on each non-D run. D emission
    # is disabled there; known D is assigned explicitly after decoding.
    emission = np.column_stack((np.log(ph), np.log1p(-ph), np.full(n, -1.e12)))
    emission *= cfg['evidence_rate_hz']/fps
    cost = cfg['team_switch_cost'] + cfg['uncertainty_switch_extra']*(1-control['control_reliability'])
    cost[control['control_onset']] *= cfg['confirmed_onset_cost_fraction']
    cost = np.clip(cost, cfg['minimum_switch_cost'], cfg['maximum_switch_cost'])
    zero = np.zeros(n)
    labels = control_model.viterbi(emission, np.where(dead_mask, 0, blocks), cost, zero, zero)
    labels[dead_mask] = 2
    if not np.array_equal(labels == 2, dead_mask):
        raise AssertionError("Staged decoder changed frozen D")
    evidence = {k: v for k, v in control.items() if isinstance(v, np.ndarray) and v.shape == (n,)}
    evidence.update({'stoppage_'+k: v for k, v in dead.items()
                     if isinstance(v, np.ndarray) and v.shape == (n,)})
    evidence.update(p_H_given_alive=ph, frozen_is_D=dead_mask, team_switch_cost=cost,
                    touch=features['touch'], ball_valid=features['ball_valid'],
                    ball_reliability=features['ball_reliability'],
                    ball_interpolated=features['ball_interpolated'])
    metadata = dict(mode='staged', D_mask_preserved=True, decoder=cfg,
                    control=control['metadata'], stoppage=dead['metadata'],
                    public_gt_used_for_inference=False,
                    calibration='Heuristic evidence, not calibrated probabilities',
                    ambiguous_HA='Forced H/A outside D; equal-evidence ties can favour H',
                    stoppage_role='Diagnostic only; per-team coverage differs from frozen pooled D')
    return labels, evidence, control['episodes'], metadata
