from __future__ import annotations
import numpy as np

MAXIMUM_GAP_SECONDS = .5


def observed_contacts(features, players, teams, blocks, fps, pitch, control_config, kernels):
    """The teammate's raw-only contact logic, with canonical GK inputs."""
    cfg = kernels.buildup.Config()
    cc = {**control_config, 'pitch_length_m': float(pitch[0]),
          'pitch_width_m': float(pitch[1]), 'interaction_distance_m': 1.}
    provisional = kernels.pipeline.control_episodes.build_control_evidence(
        {**features, 'p_dead': np.zeros(len(blocks), np.float32)},
        players, teams, blocks, fps, cc)
    ball = features['raw_ball']
    confirmed = provisional['confirmed_control_team']
    observed = (features['ball_valid'] & np.isfinite(features['ball_velocity']).all(axis=1)
                & ((ball >= 0) & (ball <= pitch)).all(axis=1) & (blocks > 0))
    distance = np.linalg.norm(players-ball[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    distance[teams[None, :] != confirmed[:, None]] = np.inf
    nearest = distance.argmin(axis=1)
    control = observed & (confirmed >= 0) & (distance[np.arange(len(ball)), nearest] <= cfg.contact_distance_m)
    key = np.where(control, nearest+1, -1)
    contacts = []
    # A contact is never joined across halves even if a synthetic fixture has
    # adjacent half blocks with no halftime frames between them.
    cuts = np.r_[0, np.flatnonzero((key[1:] != key[:-1]) | (blocks[1:] != blocks[:-1]))+1, len(key)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if key[a] < 0 or b-a < np.ceil(cfg.contact_duration_seconds*fps):
            continue
        slot = int(nearest[a])
        contacts.append(dict(start=int(a), stop=int(b), slot=slot, team=int(teams[slot])))
    return contacts, confirmed, observed


def cycles_by_half(contacts, features, confirmed, observed, blocks, home_defends_left,
                   fps, pitch, kernels):
    """Keep source circulation rules while using the GK model's own-half map."""
    cycles, transfers, directions = [], [], {}
    side = np.asarray(home_defends_left)
    if side.shape != blocks.shape or side.dtype != bool:
        raise ValueError('Expected frame-aligned boolean defensive-half map')
    for half in (1, 2):
        frames = np.flatnonzero(blocks == half)
        if not len(frames):
            continue
        if np.any(np.diff(frames) != 1) or not np.all(side[frames] == side[frames[0]]):
            raise ValueError('Each half needs one stable defensive-side assignment')
        actual = np.array([1, -1]) if side[frames[0]] else np.array([-1, 1])
        # Vendor circulation_cycles internally flips supplied first-half
        # directions for block 2. Cancel that flip to supply this half's map.
        vendor_direction = actual if half == 1 else -actual
        selected = [c for c in contacts if blocks[c['start']] == half]
        found, passes = kernels.buildup.circulation_cycles(
            selected, features['raw_ball'], observed, features['ball_speed'], confirmed,
            blocks, vendor_direction, fps, pitch, kernels.buildup.Config())
        cycles.extend(found)
        transfers.extend(passes)
        directions[str(half)] = dict(home_defends_left=bool(side[frames[0]]),
                                    H_attacks_x=int(actual[0]), A_attacks_x=int(actual[1]))
    return cycles, transfers, directions


def infer(features, players, teams, blocks, fps, pitch, original_dead, keeper_mask,
          home_defends_left, line, setpiece, model_config, kernels):
    """Pure array path; suitable for independent synthetic behavior checks."""
    players, teams, blocks = np.asarray(players), np.asarray(teams), np.asarray(blocks)
    original_dead = np.asarray(original_dead, bool)
    line, setpiece = np.asarray(line, bool), np.asarray(setpiece, bool)
    n = len(blocks)
    if (n == 0 or players.ndim != 3 or players.shape[0] != n or players.shape[-1] != 2
            or teams.shape != (players.shape[1],) or np.asarray(keeper_mask).shape != teams.shape
            or any(a.shape != (n,) for a in (original_dead, line, setpiece))
            or np.any(original_dead & (blocks <= 0))):
        raise ValueError('Misaligned combined model inputs')
    contacts, confirmed, observed = observed_contacts(
        features, players, teams, blocks, fps, pitch, model_config.get('control', {}), kernels)
    cycles, transfers, directions = cycles_by_half(contacts, features, confirmed, observed,
        blocks, home_defends_left, fps, pitch, kernels)
    revised, held_team, audit, support = kernels.buildup.refine(
        original_dead, line, setpiece, blocks, cycles, fps, kernels.buildup.Config())
    before, evidence, episodes, model = kernels.pipeline.infer_staged(
        features, players, teams, blocks, fps, pitch, revised, keeper_mask, model_config)
    held = held_team >= 0
    if np.any(held & (confirmed >= 0) & (confirmed != held_team)):
        raise ValueError('Buildup hold conflicts with observed control')
    before[held] = held_team[held]
    final_dead, gaps = kernels.merge.close_dead_gaps(revised, blocks, fps, MAXIMUM_GAP_SECONDS)
    bridge = final_dead & ~revised
    final = before.copy()
    final[bridge] = 2
    if (not np.array_equal(before == 2, revised) or not np.array_equal(final == 2, final_dead)
            or not np.all(final[blocks <= 0] == 3)
            or not np.isin(final[blocks > 0], [0, 1, 2]).all()
            or not np.array_equal(final[~bridge], before[~bridge])):
        raise ValueError('Combined model label/half contract failed')
    evidence = {**evidence, 'decoder_revised_D': revised, 'original_D': original_dead,
        'revised_D': revised, 'short_D_gap_bridge': bridge, 'final_is_D': final_dead,
        'frozen_is_D': final_dead, 'pred_code': final, 'final_buildup_owner': held_team}
    return dict(labels=final, before_merge=before, evidence=evidence, contacts=contacts,
        confirmed_control=confirmed, raw_observed=observed, cycles=cycles, transfers=transfers,
        revised_D=revised, buildup_team=support, held_team=held_team, guard_audit=audit,
        merged_gap_audit=gaps, episodes=episodes, model=model, directions=directions)
