from __future__ import annotations
import json
from pathlib import Path
import warnings
import numpy as np

CONFIG_PATH = Path(__file__).with_name('keeper_selection_config.json')


DEFAULT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))


SELECTION_REASONS = {
    0: 'no candidate in this defending half, or inactive block',
    1: 'only finite candidate in this defending half',
    2: 'highest robust activity beyond the tie margin',
    3: 'activity near tie: previous selected source retained',
    4: 'activity near tie: closest to own goal, then lowest slot',
}


def _runs(values):
    cuts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(values)]
    return [(int(a), int(b), int(values[a])) for a, b in zip(cuts[:-1], cuts[1:])]


def _config(overrides):
    cfg = {**DEFAULT_CONFIG, **(overrides or {})}
    if set(cfg) != set(DEFAULT_CONFIG) or cfg['method'] != 'side_activity_selection':
        raise ValueError('Unknown goalkeeper selection configuration')
    if cfg['minimum_confidence'] != 0:
        raise ValueError('Selection confidence is diagnostic; threshold must be zero')
    if any(not np.isfinite(v) or v <= 0 for k, v in cfg.items()
           if k not in ('method', 'minimum_confidence')):
        raise ValueError('Selection parameters must be positive finite values')
    if (not .5 < cfg['side_min_fraction'] <= 1
            or cfg['side_min_team_slots'] != int(cfg['side_min_team_slots'])):
        raise ValueError('Invalid opening formation requirements')
    return cfg


def _side(players, teams, keepers, fps, pitch, cfg):
    """Use opening nonkeeper formation, with an explicit weaker fallback."""
    length = float(pitch[0])
    count = min(len(players), max(1, round(cfg['side_search_seconds']*fps)))
    data = players[:count]
    finite = np.isfinite(data).all(axis=2)
    finite &= ((data[..., 0] >= 0) & (data[..., 0] <= length)
               & (data[..., 1] >= 0) & (data[..., 1] <= pitch[1]))
    counts, medians, fractions = [], [], []
    for team in (0, 1):
        mask = (teams == team) & ~keepers
        x = np.where(finite[:, mask], data[:, mask, 0], np.nan)
        counts.append(np.isfinite(x).sum(axis=1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            medians.append(np.nanmedian(x, axis=1))
        fractions.append(np.divide((x < length/2).sum(axis=1), counts[-1],
                         out=np.full(count, .5), where=counts[-1] > 0))
    h, a = fractions
    usable = ((counts[0] >= cfg['side_min_team_slots'])
              & (counts[1] >= cfg['side_min_team_slots'])
              & (abs(medians[0]-medians[1]) >= cfg['side_min_median_separation_fraction']*length))
    left = usable & (h >= cfg['side_min_fraction']) & (a <= 1-cfg['side_min_fraction'])
    right = usable & (a >= cfg['side_min_fraction']) & (h <= 1-cfg['side_min_fraction'])
    orientation = np.where(left, 0, np.where(right, 1, -1))
    needed = max(1, round(cfg['side_stable_seconds']*fps))
    for first, stop, side in _runs(orientation):
        if side >= 0 and stop-first >= needed:
            support = np.minimum(abs(h[first:first+needed]-.5), abs(a[first:first+needed]-.5))*2
            return side == 0, dict(home_defends='left' if side == 0 else 'right',
                away_defends='right' if side == 0 else 'left',
                confidence=float(np.median(support)), status='stable_opening_formation',
                method='earliest sustained separated nonkeeper formation',
                evidence_start_offset=first, evidence_stop_offset_exclusive=first+needed)

    # Weak fallback still uses BOTH teams' observed field players. A temporal
    # window and observed-count weight reduce sensitivity to one extreme slot.
    valid = (counts[0] > 0) & (counts[1] > 0) & np.isfinite(medians[0]) & np.isfinite(medians[1])
    signed = .75*(h-a)+.25*np.clip((medians[1]-medians[0])/(.5*length), -1, 1)
    weights = np.minimum(1., np.minimum(counts[0], counts[1])/cfg['side_min_team_slots'])*valid
    width = min(count, max(1, round(cfg['side_fallback_seconds']*fps)))
    cs = np.r_[0., np.cumsum(np.where(valid, signed, 0.)*weights)]
    cw = np.r_[0., np.cumsum(weights)]
    support = cw[width:]-cw[:-width]
    scores = (cs[width:]-cs[:-width])/width
    if not len(scores) or not np.any(support > 0) or np.max(abs(scores)) <= 1.e-8:
        raise ValueError('Cannot infer defending halves: no directional nonkeeper opening evidence')
    first = int(np.argmax(abs(scores)))
    home_left = scores[first] > 0
    # Report a weak confidence, but it NEVER masks selected coordinates.
    confidence = min(.49, float(abs(scores[first])))
    return bool(home_left), dict(home_defends='left' if home_left else 'right',
        away_defends='right' if home_left else 'left', confidence=confidence,
        status='weak_opening_fallback', method='strongest weighted opening split window',
        evidence_start_offset=first, evidence_stop_offset_exclusive=first+width)


def _activity(positions, fps, length, cfg):
    """Robust velocity magnitude, never crossing missing/jump/half-field cuts.

    Coordinates remain untouched. A jump only breaks the motion-estimation run.
    A singleton observation has zero activity with activity_valid=False; it can
    still be selected as the only observed keeper in its physical half.
    """
    n, k = positions.shape[:2]
    available = np.isfinite(positions).all(axis=2)
    physical_half = np.where(available, (positions[..., 0] >= length/2).astype(np.int8), -1)
    activity = np.zeros((n, k), np.float32)
    activity_valid = np.zeros((n, k), bool)
    jump = np.zeros((n, k), bool)
    radius = max(1, round(cfg['activity_velocity_radius_seconds']*fps))
    median_radius = max(0, round(cfg['activity_smooth_seconds']*fps/2))
    for slot in range(k):
        pos = positions[:, slot]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)*fps
        bad_edge = available[:-1, slot] & available[1:, slot] & (step > cfg['max_activity_speed_mps'])
        jump[:-1, slot] |= bad_edge
        jump[1:, slot] |= bad_edge
        changes = (physical_half[1:, slot] != physical_half[:-1, slot]) | bad_edge
        cuts = np.r_[0, np.flatnonzero(changes)+1, n]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if not available[lo, slot] or hi-lo < 2:
                continue
            t = np.arange(lo, hi)
            before, after = np.maximum(lo, t-radius), np.minimum(hi-1, t+radius)
            speed = np.linalg.norm(pos[after]-pos[before], axis=1)*fps/(after-before)
            r = min(median_radius, hi-lo-1)
            # NaN padding shrinks windows at segment ends without borrowing data.
            padded = np.pad(speed, (r, r), constant_values=np.nan)
            windows = np.lib.stride_tricks.sliding_window_view(padded, 2*r+1)
            activity[lo:hi, slot] = np.nanmedian(windows, axis=1)
            activity_valid[lo:hi, slot] = True
    return activity, activity_valid, physical_half, jump


def preprocess(players, teams, blocks, fps, pitch, keeper_mask, config=None, ball=None,
               formation_players=None):
    """Select one raw candidate in every occupied defending half, per frame.

    With two candidates in different halves, both are selected. With two in one
    half, select the higher activity candidate; a near tie retains that half's
    previous selection, otherwise choose the goal-nearest candidate. The half
    without an observation stays missing. Ball position only annotates whether
    a selected keeper is in the event/ball half; it does not change coordinates
    or override a candidate's physical half.
    """
    cfg = _config(config)
    players = np.asarray(players)
    teams, blocks = np.asarray(teams), np.asarray(blocks)
    keepers = np.asarray(keeper_mask, bool)
    if (players.ndim != 3 or players.shape[2] != 2 or not len(players)
            or teams.shape != (players.shape[1],) or keepers.shape != teams.shape
            or blocks.shape != (len(players),) or not np.isin(teams, [0, 1]).all()
            or not np.isfinite(fps) or fps <= 0 or len(pitch) != 2
            or not np.isfinite(pitch).all() or min(pitch) <= 0):
        raise ValueError('Invalid goalkeeper selection input contract')
    candidates = np.flatnonzero(keepers)
    if len(candidates) != 2:
        raise ValueError('Expected exactly two metadata goalkeeper candidates')
    if np.any(~np.isfinite(blocks)) or np.any(blocks < 0) or np.any(blocks != blocks.astype(int)):
        raise ValueError('Half block IDs must be nonnegative integers')
    runs = _runs(blocks)
    positive = [block for _, _, block in runs if block > 0]
    if len(positive) != len(set(positive)):
        raise ValueError('Each half block must occupy one contiguous run')
    n, p, _ = players.shape
    formation_players = players if formation_players is None else np.asarray(formation_players)
    if formation_players.shape != players.shape:
        raise ValueError('Opening formation/player shape mismatch')
    if ball is not None and np.asarray(ball).shape != (n, 2):
        raise ValueError('Ball/frame shape mismatch')
    ball_half = np.full(n, -1, np.int8)
    if ball is not None:
        ball = np.asarray(ball)
        finite_ball = np.isfinite(ball).all(axis=1) & (blocks > 0)
        ball_half[finite_ball] = (ball[finite_ball, 0] >= pitch[0]/2).astype(np.int8)
    roles = np.full((n, 2, 2), np.nan, np.float32)
    source = np.full((n, 2), -1, np.int16)
    confidence = np.zeros((n, 2), np.float32)
    reason = np.zeros((n, 2), np.int8)
    selected_activity = np.full((n, 2), np.nan, np.float32)
    candidate_activity = np.zeros((n, 2), np.float32)
    activity_valid = np.zeros((n, 2), bool)
    candidate_half = np.full((n, 2), -1, np.int8)
    jump = np.zeros((n, 2), bool)
    home_defends_left = np.zeros(n, bool)
    side_confidence = np.zeros(n, np.float32)
    duplicate = np.zeros(n, bool)
    on_ball_side = np.zeros((n, 2), bool)
    summary = {}
    for lo, hi, block in runs:
        if block <= 0:
            continue
        home_left, meta = _side(formation_players[lo:hi], teams, keepers, fps, pitch, cfg)
        summary[str(block)] = dict(meta, start_frame=lo, stop_frame_exclusive=hi)
        home_defends_left[lo:hi] = home_left
        side_confidence[lo:hi] = meta['confidence']
        positions = players[lo:hi, candidates].astype(np.float32)
        activity, valid_activity, halves, jumps = _activity(positions, fps, pitch[0], cfg)
        candidate_activity[lo:hi], activity_valid[lo:hi] = activity, valid_activity
        candidate_half[lo:hi], jump[lo:hi] = halves, jumps
        duplicate[lo:hi] = (halves[:, 0] >= 0) & (halves[:, 0] == halves[:, 1])
        previous = [-1, -1]  # reset at every half and every unobserved role frame
        role_half = (0, 1) if home_left else (1, 0)
        for local_t in range(hi-lo):
            t = lo+local_t
            for role, side in enumerate(role_half):
                options = np.flatnonzero(halves[local_t] == side)
                if not len(options):
                    previous[role] = -1
                    continue
                if len(options) == 1:
                    chosen, why, strength = int(options[0]), 1, 1.
                else:
                    scores = activity[local_t, options]
                    best = float(scores.max())
                    tied = options[scores >= best-cfg['activity_tie_mps']]
                    if len(tied) == 1:
                        chosen, why = int(tied[0]), 2
                        gap = float(np.ptp(scores))
                        strength = gap/(gap+cfg['activity_tie_mps'])
                    elif previous[role] in tied:
                        chosen, why, strength = previous[role], 3, .25
                    else:
                        distance = positions[local_t, tied, 0]
                        if side == 1:
                            distance = pitch[0]-distance
                        chosen = int(tied[np.argmin(distance)])
                        why, strength = 4, .25
                previous[role] = chosen
                roles[t, role] = positions[local_t, chosen]
                source[t, role] = candidates[chosen]
                confidence[t, role] = meta['confidence']*strength
                reason[t, role] = why
                selected_activity[t, role] = activity[local_t, chosen]
                on_ball_side[t, role] = ball_half[t] == side
    corrected = np.concatenate((players.astype(np.float32).copy(), roles), axis=1)
    corrected[:, candidates] = np.nan
    reassigned = np.zeros((n, 2), bool)
    for role in (0, 1):
        known = source[:, role] >= 0
        reassigned[known, role] = teams[source[known, role]] != role
    return dict(players=corrected, teams=np.r_[teams, 0, 1].astype(np.int8),
        keeper_mask=np.r_[keepers, True, True], source_slot=source,
        confidence=confidence, side_confidence=side_confidence, role_positions=roles,
        side_by_half=summary, home_defends_left=home_defends_left, ball_half=ball_half,
        selection_reason=reason, selected_activity=selected_activity,
        candidate_activity=candidate_activity, candidate_slots=candidates,
        activity_valid=activity_valid, candidate_half=candidate_half,
        activity_jump=jump, selection_on_ball_side=on_ball_side,
        duplicate_ambiguous=duplicate, mirrored_ambiguous=np.zeros(n, bool),
        reassigned=reassigned, config=cfg,
        metadata=dict(method='side_activity_selection', role_order=['H_GK', 'A_GK'],
            virtual_slot_indices=[p, p+1], source_slot_index_base=0,
            confidence='diagnostic side support and activity contrast; never used to reject coordinates',
            selection_reason_labels={str(k): v for k, v in SELECTION_REASONS.items()},
            candidate_selection='exactly two provided metadata goalkeeper role slots',
            goalkeeper_team_metadata_used_for_inference=False,
            coordinate_reflection=False, coordinate_interpolation=False, input_mutated=False,
            public_gt_used=False, ball_used='physical event-half annotation only; missing ball never suppresses selection',
            guarantee='Exactly one observed coordinate per occupied physical half, during each active half',
            assumptions=['Each physical half belongs to the defending goalkeeper team.',
                         'Opening nonkeeper team formation indicates defending direction.',
                         'Among same-half candidates, more robust local activity is more useful.'],
            limitations=['Real goalkeeper midfield crossings violate the hard own-half model.',
                         'Duplicate moving false detections may win activity selection.',
                         'Weak opening fallback can select the wrong defending orientation.',
                         'Finite outlier coordinates can be selected; jump flags only protect activity.',
                         'No candidate in a half means that keeper coordinate cannot be selected.']))
