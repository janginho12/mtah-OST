from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    minimum_runners: int = 4
    motion_window_seconds: float = .8
    minimum_speed_mps: float = 2.
    maximum_speed_mps: float = 12.
    minimum_radial_speed_mps: float = 1.5
    minimum_heading_cosine: float = .7
    corner_radius_m: float = 25.
    minimum_convergence_seconds: float = .6
    duplicate_distance_m: float = .75
    duplicate_velocity_difference_mps: float = .75
    debounce_seconds: float = 10.
    maximum_restart_search_seconds: float = 180.
    minimum_goal_to_center_seconds: float = 3.
    center_radius_m: float = 2.5
    center_stationary_speed_mps: float = .8
    center_stationary_seconds: float = .4
    kicker_distance_m: float = 2.
    departure_distance_m: float = 2.
    departure_search_seconds: float = 1.5
    receiver_search_seconds: float = 6.
    receiver_distance_m: float = 2.
    receiver_center_distance_m: float = 3.
    receiver_control_seconds: float = .2
    distinct_player_distance_m: float = 2.
    minimum_ball_reliability: float = .65


CANDIDATE_COLUMNS = ['candidate_id', 'half', 'scoring_team', 'conceding_team',
    'corner_index', 'start_frame', 'end_frame_exclusive', 'peak_count', 'status',
    'goal_event_id', 'reason']


EVENT_COLUMNS = ['id', 'candidate_id', 'half', 'scoring_team', 'conceding_team',
    'corner_index', 'start_frame', 'corner_end_frame_exclusive', 'center_start_frame',
    'kick_frame', 'receive_frame', 'receiver_end_frame_exclusive', 'kicker_slot',
    'receiver_slot', 'cleared_prior_D_end_frame']


def runs(values, blocks=None):
    values = np.asarray(values)
    if not len(values):
        return []
    change = values[1:] != values[:-1]
    if blocks is not None:
        change |= np.asarray(blocks)[1:] != np.asarray(blocks)[:-1]
    edges = np.r_[0, np.flatnonzero(change)+1, len(values)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def motion(players, blocks, fps, cfg):
    """Only continuous, finite same-slot observations support a velocity window."""
    p, blocks = np.asarray(players), np.asarray(blocks)
    n, slots, _ = p.shape
    width = max(1, round(cfg.motion_window_seconds*fps))
    valid = np.isfinite(p).all(axis=2) & (blocks[:, None] > 0)
    edges = np.zeros((n, slots), bool)
    steps = np.linalg.norm(p[1:]-p[:-1], axis=2)*fps
    edges[1:] = (valid[1:] & valid[:-1] & (blocks[1:, None] == blocks[:-1, None])
                 & (steps <= cfg.maximum_speed_mps))
    invalid = np.cumsum(~edges, axis=0)
    velocity = np.full(p.shape, np.nan, np.float32)
    if n > width:
        good = valid[width:] & valid[:-width] & ((invalid[width:]-invalid[:-width]) == 0)
        v = (p[width:]-p[:-width])*(fps/width)
        velocity[width:] = np.where(good[..., None], v, np.nan)
    return velocity, edges


def deduplicated_count(points, velocities, indices, priority, cfg):
    """Count spatial/motion-distinct observations, not permanent player IDs."""
    selected = []
    order = indices[np.argsort(-priority[indices], kind='stable')]
    for slot in order:
        if any(np.linalg.norm(points[slot]-points[other]) <= cfg.duplicate_distance_m
               and np.linalg.norm(velocities[slot]-velocities[other]) <= cfg.duplicate_velocity_difference_mps
               for other in selected):
            continue
        selected.append(int(slot))
    return len(selected)


def corner_candidates(players, teams, keeper_mask, blocks, fps, pitch, cfg=None):
    cfg = cfg or Config()
    p, teams, keepers, blocks = map(np.asarray, (players, teams, keeper_mask, blocks))
    if (p.ndim != 3 or p.shape[-1] != 2 or teams.shape != (p.shape[1],)
            or keepers.shape != teams.shape or keepers.dtype != bool
            or blocks.shape != (len(p),) or not np.isin(teams, [0, 1]).all()
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Invalid goal motion input axes')
    velocity, edges = motion(p, blocks, fps, cfg)
    speed = np.linalg.norm(velocity, axis=2)
    counts = np.zeros((len(p), 2, 4), np.int8)
    L, W = map(float, pitch)
    corners = np.array([[0., 0.], [L, 0.], [0., W], [L, W]])
    plausible = np.isfinite(p).all(axis=2) & ((p >= -5.) & (p <= np.array([L, W])+5.)).all(axis=2)
    rows = []
    for corner_index, corner in enumerate(corners):
        delta = corner-p
        distance = np.linalg.norm(delta, axis=2)
        radial = np.sum(velocity*delta, axis=2)/np.maximum(distance, 1e-6)
        cosine = radial/np.maximum(speed, 1e-6)
        support = (plausible & (speed >= cfg.minimum_speed_mps) & (speed <= cfg.maximum_speed_mps)
                   & (distance <= cfg.corner_radius_m) & (radial >= cfg.minimum_radial_speed_mps)
                   & (cosine >= cfg.minimum_heading_cosine) & ~keepers[None, :] & (blocks[:, None] > 0))
        for team in (0, 1):
            mask = support & (teams[None, :] == team)
            raw_count = mask.sum(axis=1)
            counts[:, team, corner_index] = raw_count
            for f in np.flatnonzero(raw_count >= 2):
                counts[f, team, corner_index] = deduplicated_count(
                    p[f], velocity[f], np.flatnonzero(mask[f]), radial[f], cfg)
            yes = counts[:, team, corner_index] >= cfg.minimum_runners
            for a, b in runs(yes, blocks):
                if not yes[a] or b-a < np.ceil(cfg.minimum_convergence_seconds*fps):
                    continue
                rows.append(dict(half=int(blocks[a]), scoring_team='HA'[team],
                    conceding_team='HA'[1-team], corner_index=corner_index, start_frame=a,
                    end_frame_exclusive=b, peak_count=int(counts[a:b, team, corner_index].max()),
                    status='unconfirmed', goal_event_id=-1, reason='corner_convergence_without_confirmed_restart'))
    rows.sort(key=lambda r: (r['start_frame'], -r['peak_count'], r['scoring_team'], r['corner_index']))
    merged = []
    for row in rows:
        previous = next((r for r in reversed(merged) if r['half'] == row['half']
                         and r['scoring_team'] == row['scoring_team'] and r['corner_index'] == row['corner_index']), None)
        if previous and row['start_frame']-previous['end_frame_exclusive'] <= cfg.debounce_seconds*fps:
            previous['end_frame_exclusive'] = max(previous['end_frame_exclusive'], row['end_frame_exclusive'])
            previous['peak_count'] = max(previous['peak_count'], row['peak_count'])
        else:
            merged.append(row)
    for index, row in enumerate(merged, 1):
        row['candidate_id'] = index
        # Ambiguous simultaneous opposing-team convergence is not a scored-team observation.
        if any(other is not row and other['half'] == row['half'] and other['scoring_team'] != row['scoring_team']
               and max(row['start_frame'], other['start_frame']) < min(row['end_frame_exclusive'], other['end_frame_exclusive'])
               for other in merged):
            row.update(status='ambiguous_teams', reason='opposing_teams_converge_simultaneously')
    return merged, counts, edges


def _nearest(players, teams, ball, team):
    distance = np.linalg.norm(players-ball[:, None], axis=2)
    distance[:, teams != team] = np.inf
    distance[~np.isfinite(distance)] = np.inf
    slot = distance.argmin(axis=1)
    return slot, distance[np.arange(len(ball)), slot]


def apply_intervals(original_dead, blocks, events):
    """D until reception; old continuing D may end there, future D stays intact."""
    original = np.asarray(original_dead, bool)
    revised = original.copy()
    goal_d = np.zeros(len(original), bool)
    protected = np.zeros(len(original), bool)
    restart_team = np.full(len(original), -1, np.int8)
    updated = []
    for event in events:
        event = dict(event)
        a, r, stop = event['start_frame'], event['receive_frame'], event['receiver_end_frame_exclusive']
        if not (0 <= a < r < stop <= len(original)) or not np.all(blocks[a:stop] == event['half']):
            raise ValueError('Goal interval crosses half or has an invalid reception boundary')
        revised[a:r] = True
        goal_d[a:r] = True
        # Truncate only a D component already active at kick time. A later
        # separate stoppage is not erased by this new restart observation.
        clear_end = r
        if original[r]:
            lo = r
            while lo > 0 and original[lo-1] and blocks[lo-1] == event['half']:
                lo -= 1
            if lo <= event['kick_frame']:
                clear_end = r
                while clear_end < len(original) and original[clear_end] and blocks[clear_end] == event['half']:
                    clear_end += 1
                revised[r:clear_end] = False
        revised[r:stop] = False
        protected[r:stop] = True
        restart_team[r:stop] = 'HA'.index(event['conceding_team'])
        event['cleared_prior_D_end_frame'] = int(clear_end)
        updated.append(event)
    return revised, goal_d, protected, restart_team, updated
