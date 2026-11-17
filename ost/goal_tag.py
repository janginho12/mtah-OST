"""Automatic GOALS tag AND corner convergence -> latched D -> centre reception.

Tags are noisy input evidence, never ground truth. Once the AND gate has fired,
missing restart evidence cannot undo D. Only a same-half centre-origin pass to
a distinct conceding-team receiver releases the latch; half boundaries reset it.
No provided-match timestamps or public GT enter these functions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from . import goal_corner as motion


@dataclass(frozen=True)
class Config(motion.Config):
    # Association allowance, not a manual time offset or exact goal timestamp.
    tag_pre_seconds: float = 2.
    tag_post_seconds: float = 20.


CANDIDATE_COLUMNS = motion.CANDIDATE_COLUMNS + ['tag_id', 'tag_gap_seconds']
EVENT_COLUMNS = motion.EVENT_COLUMNS + ['tag_id', 'tag_start_frame', 'tag_end_frame',
    'end_frame_exclusive', 'release_status']
TAG_COLUMNS = ['tag_id', 'half', 'start_frame', 'end_frame_exclusive', 'team',
    'association_start', 'association_end', 'status', 'candidate_id']


def team_code(value):
    value = str(value or '').strip().casefold()
    if value in ('h', 'home', 'barcelona', 'fc barcelona'):
        return 'H'
    if value in ('a', 'away', 'athletic club', 'atletic club', 'athletic bilbao'):
        return 'A'
    return '?'


def goal_tags(tags, blocks, fps, cfg):
    """Accept already video-frame-aligned tags from the official local loader."""
    n = len(blocks)
    rows = []
    for tag_id, tag in enumerate(tags, 1):
        if str(tag.get('code') or '').strip().casefold() not in ('goal', 'goals'):
            continue  # Goal Kick / Kick Off (After Goal) are not GOALS tags.
        a, b = tag['start'], tag['end']
        if (isinstance(a, bool) or isinstance(b, bool) or not np.isfinite([a, b]).all()
                or int(a) != a or int(b) != b or b <= a):
            raise ValueError('GOALS tag must have a finite half-open video frame interval')
        for h in (1, 2):
            frames = np.flatnonzero(blocks == h)
            if not len(frames):
                continue
            lo, hi = int(frames[0]), int(frames[-1])+1
            # Raw tag must itself intersect this half: padded halftime tags do
            # not become a new second-half event.
            x, y = max(int(a), lo, 0), min(int(b), hi, n)
            if x >= y:
                continue
            rows.append(dict(tag_id=tag_id, half=h, start_frame=x, end_frame_exclusive=y,
                team=team_code(tag.get('team')), association_start=max(lo, x-round(cfg.tag_pre_seconds*fps)),
                association_end=min(hi, y+round(cfg.tag_post_seconds*fps)),
                status='no_supported_corner', candidate_id=-1))
    return rows


def pair_candidates(candidates, tags, fps):
    """One tag supports at most one latch; known tag-team disagreement rejects."""
    candidates = [dict(r, tag_id=-1, tag_gap_seconds=np.nan) for r in candidates]
    tags = [dict(r) for r in tags]
    used = set()
    for row in candidates:
        row.update(goal_event_id=-1)
        if row['status'] == 'ambiguous_teams':
            continue
        if any(old is not row and old['status'] == 'accepted' and old['half'] == row['half']
               and old['start_frame'] == row['start_frame'] for old in candidates):
            row.update(status='not_applied', reason='duplicate_simultaneous_corner_candidate')
            continue
        a, b = row['start_frame'], row['end_frame_exclusive']
        nearby = [t for t in tags if t['half'] == row['half']
                  and max(a, t['association_start']) < min(b, t['association_end'])]
        compatible = [t for t in nearby if t['team'] in ('?', row['scoring_team'])]
        available = [t for t in compatible if (t['tag_id'], t['half']) not in used]
        if not available:
            reason = ('goal_tag_already_used' if compatible else
                      'goal_tag_team_conflicts_with_corner_team' if nearby else 'no_nearby_GOALS_tag')
            row.update(status='not_applied', reason=reason)
            continue
        gap = lambda t: max(0, t['start_frame']-b, a-t['end_frame_exclusive'])
        chosen = min(available, key=lambda t: (gap(t), abs(a-t['start_frame']), t['tag_id']))
        if any((t['tag_id'], t['half']) in used and t['half'] == chosen['half']
               and max(t['start_frame'], chosen['start_frame']) < min(t['end_frame_exclusive'], chosen['end_frame_exclusive'])
               for t in tags):
            chosen.update(status='duplicate_overlapping_tag')
            row.update(status='not_applied', reason='duplicate_overlapping_GOALS_tag')
            continue
        used.add((chosen['tag_id'], chosen['half']))
        chosen.update(status='paired', candidate_id=row['candidate_id'])
        row.update(status='accepted', tag_id=chosen['tag_id'], tag_gap_seconds=gap(chosen)/fps,
            reason='GOALS_tag_AND_same_team_corner_convergence; D_latched_until_centre_reception')
    return candidates, tags


def kickoff_window(tags, goal_tag, after, end):
    """Associate only an automatic after-goal kick-off with this paired goal."""
    windows = []
    for tag in tags:
        if (str(tag.get('code', '')).strip().casefold() != 'set pieces'
                or str(tag.get('type', '')).strip().casefold() != 'kick off (after goal)'):
            continue
        a = max(int(tag['start']), after)
        b = min(int(tag['end']), end)
        if a < b and max(a, goal_tag['association_start']) < min(b, goal_tag['association_end']):
            windows.append((a, b))
    if windows:
        return min(windows, key=lambda w: (max(w[0] - goal_tag['end_frame_exclusive'],
                                                goal_tag['end_frame_exclusive'] - w[1], 0), w))
    return None


def find_centre_reception(row, features, players, teams, blocks, fps, pitch,
                          confirmed_team, player_edges, end, cfg, window=None):
    """Track the raw centre pass to the first distinct team-mate with control."""
    half, team = row['half'], 'HA'.index(row['conceding_team'])
    ball = np.asarray(features['raw_ball'])
    valid = (np.asarray(features['ball_valid'], bool) & np.isfinite(ball).all(axis=1)
             & (np.asarray(features['ball_reliability']) >= cfg.minimum_ball_reliability)
             & (blocks == half))
    speed = np.asarray(features['ball_speed'])
    centre_distance = np.linalg.norm(ball-np.asarray(pitch)/2, axis=1)
    slots, distances = motion._nearest(players, teams, ball, team)
    earliest = max(row['end_frame_exclusive'],
                   row['start_frame']+round(cfg.minimum_goal_to_center_seconds*fps))
    lo, hi = (earliest, end) if window is None else (max(earliest, window[0]), min(end, window[1]))
    empty = dict(center_start_frame=-1, kick_frame=-1, receive_frame=-1,
        receiver_end_frame_exclusive=-1, kicker_slot=-1, receiver_slot=-1,
        cleared_prior_D_end_frame=-1)
    if lo >= hi:
        return empty
    contact = valid & (centre_distance <= cfg.center_radius_m) & (distances <= cfg.kicker_distance_m)
    still = contact & np.isfinite(speed) & (speed <= cfg.center_stationary_speed_mps)
    prep_frames = max(1, int(np.ceil(cfg.center_stationary_seconds * fps)))
    key = np.where(still[lo:hi], slots[lo:hi] + 1, -1)
    anchors = [(lo+a, lo+b-1) for a, b in motion.runs(key)
               if key[a] >= 0 and b-a >= prep_frames]
    if window is not None:
        # During occlusion the first ball sample can already be moving. Only
        # accept a new observed track starting at the centre on the kicker's
        # feet inside a matching restart tag. A pass crossing midfield while
        # already tracked cannot supply this origin.
        _, other_distance = motion._nearest(players[lo:hi], teams, ball[lo:hi], 1-team)
        for offset in np.flatnonzero(contact[lo:hi] & (distances[lo:hi] < other_distance)):
            anchor = lo + int(offset)
            if anchor >= prep_frames and not valid[anchor-prep_frames:anchor].any():
                anchors.append((anchor, anchor))
    partial = empty.copy()
    for a, anchor in sorted(set(anchors), key=lambda pair: (pair[1], pair[0])):
        kicker, b = int(slots[anchor]), anchor+1
        departure_end = min(end, b+max(1, round(cfg.departure_search_seconds*fps)))
        moved = np.linalg.norm(ball[b:departure_end]-ball[anchor], axis=1)
        departure = np.flatnonzero(valid[b:departure_end] & (moved >= cfg.departure_distance_m)
                                   & (centre_distance[b:departure_end] > centre_distance[anchor]))
        if not len(departure):
            continue
        kick = b+int(departure[0])
        if not valid[anchor:kick+1].all():
            continue
        observed = dict(empty, center_start_frame=a, kick_frame=kick, kicker_slot=kicker+1)
        if partial['kick_frame'] < 0:
            partial = observed
        stop = min(end, kick+max(1, round(cfg.receiver_search_seconds*fps))+1)
        # Losing the raw trajectory, the kicker, or control to the opponent
        # breaks the pass. A later regaining team cannot complete this kick-off.
        path = slice(anchor+1, stop)
        broken = np.flatnonzero(~valid[path] | ~player_edges[path, kicker] | (confirmed_team[path] == 1-team))
        if len(broken):
            stop = anchor+1+int(broken[0])
        if stop <= kick+1:
            continue
        frames = np.arange(kick+1, stop)
        nearest = slots[frames]
        receiver_ok = ((confirmed_team[frames] == team) & (distances[frames] <= cfg.receiver_distance_m)
            & (centre_distance[frames] >= cfg.receiver_center_distance_m) & (nearest != kicker))
        separation = np.linalg.norm(players[frames, nearest]-players[frames, kicker], axis=1)
        receiver_ok &= np.isfinite(separation) & (separation >= cfg.distinct_player_distance_m)
        receiver_key = np.where(receiver_ok, nearest+1, -1)
        for first, last in motion.runs(receiver_key):
            if receiver_key[first] < 0 or last-first < np.ceil(cfg.receiver_control_seconds*fps):
                continue
            ra, rb = kick+1+first, kick+1+last
            receiver = int(slots[ra])
            if (not np.isfinite(players[anchor, receiver]).all()
                    or not player_edges[anchor+1:ra+1, receiver].all()
                    or np.linalg.norm(players[anchor, receiver]-players[anchor, kicker]) < cfg.distinct_player_distance_m):
                continue
            return dict(observed, receive_frame=ra, receiver_end_frame_exclusive=rb,
                        receiver_slot=receiver+1, cleared_prior_D_end_frame=ra)
    return partial


def detect(features, players, teams, keeper_mask, blocks, fps, pitch,
           confirmed_team, original_dead, tags, cfg=None):
    cfg = cfg or Config()
    blocks = np.asarray(blocks)
    n = len(blocks)
    candidates, counts, edges = motion.corner_candidates(players, teams, keeper_mask, blocks, fps, pitch, cfg)
    candidates, tag_rows = pair_candidates(candidates, goal_tags(tags, blocks, fps, cfg), fps)
    latched = [r for r in candidates if r['status'] == 'accepted']
    events = []
    for i, row in enumerate(latched):
        half_end = int(np.flatnonzero(blocks == row['half'])[-1])+1
        next_start = next((r['start_frame'] for r in latched[i+1:] if r['half'] == row['half']), half_end)
        end = min(half_end, next_start)
        tag = next(t for t in tag_rows if t['candidate_id'] == row['candidate_id'])
        window = kickoff_window(tags, tag, row['end_frame_exclusive'], end)
        result = find_centre_reception(row, features, players, teams, blocks, fps, pitch,
                                      confirmed_team, edges, end, cfg, window)
        received = result['receive_frame'] >= 0
        row['goal_event_id'] = len(events)+1
        events.append(dict(id=row['goal_event_id'], candidate_id=row['candidate_id'],
            half=row['half'], scoring_team=row['scoring_team'], conceding_team=row['conceding_team'],
            corner_index=row['corner_index'], start_frame=row['start_frame'],
            corner_end_frame_exclusive=row['end_frame_exclusive'], tag_id=row['tag_id'],
            tag_start_frame=tag['start_frame'], tag_end_frame=tag['end_frame_exclusive'], **result,
            end_frame_exclusive=result['receive_frame'] if received else end,
            release_status='received' if received else ('held_to_half_end' if end == half_end else 'held_until_next_goal_evidence')))
    revised, goal_d, protected, restart_team, received_rows = motion.apply_intervals(
        original_dead, blocks, [e for e in events if e['receive_frame'] >= 0])
    for updated in received_rows:
        next(e for e in events if e['id'] == updated['id']).update(updated)
    channels = dict(goal_candidate=np.zeros(n, bool), goal_pending=np.zeros(n, bool),
        goal_D_override=goal_d, goal_kick=np.zeros(n, bool), goal_receive=np.zeros(n, bool),
        goal_receiver_protected=protected, goal_original_D=np.asarray(original_dead, bool),
        goal_revised_D=revised, goal_scoring_team=np.full(n, -1, np.int8),
        goal_corner_index=np.full(n, -1, np.int8), goal_corner_runner_count=counts.max(axis=(1, 2)),
        goal_tag=np.zeros(n, bool), goal_AND_trigger=np.zeros(n, bool),
        goal_latched_D=np.zeros(n, bool), goal_unresolved_D=np.zeros(n, bool),
        goal_tag_id=np.full(n, -1, np.int32))
    for tag in tag_rows:
        channels['goal_tag'][tag['start_frame']:tag['end_frame_exclusive']] = True
    for row in candidates:
        a, b = row['start_frame'], row['end_frame_exclusive']
        channels['goal_candidate'][a:b] = True
        channels['goal_scoring_team'][a:b] = 'HA'.index(row['scoring_team'])
        channels['goal_corner_index'][a:b] = row['corner_index']
        channels['goal_corner_runner_count'][a:b] = counts[a:b, 'HA'.index(row['scoring_team']), row['corner_index']]
    for event in events:
        a, b = event['start_frame'], event['end_frame_exclusive']
        if not a < b or not np.all(blocks[a:b] == event['half']):
            raise ValueError('Invalid goal latch interval')
        revised[a:b] = True
        goal_d[a:b] = True
        channels['goal_AND_trigger'][a] = True
        channels['goal_latched_D'][a:b] = True
        channels['goal_pending'][a:b] = True
        if event['receive_frame'] < 0:
            channels['goal_unresolved_D'][a:b] = True
        if event['kick_frame'] >= 0:
            channels['goal_kick'][event['kick_frame']] = True
        if event['receive_frame'] >= 0:
            channels['goal_receive'][event['receive_frame']] = True
        channels['goal_scoring_team'][a:b] = 'HA'.index(event['scoring_team'])
        channels['goal_corner_index'][a:b] = event['corner_index']
        channels['goal_corner_runner_count'][a:b] = counts[a:b, 'HA'.index(event['scoring_team']), event['corner_index']]
        channels['goal_tag_id'][a:b] = event['tag_id']
    if (np.any(goal_d & protected) or np.any(goal_d[blocks <= 0])
            or not np.all(revised[goal_d])):
        raise ValueError('Goal D latch conflicts with receiver/half boundaries')
    config = asdict(cfg)
    config.pop('maximum_restart_search_seconds')  # No timeout may release D.
    return dict(candidates=pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS),
        events=pd.DataFrame(events, columns=EVENT_COLUMNS), tags=pd.DataFrame(tag_rows, columns=TAG_COLUMNS),
        channels=channels, revised_D=revised, restart_team=restart_team,
        receiver_protected=protected, config=config,
        semantics='GOALS_AND_corner; latch_to_centre_origin_distinct_receiver_control_or_half_end')
