from __future__ import annotations
import numpy as np
import pandas as pd
from . import goal_tag

def infer(features, players, teams, blocks, keeper_mask, fps, pitch, original_dead,
          held_team, previous_labels, previous_before_merge, previous_evidence,
          model_config, kernels, config=None, tags=()):
    """Pure-array inference; no accepted goal is an exact old-label no-op."""
    n = len(blocks)
    if (np.asarray(original_dead).shape != (n,) or np.asarray(held_team).shape != (n,)
            or np.asarray(previous_labels).shape != (n,) or np.asarray(previous_before_merge).shape != (n,)):
        raise ValueError('Goal inference axes differ')
    control_cfg = {**model_config.get('control', {}), 'pitch_length_m': float(pitch[0]),
                   'pitch_width_m': float(pitch[1]), 'interaction_distance_m': 1.}
    provisional = kernels.pipeline.control_episodes.build_control_evidence(
        {**features, 'p_dead': np.zeros(n, np.float32)}, players, teams, blocks, fps, control_cfg)
    goal = goal_tag.detect(features, players, teams, keeper_mask, blocks, fps, pitch,
        provisional['confirmed_control_team'], original_dead, tags, config)
    if goal['events'].empty:
        final, before = np.array(previous_labels, copy=True), np.array(previous_before_merge, copy=True)
        evidence = {key: np.array(value, copy=True) for key, value in previous_evidence.items()}
        episodes, model, gaps = [], {}, pd.DataFrame()
    else:
        before, evidence, episodes, model = kernels.pipeline.infer_staged(
            features, players, teams, blocks, fps, pitch, goal['revised_D'], keeper_mask, model_config)
        hold = (held_team >= 0) & ~goal['revised_D'] & ~goal['receiver_protected']
        before[hold] = held_team[hold]
        protected = goal['receiver_protected']
        before[protected] = goal['restart_team'][protected]
        before[goal['channels']['goal_D_override']] = 2
        merged_dead, gaps = kernels.merge.close_dead_gaps(goal['revised_D'], blocks, fps, .5)
        # Receiver control has priority over the older generic short-live-hole
        # suppression. Reject an entire bridge that overlaps that protected run.
        if not gaps.empty:
            keep = []
            for row in gaps.itertuples():
                a, b = int(row.start_frame), int(row.end_frame_exclusive)
                allow = not protected[a:b].any()
                keep.append(allow)
                if not allow:
                    merged_dead[a:b] = goal['revised_D'][a:b]
            gaps = gaps.copy()
            gaps['applied_after_receiver_priority'] = keep
        final = before.copy()
        final[merged_dead] = 2
        final[protected] = goal['restart_team'][protected]
        final[goal['channels']['goal_D_override']] = 2
        evidence = {**evidence, 'decoder_goal_D': goal['revised_D'],
            'short_D_gap_bridge': merged_dead & ~goal['revised_D'], 'final_buildup_owner': held_team}
    if (not np.isin(final[blocks > 0], [0, 1, 2]).all() or not np.all(final[blocks <= 0] == 3)
            or not np.all(final[goal['channels']['goal_D_override']] == 2)
            or not np.array_equal(final[goal['receiver_protected']], goal['restart_team'][goal['receiver_protected']])):
        raise ValueError('Goal reception/half/label contract failed')
    goal['channels']['goal_final_D'] = final == 2
    evidence.update(goal['channels'])
    evidence.update(pred_code=final, frozen_is_D=final == 2, final_is_D=final == 2)
    return dict(labels=final, before_merge=before, evidence=evidence, goal=goal,
                episodes=episodes, model=model, merged_gap_audit=gaps,
                no_op=goal['events'].empty)
