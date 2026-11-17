from __future__ import annotations
import numpy as np
from .hybrid_features import _runs

DEFAULT_DECODER = dict(
    evidence_rate_hz=5., team_switch_cost=1.4, uncertainty_switch_extra=1.2,
    alive_dead_cost=1.6, dead_alive_cost=1.2,
    minimum_switch_cost=.15, maximum_switch_cost=4.,
    confirmed_onset_cost_fraction=.3, weak_tag_weight=.12,
    probability_clip=.005,
)


def validate_config(config=None):
    cfg = {**DEFAULT_DECODER, **(config or {})}
    if set(cfg) != set(DEFAULT_DECODER):
        raise ValueError('Unknown decoder configuration key')
    if not all(isinstance(v, (float, int)) and np.isfinite(v) for v in cfg.values()):
        raise ValueError('Decoder values must be finite numbers')
    if not (0 < cfg['probability_clip'] < .5 and cfg['evidence_rate_hz'] > 0
            and 0 < cfg['minimum_switch_cost'] <= cfg['maximum_switch_cost']
            and 0 < cfg['confirmed_onset_cost_fraction'] <= 1
            and 0 <= cfg['weak_tag_weight'] <= 1):
        raise ValueError('Invalid decoder scale, probability or cost')
    if any(cfg[k] < 0 for k in ('team_switch_cost', 'uncertainty_switch_extra',
                               'alive_dead_cost', 'dead_alive_cost')):
        raise ValueError('Transition costs cannot be negative')
    return cfg


def viterbi(emission, blocks, team_cost, into_dead, out_dead):
    """O(9*N), independent contiguous half blocks; 0=H,1=A,2=D,3=X.

Every frame permits every transition. Candidate boundaries reduce a cost but
never force or forbid a transition. There is no hard minimum spell duration.
"""
    emission = np.asarray(emission, float)
    blocks = np.asarray(blocks)
    n = len(blocks)
    if emission.shape != (n, 3) or not np.isfinite(emission).all():
        raise ValueError('Invalid emission shape or nonfinite values')
    costs = [np.asarray(v, float) for v in (team_cost, into_dead, out_dead)]
    if any(v.shape != (n,) or not np.isfinite(v).all() or (v < 0).any() for v in costs):
        raise ValueError('Invalid transition cost arrays')
    labels = np.full(n, 3, np.int8)
    destination = np.arange(3)
    matrix = np.zeros((3, 3))
    for lo, hi in _runs(blocks):
        if blocks[lo] <= 0:
            continue
        back = np.zeros((hi-lo, 3), np.int8)
        value = emission[lo].copy()
        for j, t in enumerate(range(lo+1, hi), 1):
            matrix[0, 1] = matrix[1, 0] = costs[0][t]
            matrix[:2, 2] = costs[1][t]
            matrix[2, :2] = costs[2][t]
            candidate = value[:, None] - matrix
            back[j] = np.argmax(candidate, axis=0)
            value = candidate[back[j], destination] + emission[t]
            value -= value.max()
        labels[hi-1] = np.argmax(value)
        for j in range(hi-lo-1, 0, -1):
            labels[lo+j-1] = back[j, labels[lo+j]]
    return labels
