"""Build match observations and stoppage evidence from the three tactical assets."""
from io import StringIO

import numpy as np
import pandas as pd

from starter import detect_dead_ball as dead
from starter import detect_dead_ball_pooled as pooled
from starter import refine_dead_ball_restarts as restarts
from starter import refine_dead_ball_setpieces as setpieces
from starter import repair_player_tracks as repair


def table_boundary(table):
    """Match the decimal precision and empty-field handling of the staged CSV interface."""
    buffer = StringIO()
    table.to_csv(buffer, index=False, na_rep="NaN")
    buffer.seek(0)
    return pd.read_csv(buffer)


def prepare_stoppages(normalized, players, ball, ball_table, teams, keepers, directions, halves, fps, pitch):
    config = dead.Config().validate()
    stability = pooled.StabilityConfig().validate()
    restart_config = restarts.RestartConfig().validate()
    motion_rows, line_rows, speed_tables, restart_rows = [], [], [], []
    for half, first, last in halves:
        window = slice(first, last + 1)
        speeds, _ = dead.motion_features(normalized[window], first, half, teams, keepers, fps, pitch, config)
        lows, _, support = pooled.pooled_events(speeds, (first, last + 1), fps, config, stability, True)
        speeds["pooled_stable_support"] = support
        lines, _ = dead.line_events(ball[window], half, first, fps, pitch, config)
        speed_tables.append(speeds)
        motion_rows.extend(lows)
        line_rows.extend(lines)
        restart_rows.extend(restarts.find_proposals(ball[window], players[window], teams, half, first,
                                                     fps, pitch, directions, restart_config))
    # Separate parsing boundaries preserve the release model's empty values and numerical precision.
    line_records = table_boundary(dead.event_table(line_rows, fps)).to_dict("records")
    records = line_records + motion_rows
    events = dead.event_table(records, fps)
    if records:
        extra = pd.DataFrame(records).sort_values(["half", "start_frame", "method"], kind="stable").reset_index(drop=True)
        for name in ("all_before_mps", "all_low_mps", "all_drop_mps", "stable_required"):
            events[name] = extra[name] if name in extra else np.nan
    events = table_boundary(events)
    intervals = table_boundary(dead.merge_intervals(events, fps))
    proposals = pd.DataFrame(restart_rows, columns=restarts.PROPOSAL_COLUMNS).sort_values(["half", "frame", "restart_type"])
    proposals["proposal_id"] = np.arange(1, len(proposals) + 1)
    revised, _ = restarts.refine(events, intervals, proposals, ball, halves, fps, pitch, restart_config)
    speeds = table_boundary(pd.concat(speed_tables, ignore_index=True))
    mixed = ball_table[["output_x_m", "output_y_m"]].to_numpy()
    final, _, _ = setpieces.refine_setpieces(
        table_boundary(revised.replace([np.inf, -np.inf], np.nan)),
        table_boundary(dead.merge_intervals(revised, fps).replace([np.inf, -np.inf], np.nan)),
        speeds, ball, mixed, players, teams, directions, halves, fps, pitch,
        setpieces.SetPieceConfig().validate(), restart_config)
    frames = dead.frame_table(len(ball), halves, final)
    protected = np.zeros(len(ball), bool)
    for event in table_boundary(final).itertuples():
        if pd.notna(event.setpiece_proposal_id) or (isinstance(event.restart_type, str)
                and event.restart_type not in ("", "unknown")):
            protected[int(event.start_frame):int(event.end_frame_exclusive)] = True
    return frames, protected


def observed_players(players, raw_float32, teams, keepers, halves, fps, pitch):
    """Keep measured, unambiguous outfield coordinates and raw keeper candidates."""
    config = repair.Config()
    result = np.full(players.shape, np.nan, np.float32)
    for half, first, last in halves:
        window = slice(first, last + 1)
        clean, provenance = repair.cleanup(players[window], fps, pitch, config)
        _, uncertain = repair.identity_candidates(clean, teams, keepers, fps, config, half, first)
        stitches = repair.stitch_candidates(clean, teams, keepers, fps, config, half, first, pitch)
        uncertain = repair.mark_stitch_uncertainty(uncertain, stitches, first, fps, config)
        clean[(provenance != 0) | uncertain] = np.nan
        clean[:, keepers] = raw_float32[window, keepers]
        result[window] = clean
    return result
