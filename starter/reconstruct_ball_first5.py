import numpy as np
import pandas as pd
from starter.detect_dead_ball import Config, line_events
from starter.reconstruct_ball_neighbors import fixed_distance_weights, missing_runs, reconstruct_gap

def fill_window(players, ball, start_frame, fps, pitch, distance_weighted=False, half=1):
    """Use fixed nearest-three displacement, excluding raw line-out gap evidence."""
    raw = np.asarray(ball, dtype=float)
    observed = np.isfinite(raw).all(axis=1)
    outside = observed & ((raw < 0) | (raw > np.asarray(pitch))).any(axis=1)
    inputs = raw.copy()
    inputs[~observed] = np.nan
    estimates = np.full_like(raw, np.nan)
    status = np.where(outside, "observed_outside", np.where(observed, "observed", "unprocessed")).astype(object)
    gap_ids = np.full(len(raw), -1, dtype=int)
    counts = np.zeros(len(raw), dtype=int)
    line_config = Config().validate()
    _, line_audit = line_events(raw, half, start_frame, fps, pitch, line_config)
    outward_gaps = {int(row["start_frame"]): row for row in line_audit if row["accepted"]}
    gaps = []
    for gap_id, (a, b) in enumerate(missing_runs(raw)):
        gap_ids[a:b] = gap_id
        selected = np.array([], dtype=int)
        prediction = np.full((min(b+1, len(raw))-a, 2), np.nan)
        used = np.zeros(len(prediction), dtype=int)
        if a and outside[a-1]:
            reason = "excluded_outside_anchor"
        elif start_frame+a in outward_gaps:
            reason = "excluded_near_line_outward_dropout"
        else:
            selected, prediction, used, reason = reconstruct_gap(
                players, raw, a, b, pitch, distance_weighted=distance_weighted)
        valid = np.isfinite(prediction[:b-a]).all(axis=1)
        indexes = np.arange(a, b)[valid]
        inputs[indexes] = prediction[:b-a][valid]
        estimates[a:b] = prediction[:b-a]
        counts[a:b] = used[:b-a]
        status[a:b] = (np.where(valid, "player_interpolated", "no_selected_players_visible")
                       if reason == "attempted" else reason)
        endpoint_error = (float(np.linalg.norm(prediction[-1]-raw[b]))
                          if b < len(raw) and np.isfinite(prediction[-1]).all() else None)
        gaps.append({"gap_id": gap_id, "start_frame": start_frame+a,
                     "end_frame_exclusive": start_frame+b, "duration_seconds": (b-a)/fps,
                     "status": reason, "filled_frames": int(valid.sum()),
                     "unfilled_frames": int((~valid).sum()),
                     "selected_slots_1based": ";".join(str(int(i)+1) for i in selected),
                     "weights_at_anchor": (";".join(f"{w:.8f}" for w in
                         (fixed_distance_weights(players[a-1, selected], raw[a-1]) if distance_weighted else np.full(3, 1/3)))
                         if len(selected) else ""),
                     "reappearance_discrepancy_m": endpoint_error,
                     "gap_clipped_at_window_end": b == len(raw),
                     "estimated_outside_frames": int((valid & ((prediction[:b-a] < 0) |
                                                       (prediction[:b-a] > pitch)).any(axis=1)).sum())})
    table = pd.DataFrame({"frame": start_frame+np.arange(len(raw)), "elapsed_seconds": np.arange(len(raw))/fps,
                          "raw_x_m": raw[:, 0], "raw_y_m": raw[:, 1],
                          "estimated_x_m": estimates[:, 0], "estimated_y_m": estimates[:, 1],
                          "input_x_m": inputs[:, 0], "input_y_m": inputs[:, 1],
                          "source_status": status, "gap_id": gap_ids, "valid_selected_players": counts})
    np.testing.assert_array_equal(inputs[observed], raw[observed])
    return table, pd.DataFrame(gaps), pd.DataFrame(line_audit)
