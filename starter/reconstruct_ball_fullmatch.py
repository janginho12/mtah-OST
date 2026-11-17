import numpy as np
import pandas as pd
from starter.reconstruct_ball_first5 import fill_window

def reconstruct_fullmatch(players, ball, blocks, fps, pitch):
    n = len(ball)
    result = pd.DataFrame({"frame": np.arange(n), "half": np.zeros(n, int),
                           "raw_x_m": ball[:, 0], "raw_y_m": ball[:, 1],
                           "estimated_x_m": np.nan, "estimated_y_m": np.nan,
                           "input_x_m": ball[:, 0], "input_y_m": ball[:, 1],
                           "source_status": "outside_scope", "gap_id": -1,
                           "valid_selected_players": 0, "elapsed_seconds": np.nan})
    gaps, audits, offset = [], [], 0
    for half, first, last in blocks:
        part, report, audit = fill_window(players[first:last+1], ball[first:last+1], first,
                                         fps, pitch, distance_weighted=True, half=half)
        mask = part.gap_id >= 0
        part.loc[mask, "gap_id"] += offset
        if len(report):
            report["gap_id"] += offset
            report["half"] = half
            gaps.append(report)
        offset += len(report)
        result.loc[first:last, part.columns] = part.to_numpy()
        result.loc[first:last, "half"] = half
        if len(audit):
            audits.append(audit)
    result["output_x_m"] = result.input_x_m
    result["output_y_m"] = result.input_y_m
    result["is_raw_observed"] = np.isfinite(ball).all(axis=1)
    result["is_interpolated"] = result.source_status.eq("player_interpolated")
    xy = result[["output_x_m", "output_y_m"]].to_numpy(float)
    result["estimated_outside_pitch"] = result.is_interpolated & ((xy < 0) | (xy > pitch)).any(axis=1)
    np.testing.assert_array_equal(xy[result.is_raw_observed], ball[result.is_raw_observed])
    return result, pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(), pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
