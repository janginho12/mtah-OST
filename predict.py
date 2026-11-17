"""Regenerate the OST Phase 1 possession file on CPU."""
import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from games import GAME, GAMES, half_video_blocks, require
from evaluation import possession_metrics
from ost import buildup_possession, goal_possession, keeper_selection
from ost.prepare import observed_players, prepare_stoppages, table_boundary
from starter import merge_short_dead_gaps, refine_dead_ball_buildup
from starter.control_integration import pipeline
from starter.detect_dead_ball import read_tracking
from starter.load_smart_tagging import load as load_tags
from starter.motion_candidates import metadata
from starter.reconstruct_ball_fullmatch import reconstruct_fullmatch


def status(message):
    print(message, flush=True)


def predict(data_dir):
    info = GAMES[GAME]
    fps, pitch = info["fps"], np.asarray(info["pitch"])
    halves = half_video_blocks(GAME)
    meta, raw, _ = require(["atd_meta", "atd_raw", "smart_tagging"], GAME, str(data_dir))
    teams, keepers, directions = metadata(meta)
    status("1/6  Reading the tactical observations")
    normalized, normalized_ball = read_tracking(raw, info["n_video_frames"], len(teams))
    formation = normalized.astype(np.float32) * pitch.astype(np.float32)
    players, ball = normalized * pitch, normalized_ball * pitch
    status("2/6  Reconstructing eligible ball gaps within each half")
    ball_table, _, _ = reconstruct_fullmatch(players, ball, halves, fps, pitch)
    ball_table = table_boundary(ball_table[["raw_x_m", "raw_y_m", "output_x_m", "output_y_m", "is_interpolated"]])
    status("3/6  Detecting stoppages and refining restart boundaries")
    dead, setpiece = prepare_stoppages(normalized, players, ball, ball_table, teams, keepers, directions, halves, fps, pitch)
    del normalized, normalized_ball
    blocks = dead.half.to_numpy(np.int8)
    status("4/6  Checking track continuity and assigning goalkeeper roles")
    observed = observed_players(players, formation, teams, keepers, halves, fps, pitch)
    del players
    raw_ball = ball_table[["raw_x_m", "raw_y_m"]].to_numpy(np.float32)
    roles = keeper_selection.preprocess(observed, teams, blocks, fps, pitch, keepers,
                                        ball=raw_ball, formation_players=formation)
    del observed, formation
    corrected, teams, keepers = roles["players"], roles["teams"], roles["keeper_mask"]
    features, _ = pipeline.extract_features(raw_ball, ball_table[["output_x_m", "output_y_m"]].to_numpy(np.float32),
        ball_table.is_interpolated.to_numpy(bool), corrected, teams, blocks, fps, pitch)
    kernels = SimpleNamespace(buildup=refine_dead_ball_buildup, merge=merge_short_dead_gaps, pipeline=pipeline)
    status("5/6  Decoding control and checking slow passing sequences")
    base = buildup_possession.infer(features, corrected, teams, blocks, fps, pitch,
        dead.is_d_candidate.to_numpy(bool), keepers, roles["home_defends_left"],
        (dead.line_dropout | dead.observed_line_exit).to_numpy(bool), setpiece, {}, kernels)
    tags = [dict(start=t["start_frame"], end=t["end_frame"], code=t["code"],
                 team=t["tags"].get("Team", ""), type=t["tags"].get("Type", ""))
            for t in load_tags(data_dir=str(data_dir), fps=fps)]
    status("6/6  Resolving corroborated goal stoppages and centre restarts")
    final = goal_possession.infer(features, corrected, teams, blocks, keepers, fps, pitch,
        base["revised_D"], base["held_team"], base["labels"], base["before_merge"],
        base["evidence"], {}, kernels, tags=tags)["labels"]
    if not (np.isin(final[blocks > 0], [0, 1, 2]).all() and (final[blocks == 0] == 3).all()):
        raise ValueError("Output violates the half or label contract")
    return final


def write_submission(output, labels, verify=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent,
                                         prefix=".ost-", suffix=".csv", delete=False) as handle:
            temp = Path(handle.name)
            handle.write("frame,label\n")
            handle.writelines(f"{frame},{'HADX'[label]}\n" for frame, label in enumerate(labels))
        reason = possession_metrics.validate_submission(possession_metrics.read_label_csv(temp), GAMES[GAME]["n_video_frames"])
        if reason:
            raise ValueError(reason)
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        if verify:
            expected = (Path(__file__).with_name("submission.sha256")).read_text(encoding="ascii").split()[0]
            if digest != expected:
                raise ValueError(f"Output checksum mismatch: {digest}")
        try:
            # Windows rename is atomic and fails if the destination exists.
            # On POSIX a hard link provides the same no-clobber guarantee.
            if os.name == "nt":
                os.rename(temp, output)
            else:
                os.link(temp, output)
        except FileExistsError:
            if hashlib.sha256(output.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Output already exists with different contents: {output}")
        return digest
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MTAH_DATA_DIR") or "data"),
                        help="Competition bundle containing the tactical subdirectory")
    parser.add_argument("--output", type=Path, default=Path("outputs/OST.csv"))
    parser.add_argument("--verify", action="store_true", help="Require byte identity with the released submission")
    args = parser.parse_args()
    digest = write_submission(args.output, predict(args.data_dir), args.verify)
    status(f"Validated {args.output}  SHA-256 {digest}")


if __name__ == "__main__":
    main()
