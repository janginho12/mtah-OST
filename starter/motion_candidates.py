from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from starter.baseline import team_of_each_track

@dataclass(frozen=True)
class Settings:
    bin_frames: int = 5
    max_speed_mps: float = 12.0
    min_direction_speed_mps: float = 0.5
    acceleration_mps2: float = 3.0
    reversal_degrees: float = 90.0
    min_players: int = 6
    max_players: int = 10
    context_seconds: float = 3.0
    min_context_fraction: float = 0.8
    change_mps: float = 0.5
    separation_seconds: float = 3.0


def metadata(path):
    root = ET.parse(path).getroot()
    players = list(root.iter("Player"))
    refs = list(root.iter("PlayerChannelRef"))
    expected = ["player{}_{}".format(p.attrib["id"], channel)
                for p in players for channel in ("x", "y", "speed")]
    if [r.attrib["playerChannelId"] for r in refs] != expected:
        raise ValueError("Raw channel order is incompatible with x,y,speed slots")
    keepers = np.array([
        any(q.findtext("Name") == "position_type" and q.findtext("Value") == "Goalkeeper"
            for q in p.findall("./ProviderPlayerParameters/ProviderParameter"))
        for p in players
    ])
    directions = []
    for team in root.iter("Team"):
        value = next((q.findtext("Value") for q in
                      team.findall("./ProviderTeamsParameters/ProviderParameter")
                      if q.findtext("Name") == "attack_direction_first_half"), None)
        if value not in ("left_to_right", "right_to_left"):
            raise ValueError("Missing/unsupported first-half attack direction")
        directions.append(1 if value == "left_to_right" else -1)
    if sorted(directions) != [-1, 1]:
        raise ValueError("Expected opposite team attack directions")
    return team_of_each_track(path), keepers, np.array(directions)


def bin_positions(points, start, fps, pitch, settings):
    """Return meter coordinates, center frames, and explicit rejection counts."""
    if fps % settings.bin_frames:
        raise ValueError("Aggregation must divide the source FPS")
    p = points.copy() * np.asarray(pitch)
    observed = np.isfinite(p).all(axis=2)
    inside = observed & (p >= 0).all(axis=2) & (p <= pitch).all(axis=2)
    valid = inside.copy()
    edge_speed = np.linalg.norm(np.diff(p, axis=0), axis=2) * fps
    jumps = inside[:-1] & inside[1:] & (edge_speed > settings.max_speed_mps)
    valid[:-1] &= ~jumps
    valid[1:] &= ~jumps
    p[~valid] = np.nan
    size = settings.bin_frames
    offset = (-start) % size
    count = max(0, (len(p) - offset) // size)
    if not count:
        raise ValueError("Interval has no complete aggregation bins")
    chunk = p[offset:offset + count * size].reshape(count, size, p.shape[1], 2)
    complete = np.isfinite(chunk).all(axis=(1, 3))
    averaged = np.where(np.isfinite(chunk), chunk, 0).mean(axis=1)
    averaged[~complete] = np.nan
    centers = start + offset + np.arange(count) * size + size // 2
    return averaged, centers, {
        "raw_observed_slot_frames": int(observed.sum()),
        "outside_pitch_slot_frames": int((observed & ~inside).sum()),
        "implausible_raw_edges": int(jumps.sum()),
        "valid_aggregate_slot_bins": int(complete.sum()),
        "discarded_edge_frames": int(len(p) - count * size),
    }


def kinematics(points, dt, settings):
    velocity = np.full_like(points, np.nan)
    velocity[1:] = np.diff(points, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=2)
    velocity[speed > settings.max_speed_mps] = np.nan
    speed = np.linalg.norm(velocity, axis=2)
    acceleration = np.full_like(points, np.nan)
    acceleration[1:] = np.diff(velocity, axis=0) / dt
    angle = np.full(speed.shape, np.nan)
    good = ((speed[1:] >= settings.min_direction_speed_mps)
            & (speed[:-1] >= settings.min_direction_speed_mps)
            & np.isfinite(speed[1:]) & np.isfinite(speed[:-1]))
    dot = np.sum(velocity[1:] * velocity[:-1], axis=2)
    ratio = np.full(dot.shape, np.nan)
    np.divide(dot, speed[1:] * speed[:-1], out=ratio, where=good)
    angle[1:] = np.degrees(np.arccos(np.clip(ratio, -1, 1)))
    return velocity, acceleration, angle
