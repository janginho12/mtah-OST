import xml.etree.ElementTree as ET
import numpy as np

HOME, AWAY = 0, 1

def team_of_each_track(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The ATD metadata lists both teams and assigns every track slot to one of them. The first team
    in the file is the home side -- the FIFA/EPTS convention, and the one kloppy follows when it
    reports metadata.teams. For this match that is Barcelona, which DATA.md states independently.

    Note what these slots are NOT: there are 46 of them for 22 players on the pitch, they are
    named "Track 1".."Track 46", and their ShirtNumber is just the slot index. A slot is a
    tracklet, not a person, and it does not follow one player through the match.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)
