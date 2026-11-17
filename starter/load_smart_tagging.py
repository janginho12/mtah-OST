import xml.etree.ElementTree as ET
from games import GAME, GAMES, require

def pattern_path(game=GAME, data_dir=None):
    """Path to the Smart Tagging XML, with a readable error if it is not there."""
    return require(["smart_tagging"], game, data_dir)[0]


def load(game=GAME, data_dir=None, fps=None):
    """[{code, start_frame, end_frame, start_s, end_s, tags}] in file order.

    Frames are half-open on the right the way the rest of this repository treats intervals, so
    end_frame is the first frame *after* the phase. Rounding is to the nearest frame: the tags are
    accurate to about a second, so a half-frame either way is not the error that matters.
    """
    fps = fps or GAMES[game]["fps"]
    out = []
    for instance in ET.parse(pattern_path(game, data_dir)).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        out.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            # A group can legitimately appear more than once; last one wins, which is fine for
            # the single-valued groups (Team, Half, Type) this file uses.
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    return out
