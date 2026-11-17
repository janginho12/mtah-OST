DEFAULT_ACTIVITY = {}


def _keeper_mask(*args, **kwargs):
    raise ValueError("Supply goalkeeper_mask from ATD metadata; upstream role inference is absent")
