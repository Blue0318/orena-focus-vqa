"""
Data loader — wraps orena-focus FocusDataset with config-driven setup.
"""
from focus import FocusDataset, DatasetSplit, Track


TRACK_MAP = {
    "FRAME": Track.FRAME,
    "SEGMENT": Track.SEGMENT,
    "PROCEDURE": Track.PROCEDURE,
}

SPLIT_MAP = {
    "train": DatasetSplit.TRAIN,
    "test": DatasetSplit.TEST,
}


def load_dataset(cfg: dict) -> FocusDataset:
    """Load FocusDataset from a config dict (parsed YAML)."""
    track = TRACK_MAP[cfg["track"]]
    split = SPLIT_MAP[cfg["split"]]
    return FocusDataset(cfg["dataset"], split, track)


def get_requests_and_references(ds: FocusDataset):
    """Return parallel lists of requests and references."""
    requests, references = [], []
    for req, ref in ds:
        requests.append(req)
        references.append(ref)
    return requests, references
