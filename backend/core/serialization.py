"""Safe conversion of NumPy values at the API boundary."""

from collections.abc import Mapping

import numpy as np


def to_jsonable(value):
    """Convert NumPy containers to JSON-compatible, finite Python values."""
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("simulation produced non-finite numerical output")
        return value.tolist()
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            raise ValueError("simulation produced non-finite numerical output")
        return scalar
    if isinstance(value, Mapping):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("simulation produced non-finite numerical output")
    return value
