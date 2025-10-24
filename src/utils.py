import json
from typing import Any


def to_json_serializable(obj: Any) -> Any:
    """Try to convert common non-serializable objects into JSON-serializable forms.

    For now this is a simple wrapper around json.dumps for known simple cases.
    """
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        # Fallback: convert to string representation
        return str(obj)
