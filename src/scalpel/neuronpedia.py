"""Optional Neuronpedia label lookup.

When a released SAE has a Neuronpedia source id (e.g. ``gpt2-small/7-res-jb``),
we can fetch the community/auto-interp label for a feature to annotate discovery
results. This is best-effort and network-dependent: failures degrade to ``None``
rather than raising, and the HTTP call is injectable so tests never touch the
network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.request import Request, urlopen

BASE_URL = "https://www.neuronpedia.org/api/feature"

HttpGet = Callable[[str, float], str]


def _default_http_get(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "scalpel/0.1"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
        return response.read().decode("utf-8")


def _parse_label(body: str) -> str | None:
    data = json.loads(body)
    explanations = data.get("explanations") or []
    if not explanations:
        return None
    description = explanations[0].get("description")
    return str(description) if description else None


def fetch_label(
    neuronpedia_id: str,
    feature_index: int,
    *,
    timeout: float = 10.0,
    http_get: HttpGet = _default_http_get,
) -> str | None:
    """Return the Neuronpedia label for a feature, or ``None`` on any failure."""
    url = f"{BASE_URL}/{neuronpedia_id}/{feature_index}"
    try:
        return _parse_label(http_get(url, timeout))
    except Exception:
        # Network error, unexpected payload, missing feature — all non-fatal.
        return None
