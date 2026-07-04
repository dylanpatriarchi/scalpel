"""Ollama LLM-judge for the effect metric.

Scoring "how strongly does this text express concept X" is text-in / text-out,
which is exactly what a local Ollama model is good at — no residual-stream access
needed, no API keys, no cost. The judge asks for an integer 0-10 and normalises
it to ``[0, 1]``.

The HTTP transport is injectable so tests never touch the network, and every
failure path degrades to ``None`` rather than raising.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from urllib.request import Request, urlopen

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"

PROMPT_TEMPLATE = (
    "You are grading text for the presence of a concept.\n"
    "On a scale of 0 to 10, how strongly does the TEXT express or relate to the "
    'concept "{concept}"?\n'
    "0 means not at all, 10 means the text is entirely about it.\n"
    "Reply with ONLY a single integer from 0 to 10, nothing else.\n\n"
    "TEXT: {text}\n\nSCORE:"
)

# (url, json_body, timeout) -> response body
HttpPost = Callable[[str, bytes, float], str]


def _default_http_post(url: str, body: bytes, timeout: float) -> str:
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 (local http)
        return response.read().decode("utf-8")


def parse_score(response: str, *, scale: int = 10) -> float | None:
    """Extract the first number from a judge reply and normalise to ``[0, 1]``."""
    match = re.search(r"\d+(?:\.\d+)?", response)
    if match is None:
        return None
    value = max(0.0, min(float(scale), float(match.group())))
    return value / scale


class OllamaJudge:
    """A thin, robust client over Ollama's ``/api/generate`` endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        *,
        timeout: float = 30.0,
        http_post: HttpPost = _default_http_post,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.http_post = http_post
        self.prompt_template = prompt_template

    def score(self, text: str, concept: str) -> float | None:
        """Return a ``[0, 1]`` concept score, or ``None`` on any failure."""
        prompt = self.prompt_template.format(concept=concept, text=text)
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        try:
            raw = self.http_post(f"{self.host}/api/generate", body, self.timeout)
            data = json.loads(raw)
            return parse_score(str(data.get("response", "")))
        except Exception:
            return None

    def available(self) -> bool:
        """Best-effort check that the judge responds at all."""
        return self.score("ping", "anything") is not None
