"""Lyric planning utilities for the section-based pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an experienced songwriter. Compose concise lyrics for a section of a "
    "song. Style cues: {style}. Musical key: {key}. Tempo: {bpm} BPM. The section "
    "tag is '{tag}'. Avoid the following topics: {negative}. Generate at most {lines} "
    "lines with natural phrasing."
)


def _post_lyrics(prompt: str) -> Optional[List[str]]:
    endpoint = os.getenv("LYRIC_LLM_ENDPOINT") or os.getenv("NPU_LLM_ENDPOINT")
    if not endpoint:
        return None

    try:
        response = requests.post(endpoint, json={"prompt": prompt}, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure path
        LOGGER.error("Lyric LLM request failed: %s", exc)
        return None

    try:
        data = response.json()
    except json.JSONDecodeError:  # pragma: no cover - unexpected response
        LOGGER.error("Lyric LLM returned non-JSON payload")
        return None

    if isinstance(data, dict):
        content = data.get("text") or data.get("lyrics") or data.get("content")
        if isinstance(content, list):
            return [str(line).strip() for line in content if str(line).strip()]
        if isinstance(content, str):
            return [line.strip() for line in content.splitlines() if line.strip()]
    elif isinstance(data, list):  # pragma: no cover - alternative response
        return [str(item).strip() for item in data if str(item).strip()]

    LOGGER.error("Unexpected lyric service response: %s", data)
    return None


def _fallback_lyrics(
    base_style: str,
    key: str,
    bpm: int,
    sections: List[Dict[str, object]],
    seed: Optional[int],
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    style_tokens = [token.strip() for token in base_style.split(",") if token.strip()]
    if not style_tokens:
        style_tokens = ["mellow", "dreamy"]

    templates = [
        "{adjective} rhythms guide my {section}",
        "In {key} we move with hearts aligned",
        "Echoes at {bpm} BPM, softly entwined",
        "{section} lights the night, we keep in time",
    ]

    results: Dict[str, List[str]] = {}
    for item in sections:
        name = str(item.get("name", "section"))
        lines = []
        for idx in range(2):
            template = templates[
                (idx + rng.randint(0, len(templates) - 1)) % len(templates)
            ]
            adjective = rng.choice(style_tokens)
            lines.append(
                template.format(adjective=adjective, section=name, key=key, bpm=bpm)
            )
        results[name] = lines
    return results


def plan_lyrics(
    base_style: str,
    key: str,
    bpm: int,
    sections: List[Dict[str, object]],
    negative: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Generate lyrics per section.

    The function first attempts to reach an external LLM endpoint defined by
    ``LYRIC_LLM_ENDPOINT`` or ``NPU_LLM_ENDPOINT``. When the service is unavailable a
    deterministic, seedable fallback keeps the pipeline operational.
    """

    negative_text = negative or "(none)"
    results: Dict[str, List[str]] = {}

    for section in sections:
        tag = str(section.get("name", "section"))
        prompt = PROMPT_TEMPLATE.format(
            style=base_style,
            key=key,
            bpm=bpm,
            tag=tag,
            negative=negative_text,
            lines=4,
        )
        LOGGER.info("Requesting lyrics for section '%s'", tag)
        lines = _post_lyrics(prompt)
        if not lines:
            LOGGER.warning("Falling back to rule-based lyrics for section '%s'", tag)
            # We postpone fallback generation until later to keep deterministic output.
            results = _fallback_lyrics(base_style, key, bpm, sections, seed)
            break
        results[tag] = lines[:4]

    if not results:
        results = _fallback_lyrics(base_style, key, bpm, sections, seed)

    return results
