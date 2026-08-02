"""Spot segments whose audio cannot plausibly hold their text.

Every engine here occasionally swallows the tail of a sentence — a few words
simply do not get spoken. The failure is objective and cheap to detect: the
audio comes back too short for the text it was given.

Turning that into a rule needs an expected speech rate, and hardcoding one per
language, engine and voice would be guesswork that ages badly. Instead the
chapter calibrates itself: almost every segment is fine, so the median rate
across the chapter *is* the expected rate, and a segment far above it spoke far
fewer characters than it was handed.

This deliberately says nothing about stress or intonation. Those are matters of
taste, and the reader's alternative-takes panel is where they belong.
"""

from __future__ import annotations

import statistics

# Below this the rate is dominated by leading and trailing silence, so a short
# line looks arbitrarily fast or slow. "Igen." carries no signal.
MIN_TEXT_CHARS = 25

# A segment is suspect when it speaks this many times faster than the chapter
# does on average. 1.6 leaves room for genuinely brisk lines while catching a
# sentence that lost a clause.
SUSPECT_RATE_RATIO = 1.6

# The median needs something to be a median of. Under this many usable
# segments the chapter cannot calibrate itself and nothing is flagged.
MIN_REFERENCE_SEGMENTS = 6

# Audio this short for real text means nothing usable came back at all.
SILENT_MAX_SECONDS = 0.15


def speech_rate(text: str, duration_sec: float | None) -> float | None:
    """Characters of text per second of audio, or None when meaningless."""
    if not duration_sec or duration_sec <= 0:
        return None
    length = len(str(text or "").strip())
    if length < MIN_TEXT_CHARS:
        return None
    return length / duration_sec


def median_speech_rate(segments: list[dict]) -> float | None:
    """The chapter's own pace, or None when too little of it is usable."""
    rates = [
        rate
        for segment in segments
        for rate in (speech_rate(segment.get("text"), segment.get("duration_sec")),)
        if rate is not None
    ]
    if len(rates) < MIN_REFERENCE_SEGMENTS:
        return None
    return statistics.median(rates)


def find_suspects(
    segments: list[dict], *, rate_ratio: float = SUSPECT_RATE_RATIO
) -> list[dict]:
    """Segments whose audio is too short for their text.

    ``segments`` are dicts with ``text`` and ``duration_sec``. The display text
    is used rather than the enriched text, because bracketed expression tags
    are characters no engine speaks and would inflate the expected duration.
    """
    suspects: list[dict] = []
    median = median_speech_rate(segments)

    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "").strip()
        duration = segment.get("duration_sec")
        if not segment.get("audio_path"):
            continue
        if len(text) >= MIN_TEXT_CHARS and (not duration or duration <= SILENT_MAX_SECONDS):
            suspects.append({
                "index": index,
                "reason": "no audio came back for this sentence",
                "rate": None,
                "expected_rate": median,
            })
            continue
        if median is None:
            continue
        rate = speech_rate(text, duration)
        if rate is not None and rate > median * rate_ratio:
            suspects.append({
                "index": index,
                "reason": (
                    f"spoken {rate / median:.1f}x faster than the rest of the "
                    f"chapter, so words are probably missing"
                ),
                "rate": rate,
                "expected_rate": median,
            })
    return suspects


def is_acceptable(
    text: str,
    duration_sec: float | None,
    median: float | None,
    *,
    rate_ratio: float = SUSPECT_RATE_RATIO,
) -> bool:
    """Whether a freshly rendered take is back inside the chapter's pace."""
    if not duration_sec or duration_sec <= SILENT_MAX_SECONDS:
        return False
    if median is None:
        return True
    rate = speech_rate(text, duration_sec)
    return rate is None or rate <= median * rate_ratio
