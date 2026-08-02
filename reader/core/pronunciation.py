"""
Pronunciation lexicon for TTS text.

Rewrites names the way they should be *spoken* without touching the text the
reader displays. A Hungarian book full of Westerosi names is the motivating
case: "Westeros" read with Hungarian letter values sounds wrong, while
"Veszterosz" hits the intended English pronunciation.

The lexicon is a plain settings string, one rule per line::

    # foreign name  =  how it should sound
    Westeros = Veszterosz
    Aegon    = Egon

Hungarian is agglutinative, so a rule matches the stem and keeps whatever
suffix follows it, with or without the hyphen that foreign names often take:
``Westerosban`` -> ``Veszteroszban``, ``Aegon-nak`` -> ``Egon-nak``.

Longer sources win over shorter ones, so an explicit ``Alyssa`` rule is not
shadowed by an ``Alys`` rule.
"""

from __future__ import annotations

import hashlib
import logging
import re

log = logging.getLogger(__name__)

SETTING_KEY = "pronunciation_dict"

# Spans that must never be rewritten: OmniVoice bracket tags ([laugh]) and
# Higgs control tokens (<|emotion:joy|>).
_PROTECTED_RE = re.compile(r"\[[^\[\]]*\]|<\|[^|]*\|>")
_SEPARATOR_RE = re.compile(r"\s*(?:=>|->|=|→)\s*")
_HU_LETTERS = "A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű"
# A rule matches the stem plus any Hungarian suffix glued to it, optionally
# after the hyphen used with foreign names ("Aegon-nak").
_SUFFIX_PATTERN = r"(-?[a-záéíóöőúüű]*)"

_COMPILED_CACHE: dict[str, tuple[re.Pattern, dict[str, str]] | None] = {}


def parse_lexicon(raw: str | None) -> list[tuple[str, str]]:
    """Parse the settings string into (source, replacement) pairs.

    Blank lines and ``#`` comments are ignored, as are rules with an empty
    side — a half-typed line in the settings box must not break synthesis.
    """
    if not raw or not raw.strip():
        return []

    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _SEPARATOR_RE.split(line, maxsplit=1)
        if len(parts) != 2:
            continue
        source, replacement = parts[0].strip(), parts[1].strip()
        if not source or not replacement:
            continue
        if source.lower() in seen:
            continue
        seen.add(source.lower())
        entries.append((source, replacement))
    return entries


def combine_lexicons(*sources: str | None) -> str:
    """Merge lexicons, earlier sources winning on a repeated source word.

    Rules are deduplicated by first occurrence, so passing the book's own
    lexicon before the global one lets a book override a shared rule while
    still inheriting everything it does not mention.
    """
    return "\n".join(source for source in sources if source and source.strip())


def _compile(raw: str | None):
    entries = parse_lexicon(raw)
    if not entries:
        return None

    # Longest source first: regex alternation is first-match, so this is what
    # keeps "Alyssa" from being matched as "Alys" + suffix "sa".
    entries.sort(key=lambda pair: len(pair[0]), reverse=True)
    alternation = "|".join(re.escape(source) for source, _ in entries)
    try:
        pattern = re.compile(
            rf"(?<![{_HU_LETTERS}])({alternation}){_SUFFIX_PATTERN}",
            re.IGNORECASE,
        )
    except re.error as exc:  # pragma: no cover - re.escape makes this unlikely
        log.warning("Could not compile pronunciation lexicon: %s", exc)
        return None
    return pattern, {source.lower(): replacement for source, replacement in entries}


def _compiled(raw: str | None):
    key = raw or ""
    if key not in _COMPILED_CACHE:
        if len(_COMPILED_CACHE) > 8:
            _COMPILED_CACHE.clear()
        _COMPILED_CACHE[key] = _compile(raw)
    return _COMPILED_CACHE[key]


def _match_case(matched: str, replacement: str) -> str:
    """Carry the written form's capitalization over to the replacement."""
    if matched.isupper() and len(matched) > 1:
        return replacement.upper()
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    if matched.islower():
        return replacement.lower()
    return replacement


def _rewrite(text: str, compiled) -> str:
    pattern, mapping = compiled

    def _replace(match: re.Match) -> str:
        stem, suffix = match.group(1), match.group(2) or ""
        replacement = mapping.get(stem.lower())
        if replacement is None:  # pragma: no cover - alternation guarantees a hit
            return match.group()
        return _match_case(stem, replacement) + suffix

    return pattern.sub(_replace, text)


def lexicon_text(raw: str | None = None) -> str:
    """The configured lexicon, read from settings unless passed explicitly."""
    if raw is not None:
        return raw
    try:
        from core.settings import get

        return str(get(SETTING_KEY, "") or "")
    except Exception:
        return ""


def lexicon_version(raw: str | None = None) -> str:
    """Short hash of the active lexicon, for audio cache keys.

    Editing a rule has to invalidate previously synthesized audio, otherwise
    the old pronunciation keeps playing from cache. Reordering the rules does
    not: match precedence comes from rule length, not from file order, so
    sorting the list must not re-synthesize a whole book.
    """
    entries = parse_lexicon(lexicon_text(raw))
    if not entries:
        return "0"
    payload = "|".join(
        f"{source}={replacement}" for source, replacement in sorted(entries)
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def entries(raw: str | None = None) -> list[dict]:
    """The active rules as objects, for the editor and the reader."""
    return [
        {"source": source, "spoken": spoken}
        for source, spoken in parse_lexicon(lexicon_text(raw))
    ]


def pattern_source(raw: str | None = None) -> str:
    """Regex source of the active lexicon, empty when there are no rules.

    The reader marks the same words the engine rewrites, so it reuses this
    pattern instead of reimplementing stem-plus-suffix matching in JavaScript
    and slowly drifting away from it.
    """
    compiled = _compiled(lexicon_text(raw))
    return compiled[0].pattern if compiled else ""


def apply_pronunciation(text: str, raw: str | None = None) -> str:
    """Apply the lexicon to text on its way to the TTS model.

    Bracket tags and control tokens are left untouched so enrichment markup
    survives the rewrite.
    """
    if not text or not text.strip():
        return text

    compiled = _compiled(lexicon_text(raw))
    if compiled is None:
        return text

    out: list[str] = []
    last = 0
    for match in _PROTECTED_RE.finditer(text):
        if match.start() > last:
            out.append(_rewrite(text[last:match.start()], compiled))
        out.append(match.group())
        last = match.end()
    out.append(_rewrite(text[last:], compiled))
    return "".join(out)
