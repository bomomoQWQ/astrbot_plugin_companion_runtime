"""Cut one proactive line into the several messages a person would send.

Why this module exists
----------------------
The host already segments its own replies. ``result_decorate`` cuts each ``Plain``
component of an LLM result into bubbles and ``respond`` sends them one at a time
with a pause in between, all driven by ``platform_settings.segmented_reply``. A
proactive message never passes through that pipeline -- it is delivered with
``Context.send_message`` -- so a render that came back as two lines arrived as
**one** bubble with an embedded newline, while her replies to the same person
arrived as two.

The writing side already says what it wants: the Runtime's render style is
"我说话就一两句，通常 30 字以内。没说完就再发一条，别堆成一大段". So the intent
existed and was lost at delivery, which is the layer this module fixes.

What it decides, and what it deliberately does not
--------------------------------------------------
* **The split rule is the host's**, read from the host config at send time, so
  one authority governs "how this deployment segments a message": turning the
  host's 分段回复 off turns it off here too (``outbox_segmented_reply=inherit``).
* **The text is never edited.** The host's own path also applies
  ``content_cleanup_rule`` (``[。]+$``) to each bubble; this module does not,
  because the Runtime authored the line and the adapter's contract is to deliver
  it rather than to rewrite it. Beats are only ``.strip()``-ed, which is what
  makes the cut itself possible.
* **No text is dropped.** When a line splits into more beats than ``max_parts``,
  the tail is merged back into the last bubble instead of being truncated: a
  safety cap must not be able to eat half a sentence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

#: ``inherit`` follows the host's segmented-reply switch; ``on`` segments even
#: when the host has it off; ``off`` delivers one bubble exactly as before.
SEGMENT_MODE_INHERIT = "inherit"
SEGMENT_MODE_ON = "on"
SEGMENT_MODE_OFF = "off"
SEGMENT_MODES = (SEGMENT_MODE_INHERIT, SEGMENT_MODE_ON, SEGMENT_MODE_OFF)

#: Ceiling on bubbles per proactive message. The lease heartbeat keeps the lease
#: alive, but the *send* itself runs under ``send_timeout_ms``, and the pauses are
#: real: at the host's 1.6 s ceiling ten beats would already spend 14 s of a 20 s
#: budget on waiting. Four is what her style line asks for ("一两句"), and anything
#: longer is merged rather than cut.
DEFAULT_MAX_PARTS = 4

#: The host's own defaults (``astrbot/core/config/default.py``), used only when the
#: host config is missing or unreadable. They are repeated rather than imported
#: because this package must not import AstrBot.
DEFAULT_REGEX = r"[^\n]+"
DEFAULT_SPLIT_WORDS = ("。", "？", "！", "~", "…")
DEFAULT_WORDS_THRESHOLD = 60
DEFAULT_INTERVAL = (0.8, 1.6)
DEFAULT_INTERVAL_METHOD = "random"
DEFAULT_LOG_BASE = 2.6

#: What the host falls back to when the configured regex does not compile.
FALLBACK_REGEX = r".*?[。？！~…]+|.+$"

#: The host's config path for everything above.
HOST_SEGMENT_SETTINGS_PATH = ("platform_settings", "segmented_reply")


def host_segment_settings(config: Any) -> Mapping[str, Any]:
    """Return the host's ``platform_settings.segmented_reply`` mapping.

    Args:
        config: The host config object (``Context.get_config()``), a mapping, or
            anything else. Never raises: a config that cannot be read means "the
            host has no segmentation configured", which is answered with the
            defaults rather than with an exception on the delivery path.

    Returns:
        The settings mapping, or an empty mapping when it is absent.
    """
    cursor: Any = config
    for key in HOST_SEGMENT_SETTINGS_PATH:
        if isinstance(cursor, Mapping):
            cursor = cursor.get(key)
        else:
            cursor = getattr(cursor, key, None)
        if cursor is None:
            return {}
    return cursor if isinstance(cursor, Mapping) else {}


def word_count(text: str) -> int:
    """Count the words of ``text`` the way the host's interval maths does.

    CJK text has no spaces, so counting whitespace-separated tokens would make
    every Chinese beat look like one word and flatten the pause curve to a
    constant; the host counts alphanumerics there instead.
    """
    if all(ord(char) < 128 for char in text):
        return len(text.split())
    return len([char for char in text if char.isalnum()])


@dataclass(frozen=True)
class SegmentPolicy:
    """How one proactive line becomes several messages.

    Attributes:
        enabled: Whether to split at all.
        mode: ``regex`` or ``words`` (the host's ``split_mode``).
        regex: The split pattern used in ``regex`` mode.
        split_words: Sentence-final markers used in ``words`` mode.
        words_count_threshold: A line longer than this is sent as one bubble --
            the host's own rule, kept because splitting a paragraph the model
            chose to write as a paragraph is worse than one long message.
        interval: ``(min, max)`` pause in seconds between two bubbles.
        interval_method: ``random`` or ``log`` (longer beats pause a little more).
        log_base: Base of the ``log`` method's curve.
        max_parts: Ceiling on bubbles; the tail is merged, never truncated.
    """

    enabled: bool = False
    mode: str = "regex"
    regex: str = DEFAULT_REGEX
    split_words: tuple[str, ...] = DEFAULT_SPLIT_WORDS
    words_count_threshold: int = DEFAULT_WORDS_THRESHOLD
    interval: tuple[float, float] = DEFAULT_INTERVAL
    interval_method: str = DEFAULT_INTERVAL_METHOD
    log_base: float = DEFAULT_LOG_BASE
    max_parts: int = DEFAULT_MAX_PARTS

    @classmethod
    def from_host_settings(
        cls,
        settings: Mapping[str, Any] | None,
        *,
        mode: str = SEGMENT_MODE_INHERIT,
        max_parts: int = DEFAULT_MAX_PARTS,
    ) -> SegmentPolicy:
        """Build a policy from the host's ``segmented_reply`` mapping.

        Args:
            settings: The host's ``platform_settings.segmented_reply`` mapping, or
                ``None``/empty when it could not be read.
            mode: One of :data:`SEGMENT_MODES`.
            max_parts: Ceiling on bubbles; ``0`` or less means no ceiling.

        Returns:
            The policy. Anything unreadable falls back to the host's own default
            for that key, so a partial config still produces a sane split.
        """
        raw: Mapping[str, Any] = settings if isinstance(settings, Mapping) else {}
        resolved_mode = mode if mode in SEGMENT_MODES else SEGMENT_MODE_INHERIT
        enabled = (
            True
            if resolved_mode == SEGMENT_MODE_ON
            else bool(raw.get("enable", False))
            if resolved_mode == SEGMENT_MODE_INHERIT
            else False
        )

        split_mode = str(raw.get("split_mode") or "regex").strip().lower()
        if split_mode not in ("regex", "words"):
            split_mode = "regex"

        regex = str(raw.get("regex") or DEFAULT_REGEX)
        words = raw.get("split_words")
        if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
            split_words = DEFAULT_SPLIT_WORDS
        else:
            split_words = tuple(str(word) for word in words if str(word))

        threshold = _as_int(raw.get("words_count_threshold"), DEFAULT_WORDS_THRESHOLD)

        low, high = _parse_interval(raw.get("interval"))
        interval_method = str(raw.get("interval_method") or DEFAULT_INTERVAL_METHOD).strip()
        if interval_method not in ("random", "log"):
            interval_method = DEFAULT_INTERVAL_METHOD

        return cls(
            enabled=enabled,
            mode=split_mode,
            regex=regex,
            split_words=split_words,
            words_count_threshold=max(threshold, 0),
            interval=(low, high),
            interval_method=interval_method,
            log_base=_as_float(raw.get("log_base"), DEFAULT_LOG_BASE),
            max_parts=max(_as_int(max_parts, DEFAULT_MAX_PARTS), 0),
        )

    def beats(self, text: str) -> list[str]:
        """Split ``text`` into the bubbles it should be delivered as.

        Args:
            text: The authorized message text (what the Runtime rendered).

        Returns:
            One or more non-empty bubbles. Always exactly ``[text.strip()]`` when
            segmentation is off, when the line is short enough to be one bubble,
            or when the split produces nothing usable.
        """
        body = text.strip()
        if not body:
            return []
        if not self.enabled:
            return [body]
        if self.words_count_threshold > 0 and len(body) > self.words_count_threshold:
            return [body]

        parts = [part.strip() for part in self._split(body)]
        parts = [part for part in parts if part]
        if len(parts) <= 1:
            return [body]
        if 0 < self.max_parts < len(parts):
            head = parts[: self.max_parts - 1]
            tail = "\n".join(parts[self.max_parts - 1 :])
            parts = [*head, tail]
        return parts

    def delays(self, beats: Sequence[str], rng: Random) -> list[float]:
        """Return the pause to keep *before* each beat after the first.

        The first bubble goes out immediately: there is no earlier bubble for it
        to be a follow-up of, and the Runtime already waited its own turn before
        leasing the send.
        """
        return [self.delay_for(beat, rng) for beat in beats[1:]]

    def delay_for(self, beat: str, rng: Random) -> float:
        """Return one pause in seconds, following the host's chosen curve."""
        low, high = self.interval
        if self.interval_method == "log" and self.log_base > 1.0:
            steps = math.log(word_count(beat) + 1, self.log_base)
            return rng.uniform(steps, steps + 0.5)
        return rng.uniform(low, high)

    def _split(self, text: str) -> list[str]:
        """Apply the host's split rule, without any of its text cleanup."""
        if self.mode == "words":
            return self._split_by_words(text)
        try:
            return re.findall(self.regex, text, re.DOTALL | re.MULTILINE)
        except re.error:
            # Mirrors the host: a bad pattern degrades to the default one rather
            # than failing the delivery.
            return re.findall(FALLBACK_REGEX, text, re.DOTALL | re.MULTILINE)

    def _split_by_words(self, text: str) -> list[str]:
        """Split at sentence-final markers, dropping the marker itself."""
        if not self.split_words:
            return [text]
        escaped = sorted((re.escape(word) for word in self.split_words), key=len, reverse=True)
        pattern = re.compile(rf"(.*?({'|'.join(escaped)})|.+$)", re.DOTALL)
        parts: list[str] = []
        for match in pattern.findall(text):
            content = match[0] if isinstance(match, tuple) else match
            if not isinstance(content, str):
                continue
            for word in self.split_words:
                if content.endswith(word):
                    content = content[: -len(word)]
                    break
            if content.strip():
                parts.append(content)
        return parts or [text]


def _as_int(value: Any, default: int) -> int:
    """Coerce ``value`` to ``int``, falling back to ``default``."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """Coerce ``value`` to ``float``, falling back to ``default``."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_interval(raw: Any) -> tuple[float, float]:
    """Parse the host's ``"0.8,1.6"`` interval string.

    Args:
        raw: ``"min,max"``, a pair, or anything else.

    Returns:
        A ``(low, high)`` pair with ``low <= high``; the host's default when the
        value cannot be read or is negative.
    """
    low: Any = None
    high: Any = None
    if isinstance(raw, str):
        pieces = [piece for piece in raw.replace(" ", "").split(",") if piece]
        if len(pieces) >= 2:
            low, high = pieces[0], pieces[1]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        low, high = raw[0], raw[1]
    low_value = _as_float(low, DEFAULT_INTERVAL[0])
    high_value = _as_float(high, DEFAULT_INTERVAL[1])
    if low_value < 0 or high_value < 0:
        return DEFAULT_INTERVAL
    return (low_value, high_value) if low_value <= high_value else (high_value, low_value)
