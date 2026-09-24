"""Segmentation tests: the host's split rule applied to a proactive message.

The rule is read from the host config, so the interesting cases are the ones where
the host config is absent, partial, disabled, or wrong -- and the two invariants
that must hold whatever it says: the text is never rewritten, and no part of it is
ever dropped.
"""

from __future__ import annotations

import unittest
from random import Random

from companion_runtime.segments import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_PARTS,
    SEGMENT_MODE_INHERIT,
    SEGMENT_MODE_OFF,
    SEGMENT_MODE_ON,
    SegmentPolicy,
    host_segment_settings,
    word_count,
)

#: The host's shipped configuration (``astrbot/core/config/default.py``), which is
#: what the beta actually runs with.
HOST_ENABLED = {
    "enable": True,
    "only_llm_result": True,
    "interval_method": "random",
    "interval": "0.8,1.6",
    "log_base": 2.6,
    "words_count_threshold": 60,
    "split_mode": "regex",
    "regex": r"[^\n]+",
    "split_words": ["。", "？", "！", "~", "…"],
    "content_cleanup_rule": "[。]+$",
}

TWO_LINES = "突然想起你那个“急”字。\n我知道你很急，但你先别急。"


def _policy(settings: dict | None, **kwargs) -> SegmentPolicy:
    return SegmentPolicy.from_host_settings(settings, **kwargs)


class HostSettingsTests(unittest.TestCase):
    def test_the_host_section_is_read_from_the_config_mapping(self) -> None:
        config = {"platform_settings": {"segmented_reply": HOST_ENABLED}}
        self.assertEqual(host_segment_settings(config)["regex"], r"[^\n]+")

    def test_a_missing_or_foreign_config_is_not_an_error(self) -> None:
        """The delivery path must not fail because the config is unreadable."""
        for config in (None, {}, {"platform_settings": None}, 42, "nonsense"):
            with self.subTest(config=config):
                self.assertEqual(host_segment_settings(config), {})


class SplitBehaviourTests(unittest.TestCase):
    def test_two_lines_become_two_messages(self) -> None:
        """This is the whole bug: it used to be one bubble with a newline in it."""
        beats = _policy(HOST_ENABLED).beats(TWO_LINES)
        self.assertEqual(beats, ["突然想起你那个“急”字。", "我知道你很急，但你先别急。"])

    def test_a_single_line_stays_a_single_message(self) -> None:
        self.assertEqual(_policy(HOST_ENABLED).beats("在忙吗"), ["在忙吗"])

    def test_the_host_switch_is_the_authority(self) -> None:
        """Turning 分段回复 off in the WebUI has to turn it off here too."""
        self.assertEqual(_policy({**HOST_ENABLED, "enable": False}).beats(TWO_LINES), [TWO_LINES])

    def test_mode_on_segments_even_when_the_host_has_it_off(self) -> None:
        policy = _policy({**HOST_ENABLED, "enable": False}, mode=SEGMENT_MODE_ON)
        self.assertEqual(len(policy.beats(TWO_LINES)), 2)

    def test_mode_off_delivers_one_message_even_when_the_host_is_on(self) -> None:
        policy = _policy(HOST_ENABLED, mode=SEGMENT_MODE_OFF)
        self.assertEqual(policy.beats(TWO_LINES), [TWO_LINES])

    def test_mode_inherit_is_the_default(self) -> None:
        self.assertEqual(SEGMENT_MODE_INHERIT, "inherit")
        self.assertEqual(_policy(HOST_ENABLED).enabled, True)
        self.assertEqual(_policy({}).enabled, False)

    def test_a_long_line_is_left_alone(self) -> None:
        """The host's own rule: past ``words_count_threshold`` a paragraph stays one."""
        long_line = "啊" * 61 + "\n" + "嗯"
        self.assertEqual(_policy(HOST_ENABLED).beats(long_line), [long_line])

    def test_an_empty_message_has_no_beats(self) -> None:
        self.assertEqual(_policy(HOST_ENABLED).beats("   \n  "), [])

    def test_a_bad_regex_degrades_instead_of_failing(self) -> None:
        policy = _policy({**HOST_ENABLED, "regex": "[unclosed"})
        self.assertEqual(len(policy.beats("第一句。第二句。")), 2)

    def test_words_mode_splits_at_the_configured_markers(self) -> None:
        policy = _policy({**HOST_ENABLED, "split_mode": "words"})
        self.assertEqual(policy.beats("在忙吗？我先睡了。"), ["在忙吗", "我先睡了"])

    def test_more_beats_than_the_cap_are_merged_not_truncated(self) -> None:
        """A safety cap must never be able to eat half a sentence."""
        text = "一。\n二。\n三。\n四。\n五。\n六。"
        beats = _policy(HOST_ENABLED, max_parts=3).beats(text)
        self.assertEqual(len(beats), 3)
        self.assertEqual("".join(beats).count("。"), text.count("。"))
        self.assertIn("四。", beats[-1])

    def test_the_cap_defaults_to_four(self) -> None:
        self.assertEqual(DEFAULT_MAX_PARTS, 4)
        self.assertEqual(_policy(HOST_ENABLED).max_parts, 4)

    def test_text_is_never_rewritten(self) -> None:
        """刻意不比宿主多做一步「清理标点」：文本是 Runtime 写的，适配器只送不改。"""
        beats = _policy(HOST_ENABLED).beats(TWO_LINES)
        self.assertEqual("\n".join(beats), TWO_LINES)
        self.assertTrue(beats[0].endswith("。"))

    def test_the_rendered_lines_are_kept_even_without_a_host_config(self) -> None:
        """A host config that cannot be read must not become a crash or a rewrite."""
        self.assertEqual(_policy(None, mode=SEGMENT_MODE_ON).beats(TWO_LINES), TWO_LINES.split("\n"))


class DelayTests(unittest.TestCase):
    def test_one_pause_per_follow_up_message(self) -> None:
        policy = _policy(HOST_ENABLED)
        beats = policy.beats(TWO_LINES)
        delays = policy.delays(beats, Random(0))
        self.assertEqual(len(delays), len(beats) - 1)

    def test_a_single_message_waits_nothing(self) -> None:
        policy = _policy(HOST_ENABLED)
        self.assertEqual(policy.delays(["在忙吗"], Random(0)), [])

    def test_delays_stay_inside_the_configured_window(self) -> None:
        policy = _policy(HOST_ENABLED)
        rng = Random(7)
        for _ in range(50):
            delay = policy.delay_for("在忙吗", rng)
            self.assertGreaterEqual(delay, 0.8)
            self.assertLessEqual(delay, 1.6)

    def test_an_unreadable_interval_falls_back_to_the_host_default(self) -> None:
        policy = _policy({**HOST_ENABLED, "interval": None})
        self.assertEqual(policy.interval, DEFAULT_INTERVAL)
        policy = _policy({**HOST_ENABLED, "interval": "-1,-2"})
        self.assertEqual(policy.interval, DEFAULT_INTERVAL)

    def test_the_log_method_pauses_longer_for_longer_lines(self) -> None:
        policy = _policy({**HOST_ENABLED, "interval_method": "log"})
        rng = Random(3)
        short = policy.delay_for("嗯", rng)
        longer = policy.delay_for("今天真的有点累了想早点睡", rng)
        self.assertGreater(longer, short)

    def test_word_count_counts_cjk_characters_not_whitespace_tokens(self) -> None:
        self.assertEqual(word_count("在忙吗"), 3)
        self.assertEqual(word_count("busy now"), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
