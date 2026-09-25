"""Integration tests for ``main.py`` against a stubbed AstrBot surface.

The stub under ``tests/stubs/astrbot`` mirrors the AstrBot 4.28 public API this
plugin imports, so these tests can verify the *wiring* of the adapter without a
real AstrBot checkout:

* observed messages become event reports,
* ``on_llm_request`` injects a temporary ``TextPart`` behind the strict deadline,
* the outbox loop renders with the session's current provider,
* a send happens only after the Runtime authorizes it,
* ``initialize`` / ``terminate`` leave no background tasks behind.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from companion_runtime.protocol import (
    ACTION_RENDER,
    ACTION_SEND,
    AuthorizeDecision,
    ContextSnapshot,
    LeasedAction,
)

from tests.fakes import FakeTransport, wait_until

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
PACKAGE_NAME = "astrbot_plugin_companion_runtime"

SESSION = "webchat:FriendMessage:user-1"


def _load_plugin_module() -> Any:
    """Import the plugin's ``main`` module with relative imports intact."""
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))
    if str(STUBS_DIR) not in sys.path:
        sys.path.insert(0, str(STUBS_DIR))
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(PLUGIN_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.main")


class StubMessageEvent:
    """Minimal stand-in for ``AstrMessageEvent`` with the fields we read."""

    def __init__(
        self,
        *,
        text: str = "hi",
        session: str = SESSION,
        wake: bool = True,
        result_text: str = "",
        message_type: str = "FriendMessage",
        platform: str = "webchat",
    ) -> None:
        self.unified_msg_origin = session
        self.message_str = text
        self.message_obj = SimpleNamespace(message_id="msg-1")
        self.is_at_or_wake_command = wake
        self._result_text = result_text
        #: Set when a handler stops this event, so a test can tell a superseded
        #: request from the one that actually gets answered.
        self.stopped = False
        #: AstrBot's real enum values ("FriendMessage", "GroupMessage", ...), not
        #: the plain scope words the Runtime protocol uses.
        self._message_type = message_type
        #: A cron run arrives as its own platform ("cron"), which changes the rules for
        #: delivering what the character says.
        self._platform = platform
        self._extras: dict[str, Any] = {}

    def set_extra(self, key: str, value: Any) -> None:
        self._extras[key] = value

    def stop_event(self) -> None:
        self.stopped = True

    def is_stopped(self) -> bool:
        """Mirror ``AstrMessageEvent.is_stopped``."""
        return self.stopped

    def get_extra(self, key: str | None = None, default: Any = None) -> Any:
        if key is None:
            return self._extras
        return self._extras.get(key, default)

    def get_platform_name(self) -> str:
        return self._platform if hasattr(self, "_platform") else "webchat"

    def get_message_type(self) -> Any:
        return SimpleNamespace(value=self._message_type)

    def get_sender_id(self) -> str:
        return "user-1"

    def get_sender_name(self) -> str:
        return "User"

    def get_self_id(self) -> str:
        return "bot-1"

    def get_group_id(self) -> str:
        return ""

    def get_result(self) -> Any:
        if not self._result_text:
            return None
        from astrbot.api.event import MessageEventResult
        from astrbot.api.message_components import Plain

        return MessageEventResult([Plain(self._result_text)])

    def plain_result(self, text: str) -> Any:
        """Mirror ``AstrMessageEvent.plain_result``."""
        from astrbot.api.event import MessageEventResult
        from astrbot.api.message_components import Plain

        return MessageEventResult([Plain(text)])


class StubContext:
    """Stand-in for the AstrBot ``Context`` handed to the plugin."""

    def __init__(
        self,
        *,
        provider_id: str = "openai/gpt-4o",
        completion: str = "主动消息",
        send_delay_s: float = 0.0,
        plugin_whitelist: Any = None,
        segmented_reply: dict[str, Any] | None = None,
        send_fail_after: int | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.completion = completion
        #: Lets a test hold a delivery in flight while it terminates the plugin.
        self.send_delay_s = send_delay_s
        self.plugin_whitelist = plugin_whitelist
        #: The host's ``platform_settings.segmented_reply`` mapping, i.e. what the
        #: adapter reads to cut a proactive message into bubbles. ``None`` means the
        #: host config carries no such section at all.
        self.segmented_reply = segmented_reply
        #: Raise after this many accepted sends, so a test can pin what happens when
        #: only part of a segmented message can be delivered.
        self.send_fail_after = send_fail_after
        self.sent: list[tuple[str, Any]] = []
        self.generated: list[dict[str, Any]] = []
        self.provider_lookups: list[str] = []
        self.send_started = asyncio.Event()
        #: The host's conversation store; the plugin appends delivered proactive lines here.
        self.conversation_manager = StubConversationManager()

    def get_config(self) -> dict[str, Any]:
        """Mirror ``Context.get_config()`` for the whitelist and segmentation reads."""
        config: dict[str, Any] = {}
        if self.plugin_whitelist is not None:
            config["plugin_set"] = self.plugin_whitelist
        if self.segmented_reply is not None:
            config["platform_settings"] = {"segmented_reply": dict(self.segmented_reply)}
        return config

    async def get_current_chat_provider_id(self, umo: str | None = None) -> str:
        self.provider_lookups.append(str(umo))
        return self.provider_id

    async def llm_generate(self, **kwargs: Any) -> Any:
        from astrbot.api.provider import LLMResponse

        self.generated.append(kwargs)
        return LLMResponse(self.completion)

    async def send_message(self, session: Any, chain: Any) -> bool:
        self.send_started.set()
        if self.send_delay_s:
            await asyncio.sleep(self.send_delay_s)
        if self.send_fail_after is not None and len(self.sent) >= self.send_fail_after:
            raise RuntimeError("send_message failed")
        self.sent.append((str(session), chain))
        return True


class StubConversation:
    """Stand-in for one AstrBot conversation row."""

    def __init__(self, content: list[dict[str, Any]] | None = None) -> None:
        self.content = list(content or [])


class StubConversationManager:
    """Stand-in for ``Context.conversation_manager``.

    Records what the plugin appended, because the point of the write-back is that the
    *host* history ends up holding the line the character said out of band -- AstrBot's
    own ``send_message`` writes group LTM only.
    """

    def __init__(
        self,
        *,
        content: list[dict[str, Any]] | None = None,
        conversation_id: str | None = "conv-1",
        fail_with: Exception | None = None,
    ) -> None:
        self.conversation = StubConversation(content)
        self.conversation_id = conversation_id
        self.fail_with = fail_with
        #: Every history the plugin wrote, newest last.
        self.updated: list[list[dict[str, Any]]] = []

    async def get_curr_conversation_id(self, umo: str) -> str | None:
        return self.conversation_id

    async def get_conversation(
        self, umo: str, conversation_id: str | None = None
    ) -> StubConversation:
        if self.fail_with is not None:
            raise self.fail_with
        return self.conversation

    async def update_conversation(
        self,
        *,
        unified_msg_origin: str,
        conversation_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.conversation.content = list(history or [])
        self.updated.append(list(history or []))


class StubRuntimeTransport(FakeTransport):
    """Fake transport that accepts the constructor the plugin uses."""

    instances: list[StubRuntimeTransport] = []

    def __init__(
        self,
        *,
        settings: Any = None,
        base_url: str | None = None,
        log: Any = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        #: The address this client points at; one client is built per target.
        self.base_url = base_url or (settings.base_url if settings is not None else "")
        self.log = log
        StubRuntimeTransport.instances.append(self)


class PluginIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the plugin end to end against the AstrBot stub."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.main = _load_plugin_module()
        cls.filters = importlib.import_module("astrbot.api.event.filter")

    def setUp(self) -> None:
        real_astrbot = sys.modules.get("astrbot")
        if real_astrbot is not None and not str(getattr(real_astrbot, "__file__", "")).startswith(
            str(STUBS_DIR),
        ):
            self.skipTest("a real AstrBot installation is importable; stub would mislead")

    async def asyncSetUp(self) -> None:
        self._original_transport = self.main.AiohttpRuntimeTransport
        self.main.AiohttpRuntimeTransport = StubRuntimeTransport
        StubRuntimeTransport.instances = []
        self.context = StubContext()
        self.plugin: Any = None

    async def asyncTearDown(self) -> None:
        if self.plugin is not None:
            await self.plugin.terminate()
        self.main.AiohttpRuntimeTransport = self._original_transport
        # The scope filter instance is shared module state; a test that leaves it
        # widened would silently change AstrBot's pipeline for the next one.
        self.main._ObservationScopeFilter.observe_all = False

    async def _plugin(self, **config_overrides: Any) -> Any:
        config: dict[str, Any] = {
            "runtime_base_url": "http://127.0.0.1:8799",
            "adapter_id": "test-adapter",
            "context_timeout_ms": 50,
            "outbox_poll_interval_ms": 200,
            "request_timeout_ms": 300,
            "render_timeout_ms": 1000,
            "send_timeout_ms": 1000,
        }
        config.update(config_overrides)
        plugin = self.main.CompanionRuntimePlugin(context=self.context, config=config)
        await plugin.initialize()
        self.plugin = plugin
        return plugin

    def _handler(self, name: str) -> Any:
        return self.filters.handler_by_name(name)

    # -- observation -------------------------------------------------------

    async def test_observed_message_is_reported_to_the_runtime(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="今晚可能不来了"),
        )

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        body = transport.event_bodies[0]
        self.assertEqual(body["adapter_id"], "test-adapter")
        record = body["events"][0]
        self.assertEqual(record["kind"], "user_message")
        self.assertEqual(record["text"], "今晚可能不来了")
        self.assertEqual(record["session"], SESSION)
        self.assertEqual(record["platform"], "webchat")
        self.assertTrue(record["preempts_proactive"])

    async def test_a_message_another_plugin_dropped_is_not_reported(self) -> None:
        """A stopped event never becomes a user turn in the Runtime.

        AstrBot breaks the plugin handler chain once an event is stopped, and the
        observation handler runs at a deliberately low priority so a filter plugin can
        drop a message first. The guard is also checked inside the handler, which is
        what this test pins. Measured on a real QQ: the tester's client auto-reply was
        「。」, so every message she sent came back as ``[自动回复] 。``, each one was
        filed as a user message, and she answered roughly 1300 of them in one morning.
        """
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        event = StubMessageEvent(text="[自动回复] 。")
        event.stop_event()
        await self._handler("on_message_observed")(self.plugin, event)

        self.assertEqual(transport.event_bodies, [])
        self.assertEqual(len(self.plugin._queue), 0)

    async def test_non_wake_message_is_not_reported_in_wake_mode(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="群里的闲聊", wake=False),
        )

        self.assertEqual(transport.event_bodies, [])
        self.assertEqual(len(self.plugin._queue), 0)

    async def test_non_wake_message_is_reported_in_all_mode(self) -> None:
        await self._plugin(observe_mode="all")
        transport = StubRuntimeTransport.instances[-1]

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="群里的闲聊", wake=False),
        )

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        self.assertFalse(transport.event_bodies[0]["events"][0]["wake"])

    async def test_repeated_start_failures_stop_retrying(self) -> None:
        class ExplodingTransport:
            calls = 0

            def __init__(self, **kwargs: Any) -> None:
                del kwargs
                type(self).calls += 1
                raise RuntimeError("boom")

        self.main.AiohttpRuntimeTransport = ExplodingTransport
        plugin = self.main.CompanionRuntimePlugin(
            context=self.context,
            config={"runtime_base_url": "http://127.0.0.1:8799", "observe_mode": "all"},
        )
        self.plugin = plugin
        handler = self._handler("on_message_observed")

        for _ in range(6):
            await handler(plugin, StubMessageEvent())

        # Three attempts, then the adapter stays quiet instead of logging on
        # every single message.
        self.assertEqual(ExplodingTransport.calls, 3)
        self.assertTrue(plugin._gave_up)
        # Giving up is not a running state, and it must not keep AstrBot's own
        # pipeline widened for an adapter that never came up.
        self.assertFalse(plugin._started)
        self.assertFalse(self.main._ObservationScopeFilter.observe_all)

    async def test_scope_filter_never_wakes_a_sleeping_bot(self) -> None:
        await self._plugin()
        registration = self.filters.registration_for("on_message_observed")
        scope_filter = registration.filter_instance

        self.assertFalse(scope_filter.filter(StubMessageEvent(wake=False), {}))
        self.assertTrue(scope_filter.filter(StubMessageEvent(wake=True), {}))

    async def test_scope_filter_observes_everything_when_opted_in(self) -> None:
        await self._plugin(observe_mode="all")
        scope_filter = self.filters.registration_for("on_message_observed").filter_instance

        self.assertTrue(scope_filter.filter(StubMessageEvent(wake=False), {}))

    async def test_disabled_adapter_never_widens_the_host_pipeline(self) -> None:
        """``observe_all`` on a disabled plugin would wake every group message.

        The filter instance is created once per module and shared by every event,
        so a stale ``True`` left behind by an earlier instance, or published by a
        plugin that is switched off, keeps AstrBot marking non-wake messages as
        wake events on nobody's behalf.
        """
        scope_filter = self.filters.registration_for("on_message_observed").filter_instance
        self.main._ObservationScopeFilter.observe_all = True

        plugin = await self._plugin(enabled=False, observe_mode="all")

        self.assertIsNone(plugin._queue)
        self.assertFalse(self.main._ObservationScopeFilter.observe_all)
        self.assertFalse(scope_filter.filter(StubMessageEvent(wake=False), {}))
        self.assertTrue(scope_filter.filter(StubMessageEvent(wake=True), {}))

    async def test_terminate_resets_the_observation_scope(self) -> None:
        plugin = await self._plugin(observe_mode="all")
        scope_filter = self.filters.registration_for("on_message_observed").filter_instance
        self.assertTrue(scope_filter.filter(StubMessageEvent(wake=False), {}))

        await plugin.terminate()

        self.assertFalse(scope_filter.filter(StubMessageEvent(wake=False), {}))

    async def test_message_type_uses_the_runtime_vocabulary(self) -> None:
        """AstrBot reports chat *classes*; the Runtime needs private/group/other."""
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        for raw, expected in (
            ("FriendMessage", "private"),
            ("GroupMessage", "group"),
            ("OtherMessage", "other"),
        ):
            with self.subTest(message_type=raw):
                transport.event_bodies.clear()
                await self._handler("on_message_observed")(
                    plugin,
                    StubMessageEvent(text="hi", message_type=raw),
                )
                self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
                record = transport.event_bodies[0]["events"][0]
                self.assertEqual(record["message_type"], expected)

    async def test_assistant_message_is_reported(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        await self._handler("on_after_message_sent")(
            self.plugin,
            StubMessageEvent(result_text="我在听"),
        )

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        record = transport.event_bodies[0]["events"][0]
        self.assertEqual(record["kind"], "assistant_message")
        self.assertEqual(record["text"], "我在听")

    async def test_streamed_turn_is_reported_from_the_llm_response(self) -> None:
        """With streaming on, neither pre-send nor post-send hook ever fires."""
        from astrbot.api.provider import LLMResponse

        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        event = StubMessageEvent()

        await self._handler("on_llm_response")(
            self.plugin,
            event,
            LLMResponse("我听见了，慢慢说"),
        )

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        record = transport.event_bodies[0]["events"][0]
        self.assertEqual(record["kind"], "assistant_message")
        self.assertEqual(record["text"], "我听见了，慢慢说")
        self.assertEqual(record["session"], SESSION)

    async def test_a_reported_turn_is_not_reported_twice(self) -> None:
        """A host that streams *and* runs the delivery hook must still send one."""
        from astrbot.api.provider import LLMResponse

        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        event = StubMessageEvent(result_text="我听见了，慢慢说")

        await self._handler("on_llm_response")(self.plugin, event, LLMResponse("我听见了，慢慢说"))
        await self._handler("on_after_message_sent")(self.plugin, event)

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        self.assertEqual(len(transport.event_bodies), 1)
        self.assertEqual(transport.event_bodies[0]["events"][0]["text"], "我听见了，慢慢说")

    async def test_a_tool_delivered_turn_reports_what_went_out(self) -> None:
        """``completion_text`` is the model's narration when the tool did the delivering.

        Measured on the beta (2026-09-25): the send tool delivered 「洗完没有呀……都半个多小时了」
        while the completion read 「已经发了。就一条，软的、拖着尾音的，问他洗完没有」. Reporting the
        completion taught the Runtime she had said something the user never saw, and the
        render prompt then fed that narration back to her as "what you just said".
        """
        from astrbot.api.provider import LLMResponse

        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        event = StubMessageEvent()
        event.set_extra(
            "_send_message_to_user_current_session_plain_texts",
            ["洗完没有呀……都半个多小时了。", "头发擦了没有。"],
        )

        await self._handler("on_llm_response")(
            self.plugin,
            event,
            LLMResponse("已经发了。就一条，软的、拖着尾音的，问他洗完没有。"),
        )

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        record = transport.event_bodies[0]["events"][0]
        self.assertEqual(record["kind"], "assistant_message")
        self.assertEqual(record["text"], "洗完没有呀……都半个多小时了。\n头发擦了没有。")

    async def test_a_turn_without_a_tool_send_still_reports_the_completion(self) -> None:
        """The delivered texts only take precedence when there are any."""
        from astrbot.api.provider import LLMResponse

        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        event = StubMessageEvent()
        event.set_extra("_send_message_to_user_current_session_plain_texts", [])

        await self._handler("on_llm_response")(self.plugin, event, LLMResponse("那我等你回来。"))

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        self.assertEqual(transport.event_bodies[0]["events"][0]["text"], "那我等你回来。")

    async def test_a_reply_without_an_llm_turn_is_still_reported(self) -> None:
        """A command's output never reaches the LLM-response hook."""
        from astrbot.api.provider import LLMResponse

        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        empty = StubMessageEvent(result_text="companion Runtime adapter\n- state: running")
        await self._handler("on_llm_response")(self.plugin, empty, LLMResponse(""))
        await self._handler("on_after_message_sent")(self.plugin, empty)

        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        record = transport.event_bodies[0]["events"][0]
        self.assertEqual(record["kind"], "assistant_message")
        self.assertIn("companion Runtime adapter", record["text"])

    async def test_an_unwhitelisted_plugin_warns_at_startup(self) -> None:
        """A whitelist that omits this plugin kills every hook, silently."""
        self.context = StubContext(plugin_whitelist=["some_other_plugin"])
        plugin = await self._plugin()
        plugin.name = "astrbot_plugin_companion_runtime"

        with self.assertLogs("astrbot.plugin.stub", level="WARNING") as captured:
            await plugin.initialize()

        self.assertIn("plugin whitelist", "\n".join(captured.output))

    async def test_a_whitelisted_plugin_stays_quiet(self) -> None:
        self.context = StubContext(plugin_whitelist=["*"])
        plugin = await self._plugin()
        plugin.name = "astrbot_plugin_companion_runtime"

        with self.assertNoLogs("astrbot.plugin.stub", level="WARNING"):
            await plugin.initialize()

    # -- session routing ---------------------------------------------------

    async def test_sessions_route_to_their_own_runtime(self) -> None:
        """Several people, each 1v1 with the same bot, need separate Runtimes.

        A Runtime holds one character's long-term memory (``memories`` has no
        conversation column), so two people sharing one instance would share one
        memory of "the user".
        """
        other_session = "webchat:FriendMessage:user-2"
        await self._plugin(session_routes={other_session: "http://127.0.0.1:8899"})
        self.assertEqual(len(StubRuntimeTransport.instances), 2)
        default, routed = StubRuntimeTransport.instances
        self.assertEqual(default.base_url, "http://127.0.0.1:8799")
        self.assertEqual(routed.base_url, "http://127.0.0.1:8899")

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="我是小周", session=other_session),
        )
        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="我是小林", session=SESSION),
        )

        self.assertTrue(
            await wait_until(
                lambda: len(routed.event_bodies) == 1 and len(default.event_bodies) == 1,
            ),
        )
        self.assertEqual(routed.event_bodies[0]["events"][0]["text"], "我是小周")
        self.assertEqual(default.event_bodies[0]["events"][0]["text"], "我是小林")
        # The context caches must not be shared either: reading the default one
        # would inject another instance's state into this person's turn.
        self.assertIsNot(self.plugin._bridge_for(SESSION), self.plugin._bridge_for(other_session))

    async def test_unrouted_sessions_use_the_default_runtime(self) -> None:
        """A single-Runtime deployment must behave exactly as before."""
        await self._plugin()
        self.assertEqual(len(StubRuntimeTransport.instances), 1)
        self.assertEqual(tuple(self.plugin._targets), ("http://127.0.0.1:8799",))
        self.assertIs(self.plugin._bridge_for(SESSION), self.plugin._bridge)

        await self._handler("on_message_observed")(self.plugin, StubMessageEvent(text="hi"))

        transport = StubRuntimeTransport.instances[-1]
        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))
        self.assertEqual(transport.event_bodies[0]["events"][0]["text"], "hi")

    async def test_every_routed_runtime_gets_its_own_outbox_poller(self) -> None:
        """Each Runtime's outbox must be leased by a poller pointing at it."""
        await self._plugin(
            session_routes={"webchat:FriendMessage:user-2": "http://127.0.0.1:8899"},
        )
        self.assertEqual(len(self.plugin._targets), 2)
        self.assertEqual(len(self.plugin._tasks), 2)
        for target in self.plugin._targets.values():
            self.assertIsNotNone(target.outbox)
            self.assertIs(target.outbox._transport, target.transport)
        self.assertIs(self.plugin._outbox, self.plugin._targets["http://127.0.0.1:8799"].outbox)

    async def test_session_routes_are_prefix_matched_longest_first(self) -> None:
        """A narrow route must win over a wide one, and bad entries must not land."""
        settings = self.main.Settings.from_mapping(
            {
                "runtime_base_url": "http://a:8787",
                "session_routes": {
                    "webchat:": "http://b:8787",
                    "webchat:FriendMessage:user-2": "http://c:8787",
                    "bad": "not-a-url",
                    "noop": "http://a:8787",
                },
            },
        )

        self.assertEqual(settings.target_for("webchat:FriendMessage:user-2"), "http://c:8787")
        self.assertEqual(settings.target_for("webchat:GroupMessage:9"), "http://b:8787")
        self.assertEqual(settings.target_for("aiocqhttp:FriendMessage:1"), "http://a:8787")
        self.assertEqual(
            settings.targets,
            ("http://a:8787", "http://c:8787", "http://b:8787"),
        )
        self.assertEqual([prefix for prefix, _ in settings.session_routes], [
            "webchat:FriendMessage:user-2",
            "webchat:",
        ])
        self.assertTrue(
            any("session_routes" in issue and "'bad'" in issue for issue in settings.issues),
            settings.issues,
        )

    async def test_the_registry_adds_a_target_without_a_restart(self) -> None:
        """Adding a person must be a fleet-side action, not an AstrBot restart.

        The adapter asks the fleet which Runtime serves a session it has not seen.
        """
        other_session = "webchat:FriendMessage:user-9"
        await self._plugin(route_registry_url="http://127.0.0.1:8898")
        registry = StubRuntimeTransport.instances[-1]
        self.assertEqual(registry.base_url, "http://127.0.0.1:8898")
        registry.routes = {other_session: "http://127.0.0.1:8899"}

        self.assertEqual(await self.plugin._sync_registry_once(), 1)
        self.assertEqual(len(StubRuntimeTransport.instances), 3)
        routed = StubRuntimeTransport.instances[-1]
        self.assertEqual(routed.base_url, "http://127.0.0.1:8899")
        self.assertIsNotNone(self.plugin._targets["http://127.0.0.1:8899"].outbox)

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="刚来报到", session=other_session),
        )

        self.assertTrue(await wait_until(lambda: len(routed.event_bodies) == 1))
        self.assertEqual(routed.event_bodies[0]["events"][0]["text"], "刚来报到")
        # The registry's own client must never receive message traffic.
        self.assertEqual(registry.event_bodies, [])

    async def test_a_static_route_beats_the_registry(self) -> None:
        """An operator's explicit route is a decision; the registry only fills gaps."""
        session = "webchat:FriendMessage:user-2"
        await self._plugin(
            session_routes={session: "http://127.0.0.1:8897"},
            route_registry_url="http://127.0.0.1:8898",
        )
        registry = StubRuntimeTransport.instances[-1]
        registry.routes = {session: "http://127.0.0.1:8899"}

        self.assertEqual(await self.plugin._sync_registry_once(), 1)

        self.assertEqual(self.plugin._target_url(session), "http://127.0.0.1:8897")

    async def test_an_unreachable_registry_falls_back_to_the_default(self) -> None:
        """A fleet that is down must not take routing with it."""
        await self._plugin(route_registry_url="http://127.0.0.1:8898")
        registry = StubRuntimeTransport.instances[-1]
        registry.route_error = RuntimeError("connection refused")

        self.assertEqual(await self.plugin._sync_registry_once(), 0)
        self.assertEqual(self.plugin._target_url("webchat:FriendMessage:user-9"), "http://127.0.0.1:8799")

        transport = StubRuntimeTransport.instances[0]
        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="hi", session="webchat:FriendMessage:user-9"),
        )
        self.assertTrue(await wait_until(lambda: len(transport.event_bodies) == 1))

    async def test_a_message_for_an_unprovisioned_person_waits_for_the_registry(self) -> None:
        """Never file one person's words into another person's Runtime.

        A person provisioned moments ago is not in the registry cache yet. Falling
        back to the default Runtime would blend them, so the report must wait for
        the registry (the queue retries) and then go to the right instance.
        """
        newcomer = "webchat:FriendMessage:user-9"
        await self._plugin(
            route_registry_url="http://127.0.0.1:8898",
            route_sync_interval_ms=2000,
        )
        registry = StubRuntimeTransport.instances[-1]  # built after the targets
        default = StubRuntimeTransport.instances[0]
        # The registry is reachable but does not know this person yet.
        self.assertEqual(await self.plugin._sync_registry_once(), 0)

        # First message arrives before the registry knows about them.
        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="刚来报到", session=newcomer),
        )
        self.assertEqual(default.event_bodies, [], "the default Runtime must not receive it")
        self.assertEqual(len(self.plugin._queue), 1, "the report must wait in the queue")

        # The registry answers; the retry resolves and delivers to the right place.
        registry.routes = {newcomer: "http://127.0.0.1:8899"}
        self.assertTrue(
            await wait_until(
                lambda: len(StubRuntimeTransport.instances) == 3
                and StubRuntimeTransport.instances[-1].event_bodies,
                timeout_s=10.0,
            ),
            "the queued report must reach the newcomer's Runtime once routing is known",
        )
        routed = StubRuntimeTransport.instances[-1]
        self.assertEqual(routed.base_url, "http://127.0.0.1:8899")
        self.assertEqual(routed.event_bodies[0]["events"][0]["session"], newcomer)
        self.assertEqual(default.event_bodies, [])

    async def test_an_unknown_person_is_provisioned_automatically(self) -> None:
        """A tester's first message must not need an operator.

        With ``route_auto_provision`` the adapter asks the fleet to create the
        instance, then the waiting report is delivered into it -- the whole point of
        the closed beta being "just talk to it".
        """
        newcomer = "webchat:FriendMessage:user-9"
        await self._plugin(
            route_registry_url="http://127.0.0.1:8898",
            route_auto_provision=True,
            route_sync_interval_ms=2000,
        )
        registry = StubRuntimeTransport.instances[-1]
        self.assertEqual(await self.plugin._sync_registry_once(), 0)  # reachable, knows nobody

        registry.provision_result = "http://127.0.0.1:8899"

        async def provision(session: str, *, timeout_s: float) -> str:
            registry.routes = {session: "http://127.0.0.1:8899"}
            return "http://127.0.0.1:8899"

        registry.provision_session = provision  # type: ignore[assignment]

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="第一次说话", session=newcomer),
        )

        self.assertTrue(
            await wait_until(
                lambda: any(
                    item.base_url == "http://127.0.0.1:8899" and item.event_bodies
                    for item in StubRuntimeTransport.instances
                ),
                timeout_s=10.0,
            ),
            "the newcomer's first message must reach the instance created for them",
        )
        self.assertEqual(StubRuntimeTransport.instances[0].event_bodies, [])

    async def test_auto_provision_is_off_by_default(self) -> None:
        """Growing the fleet is a deployment decision, not an adapter default."""
        await self._plugin(route_registry_url="http://127.0.0.1:8898")
        registry = StubRuntimeTransport.instances[-1]
        await self.plugin._sync_registry_once()

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="陌生人", session="webchat:FriendMessage:user-9"),
        )
        await asyncio.sleep(0.1)

        self.assertEqual(registry.provision_requests, [])

    async def test_an_empty_message_event_is_not_reported(self) -> None:
        """AstrBot turns OneBot notices (pokes, requests) into empty message events.

        Measured on a real QQ: seven pokes became seven "user said nothing" reports
        and the Runtime filed an empty ``用户说：`` fact for each.
        """
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]

        await self._handler("on_message_observed")(
            self.plugin,
            StubMessageEvent(text="", message_type="FriendMessage"),
        )
        await asyncio.sleep(0.05)

        self.assertEqual(transport.event_bodies, [], "an empty event must not be reported")
        self.assertEqual(len(self.plugin._queue), 0)
        self.assertEqual(self.plugin._skipped_empty, 1)

    async def test_disabled_plugin_does_nothing(self) -> None:
        plugin = await self._plugin(enabled=False)
        self.assertIsNone(plugin._queue)
        self.assertEqual(StubRuntimeTransport.instances, [])

        await self._handler("on_message_observed")(plugin, StubMessageEvent())
        request = SimpleNamespace(extra_user_content_parts=[])
        await self._handler("on_llm_request")(plugin, StubMessageEvent(), request)
        self.assertEqual(request.extra_user_content_parts, [])

    # -- input debounce ----------------------------------------------------

    async def test_debounce_is_off_by_default(self) -> None:
        await self._plugin()

        event = StubMessageEvent(text="在吗")
        await self._handler("on_waiting_llm_request")(self.plugin, event)

        self.assertEqual(event.message_str, "在吗")
        self.assertFalse(event.stopped)
        self.assertEqual(self.plugin._input_bursts, {})

    async def test_the_debounce_is_registered_before_the_session_lock(self) -> None:
        """It must sit on the waiting hook, not on ``on_llm_request``.

        AstrBot calls ``OnWaitingLLMRequestEvent`` first and only then takes the
        per-session lock; ``on_llm_request`` runs inside that lock. A debounce waiting
        inside the lock can never observe the next message of the burst, because that
        message's handler cannot start until this turn has finished - which is exactly
        the failure measured in the beta (two messages two seconds apart, two answers).
        """
        await self._plugin(input_debounce_ms=60)

        self.assertEqual(
            self.filters.registration_for("on_waiting_llm_request").kind,
            "on_waiting_llm_request",
        )
        self.assertFalse(
            any(
                registration.name == "on_waiting_llm_request"
                and registration.kind == "on_llm_request"
                for registration in self.filters.REGISTRATIONS
            )
            or any(
                registration.kind == "on_llm_request"
                and registration.name.endswith("debounce")
                for registration in self.filters.REGISTRATIONS
            ),
            "the debounce must not be registered on on_llm_request",
        )

    async def test_a_burst_of_messages_is_answered_once(self) -> None:
        await self._plugin(input_debounce_ms=60)

        texts = ("在吗", "睡了没", "?")
        events = [StubMessageEvent(text=text) for text in texts]
        handler = self._handler("on_waiting_llm_request")

        async def fire(index: int) -> None:
            # Stagger the arrivals so the burst is unambiguous regardless of how
            # the event loop schedules the tasks.
            await asyncio.sleep(index * 0.01)
            await handler(self.plugin, events[index])

        await asyncio.gather(*(fire(index) for index in range(len(texts))))

        self.assertEqual(events[-1].message_str, "在吗\n睡了没\n?")
        self.assertFalse(events[-1].stopped, "the newest message must be answered")
        self.assertTrue(events[0].stopped, "a superseded message must not be answered")
        self.assertTrue(events[1].stopped)
        self.assertEqual(self.plugin._input_bursts, {}, "a settled burst must not leak")

    async def test_only_the_newest_parts_survive_the_char_limit(self) -> None:
        await self._plugin(input_debounce_ms=60, input_debounce_max_chars=10)

        events = [StubMessageEvent(text=text) for text in ("0123456789", "abc")]
        handler = self._handler("on_waiting_llm_request")

        async def fire(index: int) -> None:
            await asyncio.sleep(index * 0.01)
            await handler(self.plugin, events[index])

        await asyncio.gather(*(fire(index) for index in range(len(events))))

        # 13 chars would exceed the 10-char budget, so the oldest message goes.
        self.assertEqual(events[-1].message_str, "abc")

    async def test_a_second_burst_after_the_window_is_a_new_turn(self) -> None:
        await self._plugin(input_debounce_ms=30)

        handler = self._handler("on_waiting_llm_request")
        first = StubMessageEvent(text="第一句")
        second = StubMessageEvent(text="第二句")

        await handler(self.plugin, first)
        await handler(self.plugin, second)

        self.assertEqual(first.message_str, "第一句")
        self.assertEqual(second.message_str, "第二句")

    # -- injection ---------------------------------------------------------

    async def test_context_is_injected_as_a_temporary_part(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        # Patch v0.2: the injected block is the character's long-term state
        # before the turn (background), never an instruction for this turn.
        section = "【进入本轮前的长期状态（背景）】"
        transport.snapshot = ContextSnapshot(text=f"{section}\n克制，想联系", version="9")

        from astrbot.api.provider import ProviderRequest

        request = ProviderRequest(prompt="在吗")
        await self._handler("on_llm_request")(self.plugin, StubMessageEvent(), request)

        self.assertEqual(len(request.extra_user_content_parts), 1)
        part = request.extra_user_content_parts[0]
        self.assertTrue(part._no_save, "hidden context must never be persisted")
        self.assertTrue(part.text.startswith("<companion_runtime_context"))
        self.assertIn(section, part.text)
        self.assertIn('version="9"', part.text)

    async def test_injection_times_out_without_touching_the_request(self) -> None:
        await self._plugin(context_timeout_ms=50)
        transport = StubRuntimeTransport.instances[-1]
        transport.snapshot = ContextSnapshot(text="too late")
        transport.context_delay_s = 0.5

        from astrbot.api.provider import ProviderRequest

        request = ProviderRequest(prompt="在吗")
        await self._handler("on_llm_request")(self.plugin, StubMessageEvent(), request)

        self.assertEqual(request.extra_user_content_parts, [])
        self.assertGreaterEqual(self.plugin._bridge.stats.timeouts, 1)

    async def test_injection_survives_a_runtime_outage(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.context_error = RuntimeError("connection refused")

        from astrbot.api.provider import ProviderRequest

        request = ProviderRequest(prompt="在吗")
        await self._handler("on_llm_request")(self.plugin, StubMessageEvent(), request)

        self.assertEqual(request.extra_user_content_parts, [])

    async def test_the_injection_carries_the_host_delivery_note(self) -> None:
        """Plain text goes out as the reply, not through the send tool.

        ``result_decorate`` is what splits her reply into bubbles, and a message sent
        through ``send_message_to_user`` never passes through it: measured on the beta
        (2026-09-25 21:45) four lines arrived as one QQ bubble.
        """
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.snapshot = ContextSnapshot(text="状态", version="9")

        from astrbot.api.provider import ProviderRequest

        request = ProviderRequest(prompt="在吗")
        await self._handler("on_llm_request")(self.plugin, StubMessageEvent(), request)

        text = request.extra_user_content_parts[0].text
        self.assertIn("宿主说明", text)
        self.assertIn("纯文字不要用它", text)
        self.assertLess(text.index("</companion_runtime_context>"), text.index("宿主说明"))

    async def test_a_cron_run_still_gets_the_context_block(self) -> None:
        """A cron run never passes the pipeline, so ``on_agent_begin`` carries the block.

        Measured on the beta: a 21:45 cron turn left no ``context_rendered`` event behind,
        which means she answered with her persona and the transcript but none of the
        Runtime's cognition.
        """
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.snapshot = ContextSnapshot(text="状态", version="9")
        run_context = SimpleNamespace(messages=[])

        await self._handler("on_agent_begin")(
            self.plugin,
            StubMessageEvent(platform="cron"),
            run_context,
        )

        self.assertEqual(len(run_context.messages), 1)
        content = run_context.messages[0].content[0]
        self.assertTrue(content._no_save, "hidden context must never be persisted")
        self.assertIn("状态", content.text)
        self.assertIn("定时任务叫醒的", content.text)
        self.assertIn("一条一次", content.text)
        self.assertIn("不要换行", content.text)

    async def test_a_run_that_already_has_the_block_is_left_alone(self) -> None:
        """The pipeline injected this run; a second copy would only repeat it."""
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.snapshot = ContextSnapshot(text="状态", version="9")
        event = StubMessageEvent()
        run_context = SimpleNamespace(messages=[])

        from astrbot.api.provider import ProviderRequest

        await self._handler("on_llm_request")(self.plugin, event, ProviderRequest(prompt="在吗"))
        await self._handler("on_agent_begin")(self.plugin, event, run_context)

        self.assertEqual(run_context.messages, [])

    # -- outbox ------------------------------------------------------------

    async def test_authorized_send_is_delivered_and_reported(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_SEND, payload={"text": "在忙吗"})]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: len(self.context.sent) == 1, timeout_s=3.0))

        session, chain = self.context.sent[0]
        self.assertEqual(session, SESSION)
        self.assertEqual(chain.get_plain_text(), "在忙吗")
        self.assertEqual(transport.authorize_requests[0].action_id, "act_1")
        # The message is delivered before the result is reported, so wait for the
        # report rather than assuming it lands in the same loop iteration.
        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))
        report = transport.report_bodies[0]
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["result"]["sent"])

    async def test_the_delivered_line_is_appended_to_the_host_history(self) -> None:
        """主动消息必须写回宿主历史：宿主自己只写群聊 LTM，私聊一个字都不写。

        症状（两个测试者独立报出）：用户接着回一句「？」，主模型不知道自己说过什么，
        于是答非所问；同一件事还会被反复说，因为没有任何东西告诉她"你已经说过了"。
        """
        self.context.conversation_manager = StubConversationManager(
            content=[
                {"role": "user", "content": [{"type": "text", "text": "在吗"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "在。"}]},
            ]
        )
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_SEND, payload={"text": "在忙吗"})]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))

        manager = self.context.conversation_manager
        self.assertEqual(len(manager.updated), 1)
        # 既有两轮一轮都不能少，新的一轮追加在后面。
        self.assertEqual(
            [entry["role"] for entry in manager.conversation.content],
            ["user", "assistant", "assistant"],
        )
        self.assertEqual(
            manager.conversation.content[-1],
            {"role": "assistant", "content": [{"type": "text", "text": "在忙吗"}]},
        )
        # 写回历史不能让这次投递变成失败。
        self.assertEqual(transport.report_bodies[0]["status"], "ok")
        self.assertTrue(transport.report_bodies[0]["result"]["sent"])

    async def test_a_history_write_failure_does_not_fail_the_delivery(self) -> None:
        """历史写不进去只是少了上下文，不能把"已经发出去的消息"报成失败（否则会重发）。"""
        self.context.conversation_manager = StubConversationManager(
            fail_with=RuntimeError("history exploded")
        )
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_SEND, payload={"text": "在忙吗"})]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))

        self.assertEqual(len(self.context.sent), 1, "消息本身还是发出去了")
        self.assertEqual(transport.report_bodies[0]["status"], "ok")
        self.assertTrue(transport.report_bodies[0]["result"]["sent"])

    async def test_a_two_line_proactive_message_arrives_as_two_messages(self) -> None:
        """主动消息的分段必须和宿主的「分段回复」同规则。

        症状：宿主自己的回复走 result_decorate/respond，会按 `[^\\n]+` 切成几条、
        每条之间停顿一下再发；主动消息走 `Context.send_message`，把这两个 stage 全
        绕开了，于是 Runtime 渲染成两行的话在 QQ 里变成**一条**带换行的气泡。而
        Runtime 的渲染风格里明明写着「没说完就再发一条」。
        """
        self.context.segmented_reply = {
            "enable": True,
            "split_mode": "regex",
            "regex": r"[^\n]+",
            "interval": "0.01,0.02",
        }
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [
            _leased(
                ACTION_SEND,
                payload={"text": "突然想起你那个“急”字。\n我知道你很急，但你先别急。"},
            ),
        ]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: len(self.context.sent) == 2, timeout_s=3.0))
        self.assertEqual(
            [chain.get_plain_text() for _, chain in self.context.sent],
            ["突然想起你那个“急”字。", "我知道你很急，但你先别急。"],
        )
        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))
        self.assertEqual(transport.report_bodies[0]["status"], "ok")
        self.assertEqual(transport.report_bodies[0]["result"]["segments"], 2)
        self.assertEqual(transport.report_bodies[0]["result"]["delivered_segments"], 2)
        # 一次投递仍然只写一条历史（整句），不是两条。
        self.assertEqual(
            self.context.conversation_manager.conversation.content[-1],
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "突然想起你那个“急”字。\n我知道你很急，但你先别急。",
                    },
                ],
            },
        )

    async def test_segmentation_follows_the_host_switch(self) -> None:
        """宿主关掉「分段回复」时主动消息也整条发：一个开关管两边，不留第二个权威。"""
        self.context.segmented_reply = {
            "enable": False,
            "split_mode": "regex",
            "regex": r"[^\n]+",
            "interval": "0.01,0.02",
        }
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [
            _leased(ACTION_SEND, payload={"text": "第一句。\n第二句。"}),
        ]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))
        self.assertEqual(len(self.context.sent), 1)
        self.assertEqual(self.context.sent[0][1].get_plain_text(), "第一句。\n第二句。")
        self.assertNotIn("segments", transport.report_bodies[0]["result"])

    async def test_segmentation_can_be_forced_off_for_this_adapter(self) -> None:
        """`outbox_segmented_reply=off` 时即使宿主开着也整条发（回退开关）。"""
        self.context.segmented_reply = {
            "enable": True,
            "split_mode": "regex",
            "regex": r"[^\n]+",
            "interval": "0.01,0.02",
        }
        await self._plugin(outbox_segmented_reply="off")
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [
            _leased(ACTION_SEND, payload={"text": "第一句。\n第二句。"}),
        ]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))
        self.assertEqual(len(self.context.sent), 1)

    async def test_a_partial_segmented_delivery_reports_only_what_went_out(self) -> None:
        """第二条失败时不能整条重发（第一条已经在用户那儿了），历史只记发出去的那句。

        一次投递里前面几条的发送是不可逆的：把整条报成失败，Runtime 只会重发整条，
        用户就会第二次看到已经收到的那句。
        """
        self.context.segmented_reply = {
            "enable": True,
            "split_mode": "regex",
            "regex": r"[^\n]+",
            "interval": "0.01,0.02",
        }
        self.context.send_fail_after = 1
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [
            _leased(ACTION_SEND, payload={"text": "第一句。\n第二句。"}),
        ]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))
        self.assertEqual([chain.get_plain_text() for _, chain in self.context.sent], ["第一句。"])
        self.assertEqual(transport.report_bodies[0]["status"], "ok")
        self.assertEqual(transport.report_bodies[0]["result"]["delivered_segments"], 1)
        self.assertEqual(
            self.context.conversation_manager.conversation.content[-1]["content"][0]["text"],
            "第一句。",
        )

    async def test_unauthorized_send_is_not_delivered(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_SEND, payload={"text": "在忙吗"})]
        transport.authorize_decision = AuthorizeDecision(
            authorized=False,
            reason="aborted_by_user_message",
        )

        self.assertTrue(await wait_until(lambda: len(transport.report_bodies) == 1, timeout_s=3.0))

        self.assertEqual(self.context.sent, [])
        self.assertEqual(transport.report_bodies[0]["status"], "rejected")
        self.assertEqual(transport.report_bodies[0]["error"], "aborted_by_user_message")

    async def test_render_uses_the_sessions_current_provider(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [
            _leased(ACTION_RENDER, payload={"prompt": "写一句主动问候"}),
        ]

        self.assertTrue(await wait_until(lambda: len(transport.report_bodies) == 1, timeout_s=3.0))

        self.assertEqual(self.context.provider_lookups[0], SESSION)
        self.assertEqual(self.context.generated[0]["chat_provider_id"], "openai/gpt-4o")
        self.assertEqual(self.context.generated[0]["prompt"], "写一句主动问候")
        self.assertEqual(transport.report_bodies[0]["result"]["text"], "主动消息")
        self.assertEqual(self.context.sent, [])

    async def test_a_render_carries_the_host_persona_learned_from_a_chat_turn(self) -> None:
        """A proactive render gets the system prompt AstrBot assembles for a turn.

        A render is a bare ``llm_generate`` call: AstrBot decorates a chat turn with
        ~4210 characters of assembled system prompt (persona + skills + tool hint) and
        a render with **none**, so without this a proactive line is written by a model
        that has never been told who it is. The plugin learns the text from a normal
        turn and hands it over; the host owns the persona, the plugin only remembers it.
        """
        from astrbot.api.provider import ProviderRequest

        await self._plugin()
        persona = "人格设定：苏清徽\n- 一次只说一两句，不用 Markdown"
        await self._handler("on_llm_request")(
            self.plugin, StubMessageEvent(), ProviderRequest(prompt="在吗", system_prompt=persona)
        )

        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_RENDER, payload={"prompt": "写一句主动问候"})]
        self.assertTrue(await wait_until(lambda: len(transport.report_bodies) == 1, timeout_s=3.0))

        self.assertEqual(self.context.generated[0]["system_prompt"], persona)
        # The render itself is unchanged; only what it is asked with changed.
        self.assertEqual(transport.report_bodies[0]["result"]["text"], "主动消息")

    async def test_a_render_without_a_learned_persona_still_renders(self) -> None:
        """Fail open: no chat turn yet (a brand-new user, or straight after a restart)
        means the render goes out exactly as it did before."""
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_RENDER, payload={"prompt": "写一句主动问候"})]
        self.assertTrue(await wait_until(lambda: len(transport.report_bodies) == 1, timeout_s=3.0))

        self.assertNotIn("system_prompt", self.context.generated[0])
        self.assertEqual(transport.report_bodies[0]["status"], "ok")

    async def test_the_runtime_payload_still_wins_over_the_learned_persona(self) -> None:
        """The Runtime keeps the last word: if it states a system prompt, use that."""
        from astrbot.api.provider import ProviderRequest

        await self._plugin()
        await self._handler("on_llm_request")(
            self.plugin,
            StubMessageEvent(),
            ProviderRequest(prompt="在吗", system_prompt="宿主人格"),
        )
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [
            _leased(ACTION_RENDER, payload={"prompt": "写一句", "system_prompt": "运行时指定"})
        ]
        self.assertTrue(await wait_until(lambda: len(transport.report_bodies) == 1, timeout_s=3.0))

        self.assertEqual(self.context.generated[0]["system_prompt"], "运行时指定")

    async def test_an_empty_host_system_prompt_does_not_overwrite_a_learned_one(self) -> None:
        """A turn where the host assembled nothing must not erase what was learned:
        "this turn had no prompt" and "this session never has one" are different."""
        from astrbot.api.provider import ProviderRequest

        await self._plugin()
        await self._handler("on_llm_request")(
            self.plugin, StubMessageEvent(), ProviderRequest(prompt="在吗", system_prompt="宿主人格")
        )
        await self._handler("on_llm_request")(
            self.plugin, StubMessageEvent(), ProviderRequest(prompt="再说一句", system_prompt="")
        )
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_RENDER, payload={"prompt": "写一句"})]
        self.assertTrue(await wait_until(lambda: len(transport.report_bodies) == 1, timeout_s=3.0))

        self.assertEqual(self.context.generated[0]["system_prompt"], "宿主人格")

    async def test_lease_request_declares_capabilities_and_ttl(self) -> None:
        await self._plugin(outbox_max_actions_per_poll=3, outbox_lease_ttl_ms=45000)
        transport = StubRuntimeTransport.instances[-1]

        self.assertTrue(await wait_until(lambda: bool(transport.lease_requests), timeout_s=3.0))

        request = transport.lease_requests[0]
        self.assertEqual(request.max_actions, 3)
        self.assertEqual(request.lease_ttl_ms, 45000)
        self.assertEqual(request.capabilities, (ACTION_RENDER, ACTION_SEND))

    async def test_unknown_action_type_is_skipped_and_reported(self) -> None:
        await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased("teleport", payload={})]

        self.assertTrue(await wait_until(lambda: bool(transport.report_bodies), timeout_s=3.0))

        self.assertEqual(transport.report_bodies[0]["status"], "skipped")
        self.assertIn("unsupported_action_type", transport.report_bodies[0]["error"])

    # -- lifecycle ---------------------------------------------------------

    async def test_terminate_cancels_background_work_and_is_idempotent(self) -> None:
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        tasks = list(plugin._tasks)
        self.assertTrue(tasks)

        await plugin.terminate()

        for task in tasks:
            self.assertTrue(task.done())
        self.assertEqual(plugin._tasks, [])
        self.assertIsNone(plugin._queue)
        self.assertIsNone(plugin._bridge)
        self.assertIsNone(plugin._transport)
        self.assertTrue(transport.closed)
        await plugin.terminate()  # second call must be a no-op
        self.assertTrue(plugin._stopped)

    async def test_terminate_lets_an_authorized_send_finish_and_report(self) -> None:
        """The bounded graceful window is what keeps unload from duplicating a send.

        A delivery that the Runtime already authorized is irreversible: cancelling
        it mid-flight leaves a message that may have reached the user with no
        result report, and the Runtime -- which only knows what the adapter tells
        it -- would eventually hand the same action out again.
        """
        self.context.send_delay_s = 0.3
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.actions = [_leased(ACTION_SEND, payload={"text": "在忙吗"})]
        transport.authorize_decision = AuthorizeDecision(authorized=True)

        self.assertTrue(await wait_until(lambda: self.context.send_started.is_set(), timeout_s=3.0))

        await plugin.terminate()

        self.assertEqual(len(self.context.sent), 1, "an authorized delivery must not be cancelled")
        self.assertTrue(transport.report_bodies, "the finished delivery must be reported")
        self.assertEqual(transport.report_bodies[0]["status"], "ok")
        self.assertEqual(transport.report_bodies[0]["result"]["sent"], True)

    async def test_terminate_never_revives_the_adapter(self) -> None:
        """A hook that arrives after unload must not start new workers.

        AstrBot keeps dispatching to a plugin until the reload completes, so a
        late message used to be able to wire up a fresh transport and outbox loop
        *after* terminate had already cleared them: background work that nothing
        would ever cancel again.
        """
        plugin = await self._plugin()
        await plugin.terminate()
        created = len(StubRuntimeTransport.instances)

        await self._handler("on_message_observed")(plugin, StubMessageEvent())
        request = SimpleNamespace(extra_user_content_parts=[])
        await self._handler("on_llm_request")(plugin, StubMessageEvent(), request)
        await self._handler("on_after_message_sent")(
            plugin,
            StubMessageEvent(result_text="我在听"),
        )

        self.assertEqual(len(StubRuntimeTransport.instances), created)
        self.assertIsNone(plugin._queue)
        self.assertIsNone(plugin._bridge)
        self.assertIsNone(plugin._transport)
        self.assertIsNone(plugin._outbox)
        self.assertEqual(plugin._tasks, [])
        self.assertEqual(request.extra_user_content_parts, [])

    async def test_status_reports_a_terminated_adapter_as_terminated(self) -> None:
        plugin = await self._plugin()
        await plugin.terminate()

        text = await plugin._status_text()

        self.assertIn("state: terminated", text)

    async def test_status_text_reports_counters_without_the_token(self) -> None:
        plugin = await self._plugin(runtime_token="top-secret-token")
        text = await plugin._status_text()
        self.assertIn("adapter_id: test-adapter", text)
        self.assertIn("token: configured", text)
        self.assertNotIn("top-secret-token", text)

    async def test_status_text_reports_the_cognition_levels(self) -> None:
        """Patch v0.2: the report says which level is live and how much is deferred."""
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.health = {
            "semantic_provider": {"provider": "disabled", "available": False},
            "semantics": {"unresolved": 7, "by_status": {"unresolved": 7}},
        }
        text = await plugin._status_text()
        self.assertIn("semantic_provider: disabled (available=False)", text)
        self.assertIn("7 unresolved", text)
        # Deferral is normal operation, and the wording must not read as a fault.
        self.assertIn("normal", text)

    async def test_status_reads_the_runtimes_actual_field_name(self) -> None:
        """Regression: the Runtime reports ``provider``, not ``name``.

        The first version of this probe read ``name``, which the real endpoint
        never sends, so a live deployment printed ``unknown`` while every stub
        test passed. The stub default now mirrors the real payload.
        """
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.health = {"semantic_provider": {"provider": "remote_api", "available": True}}
        text = await plugin._status_text()
        self.assertIn("semantic_provider: remote_api (available=True)", text)
        self.assertNotIn("unknown", text)

    async def test_status_tolerates_the_legacy_name_field(self) -> None:
        """Accept the old key too, so a version skew mislabels nothing."""
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.health = {"semantic_provider": {"name": "remote_api", "available": True}}
        text = await plugin._status_text()
        self.assertIn("semantic_provider: remote_api (available=True)", text)

    async def test_status_default_stub_matches_the_real_health_shape(self) -> None:
        """A stub with the wrong field names could hide a protocol mismatch."""
        plugin = await self._plugin()
        text = await plugin._status_text()
        self.assertIn("semantic_provider: disabled (available=False)", text)
        self.assertNotIn("unknown", text)

    async def test_status_stays_usable_when_the_runtime_is_down(self) -> None:
        """The probe is advisory: no /health must never break the command."""
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.health = None
        text = await plugin._status_text()
        self.assertIn("cognition: unavailable", text)
        self.assertIn("adapter_id: test-adapter", text)

    async def test_status_survives_a_raising_health_probe(self) -> None:
        plugin = await self._plugin()
        transport = StubRuntimeTransport.instances[-1]
        transport.health_error = RuntimeError("boom")
        text = await plugin._status_text()
        self.assertIn("cognition: unavailable", text)

    async def test_status_command_returns_a_plain_result(self) -> None:
        await self._plugin()
        results = [
            result
            async for result in self.plugin.companion_runtime_status(StubMessageEvent())
        ]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get_plain_text().splitlines()[0], "companion Runtime adapter")

    async def test_no_task_is_left_running_after_terminate(self) -> None:
        plugin = await self._plugin()
        outbox_task = plugin._tasks[0]
        await plugin.terminate()
        self.assertTrue(outbox_task.cancelled() or outbox_task.done())
        await asyncio.sleep(0)  # let cancellation settle


def _leased(
    action_type: str,
    *,
    payload: dict[str, Any] | None = None,
    lease_ttl_ms: int = 30000,
) -> LeasedAction:
    """Build a leased action as the Runtime would send it."""
    return LeasedAction(
        action_id="act_1",
        action_type=action_type,
        lease_id="lease_1",
        session=SESSION,
        attempt_id="attempt_1",
        lease_ttl_ms=lease_ttl_ms,
        payload=payload or {},
    )


if __name__ == "__main__":
    unittest.main()
