"""Thin AstrBot adapter for the endogenous companion Runtime.

This plugin is deliberately narrow. It does not hold psychology, memory, or
intent; the Runtime does. Its whole job is three mechanical things:

1. Observe messages and report them to the Runtime asynchronously (fail-open).
2. In ``on_llm_request``, fetch the Runtime's *current* context inside a strict
   short deadline and inject it as a temporary ``TextPart``
   (``mark_as_temp()``), so hidden psychological context never reaches permanent
   conversation history.
3. Consume the Runtime outbox under lease: ``render`` composes text with the
   session's current AstrBot provider, ``send`` delivers a message *only after*
   the Runtime authorizes it at the moment of delivery.

Failure policy: observation, injection, and reporting fail open -- a Runtime
outage must be invisible to AstrBot users. Delivery fails closed -- a message is
never sent without a live lease and a positive authorization.

No credential ships with this plugin. The optional shared token comes from the
plugin config or the ``COMPANION_RUNTIME_TOKEN`` environment variable, and
AstrBot's own provider API keys are never read, stored, or forwarded.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .astrbot_executor import AstrBotActionExecutor
from .companion_runtime.bridge import ContextBridge
from .companion_runtime.coerce import as_str
from .companion_runtime.http_client import (
    AiohttpRuntimeTransport,
    RuntimeTransportError,
    action_report_body,
)
from .companion_runtime.outbox import OutboxConsumer
from .companion_runtime.protocol import (
    EVENT_ASSISTANT_MESSAGE,
    EVENT_USER_MESSAGE,
    TRIGGER_LLM_REQUEST,
    TRIGGER_MESSAGE,
    ActionReport,
    ContextRequest,
    EventEnvelope,
    EventRecord,
    truncate_error,
)
from .companion_runtime.retry_queue import BoundedRetryQueue, QueueItem
from .companion_runtime.settings import OBSERVE_MODE_ALL, SHUTDOWN_GRACE_S, Settings

try:  # the documented import path for provider-facing content parts
    from astrbot.api.event.filter import CustomFilter
except ImportError:  # pragma: no cover - defensive
    from astrbot.core.star.filter.custom_filter import CustomFilter  # type: ignore

try:  # documented since AstrBot v4.24.0 (``TextPart.mark_as_temp``)
    from astrbot.core.agent.message import TextPart
except ImportError:  # pragma: no cover - defensive
    TextPart = None  # type: ignore[assignment]

#: Queue payload discriminators.
OP_EVENTS = "events"
OP_ACTION_RESULT = "action_result"

#: Bounded per-session memory of the last reported event id.
LAST_EVENT_ID_CACHE = 64

#: Bounded per-session memory of the host's assembled system prompt. One persona per
#: session at most, so this only has to cover a fleet's worth of sessions.
HOST_SYSTEM_PROMPT_CACHE = 64

#: How many times a failing start is retried before the adapter gives up quietly.
MAX_START_ATTEMPTS = 3

#: How many unknown sessions may trigger an automatic provision request.
AUTO_PROVISION_LIMIT = 64

#: Event extra that records an assistant turn already reported for this message.
ASSISTANT_REPORTED_EXTRA = "companion_runtime_assistant_reported"

#: Event extra where AstrBot records what ``send_message_to_user`` actually delivered in
#: this session. Its own message tool writes it and its respond stage reads it to avoid
#: sending the same text twice.
SENT_PLAIN_TEXTS_EXTRA = "_send_message_to_user_current_session_plain_texts"

#: AstrBot's plugin whitelist. A non-wildcard list is applied to every handler
#: lookup, so a plugin missing from it keeps loading and keeps running its
#: background workers while none of its hooks ever fire.
PLUGIN_WHITELIST_KEY = "plugin_set"

#: AstrBot's ``MessageType`` values mapped onto the plain vocabulary the Runtime
#: protocol documents. The enum's raw values (``FriendMessage`` and friends) are
#: AstrBot's spelling of platform message classes, not a chat scope, so passing
#: them through would leave the Runtime with nothing it can reason about.
MESSAGE_TYPE_NAMES = {
    "friendmessage": "private",
    "groupmessage": "group",
    "othermessage": "other",
}


class _ObservationScopeFilter(CustomFilter):
    """Pass only for messages AstrBot itself already treats as wake events.

    AstrBot sets ``is_at_or_wake_command`` exclusively for genuine wake
    conditions (wake prefix, @bot, @all, reply-to-bot, private chat). Plugin
    listeners never set it. Gating on it means this adapter can never turn an
    ordinary group message into a wake event, nor push non-wake traffic through
    the remaining pipeline stages.

    ``observe_all`` is published by the *live* adapter instance (see
    ``CompanionRuntimePlugin._apply_observation_scope``). It stays ``False``
    whenever no adapter is running -- never started, disabled, unable to start,
    or already terminated -- because this filter is the one thing that can widen
    AstrBot's own pipeline, and widening it for a dead adapter would be a
    behaviour change nobody asked for. See ``observe_mode`` in ``README.md``.
    """

    observe_all: bool = False

    def filter(self, event: AstrMessageEvent, cfg: Any) -> bool:
        """Return whether the adapter should observe this event."""
        if type(self).observe_all:
            return True
        return bool(getattr(event, "is_at_or_wake_command", False))


def _message_type_name(event: AstrMessageEvent) -> str:
    """Return ``private`` / ``group`` / ``other`` for an event's message type.

    Args:
        event: The AstrBot event being reported.

    Returns:
        The protocol's plain chat-scope word, or the raw (lowercased) value when
        AstrBot reports a message class this adapter does not know.
    """
    try:
        message_type = event.get_message_type()
    except Exception:
        return ""
    value = getattr(message_type, "value", message_type)
    raw = as_str(value).strip().lower()
    return MESSAGE_TYPE_NAMES.get(raw, raw)


def _event_is_stopped(event: AstrMessageEvent) -> bool:
    """Return whether another handler already stopped this event.

    AstrBot's ``AstrMessageEvent.is_stopped()`` is the documented check, but the
    adapter tolerates hosts that expose only the ``stopped`` attribute.

    Args:
        event: The AstrBot event about to be reported.

    Returns:
        ``True`` when the event must not be treated as a user turn.
    """
    checker = getattr(event, "is_stopped", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return bool(getattr(event, "stopped", False))


@dataclass
class _RuntimeTarget:
    """One Runtime this adapter talks to.

    A target owns its own HTTP client, context cache and outbox poller, because
    every one of those is scoped to a single Runtime: the cache is keyed by
    session and the poller leases that instance's outbox. Sessions that are not
    routed anywhere share the default target.
    """

    transport: Any
    bridge: Any
    outbox: Any = None
    url: str = ""


@dataclass
class _InputBurst:
    """One session's in-flight burst of consecutive user messages.

    Every request of the burst appends its own text and then races to be the
    last one standing: the handler whose ``generation`` is still current after
    the quiet window elapses answers with the merged text, and every earlier
    handler yields with ``event.stop_event()``. Nothing is cancelled explicitly
    -- an older handler always wakes up and sees that a newer message took over.
    """

    parts: list[str] = field(default_factory=list)
    generation: int = 0
    deadline: float = 0.0


class CompanionRuntimePlugin(Star):
    """Host-side thin adapter for the companion Runtime."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        """Create the plugin; no background work starts until ``initialize``."""
        super().__init__(context)
        self.config = config or {}
        self._settings = Settings.from_mapping(self.config)
        self._started = False
        self._stopped = False
        """Set by ``terminate``: the adapter is finished and never starts again."""
        self._gave_up = False
        """Set when worker construction failed too often to keep retrying."""
        self._start_failures = 0
        self._transport: AiohttpRuntimeTransport | None = None
        self._queue: BoundedRetryQueue | None = None
        self._bridge: ContextBridge | None = None
        self._outbox: OutboxConsumer | None = None
        self._executor: AstrBotActionExecutor | None = None
        #: Every Runtime this adapter talks to, keyed by base URL. The three
        #: attributes above mirror the *default* target for single-Runtime
        #: deployments, which is also what the tests and ``/companion_runtime``
        #: have always read.
        self._targets: dict[str, _RuntimeTarget] = {}
        #: Session -> Runtime URL learned from the fleet's routing registry. This is
        #: what makes "add a person" a fleet-side action: the adapter picks the new
        #: route up on its own and AstrBot never restarts.
        self._registry_routes: dict[str, str] = {}
        self._registry_transport: AiohttpRuntimeTransport | None = None
        self._route_sync_pending = False
        #: True once the registry has answered at least once. Until then an unknown
        #: session falls back to the default Runtime; afterwards it waits, because a
        #: registry that is up and silent means "this person is new", not "unknown".
        self._registry_seen = False
        #: Sessions already asked about, so one stranger cannot spam the fleet.
        self._provision_requested: set[str] = set()
        #: Empty message events seen (AstrBot's notice-to-message conversions).
        self._skipped_empty = 0
        self._tasks: list[asyncio.Task[None]] = []
        self._last_event_ids: OrderedDict[str, str] = OrderedDict()
        #: Session -> the system prompt the host assembled for that session's last
        #: normal turn. A render is a bare ``llm_generate`` call and AstrBot hands it
        #: **no** persona at all (measured: a chat turn carries ~4210 characters of
        #: assembled system prompt, a render none), so the only way a proactive
        #: message can be written by the same character is to remember what the host
        #: produced for a normal turn and pass it on. Best effort: empty means "render
        #: exactly as before".
        self._host_system_prompts: OrderedDict[str, str] = OrderedDict()
        self._host_system_prompt_logged: set[str] = set()
        self._injection_warnings: set[str] = set()
        #: Session -> the burst of consecutive messages currently being merged.
        #: Entries live only while a request waits out ``input_debounce_ms``.
        self._input_bursts: dict[str, _InputBurst] = {}
        # Take ownership of the shared scope filter straight away: a previous
        # instance may have left it widened, and AstrBot only reloads plugin
        # instances, it never resets module level state for them.
        self._apply_observation_scope()
        for issue in self._settings.issues:
            self.logger.warning("companion_runtime config: %s", issue)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start background workers (called by AstrBot after loading)."""
        self._start()
        self._warn_if_whitelisted_out()

    async def terminate(self) -> None:
        """Stop background workers and release resources.

        Called when the plugin is disabled, unloaded, or reloaded. Idempotent,
        safe to call even if :meth:`initialize` never ran, and terminal: once it
        returns, a message hook that arrives late can no longer start workers,
        because AstrBot keeps dispatching to a plugin it has already unloaded
        until the reload finishes.
        """
        self._stopped = True
        self._started = False
        self._apply_observation_scope()

        targets = self._targets
        outboxes = [target.outbox for target in targets.values() if target.outbox is not None]
        if outboxes:
            # Bounded graceful stop. Leases that are already in flight are still
            # the Runtime's actions, and cancelling one *after* the Runtime
            # authorized an irreversible send is exactly how the same proactive
            # message ends up delivered twice, so give them a short window to
            # finish and report before the tasks are cancelled. Every routed
            # Runtime gets the same window, in parallel, so the total wait stays
            # ``SHUTDOWN_GRACE_S``.
            for outbox in outboxes:
                outbox.request_stop()
            idle = await asyncio.gather(
                *(outbox.wait_idle(SHUTDOWN_GRACE_S) for outbox in outboxes),
            )
            if not all(idle):
                self.logger.warning(
                    "companion_runtime stopped with actions still in flight after %.0fs; "
                    "an authorized delivery that was already on the wire may finish "
                    "unreported, and the Runtime can then only recover it from its own "
                    "lease deadline",
                    SHUTDOWN_GRACE_S,
                )

        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Cleared only now: the graceful window above lets an in-flight delivery
        # finish and *report*, and the report needs its target to still resolve.
        self._targets = {}
        queue, self._queue = self._queue, None
        bridges = [target.bridge for target in targets.values()]
        transports = [target.transport for target in targets.values()]
        registry, self._registry_transport = self._registry_transport, None
        self._registry_routes.clear()
        bridge, self._bridge = self._bridge, None
        transport, self._transport = self._transport, None
        self._outbox = None
        self._executor = None
        self._last_event_ids.clear()

        try:
            if queue is not None:
                await queue.stop()
            for item in bridges:
                await item.aclose()
            for item in transports:
                await item.aclose()
            if registry is not None:
                await registry.aclose()
        except Exception:
            self.logger.warning("companion_runtime shutdown was not clean", exc_info=True)

    def _apply_observation_scope(self) -> None:
        """Publish this adapter's observation mode to the shared scope filter.

        The filter instance is created once by AstrBot while the decorator runs
        and is shared by every event, so the flag has to be written on every state
        change rather than only on a successful start: a stale ``True`` would keep
        widening AstrBot's own pipeline -- every non-wake group message would be
        marked as a wake event -- on behalf of an adapter that is disabled, has
        failed to start, or is already gone.
        """
        _ObservationScopeFilter.observe_all = bool(
            self._started
            and not self._stopped
            and self._settings.usable
            and self._settings.observe_mode == OBSERVE_MODE_ALL,
        )

    def _start(self) -> None:
        """Wire up workers. Synchronous and idempotent.

        Synchronous on purpose: observers and hooks must be able to lazily start
        the adapter without awaiting anything on the message path. Construction
        happens before any task is created, so a failure here leaks nothing. A
        terminated adapter is never revived.
        """
        if self._started or self._stopped or self._gave_up:
            return
        self._started = True
        if not self._settings.usable:
            self.logger.info(
                "companion_runtime adapter is inactive (disabled or unusable config)",
            )
            return
        try:
            executor = AstrBotActionExecutor(
                context=self.context,
                log=self.logger,
                system_prompt_for=self._host_system_prompt_for,
                segment_mode=self._settings.outbox_segmented_reply,
                segment_max_parts=self._settings.outbox_segment_max_parts,
            )
            queue = BoundedRetryQueue(
                sender=self._deliver,
                max_items=self._settings.queue_max_items,
                max_attempts=self._settings.queue_max_attempts,
                max_age_s=self._settings.queue_max_age_s,
                base_backoff_s=self._settings.queue_base_backoff_s,
                max_backoff_s=self._settings.queue_max_backoff_s,
                send_timeout_s=self._settings.queue_send_timeout_s,
                log=self.logger,
            )
            # One client, one context cache and one outbox poller per Runtime: the
            # cache is keyed by session and the poller leases that instance's
            # outbox, so neither can be shared across targets.
            targets: dict[str, _RuntimeTarget] = {}
            for url in self._settings.targets:
                transport = AiohttpRuntimeTransport(
                    settings=self._settings,
                    base_url=url,
                    log=self.logger,
                )
                targets[url] = _RuntimeTarget(
                    transport=transport,
                    bridge=ContextBridge(
                        transport=transport,
                        settings=self._settings,
                        log=self.logger,
                    ),
                    url=url,
                )
            consumers: list[OutboxConsumer] = []
            if self._settings.outbox_enabled:
                for url, target in targets.items():
                    consumer = OutboxConsumer(
                        transport=target.transport,
                        executor=executor,
                        reporter=self._target_reporter(url),
                        settings=self._settings,
                        log=self.logger,
                    )
                    target.outbox = consumer
                    consumers.append(consumer)
        except Exception:
            # The reason is logged *here*, with the traceback: ``_note_start_failure``
            # counts attempts and reports them, but on its own it prints only a sentence.
            # A construction failure whose cause is invisible in the log is
            # indistinguishable from a config problem, and the operator has nothing to
            # act on.
            self.logger.warning("companion_runtime adapter construction failed", exc_info=True)
            self._note_start_failure(
                "adapter could not be constructed; AstrBot behaviour is unchanged",
            )
            return

        default = self._settings.base_url
        self._targets = targets
        self._transport = targets[default].transport
        self._executor = executor
        self._queue = queue
        self._bridge = targets[default].bridge
        self._outbox = targets[default].outbox
        self._apply_observation_scope()

        try:
            for consumer in consumers:
                self._tasks.append(
                    asyncio.create_task(consumer.run(), name="companion-runtime-outbox"),
                )
            if self._settings.registry_configured:
                self._registry_transport = AiohttpRuntimeTransport(
                    settings=self._settings,
                    base_url=self._settings.route_registry_url,
                    log=self.logger,
                )
                self._tasks.append(
                    asyncio.create_task(self._route_sync_loop(), name="companion-runtime-routes"),
                )
            queue.start()
        except Exception:
            # Nothing is running yet, so dropping the wiring is enough cleanup.
            self._tasks.clear()
            self._targets = {}
            self._registry_transport = None
            self._transport = None
            self._executor = None
            self._queue = None
            self._bridge = None
            self._outbox = None
            self._note_start_failure("background workers could not be scheduled")
            return

        self.logger.info(
            "companion_runtime adapter started (adapter_id=%s, base_url=%s, "
            "observe_mode=%s, outbox=%s, debounce=%s, targets=%s, registry=%s)",
            self._settings.adapter_id,
            self._settings.base_url,
            self._settings.observe_mode,
            "on" if consumers else "off",
            # Logged because a debounce that silently did not load looks exactly
            # like a debounce that loaded and is working: both are quiet.
            "%dms" % int(self._settings.input_debounce_s * 1000.0)
            if self._settings.input_debounce_s > 0.0
            else "off",
            ", ".join(self._settings.targets),
            self._settings.route_registry_url or "off",
        )

    def _warn_if_whitelisted_out(self) -> None:
        """Warn when AstrBot's plugin whitelist leaves this plugin unwired.

        ``star_handlers_registry.get_handlers_by_event_type`` drops every handler
        of a plugin that is missing from ``plugin_set`` before the waking check
        runs. An omitted plugin therefore still loads, still logs "adapter
        started", and still polls the outbox -- while observation, injection and
        assistant reporting are all silently dead. One warning at startup is the
        only cheap defence against a failure that otherwise looks like a healthy
        adapter that happens to see nothing.
        """
        name = as_str(getattr(self, "name", ""))
        if not name:
            return
        try:
            whitelist = self.context.get_config().get(PLUGIN_WHITELIST_KEY)
        except Exception:
            return
        if not isinstance(whitelist, (list, tuple)):
            return
        entries = [as_str(entry) for entry in whitelist]
        if entries == ["*"] or name in entries:
            return
        self.logger.warning(
            "companion_runtime is not in AstrBot's plugin whitelist (plugin_set=%s), "
            "so AstrBot drops every handler of this plugin before the waking check: "
            "messages are neither observed nor injected, and no assistant message is "
            "reported, even though the adapter logs that it started. Add %r to "
            "plugin_set (AstrBot WebUI -> configuration) or set it to ['*'].",
            entries,
            name,
        )

    def _note_start_failure(self, message: str) -> None:
        """Record a failed start, giving up after a few attempts.

        Retrying forever would mean a broken adapter logging on every single
        message, so after ``MAX_START_ATTEMPTS`` the plugin stays quiet until
        AstrBot reloads it. "Gave up" is its own state: the adapter is neither
        running nor merely not started yet, and it must never widen AstrBot's
        pipeline.
        """
        self._started = False
        self._start_failures += 1
        self._gave_up = self._start_failures >= MAX_START_ATTEMPTS
        self._apply_observation_scope()
        if self._gave_up:
            self.logger.error(
                "companion_runtime %s; adapter disabled after %d attempts "
                "(reload the plugin after fixing the config)",
                message,
                self._start_failures,
            )
            return
        self.logger.warning(
            "companion_runtime %s (attempt %d/%d)",
            message,
            self._start_failures,
            MAX_START_ATTEMPTS,
        )

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    # AstrBot dispatches plugin handlers in *descending* priority and breaks the whole
    # chain once an event is stopped (``pipeline/process_stage/method/star_request.py``
    # checks ``event.is_stopped()`` before every handler). Running the observation at a
    # negative priority therefore means "another plugin already decided this message is
    # not a turn" is respected rather than fought: a word filter that drops a message
    # must also keep it out of the Runtime. Measured on a real QQ: a tester set their
    # client's auto-reply to 「。」, so every message the character sent came back as
    # ``[自动回复] 。``, each one was reported to the Runtime as a user turn, and she
    # answered ~1300 of them in a single morning.
    @filter.custom_filter(_ObservationScopeFilter, priority=-100)
    async def on_message_observed(self, event: AstrMessageEvent) -> None:
        """Report an observed user message to the Runtime.

        Runs on AstrBot's message path, so every failure is swallowed and only
        logged at debug level: a Runtime outage must never change how AstrBot
        handles the message.
        """
        try:
            if _event_is_stopped(event):
                # Defence in depth: the low priority above already keeps the handler out
                # of the chain once something stopped the event, and this keeps the
                # guarantee true even if the host ever dispatches differently.
                return
            self._start()
            queue = self._queue
            if queue is None:
                return
            wake = bool(getattr(event, "is_at_or_wake_command", False))
            if not wake and self._settings.observe_mode != OBSERVE_MODE_ALL:
                # Defence in depth: the scope filter already keeps non-wake
                # messages out, and this keeps the guarantee true even if the
                # host ever evaluates handler filters differently.
                return
            session = event.unified_msg_origin
            text = as_str(getattr(event, "message_str", "")).strip()
            if not text:
                # AstrBot turns OneBot *notice* events (a poke, a friend request, a
                # group membership change) into message events with an empty body and
                # a generated message id -- measured on a real QQ: seven pokes became
                # seven "user said nothing" reports, and the Runtime filed an empty
                # ``用户说：`` fact for each. A user message with no text carries
                # nothing to remember, so it is not reported; if a poke ever deserves
                # cognition it needs its own event kind in the protocol.
                self._skipped_empty += 1
                self.logger.debug(
                    "companion_runtime skipped an empty message event (notice?) in %s",
                    session,
                )
                return
            record = EventRecord(
                kind=EVENT_USER_MESSAGE,
                session=session,
                text=text,
                platform=as_str(event.get_platform_name()),
                message_type=_message_type_name(event),
                sender_id=as_str(event.get_sender_id()),
                sender_name=as_str(event.get_sender_name()),
                self_id=as_str(event.get_self_id()),
                group_id=as_str(event.get_group_id()),
                message_id=as_str(getattr(event.message_obj, "message_id", "")),
                wake=wake,
                # A user message immediately pauses endogenous dispatch on the
                # Runtime side (entry barrier), so the proactive system can never
                # speak before the Runtime has seen what the user just said.
                preempts_proactive=True,
            )
            self._remember_event_id(session, record.event_id)
            self._enqueue_event(record, target=self._route_target(session))
            bridge = self._bridge_for(session)
            if bridge is not None:
                bridge.prefetch(self._context_request(event, trigger=TRIGGER_MESSAGE))
        except Exception:
            self.logger.debug("companion_runtime message observation failed", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(
        self,
        event: AstrMessageEvent,
        response: LLMResponse,
    ) -> None:
        """Report the assistant turn from the model's own answer.

        AstrBot's ``respond`` stage returns straight after ``send_streaming`` and
        its ``result_decorate`` stage returns before the pre-send hook when the
        result is a ``STREAMING_RESULT``, so with streaming on -- AstrBot's
        default -- neither pre-send nor post-send hook ever fires for a reply.
        The agent runner calls this hook exactly once per run, in both delivery
        modes, with the text the model finally produced: under streaming that is
        the text the user received, and it is a better record for cognition than
        the rendered chain would be (a long reply is delivered as a rendered
        *image*, which carries no words).

        The one case where that text is *not* what the user received is a turn that
        delivered through ``send_message_to_user`` - then the completion is the model's
        narration about the message it just sent, and the delivered text is what counts
        (see :meth:`_delivered_text`).

        The turn is marked so ``after_message_sent`` cannot report it twice on a
        host that does run that hook.
        """
        try:
            text = self._delivered_text(event) or as_str(
                getattr(response, "completion_text", "")
            ).strip()
            if text and self._report_assistant(event, text):
                self._mark_assistant_reported(event)
        except Exception:
            self.logger.debug("companion_runtime assistant report failed", exc_info=True)

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent) -> None:
        """Report the message AstrBot actually delivered to the user.

        This covers what the LLM-response hook cannot see: a reply that never
        went through the agent at all, such as a command's output or another
        plugin's result. Once a turn has an LLM response, that response is the
        report, so the delivered chain is not reported a second time.
        """
        try:
            if self._assistant_reported(event):
                return
            text = self._result_text(event)
            if text:
                self._report_assistant(event, text)
        except Exception:
            self.logger.debug("companion_runtime assistant report failed", exc_info=True)

    def _report_assistant(self, event: AstrMessageEvent, text: str) -> bool:
        """Queue one assistant message; return whether it was queued."""
        self._start()
        if self._queue is None or not self._settings.report_assistant_messages:
            return False
        session = event.unified_msg_origin
        record = EventRecord(
            kind=EVENT_ASSISTANT_MESSAGE,
            session=session,
            text=text,
            platform=as_str(event.get_platform_name()),
            message_type=_message_type_name(event),
            sender_id=as_str(event.get_self_id()),
            sender_name="bot",
            self_id=as_str(event.get_self_id()),
            group_id=as_str(event.get_group_id()),
            wake=bool(getattr(event, "is_at_or_wake_command", False)),
        )
        self._remember_event_id(session, record.event_id)
        self._enqueue_event(record, target=self._route_target(session))
        return True

    def _assistant_reported(self, event: AstrMessageEvent) -> bool:
        """Return whether this turn's assistant message is already reported."""
        try:
            return bool(event.get_extra(ASSISTANT_REPORTED_EXTRA, False))
        except Exception:
            return False

    def _mark_assistant_reported(self, event: AstrMessageEvent) -> None:
        """Remember that this turn's assistant message is already reported."""
        try:
            event.set_extra(ASSISTANT_REPORTED_EXTRA, True)
        except Exception:
            self.logger.debug("companion_runtime could not mark the reported turn")

    # ------------------------------------------------------------------
    # input debounce
    # ------------------------------------------------------------------

    @filter.on_waiting_llm_request(priority=100)
    async def on_waiting_llm_request(self, event: AstrMessageEvent) -> None:
        """Answer only the last message of a burst (``input_debounce_ms``).

        This hook exists so the wait can happen *before* AstrBot takes its per-session
        lock: ``internal.py`` calls ``OnWaitingLLMRequestEvent`` and only then enters
        ``session_lock_manager.acquire_lock``, and ``on_llm_request`` runs inside that
        lock. Waiting there could therefore never work — the next message of a burst
        cannot reach its own handler until this turn has finished, so the wait always
        times out and answers anyway. Measured on a real beta: two messages two seconds
        apart produced two answers in a row.

        The superseded events are stopped; their text is merged into the surviving
        event's ``message_str`` (AstrBot builds ``req.prompt`` from it in
        ``collect_initial_request``) rather than dropped. An AstrBot conversation is
        persisted as a user/assistant *pair* once the LLM has answered, so a request
        stopped here never reaches history: cancelling the earlier events without
        merging their text would silently remove those words from the model's view even
        though the Runtime did observe them.
        """
        window = self._settings.input_debounce_s
        if window <= 0.0:
            return
        try:
            superseded = await self._coalesce_burst(event, window)
        except Exception:
            self.logger.debug("companion_runtime input debounce failed", exc_info=True)
            return
        if superseded:
            event.stop_event()

    async def _coalesce_burst(self, event: AstrMessageEvent, window: float) -> bool:
        """Wait out the quiet window and report whether a newer message won.

        Args:
            event: The event being processed.
            window: Quiet period in seconds.

        Returns:
            ``True`` when a later message of the same session superseded this one.
        """
        key = as_str(event.unified_msg_origin)
        now = asyncio.get_running_loop().time()
        burst = self._input_bursts.get(key)
        if burst is None:
            self._prune_bursts(now)
            burst = _InputBurst()
            self._input_bursts[key] = burst

        burst.generation += 1
        generation = burst.generation
        burst.deadline = now + window
        text = as_str(getattr(event, "message_str", "")).strip()
        if text:
            burst.parts.append(text)
            limit = self._settings.input_debounce_max_chars
            if limit > 0:
                # Keep the newest parts: the oldest is the cheapest to drop and
                # the tail is what the user is still waiting on.
                while len(burst.parts) > 1 and sum(len(part) for part in burst.parts) > limit:
                    burst.parts.pop(0)

        while True:
            remaining = burst.deadline - asyncio.get_running_loop().time()
            if remaining > 0.0:
                await asyncio.sleep(remaining)
            if burst.generation == generation:
                break
            return True

        merged = "\n".join(burst.parts).strip()
        self._input_bursts.pop(key, None)
        if merged:
            event.message_str = merged
        return False

    def _prune_bursts(self, now: float) -> None:
        """Drop bursts whose window closed long ago.

        The last handler of a burst normally removes its own entry; this covers
        the one that never resumed, for instance on shutdown.
        """
        stale = [key for key, burst in self._input_bursts.items() if burst.deadline < now - 60.0]
        for key in stale:
            self._input_bursts.pop(key, None)

    # ------------------------------------------------------------------
    # context injection
    # ------------------------------------------------------------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Inject the Runtime's current context as a temporary content part.

        The Runtime's hidden context is explanatory, never authoritative: it is
        appended after the user's own words as an extra provider-facing content
        part, marked temporary so it is dropped instead of being persisted.
        """
        try:
            self._start()
            self._remember_host_system_prompt(event, req)
            await self._inject_context(event, req)
        except Exception:
            self.logger.debug("companion_runtime context injection failed", exc_info=True)

    async def _inject_context(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Fetch and append the Runtime context block, fail-open on any problem."""
        bridge = self._bridge_for(as_str(event.unified_msg_origin))
        if bridge is None or not self._settings.inject_enabled:
            return
        if TextPart is None:
            self._warn_injection_once(
                "missing-textpart",
                "companion_runtime cannot inject context: astrbot.core.agent.message.TextPart "
                "is unavailable on this AstrBot version; injection is disabled",
            )
            return

        text = await bridge.text_for_llm_request(
            self._context_request(event, trigger=TRIGGER_LLM_REQUEST),
        )
        if not text:
            return

        parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            self._warn_injection_once(
                "unsupported-request",
                "companion_runtime cannot inject context: this AstrBot version exposes no "
                "ProviderRequest.extra_user_content_parts",
            )
            return

        try:
            part = TextPart(text=text)
            mark_as_temp = getattr(part, "mark_as_temp", None)
            if not callable(mark_as_temp):
                # Without mark_as_temp the hidden context could be written into
                # permanent history, which the architecture forbids. Skip instead.
                self._warn_injection_once(
                    "missing-mark-as-temp",
                    "companion_runtime cannot inject context: TextPart.mark_as_temp() is "
                    "unavailable (requires AstrBot >= 4.24); injection is disabled",
                )
                return
            parts.append(mark_as_temp())
        except Exception:
            self.logger.debug("companion_runtime could not append the context part", exc_info=True)

    # ------------------------------------------------------------------
    # Runtime calls
    # ------------------------------------------------------------------

    async def _deliver(self, item: QueueItem) -> None:
        """Deliver one queued request to the Runtime (called by the retry queue).

        Raises:
            RuntimeTransportError: When the request cannot be delivered, which
                makes the queue retry it with backoff.
        """
        transport = self._transport_for(item)
        if transport is None:
            raise RuntimeTransportError("Runtime transport is not available")
        operation = as_str(item.payload.get("op"))
        body = item.payload.get("body")
        if not isinstance(body, dict):
            raise RuntimeTransportError(f"queue item {item.idempotency_key} carries no body")
        timeout_s = self._settings.request_timeout_s
        if operation == OP_EVENTS:
            await transport.post_events(body, timeout_s=timeout_s)
        elif operation == OP_ACTION_RESULT:
            await transport.report_action(body, timeout_s=timeout_s)
        else:
            raise RuntimeTransportError(f"unknown queue operation {operation!r}")

    def _transport_for(self, item: Any) -> Any:
        """Return the client a queue item must go to, retrying until routing is known.

        When the fleet registry has not yet answered for a session, the item carries
        no target. Raising here is deliberate: the bounded retry queue will try
        again in a moment with the freshly synced routes, whereas falling back to
        the default Runtime would file one person's words into another person's
        memory -- exactly the failure per-person routing exists to prevent.
        """
        payload = item.payload
        target = as_str(payload.get("target")).strip()
        if target:
            return self._target_transport(target)
        session = as_str(payload.get("session")).strip()
        if session:
            if self._routing_is_pending(session):
                self._request_route_sync()
                self._request_provision(session)
                raise RuntimeTransportError(
                    f"routing registry has not answered for session {session!r} yet",
                )
            return self._target_transport(self._target_url(session))
        return self._target_transport(self._settings.base_url)

    def _routing_is_pending(self, session: str) -> bool:
        """Whether ``session`` should wait for the registry instead of falling back.

        Only true once the registry has proven reachable: a registry that never
        answered (or a deployment without one) must keep working exactly as before.
        """
        return (
            self._settings.registry_configured
            and self._registry_seen
            and not self._route_known(session)
        )

    def _request_provision(self, session: str) -> None:
        """Ask the fleet for a Runtime for an unknown session, once per session.

        The closed beta's promise is that a tester just talks to the bot; nobody
        wants to run a command per new person. Bounded on purpose: one request per
        session per process, and a cap on how many sessions may trigger it, so a
        stream of strangers cannot make the adapter hammer the fleet.
        """
        if self._registry_transport is None or not self._settings.route_auto_provision:
            return
        if not session or session in self._provision_requested:
            return
        if len(self._provision_requested) >= AUTO_PROVISION_LIMIT:
            return
        self._provision_requested.add(session)

        async def once() -> None:
            try:
                url = await self._registry_transport.provision_session(
                    session,
                    timeout_s=self._settings.request_timeout_s,
                )
                if url:
                    self.logger.info(
                        "companion_runtime fleet provisioned %s at %s", session, url,
                    )
                    await self._sync_registry_once()
            except Exception:
                self.logger.debug("companion_runtime provision request failed", exc_info=True)

        try:
            asyncio.get_running_loop().create_task(once())
        except RuntimeError:
            self._provision_requested.discard(session)

    def _target_reporter(self, url: str) -> Callable[[ActionReport], Awaitable[None]]:
        """Return a report coroutine bound to one Runtime.

        Each outbox poller must report back to the instance that leased the
        action: a report sent to the wrong Runtime leaves the action leased until
        its deadline, and the delivery it describes would be unknown to the
        Runtime that authorized it.
        """

        async def report(action_report: ActionReport) -> None:
            await self._report_action(action_report, target=url)

        return report

    async def _report_action(self, report: ActionReport, *, target: str | None = None) -> None:
        """Report an action outcome, deferring to the retry queue when needed.

        Never raises: a lost report is recovered by the bounded local queue, and
        ultimately by the Runtime's own lease expiry.
        """
        url = as_str(target).strip() or self._settings.base_url
        transport = self._target_transport(url)
        if transport is None:
            return
        body = action_report_body(report)
        try:
            await transport.report_action(body, timeout_s=self._settings.request_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.debug("companion_runtime action report deferred: %s", truncate_error(exc))
            queue = self._queue
            if queue is not None:
                queue.put(
                    {"op": OP_ACTION_RESULT, "body": body, "target": url},
                    key=(
                        f"result:{report.action_id}:"
                        f"{report.attempt_id or report.lease_id}:{report.status}"
                    ),
                )

    def _context_request(self, event: AstrMessageEvent, *, trigger: str) -> ContextRequest:
        """Build a context request for the event's session."""
        session = event.unified_msg_origin
        return ContextRequest(
            adapter_id=self._settings.adapter_id,
            session=session,
            trigger=trigger,
            platform=as_str(event.get_platform_name()),
            last_event_id=self._last_event_ids.get(session),
        )

    def _enqueue_event(self, record: EventRecord, *, target: str) -> None:
        """Queue one event report for ``target``; dropping beats blocking.

        ``session`` travels with the item so a target that was unknown at enqueue
        time can still be resolved when delivery is attempted.
        """
        queue = self._queue
        if queue is None:
            return
        envelope = EventEnvelope(adapter_id=self._settings.adapter_id, events=[record])
        queue.put(
            {"op": OP_EVENTS, "body": envelope.to_wire(), "target": target, "session": record.session},
            key=f"event:{record.event_id}",
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _target_transport(self, url: str) -> Any:
        """Return the HTTP client for ``url``, or ``None`` when it is unknown."""
        target = self._targets.get(url)
        if target is not None:
            return target.transport
        return None

    def _route_known(self, session: str) -> bool:
        """Whether any route (static or registry) covers ``session``."""
        return (
            self._settings.static_route_for(session) is not None
            or session in self._registry_routes
        )

    def _route_target(self, session: str) -> str:
        """Return the queue target for ``session``; ``""`` while routing is unknown.

        An empty target is not a fallback: :meth:`_transport_for` resolves it again
        at delivery time and asks the queue to retry until the registry answers.
        """
        if self._routing_is_pending(session):
            self._request_route_sync()
            self._request_provision(session)
            return ""
        return self._target_url(session)

    def _bridge_for(self, session: str) -> ContextBridge | None:
        """Return the context bridge that serves ``session``.

        The bridge's cache is keyed by session, so a routed session must read the
        cache of the Runtime it is routed to -- reading the default one would
        inject another instance's context into this person's turn.
        """
        target = self._targets.get(self._target_url(session))
        return target.bridge if target is not None else None

    def _target_url(self, session: str) -> str:
        """Return the Runtime URL for ``session``: static route, registry, default.

        Precedence is deliberate. An explicit ``session_routes`` entry is an
        operator's decision and always wins; the registry is a fleet's answer and
        fills the gaps; the default instance is the floor, so a deployment with
        neither behaves exactly as it did before routing existed.
        """
        routed = self._settings.static_route_for(session)
        if routed is not None:
            return routed
        return self._registry_routes.get(session, self._settings.base_url)

    def _ensure_target(self, url: str) -> _RuntimeTarget | None:
        """Build and start the bundle for a URL discovered at runtime.

        Targets are normally built once in :meth:`_start`; a registry can name a
        Runtime that did not exist then, so the same construction happens here.
        """
        cleaned = as_str(url).strip().rstrip("/")
        if not cleaned:
            return None
        existing = self._targets.get(cleaned)
        if existing is not None:
            return existing
        if not self._started or self._executor is None:
            return None
        transport = AiohttpRuntimeTransport(
            settings=self._settings,
            base_url=cleaned,
            log=self.logger,
        )
        target = _RuntimeTarget(
            transport=transport,
            bridge=ContextBridge(transport=transport, settings=self._settings, log=self.logger),
            url=cleaned,
        )
        self._targets[cleaned] = target
        if self._settings.outbox_enabled:
            consumer = OutboxConsumer(
                transport=transport,
                executor=self._executor,
                reporter=self._target_reporter(cleaned),
                settings=self._settings,
                log=self.logger,
            )
            target.outbox = consumer
            self._tasks.append(
                asyncio.create_task(consumer.run(), name="companion-runtime-outbox"),
            )
        self.logger.info("companion_runtime added Runtime target %s", cleaned)
        return target

    async def _sync_registry_once(self) -> int:
        """Read the registry and make every named Runtime reachable.

        Returns:
            How many routes the registry reported (0 when unavailable).
        """
        transport = self._registry_transport
        if transport is None:
            return 0
        try:
            routes = await transport.fetch_routes(timeout_s=self._settings.request_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            # The real client already swallows its own failures, but the sync must
            # stay fail-open for any transport implementation.
            self.logger.debug("companion_runtime registry read failed", exc_info=True)
            return 0
        if routes is None:
            # Unreachable: keep whatever we knew. Until it has answered once, unknown
            # sessions fall back to the default Runtime rather than waiting forever.
            return 0
        self._registry_seen = True
        self._registry_routes = routes
        for url in routes.values():
            self._ensure_target(url)
        return len(routes)

    async def _route_sync_loop(self) -> None:
        """Keep the registry view fresh; never let a failure escape."""
        while True:
            try:
                await self._sync_registry_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.debug("companion_runtime registry sync failed", exc_info=True)
            await asyncio.sleep(max(2.0, self._settings.route_sync_interval_s))

    def _request_route_sync(self) -> None:
        """Ask the registry once, off the message path, for a session we cannot place.

        Called when a message arrives for a session no route covers, so a person
        provisioned seconds ago works on their first message instead of waiting for
        the next periodic sync. Fire-and-forget: the message is handled with the
        routes we already have.
        """
        if self._registry_transport is None or self._route_sync_pending:
            return
        self._route_sync_pending = True

        async def once() -> None:
            try:
                await self._sync_registry_once()
            except Exception:
                self.logger.debug("companion_runtime registry lookup failed", exc_info=True)
            finally:
                self._route_sync_pending = False

        try:
            asyncio.get_running_loop().create_task(once())
        except RuntimeError:
            self._route_sync_pending = False

    def _remember_host_system_prompt(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Remember the host's assembled system prompt for this session.

        Read, never written: the host owns the persona. This exists because the
        render path gets none of it, and the fix for that has to live on the side
        that has it. Logging the first sighting per session is deliberate - "did the
        host actually put anything here?" is otherwise unanswerable from the logs,
        and a silent empty string looks exactly like a silent working cache.
        """
        session = as_str(event.unified_msg_origin)
        text = as_str(getattr(req, "system_prompt", "")).strip()
        if not text:
            if session and session not in self._host_system_prompt_logged:
                self._host_system_prompt_logged.add(session)
                self.logger.info(
                    "companion_runtime: the host assembled no system prompt for %s; "
                    "proactive renders will go out without one",
                    session,
                )
            return
        if self._host_system_prompts.get(session) != text:
            self._host_system_prompts[session] = text
            self._host_system_prompts.move_to_end(session)
            while len(self._host_system_prompts) > HOST_SYSTEM_PROMPT_CACHE:
                self._host_system_prompts.popitem(last=False)
        if session not in self._host_system_prompt_logged:
            self._host_system_prompt_logged.add(session)
            self.logger.info(
                "companion_runtime: remembered the host system prompt for %s (%d chars); "
                "proactive renders will use it",
                session,
                len(text),
            )

    def _host_system_prompt_for(self, session: str) -> str:
        """Return the remembered host system prompt for a session, or ``""``."""
        return self._host_system_prompts.get(as_str(session), "")

    def _remember_event_id(self, session: str, event_id: str) -> None:
        """Remember the newest reported event id for one session."""
        self._last_event_ids[session] = event_id
        self._last_event_ids.move_to_end(session)
        while len(self._last_event_ids) > LAST_EVENT_ID_CACHE:
            self._last_event_ids.popitem(last=False)

    def _warn_injection_once(self, key: str, message: str) -> None:
        """Log an injection problem once instead of on every LLM request."""
        if key in self._injection_warnings:
            return
        self._injection_warnings.add(key)
        self.logger.warning(message)

    @staticmethod
    def _result_text(event: AstrMessageEvent) -> str:
        """Return the plain text of the message AstrBot just sent, if any."""
        try:
            result = event.get_result()
        except Exception:
            return ""
        if result is None:
            return ""
        try:
            return as_str(result.get_plain_text()).strip()
        except Exception:
            return ""

    @staticmethod
    def _delivered_text(event: AstrMessageEvent) -> str:
        """Return what ``send_message_to_user`` delivered this turn, if it was used.

        ``response.completion_text`` is the model's final *text output*, and when the turn
        delivered through the send tool that output is not the message. Measured on the beta
        (2026-09-25): the tool sent 「洗完没有呀……都半个多小时了，我就盯着这个框看」 while the
        completion read 「已经发了。就一条，软的、拖着尾音的，问他洗完没有、头发擦了没」 - a
        narration *about* the message. Reporting that told the Runtime she had said something
        the user never saw, and the render prompt's "don't say this again" block then handed
        the narration back to her as something to avoid.

        AstrBot stashes the delivered plain texts on the event, so prefer those; the bubbles
        are joined with a newline, the same shape a multi-bubble reply already reports as.

        Args:
            event: The message event for this turn.

        Returns:
            The delivered text, or an empty string when the turn did not deliver by tool.
        """
        try:
            texts = event.get_extra(SENT_PLAIN_TEXTS_EXTRA, [])
        except Exception:
            return ""
        if not isinstance(texts, (list, tuple)):
            return ""
        return "\n".join(part for part in (as_str(item).strip() for item in texts) if part)

    @filter.command("companion_runtime")
    async def companion_runtime_status(self, event: AstrMessageEvent):
        """查看陪伴 Runtime 适配器状态。"""
        yield event.plain_result(await self._status_text())

    async def _status_text(self) -> str:
        """Build the status report shown by the ``/companion_runtime`` command.

        The Runtime probe is advisory and fail-open: if the sidecar has no
        ``/health``, is down, or is a version that predates patch v0.2, the report
        simply says so and every other line is unaffected. Only a whitelist of
        fields is rendered, so a credential in the payload could never reach the
        chat even if one were ever added.
        """
        settings = self._settings
        if self._stopped:
            state = "terminated"
        elif self._gave_up:
            state = "unavailable (worker start failed)"
        elif self._started:
            state = "running"
        else:
            state = "inactive"
        lines = [
            "companion Runtime adapter",
            f"- state: {state}",
            f"- adapter_id: {settings.adapter_id}",
            f"- runtime: {settings.base_url or '<unset>'}",
            f"- token: {'configured' if settings.token else 'not configured'}",
            f"- observe_mode: {settings.observe_mode}",
            f"- context deadline: {settings.context_timeout_s * 1000:.0f}ms"
            f" (ttl {settings.context_cache_ttl_s:.0f}s)",
            f"- outbox: {'on' if settings.outbox_enabled else 'off'}"
            f" every {settings.outbox_poll_interval_s:.1f}s"
            f" batch {settings.outbox_batch}",
        ]
        if settings.session_routes:
            # With several Runtimes the per-target numbers are the only way to see
            # which instance is actually serving which person.
            lines.append(f"- session routes: {len(settings.session_routes)}")
            lines.extend(
                f"  · {prefix} -> {url}" for prefix, url in settings.session_routes
            )
            for url in settings.targets:
                target = self._targets.get(url)
                if target is None or target.outbox is None:
                    lines.append(f"  · {url}: not running")
                    continue
                stats = target.outbox.stats
                lines.append(
                    f"  · {url}: {stats.leased} leased, {stats.rendered} rendered, "
                    f"{stats.sent} sent, {stats.failed} failed",
                )
        if settings.registry_configured:
            lines.append(
                f"- route registry: {settings.route_registry_url} "
                f"({len(self._registry_routes)} sessions, {len(self._targets)} targets, "
                f"every {settings.route_sync_interval_s:.0f}s)",
            )
        lines.extend(await self._semantic_status_lines())
        bridge = self._bridge
        if bridge is not None:
            stats = bridge.stats
            lines.append(
                "- context: "
                f"{stats.requests} requests, {stats.cache_hits} cache hits, "
                f"{stats.fetches} fetches, {stats.timeouts} timeouts, "
                f"{stats.errors} errors, {stats.stale_fallbacks} stale fallbacks",
            )
        outbox = self._outbox
        if outbox is not None:
            stats = outbox.stats
            lines.append(
                "- actions: "
                f"{stats.leased} leased, {stats.rendered} rendered, {stats.sent} sent, "
                f"{stats.rejected} rejected, {stats.failed} failed, "
                f"{stats.skipped} skipped, {stats.replayed} replayed, "
                f"{stats.deferred} deferred (unreported, left to lease expiry)",
            )
        queue = self._queue
        if queue is not None:
            stats = queue.stats
            lines.append(
                "- queue: "
                f"{len(queue)} pending, {stats.delivered} delivered, {stats.retried} retried, "
                f"{stats.dropped()} dropped (full {stats.dropped_full}, "
                f"failed {stats.dropped_failed}, expired {stats.dropped_expired})",
            )
        if self._skipped_empty:
            lines.append(
                f"- skipped empty events: {self._skipped_empty} "
                "(AstrBot's notice-to-message conversions, e.g. pokes)",
            )
        if settings.issues:
            lines.append("- config issues:")
            lines.extend(f"  · {issue}" for issue in settings.issues)
        return "\n".join(lines)

    async def _semantic_status_lines(self) -> list[str]:
        """Return advisory lines describing the Runtime's cognition levels.

        Patch v0.2 made the semantic provider optional and made unresolved events
        a normal state, so the wording here must not read as a fault. Anything
        unexpected degrades to a single ``unavailable`` line.
        """
        transport = self._transport
        if transport is None:
            return ["- cognition: unavailable (adapter inactive)"]
        try:
            payload = await transport.fetch_health(timeout_s=self._settings.request_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Defence in depth: the transport already swallows its own failures,
            # but the plugin's whole contract is that an advisory probe can never
            # break a command, so it does not rely on that.
            return ["- cognition: unavailable (health probe failed)"]
        if payload is None:
            return ["- cognition: unavailable (no /health response)"]

        provider = payload.get("semantic_provider")
        if isinstance(provider, dict):
            # The Runtime reports ``provider``; older drafts of the protocol used
            # ``name``. Accept both so a version skew degrades to a wrong label
            # rather than to "unknown".
            name = as_str(provider.get("provider")) or as_str(provider.get("name")) or "unknown"
            available = bool(provider.get("available"))
            lines = [f"- semantic_provider: {name} (available={available})"]
        else:
            lines = ["- semantic_provider: unknown (Runtime predates patch v0.2)"]

        semantics = payload.get("semantics")
        if isinstance(semantics, dict):
            unresolved = semantics.get("unresolved")
            lines.append(
                f"- semantics: {unresolved if unresolved is not None else '?'} unresolved "
                "(normal: the Runtime defers what it cannot settle confidently)"
            )
        return lines
