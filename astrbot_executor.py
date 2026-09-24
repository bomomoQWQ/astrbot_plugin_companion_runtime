"""AstrBot-facing execution of leased Runtime actions.

This module lives next to ``main.py`` (not inside ``companion_runtime/``) on
purpose: the ``companion_runtime`` package is kept strictly free of AstrBot
imports so its protocol, queue, bridge, and outbox logic stay unit-testable
without a running AstrBot instance. Together with ``main.py``, this is the only
place that touches AstrBot.

Only public, documented AstrBot APIs are used:

* ``Context.get_current_chat_provider_id(umo=...)`` -- the session's current
  chat provider, honouring per-session model preferences.
* ``Context.llm_generate(chat_provider_id=..., prompt=..., system_prompt=...)``
  -- the documented SDK entry point for one-shot generation.
* ``Context.send_message(session, MessageChain)`` -- proactive delivery.
* ``Context.conversation_manager`` -- putting a delivered line back into the
  conversation history, which ``send_message`` does not do for private chats.
* ``Context.get_config()`` -- reading the host's own
  ``platform_settings.segmented_reply`` so a proactive line is cut into bubbles
  by the same rule the host cuts its replies with (see
  :mod:`companion_runtime.segments`).
"""

from __future__ import annotations

import asyncio
from random import Random
from typing import Any, Callable

from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain

from .companion_runtime.coerce import as_int, as_str
from .companion_runtime.protocol import (
    LeasedAction,
    TransportUnavailable,
    is_transport_unavailable,
    truncate_error,
)
from .companion_runtime.retry_queue import NULL_LOG
from .companion_runtime.segments import (
    DEFAULT_MAX_PARTS,
    SEGMENT_MODE_INHERIT,
    SegmentPolicy,
    host_segment_settings,
)


class ActionExecutionError(RuntimeError):
    """Raised when a leased action cannot be executed on the host side."""


class AstrBotActionExecutor:
    """Executes ``render`` and ``send`` actions through public AstrBot APIs."""

    def __init__(
        self,
        *,
        context: Any,
        log: Any = NULL_LOG,
        system_prompt_for: Callable[[str], str] | None = None,
        segment_mode: str = SEGMENT_MODE_INHERIT,
        segment_max_parts: int = DEFAULT_MAX_PARTS,
        segment_rng: Random | None = None,
    ) -> None:
        """Create the executor.

        Args:
            context: The AstrBot ``Context`` handed to the plugin.
            log: Logger-like object.
            system_prompt_for: Returns the system prompt the *host* assembles for a
                session, or ``""``. A render is a bare ``llm_generate`` call: AstrBot
                does not decorate it with the persona, the tools or the reminder
                (measured - a chat turn goes out with ~4210 characters of assembled
                system prompt, a render with none), so without this a proactive
                message is written by a model that has never been told who it is.
                The plugin learns the text from a normal turn and caches it; this
                callback is how the executor asks for it.
            segment_mode: ``inherit`` (follow the host's 分段回复 switch), ``on``
                (segment even when the host has it off) or ``off``.
            segment_max_parts: Ceiling on bubbles for one proactive message; the
                tail of a longer line is merged, never truncated.
            segment_rng: Random source for the pauses between bubbles. Injectable
                so a test can pin the timing.
        """
        self._context = context
        self._log = log
        self._system_prompt_for = system_prompt_for
        self._segment_mode = segment_mode
        self._segment_max_parts = segment_max_parts
        self._segment_rng = segment_rng or Random()

    def _host_system_prompt(self, session: str) -> str:
        """Return the cached host system prompt for a session, or ``""``.

        Never raises: prompt enrichment must not be the reason a leased render fails.
        """
        if self._system_prompt_for is None:
            return ""
        try:
            return as_str(self._system_prompt_for(session)).strip()
        except Exception:  # noqa: BLE001 - a missing persona is not a delivery failure
            self._log.debug("host system prompt lookup failed", exc_info=True)
            return ""

    async def render(self, action: LeasedAction) -> dict[str, Any]:
        """Render a prompt with the session's current AstrBot chat provider.

        Args:
            action: The leased ``render`` action. ``payload['prompt']`` holds the
                fully composed prompt (the Runtime owns prompt composition);
                ``payload['system_prompt']`` is optional, and
                ``payload['max_chars']`` optionally caps the returned text.

        Returns:
            A result mapping with the rendered ``text`` plus the provider id, so
            the Runtime can record which model actually spoke.

        Raises:
            ActionExecutionError: If the payload has no prompt, the session has no
                chat provider, or generation fails.
        """
        prompt = as_str(action.payload.get("prompt")).strip()
        if not prompt:
            raise ActionExecutionError("render action payload has no prompt")

        provider_id = await self._current_provider_id(action.session)
        kwargs: dict[str, Any] = {"chat_provider_id": provider_id, "prompt": prompt}
        # What the Runtime asked for wins; otherwise the host's own persona, so the
        # proactive line is written by the same character as a normal reply.
        system_prompt = as_str(action.payload.get("system_prompt")).strip() or self._host_system_prompt(
            str(action.session)
        )
        if system_prompt:
            kwargs["system_prompt"] = system_prompt

        try:
            response = await self._context.llm_generate(**kwargs)
        except Exception as exc:
            raise ActionExecutionError(f"llm_generate failed: {truncate_error(exc)}") from exc

        text = as_str(getattr(response, "completion_text", "")).strip()
        result: dict[str, Any] = {
            "text": text,
            "provider_id": provider_id,
            "chars": len(text),
        }
        max_chars = as_int(action.payload.get("max_chars"), 0)
        if max_chars > 0 and len(text) > max_chars:
            result["text"] = text[:max_chars]
            result["truncated"] = True
        return result

    async def send(self, action: LeasedAction, text: str) -> dict[str, Any]:
        """Deliver a proactive message to the action's session.

        The line is cut into the bubbles the host itself would cut it into (see
        :mod:`companion_runtime.segments`) and each bubble is sent on its own,
        with the host's pause in between: her proactive messages then arrive the
        way her replies do, and a render that came back as two lines is two
        messages rather than one paragraph with an embedded newline.

        Args:
            action: The leased ``send`` action; ``action.session`` is the
                ``unified_msg_origin`` the message goes to.
            text: The exact text to deliver, already authorized by the Runtime.

        Returns:
            ``{"sent": bool, "chars": int, ...}``. ``sent`` is ``False`` when no
            platform matched the session, which the Runtime records as a failure
            rather than as a delivered message. A segmented delivery adds
            ``segments`` and ``delivered_segments``; ``sent`` stays ``True`` once
            at least one bubble is on the user's screen, because re-dispatching
            the action would deliver that bubble a second time.

        Raises:
            TransportUnavailable: If the platform link was down, so nothing was
                delivered and the action is worth re-dispatching. Only raised when
                *no* bubble had gone out yet.
            ActionExecutionError: If the text is empty or AstrBot rejects the
                session, under the same "nothing delivered yet" condition.
        """
        body = text.strip()
        if not body:
            raise ActionExecutionError("send action has no text")

        policy = self._segment_policy()
        beats = policy.beats(body) or [body]
        delays = policy.delays(beats, self._segment_rng)

        delivered: list[str] = []
        for index, beat in enumerate(beats):
            if index:
                await asyncio.sleep(delays[index - 1])
            try:
                accepted = await self._send_one(action, beat)
            except (TransportUnavailable, ActionExecutionError):
                if not delivered:
                    raise
                # The earlier bubbles are already on the user's screen. Reporting
                # the failure would make the Runtime re-dispatch the action and
                # repeat them, so what landed is what gets reported.
                self._log.warning(
                    "part of a proactive message could not be delivered to %s; "
                    "the %d bubble(s) already sent stay as they are",
                    action.session,
                    len(delivered),
                    exc_info=True,
                )
                break
            if not accepted:
                if not delivered:
                    return {"sent": False, "chars": len(body)}
                self._log.warning(
                    "part of a proactive message was refused by the platform for %s; "
                    "the %d bubble(s) already sent stay as they are",
                    action.session,
                    len(delivered),
                )
                break
            delivered.append(beat)

        if not delivered:
            return {"sent": False, "chars": len(body)}
        # The line is in the user's chat now, so the session's history has to know
        # it. AstrBot's own send path records group LTM only, so without this a
        # private chat's next turn is built from a history that never contains it.
        # Only what actually went out is written: a partially delivered message
        # must not leave her remembering a sentence the user never saw.
        await self._remember_own_line(action.session, "\n".join(delivered))

        result: dict[str, Any] = {"sent": True, "chars": len(body)}
        if len(beats) > 1:
            # Logged because "did the split actually happen in production" is not
            # otherwise observable: the bubbles look identical in the host log.
            self._log.info(
                "delivered a proactive message to %s as %d/%d bubbles",
                action.session,
                len(delivered),
                len(beats),
            )
            result["segments"] = len(beats)
            result["delivered_segments"] = len(delivered)
        return result

    async def _send_one(self, action: LeasedAction, beat: str) -> bool:
        """Send one bubble, translating host failures into adapter outcomes.

        Args:
            action: The leased ``send`` action.
            beat: The single bubble to deliver.

        Returns:
            Whether the platform accepted the message.

        Raises:
            TransportUnavailable: The OneBot link was down.
            ActionExecutionError: The host rejected the send for any other reason.
        """
        chain = MessageChain([Plain(beat)])
        try:
            return bool(await self._context.send_message(action.session, chain))
        except Exception as exc:
            if is_transport_unavailable(exc):
                # Nothing reached the user, and the reason is a disconnected OneBot
                # rather than a rejected message, so this must not be reported as a
                # terminal delivery failure.
                raise TransportUnavailable(
                    f"send_message transport unavailable: {truncate_error(exc)}", error=exc
                ) from exc
            raise ActionExecutionError(f"send_message failed: {truncate_error(exc)}") from exc

    def _segment_policy(self) -> SegmentPolicy:
        """Resolve the host's segmentation rule for one delivery.

        Read per send rather than cached: 分段回复 can be switched off in the WebUI
        while the bot is running, and the next proactive message should obey the
        new setting without a plugin reload. The host's global config is used, not
        a per-session override, because that is the config the host's own reply
        path reads -- one authority for both.

        Never raises: an unreadable config degrades to the host's documented
        defaults, and a config that says "no segmentation" leaves this delivery
        exactly as it was before segmentation existed.
        """
        config: Any = None
        try:
            config = self._context.get_config()
        except Exception:  # noqa: BLE001 - a missing config is not a delivery failure
            self._log.debug("host config unavailable for segmented replies", exc_info=True)
        return SegmentPolicy.from_host_settings(
            host_segment_settings(config),
            mode=self._segment_mode,
            max_parts=self._segment_max_parts,
        )

    async def _remember_own_line(self, session: str, text: str) -> None:
        """Append a delivered proactive line to the host conversation history.

        Why this exists: ``Context.send_message`` writes **group LTM only** --
        ``astrbot/core/star/context.py`` gates that write on
        ``group_message_history_enable`` and a group session -- so a private chat's
        ``conversations.content`` never learns what the character said out of band.
        Measured on the beta: her three proactive lines were absent from the host
        history while every normal reply was present, so the user's reply to a
        proactive message ("？") reached the main model as an answer to nothing
        (「？什么。昨天说不和我见的是你，现在发问号的也是你。」), and the same fact was
        re-told three times because nothing told her she had already said it.

        Best-effort by contract: the message is already in the user's chat, so a failed
        history write must degrade to the old behaviour rather than turn a delivered
        message into a reported failure -- which the Runtime would answer by sending it
        again.

        Args:
            session: The ``unified_msg_origin`` the message went to.
            text: The exact text that was delivered.
        """
        manager = getattr(self._context, "conversation_manager", None)
        if manager is None:
            return
        try:
            conversation_id = await manager.get_curr_conversation_id(session)
            if not conversation_id:
                return
            conversation = await manager.get_conversation(session, conversation_id)
            history = list(getattr(conversation, "content", None) or [])
            history.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
            await manager.update_conversation(
                unified_msg_origin=session,
                conversation_id=conversation_id,
                history=history,
            )
        except Exception as exc:  # noqa: BLE001 - history is not worth a failed delivery
            self._log.warning(
                "could not append the delivered proactive line to the conversation history: %s",
                truncate_error(exc),
            )

    async def _current_provider_id(self, session: str) -> str:
        """Resolve the current chat provider id for one session."""
        try:
            provider_id = await self._context.get_current_chat_provider_id(umo=session)
        except Exception as exc:
            raise ActionExecutionError(
                f"no chat provider for session: {truncate_error(exc)}",
            ) from exc
        resolved = as_str(provider_id).strip()
        if not resolved:
            raise ActionExecutionError("session has no chat provider")
        return resolved
