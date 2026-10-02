"""Durable proactive-send idempotency across crashes, restarts, and concurrency."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from companion_runtime.outbox import OutboxConsumer
from companion_runtime.protocol import ACTION_SEND
from companion_runtime.sent_ledger import SentLedger, SentLedgerUnavailable
from companion_runtime.settings import Settings
from tests.fakes import FakeClock, FakeExecutor, FakeTransport, ReportCollector
from tests.test_outbox import _action, _no_sleep


class _FailingLedger:
    """Ledger double that proves persistence failure stops delivery."""

    def reserve(self, **_kwargs):
        raise SentLedgerUnavailable("disk is read-only")


class SentLedgerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "sent.sqlite3"
        self.settings = Settings.from_mapping(
            {
                "adapter_id": "test-adapter",
                "outbox_poll_interval_ms": 200,
                "outbox_max_actions_per_poll": 4,
                "outbox_lease_ttl_ms": 30000,
                "outbox_max_concurrency": 1,
                "render_timeout_ms": 1000,
                "send_timeout_ms": 1000,
                "request_timeout_ms": 500,
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _consumer(
        self,
        *,
        transport: FakeTransport,
        executor: FakeExecutor,
        reporter: ReportCollector,
        ledger,
        namespace: str = "http://runtime-a",
    ) -> OutboxConsumer:
        return OutboxConsumer(
            transport=transport,
            executor=executor,
            reporter=reporter,
            settings=self.settings,
            sent_ledger=ledger,
            ledger_namespace=namespace,
            clock=FakeClock(),
            sleep=_no_sleep,
        )

    async def test_send_success_is_durable_before_runtime_ack(self) -> None:
        """A report outage after platform success leaves a durable replayable ACK."""
        action = _action(action_id="obx-1", payload={"text": "只发一次"})
        executor = FakeExecutor()
        reporter = ReportCollector(fail_times=1)
        consumer = self._consumer(
            transport=FakeTransport(actions=[action]),
            executor=executor,
            reporter=reporter,
            ledger=SentLedger(self.path),
        )

        await consumer.poll_once()

        self.assertEqual(len(executor.send_calls), 1)
        stored = SentLedger(self.path).get(
            namespace="http://runtime-a",
            session=action.session,
            outbox_id=action.action_id,
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored.state, "sent")
        self.assertTrue(stored.result["sent"])
        self.assertEqual(reporter.reports, [], "the simulated crash happened before Runtime ACK")

    async def test_restart_redelivery_replays_ack_without_platform_send(self) -> None:
        """A new consumer replays success for the same outbox id after restart."""
        action = _action(action_id="obx-2", payload={"text": "重启也只发一次"})
        first_executor = FakeExecutor()
        first = self._consumer(
            transport=FakeTransport(actions=[action]),
            executor=first_executor,
            reporter=ReportCollector(fail_times=1),
            ledger=SentLedger(self.path),
        )
        await first.poll_once()

        replay = _action(action_id="obx-2", payload={"text": "重启也只发一次"})
        replay.lease_id = "lease-after-restart"
        second_executor = FakeExecutor()
        second_reporter = ReportCollector()
        second = self._consumer(
            transport=FakeTransport(actions=[replay]),
            executor=second_executor,
            reporter=second_reporter,
            ledger=SentLedger(self.path),
        )
        await second.poll_once()

        self.assertEqual(len(first_executor.send_calls), 1)
        self.assertEqual(second_executor.send_calls, [])
        self.assertEqual(len(second_reporter.reports), 1)
        self.assertEqual(second_reporter.reports[0].lease_id, "lease-after-restart")
        self.assertTrue(second_reporter.reports[0].result["sent"])

    async def test_ledger_write_failure_is_fail_closed(self) -> None:
        """No durable reservation means no irreversible platform call."""
        action = _action(action_id="obx-3", payload={"text": "不能冒险发"})
        executor = FakeExecutor()
        reporter = ReportCollector()
        consumer = self._consumer(
            transport=FakeTransport(actions=[action]),
            executor=executor,
            reporter=reporter,
            ledger=_FailingLedger(),
        )

        await consumer.poll_once()

        self.assertEqual(executor.send_calls, [])
        self.assertEqual(reporter.reports, [])
        self.assertEqual(consumer.stats.deferred, 1)

    async def test_corrupt_ledger_is_fail_closed(self) -> None:
        """A corrupt SQLite file never degrades into an untracked platform send."""
        self.path.write_bytes(b"not-a-sqlite-database")
        action = _action(action_id="obx-4", payload={"text": "不能冒险发"})
        executor = FakeExecutor()
        consumer = self._consumer(
            transport=FakeTransport(actions=[action]),
            executor=executor,
            reporter=ReportCollector(),
            ledger=SentLedger(self.path),
        )

        await consumer.poll_once()

        self.assertEqual(executor.send_calls, [])
        self.assertEqual(consumer.stats.deferred, 1)

    async def test_same_outbox_id_is_isolated_by_runtime_and_session(self) -> None:
        """One target/session cannot suppress another target/session's delivery."""
        ledger = SentLedger(self.path)
        one = _action(action_id="same", session="scope:a", payload={"text": "A"})
        two = _action(action_id="same", session="scope:b", payload={"text": "B"})
        three = _action(action_id="same", session="scope:a", payload={"text": "C"})
        executors = [FakeExecutor(), FakeExecutor(), FakeExecutor()]
        consumers = [
            self._consumer(
                transport=FakeTransport(actions=[action]),
                executor=executor,
                reporter=ReportCollector(),
                ledger=ledger,
                namespace=namespace,
            )
            for action, executor, namespace in (
                (one, executors[0], "runtime-a"),
                (two, executors[1], "runtime-a"),
                (three, executors[2], "runtime-b"),
            )
        ]

        for consumer in consumers:
            await consumer.poll_once()

        self.assertEqual([len(item.send_calls) for item in executors], [1, 1, 1])

    async def test_concurrent_same_outbox_id_calls_platform_once(self) -> None:
        """SQLite reservation serializes two consumers racing on one outbox id."""
        ledger = SentLedger(self.path)
        action_a = _action(action_id="race", payload={"text": "一条"})
        action_b = _action(action_id="race", payload={"text": "一条"})
        action_b.lease_id = "lease-b"
        executor_a = FakeExecutor(send_delay_s=0.05)
        executor_b = FakeExecutor(send_delay_s=0.05)
        reports_a = ReportCollector()
        reports_b = ReportCollector()
        consumer_a = self._consumer(
            transport=FakeTransport(actions=[action_a]),
            executor=executor_a,
            reporter=reports_a,
            ledger=ledger,
        )
        consumer_b = self._consumer(
            transport=FakeTransport(actions=[action_b]),
            executor=executor_b,
            reporter=reports_b,
            ledger=ledger,
        )

        await asyncio.gather(consumer_a.poll_once(), consumer_b.poll_once())

        self.assertEqual(len(executor_a.send_calls) + len(executor_b.send_calls), 1)
        self.assertEqual(len(reports_a.reports) + len(reports_b.reports), 1)
        stored = ledger.get(
            namespace="http://runtime-a",
            session=action_a.session,
            outbox_id="race",
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored.state, "sent")


if __name__ == "__main__":
    unittest.main()
