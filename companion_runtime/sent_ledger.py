"""Durable at-most-once witness for irreversible proactive sends.

The Runtime outbox is intentionally at-least-once: a send whose platform call succeeded but
whose ACK was lost will be leased again.  This small SQLite ledger lives on the AstrBot side,
next to other plugin data, and makes ``outbox_id`` durable across plugin/process restarts.

A row is reserved as ``sending`` before the platform call.  A crash in that ambiguous window
is handled fail-closed: the row is never sent again automatically.  Once the platform reports
success, its result is persisted as ``sent`` before the Runtime ACK is attempted; a redelivery
then rebuilds the ACK from disk without touching the platform.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class SentLedgerUnavailable(RuntimeError):
    """The durable ledger cannot safely answer; sending must stop."""


class SentLedgerConflict(RuntimeError):
    """One durable identity was reused for different send content."""


@dataclass(frozen=True)
class SentRecord:
    """One durable platform-send state."""

    namespace: str
    session: str
    outbox_id: str
    attempt_id: str
    text_sha256: str
    state: str
    result: dict[str, Any]


class SentLedger:
    """SQLite-backed reservation and success ledger keyed by Runtime/session/outbox."""

    def __init__(self, path: Path | str) -> None:
        """Remember where the ledger lives; opening stays lazy and fail-closed.

        Args:
            path: SQLite file under AstrBot's persistent plugin-data directory.
        """
        self.path = Path(path)

    def reserve(
        self,
        *,
        namespace: str,
        session: str,
        outbox_id: str,
        attempt_id: str,
        text: str,
    ) -> tuple[str, SentRecord | None]:
        """Atomically reserve a send or read the existing durable state.

        Args:
            namespace: Runtime target identity (normally its base URL).
            session: AstrBot unified session origin.
            outbox_id: Runtime outbox identity; the platform idempotency key.
            attempt_id: Runtime attempt identity for audit/replay.
            text: Exact text about to be sent.

        Returns:
            ``("acquired", None)`` for the sole caller allowed to invoke the
            platform, ``("sent", record)`` for an ACK replay, or
            ``("sending", record)`` for an ambiguous in-flight/crashed send that
            must remain fail-closed.

        Raises:
            SentLedgerUnavailable: The file cannot be opened, migrated, or read.
            SentLedgerConflict: The same durable identity names different content.
        """
        namespace = str(namespace).strip()
        session = str(session).strip()
        outbox_id = str(outbox_id).strip()
        attempt_id = str(attempt_id).strip()
        if not namespace or not session or not outbox_id:
            raise SentLedgerConflict("namespace, session, and outbox_id are required")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5.0, isolation_level=None)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS sent_actions (
                           namespace TEXT NOT NULL,
                           session TEXT NOT NULL,
                           outbox_id TEXT NOT NULL,
                           attempt_id TEXT NOT NULL,
                           text_sha256 TEXT NOT NULL,
                           state TEXT NOT NULL CHECK(state IN ('sending','sent')),
                           result_json TEXT,
                           reserved_at TEXT NOT NULL,
                           sent_at TEXT,
                           PRIMARY KEY(namespace,session,outbox_id)
                       )""",
                )
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """SELECT attempt_id,text_sha256,state,result_json
                         FROM sent_actions
                        WHERE namespace=? AND session=? AND outbox_id=?""",
                    (namespace, session, outbox_id),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """INSERT INTO sent_actions
                           (namespace,session,outbox_id,attempt_id,text_sha256,state,result_json,reserved_at)
                           VALUES (?,?,?,?,?,'sending',NULL,?)""",
                        (namespace, session, outbox_id, attempt_id, digest, stamp),
                    )
                    conn.execute("COMMIT")
                    return "acquired", None
                if str(row[0]) != attempt_id or str(row[1]) != digest:
                    conn.execute("ROLLBACK")
                    raise SentLedgerConflict("outbox identity was reused for different send content")
                record = SentRecord(
                    namespace=namespace,
                    session=session,
                    outbox_id=outbox_id,
                    attempt_id=attempt_id,
                    text_sha256=digest,
                    state=str(row[2]),
                    result=(json.loads(row[3]) if row[3] else {}),
                )
                conn.execute("COMMIT")
                return record.state, record
        except SentLedgerConflict:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SentLedgerUnavailable(f"sent ledger unavailable: {type(exc).__name__}") from exc

    def abandon(
        self,
        *,
        namespace: str,
        session: str,
        outbox_id: str,
        attempt_id: str,
        text: str,
    ) -> None:
        """Remove a reservation after a definite non-delivery.

        Args:
            namespace: Runtime target identity.
            session: AstrBot unified session origin.
            outbox_id: Runtime outbox identity.
            attempt_id: Runtime attempt identity.
            text: Exact attempted text.

        Raises:
            SentLedgerUnavailable: The reservation cannot be updated safely.
            SentLedgerConflict: The reservation belongs to different content.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with closing(sqlite3.connect(self.path, timeout=5.0, isolation_level=None)) as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """SELECT attempt_id,text_sha256,state FROM sent_actions
                        WHERE namespace=? AND session=? AND outbox_id=?""",
                    (namespace, session, outbox_id),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return
                if str(row[0]) != attempt_id or str(row[1]) != digest or str(row[2]) != "sending":
                    conn.execute("ROLLBACK")
                    raise SentLedgerConflict("durable reservation no longer matches failed send")
                conn.execute(
                    "DELETE FROM sent_actions WHERE namespace=? AND session=? AND outbox_id=?",
                    (namespace, session, outbox_id),
                )
                conn.execute("COMMIT")
        except SentLedgerConflict:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise SentLedgerUnavailable(f"sent ledger unavailable: {type(exc).__name__}") from exc

    def mark_sent(
        self,
        *,
        namespace: str,
        session: str,
        outbox_id: str,
        attempt_id: str,
        text: str,
        result: Mapping[str, Any],
    ) -> SentRecord:
        """Persist platform success before Runtime ACK.

        Args:
            namespace: Runtime target identity.
            session: AstrBot unified session origin.
            outbox_id: Runtime outbox identity.
            attempt_id: Runtime attempt identity.
            text: Exact delivered text.
            result: Minimal JSON-compatible platform result used to replay ACK.

        Returns:
            The durable ``sent`` record.

        Raises:
            SentLedgerUnavailable: The update cannot be made durable.
            SentLedgerConflict: No matching reservation exists or content changed.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = json.dumps(dict(result), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            with closing(sqlite3.connect(self.path, timeout=5.0, isolation_level=None)) as conn:
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """SELECT attempt_id,text_sha256,state,result_json FROM sent_actions
                        WHERE namespace=? AND session=? AND outbox_id=?""",
                    (namespace, session, outbox_id),
                ).fetchone()
                if row is None or str(row[0]) != attempt_id or str(row[1]) != digest:
                    conn.execute("ROLLBACK")
                    raise SentLedgerConflict("platform success has no matching durable reservation")
                if str(row[2]) == "sent":
                    existing = json.loads(row[3]) if row[3] else {}
                    if existing != dict(result):
                        conn.execute("ROLLBACK")
                        raise SentLedgerConflict("sent result conflicts with durable success")
                else:
                    cursor = conn.execute(
                        """UPDATE sent_actions SET state='sent',result_json=?,sent_at=?
                            WHERE namespace=? AND session=? AND outbox_id=? AND state='sending'""",
                        (payload, stamp, namespace, session, outbox_id),
                    )
                    if cursor.rowcount != 1:
                        conn.execute("ROLLBACK")
                        raise SentLedgerConflict("durable reservation changed before success")
                conn.execute("COMMIT")
                return SentRecord(
                    namespace=namespace,
                    session=session,
                    outbox_id=outbox_id,
                    attempt_id=attempt_id,
                    text_sha256=digest,
                    state="sent",
                    result=dict(result),
                )
        except SentLedgerConflict:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SentLedgerUnavailable(f"sent ledger unavailable: {type(exc).__name__}") from exc

    def get(self, *, namespace: str, session: str, outbox_id: str) -> SentRecord | None:
        """Read one record for diagnostics/tests.

        Args:
            namespace: Runtime target identity.
            session: AstrBot unified session origin.
            outbox_id: Runtime outbox identity.

        Returns:
            The durable record, or ``None``.

        Raises:
            SentLedgerUnavailable: The ledger cannot be read safely.
        """
        try:
            with closing(sqlite3.connect(self.path, timeout=5.0)) as conn:
                row = conn.execute(
                    """SELECT attempt_id,text_sha256,state,result_json FROM sent_actions
                        WHERE namespace=? AND session=? AND outbox_id=?""",
                    (namespace, session, outbox_id),
                ).fetchone()
            if row is None:
                return None
            return SentRecord(
                namespace=namespace,
                session=session,
                outbox_id=outbox_id,
                attempt_id=str(row[0]),
                text_sha256=str(row[1]),
                state=str(row[2]),
                result=json.loads(row[3]) if row[3] else {},
            )
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SentLedgerUnavailable(f"sent ledger unavailable: {type(exc).__name__}") from exc


__all__ = ["SentLedger", "SentLedgerConflict", "SentLedgerUnavailable", "SentRecord"]
