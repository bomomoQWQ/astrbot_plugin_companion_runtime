"""Stub of ``astrbot.api.star``."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any


class Context:
    """Stand-in for the plugin context; the integration test supplies its own."""


class StarTools:
    """Stand-in for AstrBot's persistent plugin-data path helper."""

    _root = Path(tempfile.gettempdir()) / "astrbot-plugin-tests"

    @classmethod
    def get_data_dir(cls, plugin_name: str | None = None) -> Path:
        """Return a stable test directory for one plugin.

        Args:
            plugin_name: Explicit plugin name.

        Returns:
            Stable directory under the OS temporary directory.
        """
        path = cls._root / (plugin_name or "unknown")
        path.mkdir(parents=True, exist_ok=True)
        return path


class Star:
    """Mirrors ``Star.__init__(context, config=None)`` plus ``self.logger``."""

    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        del config
        self.context = context
        self.logger = logging.getLogger("astrbot.plugin.stub")

    async def initialize(self) -> None:
        """Called when the plugin is activated."""

    async def terminate(self) -> None:
        """Called when the plugin is disabled or reloaded."""
