"""Cooperative pause support for plugin functions.

Plugins import ``wait_if_paused`` and call it inside their loops so the
webrock pause/resume mechanism can actually suspend them at safe points.

Example usage inside a plugin::

    from webrock.pause import wait_if_paused

    PLUGIN_ID = "my_package.my_module.my_function"

    async def my_function():
        for item in items:
            await wait_if_paused(PLUGIN_ID)
            ...process item...
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

_engine: "Engine | None" = None


async def wait_if_paused(plugin_id: str) -> None:
    """Block until the plugin's pause is lifted; return immediately otherwise.

    Args:
        plugin_id: Dotted plugin path as registered with webrock
            (e.g. ``"src.data_sources.market.fill_missing"``).
    """
    if _engine is not None:
        await _engine.wait_if_paused(plugin_id)
