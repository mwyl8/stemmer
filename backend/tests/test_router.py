"""Routing + tier selection. Real cases land in Phase 1 once separators exist.

Phase 0 smoke test: the router module and its dependents import cleanly.
"""

import pytest

from backend.separators.base import Separator
from backend.separators.router import select_separator


def test_router_raises_not_implemented_before_phase_1():
    with pytest.raises(NotImplementedError):
        select_separator("music", "fast")


def test_separator_is_abstract():
    with pytest.raises(TypeError):
        Separator()
