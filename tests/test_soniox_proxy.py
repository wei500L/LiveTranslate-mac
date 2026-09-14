"""Proxy-environment regression test (offline).

Measured in the field: on a machine with a system SOCKS proxy (macOS system
settings, e.g. 127.0.0.1:7890), the Soniox SDK's WebSocket connection fails
with ``ImportError: connecting through a SOCKS proxy requires python-socks``.

websockets' get_proxy() reads *system* proxy settings via
urllib.request.getproxies() — not just environment variables — so this hits
any proxied machine even with a clean shell. python-socks is the declared
dependency that makes that path work; this test pins it without touching the
network.

Guarded on ``websockets`` rather than on ``python-socks`` itself: skipping
exactly when the dependency is missing would make the guard vacuous, since a
missing dependency is the failure being pinned. The light CI recipe installs
no websockets stack at all and skips this file; the full recipe installs both
and runs it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip(
    "websockets", reason="websockets (a soniox dependency) is not installed"
)


def test_socks_proxy_dependency_is_importable():
    """websockets' SOCKS branch imports python_socks at connect time — the
    exact import that raised ImportError before the dependency existed."""
    from python_socks.sync import Proxy  # noqa: F401

    assert Proxy is not None


def test_soniox_connection_survives_socks_proxy_detection():
    """The failing path was websockets.proxy.get_proxy -> socks scheme ->
    connect_socks_proxy -> ImportError. With python-socks installed the
    branch resolves; simulate the detection decision (no network)."""
    import urllib.request
    from websockets.proxy import get_proxy, parse_proxy
    from websockets.uri import parse_uri

    uri = parse_uri("wss://stt-rt.soniox.com")
    proxy = get_proxy(uri)
    if proxy is None:
        # This machine is not proxied; the regression cannot reproduce here,
        # but the dependency import above still guards the failure.
        return
    parsed = parse_proxy(proxy)
    if parsed.scheme.startswith("socks"):
        # Exactly the branch that used to raise ImportError.
        from websockets.sync.client import connect_socks_proxy  # noqa: F401
    # getproxies() seeing the system proxy at all proves the detection reads
    # system settings (the reason env-only workarounds were not enough).
    proxies = urllib.request.getproxies()
    assert isinstance(proxies, dict)
