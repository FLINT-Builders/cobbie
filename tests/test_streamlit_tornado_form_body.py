"""Streamlit's Tornado server must reject oversized urlencoded bodies without stalling.

`streamlit run` (see streamlit.web.bootstrap.run) starts
streamlit.web.server.server.Server. With the default configuration that server
uses Tornado, leaves server.address unset, and Tornado then listens on every
interface. RequestHandler._execute parses an application/x-www-form-urlencoded
body before the handler runs and before the XSRF check.

These tests send that body with no cookies and no XSRF token. They fail while
Tornado parses every field on the event loop, and they pass once parsing refuses
a body with more than the fixed field limit without blocking other connections.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# Keep the test from writing a Streamlit credential file into the real home
# directory, and from phoning home. Do not set server.address: the product
# does not, and the listen address is part of what is under test.
_TMP_HOME = tempfile.mkdtemp(prefix="cobbie-streamlit-home-")
os.environ["HOME"] = _TMP_HOME
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from streamlit import config  # noqa: E402
from streamlit.web.server.server import Server  # noqa: E402

# Tornado 6.5.8 refuses urlencoded bodies above 1000 fields. Stay above that,
# and use a body large enough that an unbounded parse blocks the loop.
_OVER_LIMIT_FIELDS = 5_000
_STALL_FIELDS = 2_000_000
_SMALL_FIELDS = 8
_MAX_UNSTALLED_SECONDS = 0.5

_HOST = "127.0.0.1"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind((_HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _form_body(field_count: int) -> bytes:
    return b"a=&" * field_count


def _listening_locals(port: int) -> set[str]:
    """Return local addresses from /proc for TCP listeners on port.

    /proc/net/tcp stores the address as a little-endian hex word. 00000000 is
    0.0.0.0. The port is a big-endian hex word.
    """
    port_hex = f"{port:04X}"
    found: set[str] = set()
    for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        text = Path(proc_path).read_text()
        for line in text.splitlines()[1:]:
            parts = line.split()
            local, state = parts[1], parts[3]
            address, local_port = local.split(":")
            if local_port.upper() == port_hex and state == "0A":
                found.add(address)
    return found


def _read_response(sock: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        data = sock.recv(65536)
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _status_line(response: bytes) -> bytes:
    return response.split(b"\r\n", 1)[0]


def _exchange(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    timeout: float = 30,
) -> tuple[float, bytes, bytes]:
    sock = socket.create_connection((_HOST, port), timeout)
    sock.settimeout(timeout)
    try:
        headers = [
            f"{method} {path} HTTP/1.1",
            f"Host: {_HOST}",
            "Connection: close",
        ]
        payload = b""
        if body is not None:
            headers.append("Content-Type: application/x-www-form-urlencoded")
            headers.append(f"Content-Length: {len(body)}")
            payload = body
        raw = ("\r\n".join(headers) + "\r\n\r\n").encode() + payload
        started = time.perf_counter()
        sock.sendall(raw)
        response = _read_response(sock)
        return time.perf_counter() - started, _status_line(response), response
    finally:
        sock.close()


@pytest.fixture(scope="module")
def streamlit_port() -> Iterator[int]:
    """Start the Tornado server the documented Streamlit command starts.

    global.developmentMode is forced off because a non-installed layout of the
    Streamlit package turns it on, which replaces the static-file routes with
    the Node dev server. An installed Streamlit (the one `uv run streamlit run`
    uses) leaves development mode off. server.address is left unset.
    """
    config.set_option("global.developmentMode", False)
    config.set_option("server.headless", True)
    config.set_option("browser.gatherUsageStats", False)
    port = _free_port()
    config.set_option("server.port", port)

    assert config.get_option("server.address") is None
    assert config.is_manually_set("server.address") is False
    assert config.get_option("server.useStarlette") is False

    script = Path(_TMP_HOME) / "app.py"
    script.write_text("import streamlit as st\nst.write('ok')\n", encoding="utf-8")
    server = Server(str(script), is_hello=False)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve(server))

    async def _serve(running: Server) -> None:
        await running.start()
        await running.stopped

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((_HOST, port), 0.2):
                break
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    else:
        raise RuntimeError(f"Streamlit server did not listen on {port}: {last_error}")

    # Warm the route so later timings measure parsing, not first-import work.
    _exchange(port, "GET", "/_stcore/host-config", timeout=10)  # warm routes

    yield port
    server.stop()
    thread.join(timeout=10)


def test_default_server_listens_on_every_interface(streamlit_port: int) -> None:
    listeners = _listening_locals(streamlit_port)
    assert "00000000" in listeners
    assert "00000000000000000000000000000000" in listeners


def test_small_urlencoded_post_is_parsed_before_xsrf(streamlit_port: int) -> None:
    """A normal form is still parsed, then refused for the missing XSRF token.

    Tornado logs the XSRF failure and returns a generic 403 page, so the
    assertion watches that log. A handler that rejected every POST before
    parsing would not emit it. The field-limit tests require the refusal to
    happen in the parser instead, as HTTP 400, before this check.
    """
    logger = logging.getLogger("tornado.general")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Capture()
    logger.addHandler(handler)
    try:
        elapsed, status, _response = _exchange(
            streamlit_port,
            "POST",
            "/_stcore/health",
            _form_body(_SMALL_FIELDS),
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    assert status.startswith(b"HTTP/1.1 403")
    assert elapsed < _MAX_UNSTALLED_SECONDS
    assert any("_xsrf" in message for message in messages)


def test_health_urlencoded_body_over_field_limit_does_not_stall_the_loop(
    streamlit_port: int,
) -> None:
    """An unauthenticated health POST must not block a concurrent request.

    The body is withheld by one byte so the server is idle, a control GET is
    checked, then the last byte is written and another GET is issued while the
    body parser would be running.
    """
    body = _form_body(_STALL_FIELDS)
    sock = socket.create_connection((_HOST, streamlit_port), 10)
    sock.settimeout(30)
    header = (
        "POST /_stcore/health HTTP/1.1\r\n"
        f"Host: {_HOST}\r\n"
        "Content-Type: application/x-www-form-urlencoded\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode()
    sock.sendall(header + body[:-1])

    control_elapsed, control_status, _control_response = _exchange(
        streamlit_port, "GET", "/_stcore/host-config", timeout=5
    )
    assert control_status.startswith(b"HTTP/1.1 200")
    assert control_elapsed < _MAX_UNSTALLED_SECONDS

    result: dict[str, tuple[float, bytes]] = {}

    def _finish() -> None:
        started = time.perf_counter()
        sock.sendall(body[-1:])
        response = _read_response(sock)
        sock.close()
        result["post"] = (time.perf_counter() - started, _status_line(response))

    worker = threading.Thread(target=_finish)
    worker.start()
    time.sleep(0.05)
    during_elapsed, during_status, _during_response = _exchange(
        streamlit_port, "GET", "/_stcore/host-config", timeout=30
    )
    worker.join(timeout=30)

    post_elapsed, post_status = result["post"]
    assert post_status.startswith(b"HTTP/1.1 400"), (
        post_status,
        post_elapsed,
        during_elapsed,
    )
    assert post_elapsed < _MAX_UNSTALLED_SECONDS, (post_elapsed, during_elapsed)
    assert during_status.startswith(b"HTTP/1.1 200")
    assert during_elapsed < _MAX_UNSTALLED_SECONDS, (post_elapsed, during_elapsed)


@pytest.mark.parametrize(
    "path",
    [
        "/_stcore/health",
        "/_stcore/upload_file/not-a-session/not-a-file",
        "/_stcore/stream",
        "/",
    ],
)
def test_handlers_reject_urlencoded_body_over_field_limit(
    streamlit_port: int, path: str
) -> None:
    elapsed, status, _response = _exchange(
        streamlit_port,
        "POST",
        path,
        _form_body(_OVER_LIMIT_FIELDS),
        timeout=10,
    )
    assert status.startswith(b"HTTP/1.1 400"), status
    assert elapsed < _MAX_UNSTALLED_SECONDS
