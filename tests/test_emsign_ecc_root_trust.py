"""Spec: API-key HTTPS calls must not trust chains that end at emSign ECC Root CA - G3.

These tests fail while certifi still trusts that root. They pass once the CA
bundle those calls load no longer accepts the chains.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path

import urllib3.connection

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAIN_DIR = Path(__file__).resolve().parent / "fixtures" / "emudhra_g3_chains"
CONTEXT7_HOST = "mcp.context7.com"
OPENROUTER_HOST = "openrouter.ai"
CONTEXT7_PLACEHOLDER = "placeholder-context7-api-key"
OPENROUTER_PLACEHOLDER = "placeholder-openrouter-api-key"
REDIRECTED_HOSTS = {CONTEXT7_HOST, OPENROUTER_HOST}


def _load_module(qualified_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _load_call_sites():
    """Load the two modules that send API keys, without importing unrelated packages."""
    os.environ["ROOT_PATH"] = str(REPO_ROOT)
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "FIREWORKS_API_KEY",
        "CEREBRAS_API_KEY",
    ):
        os.environ[name] = "placeholder"
    os.environ["CONTEXT7_API_KEY"] = CONTEXT7_PLACEHOLDER
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_PLACEHOLDER
    os.environ.pop("REQUESTS_CA_BUNDLE", None)
    os.environ.pop("CURL_CA_BUNDLE", None)
    os.environ.pop("SSL_CERT_FILE", None)

    src = types.ModuleType("src")
    src.__path__ = [str(REPO_ROOT / "src")]
    src.__package__ = "src"
    util = types.ModuleType("src.util")
    util.__path__ = [str(REPO_ROOT / "src" / "util")]
    util.__package__ = "src.util"
    tools = types.ModuleType("src.tools")
    tools.__path__ = [str(REPO_ROOT / "src" / "tools")]
    tools.__package__ = "src.tools"
    initial = types.ModuleType("src.tools.initial")
    initial.__path__ = [str(REPO_ROOT / "src" / "tools" / "initial")]
    initial.__package__ = "src.tools.initial"
    sys.modules["src"] = src
    sys.modules["src.util"] = util
    sys.modules["src.tools"] = tools
    sys.modules["src.tools.initial"] = initial

    _load_module("src.config", REPO_ROOT / "src" / "config.py")
    _load_module(
        "src.util.python_executor",
        REPO_ROOT / "src" / "util" / "python_executor.py",
    )
    docs = _load_module(
        "src.tools.initial.query_ifcopenshell_documentation",
        REPO_ROOT / "src" / "tools" / "initial" / "query_ifcopenshell_documentation.py",
    )
    usage = _load_module(
        "src.util.get_usage_openrouter",
        REPO_ROOT / "src" / "util" / "get_usage_openrouter.py",
    )
    docs.CONTEXT7_API_KEY = CONTEXT7_PLACEHOLDER
    return docs, usage


def _chain_files() -> list[Path]:
    chains = sorted(CHAIN_DIR.glob("*.pem"))
    assert chains, f"no chain fixtures in {CHAIN_DIR}"
    return chains


def _openssl_trusts_chain(cafile: str, chain: Path) -> bool:
    """Same purpose the TLS client handshake uses: ssl_server / X509_PURPOSE_SSL_SERVER.

    Flags keep the check on this CA file alone. OpenSSL's default trust store is not consulted,
    and revocation is not consulted, matching urllib3's default SSL context.
    """
    result = subprocess.run(
        [
            "openssl",
            "verify",
            "-purpose",
            "sslserver",
            "-no-CAfile",
            "-no-CApath",
            "-no-CAstore",
            "-CAfile",
            cafile,
            "-untrusted",
            str(chain),
            str(chain),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _exercise(call) -> tuple[list[str], list[str | None], bytes]:
    """Run one product call against a local sink and record the CA file TLS verification uses."""
    received = bytearray()
    ca_files: list[str] = []
    hostnames: list[str | None] = []
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def accept_loop() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _addr = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn.settimeout(1.0)
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    received.extend(data)
            except OSError:
                pass
            finally:
                conn.close()

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()

    original_getaddrinfo = socket.getaddrinfo
    original_wrap = urllib3.connection.ssl_wrap_socket

    def redirect_getaddrinfo(host, port_arg, *args, **kwargs):
        if host in REDIRECTED_HOSTS:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", port),
                )
            ]
        return original_getaddrinfo(host, port_arg, *args, **kwargs)

    def recording_wrap(*args, **kwargs):
        ca_files.append(kwargs.get("ca_certs"))
        hostnames.append(kwargs.get("server_hostname"))
        return original_wrap(*args, **kwargs)

    socket.getaddrinfo = redirect_getaddrinfo
    urllib3.connection.ssl_wrap_socket = recording_wrap
    try:
        call()
    finally:
        socket.getaddrinfo = original_getaddrinfo
        urllib3.connection.ssl_wrap_socket = original_wrap
        stop.set()
        listener.close()
        thread.join(timeout=2)

    return ca_files, hostnames, bytes(received)


def _assert_call_rejects_emsign_chains(
    ca_files: list[str],
    hostnames: list[str | None],
    received: bytes,
    label: str,
    allowed_hosts: set[str],
) -> None:
    assert ca_files, f"{label} never reached TLS wrapping"
    assert set(hostnames) <= allowed_hosts, (
        f"{label} wrapped TLS for an unexpected host: {hostnames!r}"
    )
    assert all(isinstance(path, str) and os.path.isfile(path) for path in ca_files), (
        f"{label} did not load a CA bundle for verification: {ca_files!r}"
    )
    assert CONTEXT7_PLACEHOLDER.encode() not in received
    assert OPENROUTER_PLACEHOLDER.encode() not in received
    trusted: list[str] = []
    for cafile in ca_files:
        for chain in _chain_files():
            if _openssl_trusts_chain(cafile, chain):
                trusted.append(f"{cafile} trusts {chain.name}")
    assert not trusted, (
        "TLS verification for this call still accepts certificate chains that end at "
        "emSign ECC Root CA - G3:\n" + "\n".join(trusted)
    )


def test_context7_documentation_query_rejects_emsign_ecc_root_g3_chains():
    docs, _usage = _load_call_sites()
    os.environ["DOC_BACKEND"] = "context7"

    def call():
        docs._query_context7("bounding box")
        docs.query_ifcopenshell_docs("bounding box")

    ca_files, hostnames, received = _exercise(call)
    _assert_call_rejects_emsign_chains(
        ca_files, hostnames, received, "context7", {CONTEXT7_HOST}
    )


def test_openrouter_usage_request_rejects_emsign_ecc_root_g3_chains():
    _docs, usage = _load_call_sites()

    def call():
        try:
            usage.get_usage_openrouter()
        except Exception:
            # The local sink is not the real API. The TLS CA file is recorded before the request fails.
            return

    ca_files, hostnames, received = _exercise(call)
    _assert_call_rejects_emsign_chains(
        ca_files, hostnames, received, "openrouter", {OPENROUTER_HOST}
    )
