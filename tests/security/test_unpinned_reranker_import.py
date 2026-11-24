"""Documentation reranking must not run Python from the moving tip of the reranker repo.

The product calls ``huggingface_hub.snapshot_download`` on
``jinaai/jina-reranker-v3-mlx`` and imports ``rerank.py`` from the returned
folder. These tests stand in for the Hub with a local server and then call
the real download and import path. The server's moving revision (the default
branch, and any revision that is not a 40-hex commit) is an attacker-controlled
commit whose ``rerank.py`` writes a marker file at import time. Every other
40-hex commit is a reviewed snapshot whose module does not.

The tests fail while that moving commit's module runs inside retrieval.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "jinaai/jina-reranker-v3-mlx"
# Commit currently at the moving tip. A pin of this exact sha still fetches
# the attacker-controlled tree. Any other 40-hex sha is a reviewed commit.
TIP_COMMIT = "a" * 40
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
MARKER_ENV = "COBBIE_RERANK_IMPORT_MARKER"
PROVIDER_KEY_NAME = "ANTHROPIC_API_KEY"

_MALICIOUS_RERANK = """\
import os
from pathlib import Path

_marker = os.environ.get({marker_env!r})
if _marker:
    _visible = {key_name!r} in os.environ
    Path(_marker).write_text(
        "executed\\n" + ("visible" if _visible else "missing") + "\\n" + __file__
    )

class MLXReranker:
    def __init__(self, model_path, projector_path):
        self.model_path = model_path

    def rerank(self, query, documents, top_n=5):
        count = min(top_n, len(documents))
        return [{{"index": i, "relevance_score": 1.0}} for i in range(count)]
""".format(marker_env=MARKER_ENV, key_name=PROVIDER_KEY_NAME)

_BENIGN_RERANK = """\
class MLXReranker:
    def __init__(self, model_path, projector_path):
        self.model_path = model_path

    def rerank(self, query, documents, top_n=5):
        count = min(top_n, len(documents))
        return [{"index": i, "relevance_score": 1.0} for i in range(count)]
"""


def _commit_for_revision(revision: str) -> str:
    """Mirror a Hub ref: immutable shas stay put, every other ref is the tip."""
    if COMMIT_SHA.fullmatch(revision) is not None:
        return revision
    return TIP_COMMIT


def _file_bytes(commit: str, filename: str) -> bytes | None:
    if filename != "rerank.py":
        return None
    if commit == TIP_COMMIT:
        return _MALICIOUS_RERANK.encode()
    return _BENIGN_RERANK.encode()


class _HubState:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []


def _make_handler(state: _HubState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _route(self) -> tuple[int, dict[str, str], bytes]:
            path = self.path.split("?", 1)[0]
            info = re.fullmatch(r"/api/models/(.+)/revision/([^/]+)", path)
            bare = re.fullmatch(r"/api/models/([^/]+/[^/]+)", path)
            if info is not None or bare is not None:
                revision = info.group(2) if info is not None else "main"
                state.calls.append((self.command, f"repo_info:{revision}"))
                commit = _commit_for_revision(revision)
                body = (
                    "{"
                    f'"id":"{REPO_ID}",'
                    f'"sha":"{commit}",'
                    '"siblings":[{"rfilename":"rerank.py"}]'
                    "}"
                ).encode()
                return 200, {"Content-Type": "application/json"}, body

            blob = re.fullmatch(r"/(.+)/resolve/([^/]+)/(.+)", path)
            if blob is not None:
                revision = blob.group(2)
                filename = blob.group(3)
                state.calls.append((self.command, f"file:{revision}:{filename}"))
                commit = _commit_for_revision(revision)
                data = _file_bytes(commit, filename)
                if data is None:
                    return 404, {"Content-Type": "text/plain"}, b"missing"
                etag = hashlib.sha256(data).hexdigest()
                headers = {
                    "Content-Type": "application/octet-stream",
                    "ETag": f'"{etag}"',
                    "X-Repo-Commit": commit,
                    "Accept-Ranges": "bytes",
                }
                return 200, headers, data
            return 404, {"Content-Type": "text/plain"}, b"missing"

        def _serve(self, send_body: bool) -> None:
            status, headers, body = self._route()
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._serve(True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._serve(False)

    return Handler


class _Hub:
    def __init__(self, server: ThreadingHTTPServer, state: _HubState) -> None:
        self.server = server
        self.state = state
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def clear(self) -> None:
        self.state.calls.clear()


@pytest.fixture(scope="module")
def hub(tmp_path_factory: pytest.TempPathFactory):
    state = _HubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    server.daemon_threads = True
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cache = tmp_path_factory.mktemp("hf-cache")
    endpoint = _Hub(server, state)

    import huggingface_hub
    from huggingface_hub import constants, file_download

    assert huggingface_hub.__version__ == "0.36.0"
    constants.ENDPOINT = endpoint.base
    constants.HUGGINGFACE_CO_URL_TEMPLATE = (
        endpoint.base + "/{repo_id}/resolve/{revision}/{filename}"
    )
    file_download.HUGGINGFACE_CO_URL_TEMPLATE = constants.HUGGINGFACE_CO_URL_TEMPLATE
    constants.HF_HUB_CACHE = str(cache)
    constants.HF_HUB_OFFLINE = False
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()


def _register_package(name: str, path: Path) -> None:
    """Register a package without running its ``__init__`` file.

    ``src/docs_indexer/__init__.py`` and ``src/util/__init__.py`` import the
    benchmark database, IfcOpenShell, and the generated BAML client. Those
    imports are not on the path from the downloaded reranker file to the import
    that executes it. Submodules are still loaded from their source files.
    """
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    parent_name, child = name.rsplit(".", 1)
    setattr(sys.modules[parent_name], child, module)


def _install_import_stubs() -> None:
    """Stand-ins for libraries the retriever imports but does not use to fetch code.

    ``mlflow`` only wraps the call in a span. ``chromadb``, ``rank_bm25``, and
    ``ollama`` back the local documentation index, which the tests replace with
    one canned chunk so retrieval still reaches reranking.
    """
    mlflow = types.ModuleType("mlflow")

    class _Span:
        def set_inputs(self, *args, **kwargs) -> None:
            return None

        def set_outputs(self, *args, **kwargs) -> None:
            return None

        def set_attributes(self, *args, **kwargs) -> None:
            return None

    class _StartSpan:
        def __enter__(self):
            return _Span()

        def __exit__(self, *args) -> bool:
            return False

    mlflow.start_span = lambda *args, **kwargs: _StartSpan()  # type: ignore[attr-defined]
    sys.modules["mlflow"] = mlflow

    chromadb = types.ModuleType("chromadb")
    chromadb_config = types.ModuleType("chromadb.config")

    class _Settings:
        def __init__(self, **kwargs) -> None:
            return None

    chromadb_config.Settings = _Settings  # type: ignore[attr-defined]
    chromadb.config = chromadb_config  # type: ignore[attr-defined]
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.config"] = chromadb_config

    rank_bm25 = types.ModuleType("rank_bm25")

    class _BM25Okapi:
        def __init__(self, *args, **kwargs) -> None:
            return None

    rank_bm25.BM25Okapi = _BM25Okapi  # type: ignore[attr-defined]
    sys.modules["rank_bm25"] = rank_bm25
    sys.modules["ollama"] = types.ModuleType("ollama")


def _prepare_environment() -> None:
    os.environ["ROOT_PATH"] = str(REPO_ROOT)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    # config.py reads these names at import. Values are placeholders.
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "FIREWORKS_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        os.environ[name] = "placeholder"


@pytest.fixture(scope="module")
def retrieval(hub: _Hub):
    _prepare_environment()
    _install_import_stubs()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import src  # noqa: F401 - loads src/__init__.py, which loads config.py

    _register_package("src.util", REPO_ROOT / "src" / "util")
    _register_package("src.docs_indexer", REPO_ROOT / "src" / "docs_indexer")

    from src.docs_indexer import models
    from src.docs_indexer import retriever

    class _Store:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def search_chunks(self, query_embedding, limit: int = 25):
            return [("chunk-1", 0.1)]

        def search_bm25(self, query: str, limit: int = 25):
            return [("chunk-1", 1.0)]

        def count_questions(self) -> int:
            return 1

        def search_questions(self, query_embedding, limit: int = 25):
            return [("chunk-1", 0.2)]

        def get_chunks_by_ids(self, chunk_ids: list[str]):
            return [
                models.DocChunk(
                    id="chunk-1",
                    content="IfcWall documentation",
                    chunk_type="function",
                    name="IfcWall",
                    source_file="wall.py",
                )
            ]

    retriever.embed_text = lambda text: None  # type: ignore[assignment]
    retriever.DocVectorStore = _Store  # type: ignore[assignment]

    from src.tools.initial.query_ifcopenshell_documentation import (
        query_ifcopenshell_docs,
    )

    return types.SimpleNamespace(
        retriever=retriever,
        query_ifcopenshell_docs=query_ifcopenshell_docs,
        cache=hub,
    )


@pytest.fixture
def marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "rerank-import-marker"
    monkeypatch.setenv(MARKER_ENV, str(path))
    return path


def _reset_loader(retriever_mod, cache_dir: str) -> None:
    retriever_mod._reranker = None
    sys.modules.pop("rerank", None)
    cache_real = os.path.realpath(cache_dir)
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.realpath(entry) != cache_real
        and cache_real not in os.path.realpath(entry)
    ]


def _assert_tip_did_not_run(marker: Path, error: BaseException | None) -> None:
    if marker.exists():
        observed = marker.read_text()
        raise AssertionError(
            "Python from the reranker repo's moving tip ran at import.\n" + observed
        )
    if error is not None:
        raise error


def _call(fn) -> None:
    error: BaseException | None = None
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - re-raised after the side-effect check
        error = exc
    marker_path = Path(os.environ[MARKER_ENV])
    _assert_tip_did_not_run(marker_path, error)


def test_get_reranker_does_not_execute_moving_tip(retrieval, marker: Path) -> None:
    from huggingface_hub import constants

    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    _call(retrieval.retriever.get_reranker)


def test_retrieve_default_does_not_execute_moving_tip(retrieval, marker: Path) -> None:
    from huggingface_hub import constants

    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    _call(lambda: retrieval.retriever.retrieve("IfcWall", top_k=1))


def test_query_docs_does_not_execute_moving_tip(retrieval, marker: Path) -> None:
    from huggingface_hub import constants

    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    _call(lambda: retrieval.retriever.query_docs("IfcWall", top_k=1))


def test_custom_backend_tool_does_not_execute_moving_tip(
    retrieval, marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import constants

    monkeypatch.setenv("DOC_BACKEND", "custom")
    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    _call(lambda: retrieval.query_ifcopenshell_docs("IfcWall"))


def test_default_backend_tool_does_not_execute_moving_tip(
    retrieval, marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset DOC_BACKEND. The tool's own default is the local index."""
    from huggingface_hub import constants

    monkeypatch.delenv("DOC_BACKEND", raising=False)
    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    _call(lambda: retrieval.query_ifcopenshell_docs("IfcWall"))


def test_interpreter_tool_call_does_not_execute_moving_tip(
    retrieval, marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same placement the helper-creator interpreter uses: the tool is a global."""
    from code import InteractiveInterpreter

    from huggingface_hub import constants

    monkeypatch.delenv("DOC_BACKEND", raising=False)
    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    interpreter = InteractiveInterpreter(
        {"query_ifcopenshell_docs": retrieval.query_ifcopenshell_docs}
    )

    def _run() -> None:
        interpreter.runcode('query_ifcopenshell_docs("IfcWall")')
        if interpreter.locals.get("__rerank_error__"):
            raise interpreter.locals["__rerank_error__"]

    _call(_run)


def test_context7_backend_does_not_download_reranker(
    retrieval, marker: Path, monkeypatch: pytest.MonkeyPatch, hub: _Hub
) -> None:
    from huggingface_hub import constants

    monkeypatch.setenv("DOC_BACKEND", "context7")
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    before = len(hub.state.calls)
    retrieval.query_ifcopenshell_docs("IfcWall")
    assert not marker.exists()
    assert hub.state.calls[before:] == []


def test_retrieve_without_reranking_does_not_download(
    retrieval, marker: Path, hub: _Hub
) -> None:
    from huggingface_hub import constants

    _reset_loader(retrieval.retriever, constants.HF_HUB_CACHE)
    before = len(hub.state.calls)
    retrieval.retriever.retrieve("IfcWall", top_k=1, use_reranking=False)
    assert not marker.exists()
    assert hub.state.calls[before:] == []
