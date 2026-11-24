"""Run agent-generated Python code in a sandboxed worker process.

Code written by the LLM agents (and the helper functions they create) never
runs in the orchestrator process, which holds the provider API keys. It runs in
``_sandbox_worker.py``, started with an environment that contains no keys and
confined by the operating system before any of that code is loaded: it can read
the Python installation and the IFC model files it was given, and nothing else;
it cannot write files, open sockets or start programs. See the worker module
for the details per platform. If the sandbox cannot be established the
interpreter refuses to start.

Tools are recreated inside the worker:

- functions from ``src/tools`` modules are rebuilt from their module source;
- tools made by :func:`source_tool` (LLM-generated helper functions) carry
  their source and are only ever defined inside the worker;
- functions decorated with :func:`host_tool` stay in the orchestrator and are
  called over the worker's pipe with JSON arguments. Use this only for trusted
  tools that need the network or credentials, such as the documentation lookup.
"""

import ast
import inspect
import io
import json
import keyword
import os
import select
import struct
import subprocess
import sys
import threading
import time
import weakref
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from dotenv import find_dotenv
from loguru import logger

from src.config import (
    FUNCTION_BOILERPLATE,
    ROOT_PATH,
    SANDBOX_MEMORY_LIMIT_MB,
    SANDBOX_TIMEOUT_SECONDS,
)

HOST_TOOL_ATTR = "__cobbie_host_tool__"
SOURCE_TOOL_ATTR = "__cobbie_tool_source__"

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_worker.py")
STARTUP_TIMEOUT_SECONDS = 120
OUTPUT_LIMIT_CHARS = 1_000_000
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct(">Q")


class SandboxError(RuntimeError):
    """The sandboxed worker failed or sent something unexpected."""


class SandboxUnavailableError(SandboxError):
    """The worker could not confine itself, so no code was run."""


class _Timeout(Exception):
    pass


def host_tool(function: Callable) -> Callable:
    """Mark a trusted tool to run in the orchestrator instead of the sandbox."""
    setattr(function, HOST_TOOL_ATTR, True)
    return function


# ---------------------------------------------------------------------------
# Tools defined from source (never executed in this process)
# ---------------------------------------------------------------------------


class _SourceText:
    """Placeholder whose repr is a source expression, for inspect.Signature."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _signature_from_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> inspect.Signature:
    def annotation(expr):
        return _SourceText(ast.unparse(expr)) if expr is not None else inspect.Parameter.empty

    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        params.append(
            inspect.Parameter(
                arg.arg,
                kind,
                default=_SourceText(ast.unparse(default)) if default is not None else inspect.Parameter.empty,
                annotation=annotation(arg.annotation),
            )
        )
    if args.vararg:
        params.append(
            inspect.Parameter(
                args.vararg.arg,
                inspect.Parameter.VAR_POSITIONAL,
                annotation=annotation(args.vararg.annotation),
            )
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=_SourceText(ast.unparse(default)) if default is not None else inspect.Parameter.empty,
                annotation=annotation(arg.annotation),
            )
        )
    if args.kwarg:
        params.append(
            inspect.Parameter(
                args.kwarg.arg,
                inspect.Parameter.VAR_KEYWORD,
                annotation=annotation(args.kwarg.annotation),
            )
        )
    return inspect.Signature(params, return_annotation=annotation(node.returns))


def public_functions(source: str, filename: str = "<tool>") -> list[str]:
    """Names of the public top-level functions in ``source`` (parsed, not run)."""
    tree = ast.parse(source, filename)
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


def source_tool(
    source: str,
    function_name: str,
    *,
    module_name: Optional[str] = None,
    filename: Optional[str] = None,
    wrap_errors: bool = False,
    doc: Optional[str] = None,
) -> Callable:
    """Describe a tool by its source code without executing it.

    The returned function exposes the tool's name, signature and docstring for
    prompt documentation. Calling it raises: the source is only executed inside
    the sandboxed interpreter. With ``module_name`` the source is loaded there as
    that module; otherwise it is executed with the default helper imports.
    """
    tree = ast.parse(source, filename or "<tool>")
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function_name
        ),
        None,
    )
    if node is None:
        raise ValueError(f"Function {function_name} not found in generated code")

    def sandbox_only(*args, **kwargs):
        raise RuntimeError(
            f"{function_name} is generated code and only runs inside the sandboxed interpreter"
        )

    sandbox_only.__name__ = function_name
    sandbox_only.__qualname__ = function_name
    sandbox_only.__module__ = module_name or "src.tools.generated"
    sandbox_only.__doc__ = doc if doc is not None else ast.get_docstring(node)
    sandbox_only.__signature__ = _signature_from_ast(node)  # type: ignore[attr-defined]
    setattr(
        sandbox_only,
        SOURCE_TOOL_ATTR,
        {
            "kind": "source",
            "source": source,
            "function": function_name,
            "module": module_name,
            "filename": filename,
            "wrap_errors": wrap_errors,
        },
    )
    return sandbox_only


def _is_tool_module(name: Optional[str]) -> bool:
    return bool(name) and name.startswith("src.tools.")  # type: ignore[union-attr]


def _collect_module_source(module, modules: Dict[str, dict]) -> None:
    if module.__name__ in modules:
        return
    filename = getattr(module, "__file__", None)
    if not filename:
        raise ValueError(f"Tool module {module.__name__} has no source file")
    modules[module.__name__] = {
        "source": Path(filename).read_text(encoding="utf-8"),
        "filename": filename,
    }
    for value in vars(module).values():
        dependency: Optional[str]
        if inspect.ismodule(value):
            dependency = value.__name__
        elif inspect.isfunction(value) or inspect.isclass(value):
            dependency = getattr(value, "__module__", None)
        else:
            continue
        if (
            dependency != module.__name__
            and _is_tool_module(dependency)
            and dependency in sys.modules
        ):
            _collect_module_source(sys.modules[dependency], modules)


def build_tool_specs(tools: Optional[Dict[str, Callable]]):
    """Translate a tools dict into worker specs, module sources and host tools."""
    specs: list[dict] = []
    modules: Dict[str, dict] = {}
    host: Dict[str, Callable] = {}
    for name, function in (tools or {}).items():
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Invalid tool name: {name!r}")
        if getattr(function, HOST_TOOL_ATTR, False) is True:
            specs.append({"name": name, "kind": "host"})
            host[name] = function
            continue
        source_spec = getattr(function, SOURCE_TOOL_ATTR, None)
        if isinstance(source_spec, dict):
            specs.append({**source_spec, "name": name})
            continue
        module = sys.modules.get(getattr(function, "__module__", None) or "")
        if (
            inspect.isfunction(function)
            and module is not None
            and _is_tool_module(module.__name__)
            and getattr(module, function.__name__, None) is function
        ):
            _collect_module_source(module, modules)
            specs.append(
                {"name": name, "kind": "module", "module": module.__name__, "attr": function.__name__}
            )
            continue
        raise ValueError(
            f"Tool {name!r} cannot run in the sandbox: define it in a src/tools module, "
            "build it with source_tool(), or mark a trusted tool with @host_tool"
        )
    return specs, modules, host


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    syntax_error: Optional[str] = None
    runtime_error: Optional[str] = None
    timed_out_after: Optional[float] = None
    failure: Optional[str] = None
    notice: Optional[str] = None


def _protected_paths() -> list[str]:
    paths = {ROOT_PATH, os.path.expanduser("~")}
    for dotenv_path in (find_dotenv(), find_dotenv(usecwd=True)):
        if dotenv_path:
            paths.add(dotenv_path)
    return sorted(os.path.realpath(p) for p in paths if p)


def _readable_files(paths: Iterable[Optional[str]]) -> list[str]:
    files = []
    for path in paths:
        if not path:
            continue
        real = os.path.realpath(path)
        if os.path.isfile(real):
            files.append(real)
        else:
            logger.warning(f"Not exposing {path!r} to the sandbox: not a file")
    return files


def _sbpl_string(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _darwin_profile(readable_files: list[str], protected: list[str]) -> str:
    def is_within(path: str, root: str) -> bool:
        try:
            return os.path.commonpath([path, root]) == root
        except ValueError:
            return False

    import site
    import sysconfig

    roots = {
        sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
        *site.getsitepackages(),
        *(v for k, v in sysconfig.get_paths().items() if k in ("stdlib", "platstdlib", "purelib", "platlib")),
        "/usr/lib", "/usr/share", "/System", "/Library/Apple", "/private/var/db/dyld",
        "/private/var/db/timezone", "/usr/share/zoneinfo",
    }
    subpaths = []
    for root in sorted(roots):
        real = os.path.realpath(root)
        if not os.path.exists(real) or real == "/" or any(is_within(p, real) for p in protected):
            continue
        subpaths.append(f"(subpath {_sbpl_string(real)})")
    # The profile applies from exec, so Python must be able to read the worker script.
    literals = [f"(literal {_sbpl_string(p)})" for p in [os.path.realpath(WORKER_PATH), *readable_files]]
    executable = os.path.realpath(sys.executable)
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            f"(allow process-exec (literal {_sbpl_string(executable)}))",
            "(allow file-read-metadata)",
            "(allow sysctl-read)",
            "(allow process-info* (target self))",
            "(allow signal (target self))",
            "(allow file-read* " + " ".join(subpaths + literals) + ' (literal "/dev/null") (literal "/dev/urandom"))',
            '(allow file-write-data (literal "/dev/null"))',
        ]
    )


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass
    for stream in (process.stdin, process.stdout):
        try:
            if stream:
                stream.close()
        except OSError:
            pass


class SandboxedInterpreter:
    """A persistent Python session that runs in a sandboxed worker process.

    Variables survive between :meth:`run` calls, like the in-process
    interpreter this replaces. If the worker dies or times out it is restarted
    on the next call and the result says that the session state was lost.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        readable_paths: Optional[Iterable[str]] = None,
        timeout_seconds: float = SANDBOX_TIMEOUT_SECONDS,
        memory_limit_mb: Optional[int] = SANDBOX_MEMORY_LIMIT_MB,
    ):
        specs, modules, self._host_tools = build_tool_specs(tools)
        readable = _readable_files([model_path, *(readable_paths or [])])
        self._protected = _protected_paths()
        self._init = {
            "type": "init",
            "model_path": model_path,
            "boilerplate": FUNCTION_BOILERPLATE,
            "tools": specs,
            "modules": modules,
            "readable_files": readable,
            "protected_paths": self._protected,
            "output_limit": OUTPUT_LIMIT_CHARS,
            "limits": {
                "memory_bytes": memory_limit_mb * 1024 * 1024 if memory_limit_mb else None,
            },
        }
        self.timeout_seconds = timeout_seconds
        self.warnings: list[str] = []
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._finalizer: Optional[weakref.finalize] = None
        self._lost_state = False
        self._start()

    # -- process management -------------------------------------------------

    def _command(self) -> list[str]:
        command = [sys.executable, "-I", "-B", WORKER_PATH]
        if sys.platform.startswith("linux"):
            return command
        if sys.platform == "darwin" and os.path.exists("/usr/bin/sandbox-exec"):
            profile = _darwin_profile(self._init["readable_files"], self._protected)
            return ["/usr/bin/sandbox-exec", "-p", profile, *command]
        raise SandboxUnavailableError(
            f"No sandbox is available on {sys.platform}; refusing to run generated code"
        )

    def _start(self) -> None:
        env = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        if os.environ.get("LD_LIBRARY_PATH"):
            env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=os.getcwd(),
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        self._process = process
        self._finalizer = weakref.finalize(self, _kill, process)
        try:
            self._send(self._init)
            reply = self._receive(time.monotonic() + STARTUP_TIMEOUT_SECONDS)
        except (_Timeout, SandboxError, OSError) as e:
            self._stop()
            raise SandboxUnavailableError(f"Sandboxed interpreter failed to start: {e}") from None
        if reply.get("type") == "fatal":
            self._stop()
            raise SandboxUnavailableError(
                f"Sandboxed interpreter refused to start: {reply.get('error')}"
            )
        if reply.get("type") != "ready":
            self._stop()
            raise SandboxUnavailableError("Sandboxed interpreter sent an unexpected reply")
        self.warnings = [str(w) for w in reply.get("warnings", [])]
        for warning in self.warnings:
            logger.warning(warning)

    def _stop(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._process = None

    def close(self) -> None:
        with self._lock:
            self._stop()

    def __enter__(self) -> "SandboxedInterpreter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol -------------------------------------------------------------

    def _send(self, message: dict) -> None:
        assert self._process is not None and self._process.stdin is not None
        data = json.dumps(message).encode("utf-8")
        payload = memoryview(_HEADER.pack(len(data)) + data)
        fd = self._process.stdin.fileno()
        try:
            while payload:
                written = os.write(fd, payload)
                payload = payload[written:]
        except (BrokenPipeError, OSError) as e:
            raise SandboxError(f"the sandboxed process is not accepting input ({e})") from None

    def _read_exact(self, size: int, deadline: Optional[float]) -> bytes:
        assert self._process is not None and self._process.stdout is not None
        fd = self._process.stdout.fileno()
        buffer = bytearray()
        while len(buffer) < size:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise _Timeout
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise _Timeout
            chunk = os.read(fd, min(size - len(buffer), 1 << 20))
            if not chunk:
                raise SandboxError("the sandboxed process exited")
            buffer += chunk
        return bytes(buffer)

    def _receive(self, deadline: Optional[float]) -> dict:
        (size,) = _HEADER.unpack(self._read_exact(_HEADER.size, deadline))
        if size > MAX_MESSAGE_BYTES:
            raise SandboxError("the sandboxed process sent an oversized message")
        try:
            message = json.loads(self._read_exact(size, deadline).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SandboxError(f"the sandboxed process sent invalid data ({e})") from None
        if not isinstance(message, dict):
            raise SandboxError("the sandboxed process sent invalid data")
        return message

    def _call_host_tool(self, message: dict) -> dict:
        name = message.get("name")
        args = message.get("args", [])
        kwargs = message.get("kwargs", {})
        function = self._host_tools.get(name) if isinstance(name, str) else None
        if function is None:
            return {"type": "host_result", "ok": False, "error": f"unknown tool {name!r}"}
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            return {"type": "host_result", "ok": False, "error": "invalid arguments"}
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                value = function(*args, **kwargs)
        except Exception as e:
            return {
                "type": "host_result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "stdout": stdout.getvalue()[:OUTPUT_LIMIT_CHARS],
                "stderr": stderr.getvalue()[:OUTPUT_LIMIT_CHARS],
            }
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
        return {
            "type": "host_result",
            "ok": True,
            "value": value,
            "stdout": stdout.getvalue()[:OUTPUT_LIMIT_CHARS],
            "stderr": stderr.getvalue()[:OUTPUT_LIMIT_CHARS],
        }

    # -- execution ------------------------------------------------------------

    def run(self, python_code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        with self._lock:
            notice = None
            if self._process is None or self._process.poll() is not None:
                self._stop()
                self._start()
                self._lost_state = True
            if self._lost_state:
                notice = (
                    "NOTE: the Python session was restarted after the previous failure; "
                    "variables from earlier code are no longer defined."
                )
                self._lost_state = False

            deadline = time.monotonic() + timeout if timeout else None
            try:
                self._send({"type": "exec", "code": python_code})
                while True:
                    message = self._receive(deadline)
                    kind = message.get("type")
                    if kind == "host_call":
                        self._send(self._call_host_tool(message))
                    elif kind == "result":
                        break
                    else:
                        raise SandboxError(f"unexpected message {kind!r}")
            except _Timeout:
                self._stop()
                self._lost_state = True
                return ExecutionResult(timed_out_after=timeout, notice=notice)
            except SandboxError as e:
                self._stop()
                self._lost_state = True
                return ExecutionResult(failure=str(e), notice=notice)

            return ExecutionResult(
                stdout=str(message.get("stdout") or ""),
                stderr=str(message.get("stderr") or ""),
                syntax_error=message.get("syntax_error"),
                runtime_error=message.get("runtime_error"),
                notice=notice,
            )
