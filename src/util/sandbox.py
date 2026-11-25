"""
Sandboxed execution of model-written Python.

Code actions returned by Cobbie, the helper creator and the helper debugger, and every
model-written helper function, run in a separate worker process (src/util/sandbox_worker.py)
that:

* starts with a minimal environment: no provider API keys or other secrets;
* is restricted with Landlock before any project, tool or model code runs: it can read
  the Python installation, the packages on ``sys.path``, the project sources under
  ``src/`` and the IFC models it was given, and can only write to a private temporary
  directory. Files directly in the repository root (such as ``.env``) are not readable
  and the tools directories are not writable;
* cannot create sockets (seccomp), so it cannot reach the network, local services or
  Unix sockets, and cannot signal or trace the parent.

Trusted helpers that need the parent's credentials or network access (the functions in
``src.tools.initial``, e.g. ``query_ifcopenshell_docs``) stay in the parent; the worker
calls them over the pipe with JSON arguments and gets their output back.

The sandbox requires Linux with Landlock. Where it is unavailable, creating an
interpreter raises SandboxUnavailableError instead of running code unsandboxed.
"""

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from loguru import logger

from src.config import FUNCTION_BOILERPLATE, ROOT_PATH
from src.util.generated_tool import GeneratedTool

WORKER_PATH = Path(__file__).with_name("sandbox_worker.py")

HOST_TOOL_MODULE_PREFIXES = ("src.tools.initial",)

IFC_SUFFIXES = (".ifc", ".ifczip", ".ifcxml")

SYSTEM_READ_DIRS = ("/usr", "/lib", "/lib64", "/lib32", "/libx32", "/bin", "/sbin", "/etc")
SYSTEM_READ_FILES = ("/dev/urandom", "/dev/random", "/dev/zero", "/proc/cpuinfo", "/proc/meminfo")
SYSTEM_READ_WRITE_FILES = ("/dev/null",)

ENV_ALLOWLIST = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "TZ")

SOURCE_TOOL_PRELUDE = FUNCTION_BOILERPLATE + """
try:
    import numpy
    import pandas
except ImportError:
    pass
"""

CONFIG_VALUES = {
    "ROOT_PATH": ROOT_PATH,
    "SRC_PATH": os.path.join(ROOT_PATH, "src"),
    "TEST_IFC_PATH": os.path.join(ROOT_PATH, "src/db/bim_models/duplex/arc.ifc"),
    "DB_PATH": os.path.join(ROOT_PATH, "src/db/db.db"),
    "DIRECTORY_IFC_MODELS_PATH": os.path.join(ROOT_PATH, "src/db/bim_models"),
    "CREATED_TOOLS_PATH": os.path.join(ROOT_PATH, "src/tools/created"),
    "INITIAL_TOOLS_PATH": os.path.join(ROOT_PATH, "src/tools/initial"),
    "MANUAL_TOOLS_PATH": os.path.join(ROOT_PATH, "src/tools/manual"),
    "FUNCTION_BOILERPLATE": FUNCTION_BOILERPLATE,
    "LOG_LEVEL": "INFO",
}


class SandboxUnavailableError(RuntimeError):
    """The sandbox cannot be set up on this machine; model-written code will not run."""


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


class _Grants:
    """Landlock grants for one worker. Never grants read access to the repository root or its ancestors."""

    def __init__(self) -> None:
        self.root = _real(ROOT_PATH)
        self.src = os.path.join(self.root, "src")
        self.home = _real(os.path.expanduser("~"))
        self.read_dirs: List[str] = []
        self.read_files: List[str] = []
        self.read_dir_only: List[str] = [self.root, self.src]
        self.read_write_dirs: List[str] = []
        self.read_write_files: List[str] = [p for p in SYSTEM_READ_WRITE_FILES if os.path.exists(p)]

    def _protected(self, path: str) -> bool:
        return _is_within(self.root, path) or _is_within(self.home, path)

    def add_read_dir(self, path: str) -> None:
        path = _real(path)
        if not os.path.isdir(path) or self._protected(path) or path in self.read_dirs:
            return
        self.read_dirs.append(path)

    def add_read_file(self, path: str) -> None:
        path = _real(path)
        if os.path.isfile(path) and path not in self.read_files:
            self.read_files.append(path)

    def add_sources(self) -> None:
        # Children of src/ one by one, so a dotfile such as src/.env stays unreadable.
        for entry in sorted(os.listdir(self.src)):
            if entry.startswith("."):
                continue
            full = os.path.join(self.src, entry)
            if os.path.isdir(full):
                self.add_read_dir(full)
            else:
                self.add_read_file(full)

    def add_python(self, sys_path: Iterable[str]) -> List[str]:
        prefixes = {_real(p) for p in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix)}
        for prefix in prefixes:
            self.add_read_dir(prefix)
        kept = []
        for entry in sys_path:
            if not entry:
                continue
            real = _real(entry)
            if real == self.root or _is_within(real, self.src):
                kept.append(entry)
                continue
            under_prefix = any(_is_within(real, p) for p in prefixes)
            is_packages = any(part in ("site-packages", "dist-packages") for part in Path(real).parts)
            if under_prefix or (is_packages and not _is_within(real, self.root)):
                self.add_read_dir(real)
                kept.append(entry)
        return kept

    def add_ifc(self, path: str) -> None:
        if path and path.lower().endswith(IFC_SUFFIXES):
            self.add_read_file(path)


def _tool_spec(name: str, tool: Callable) -> Dict[str, Any]:
    if isinstance(tool, GeneratedTool):
        return {"kind": "source", "name": name, "module": tool.module_name, "source": tool.source}

    module_name = getattr(tool, "__module__", None) or ""
    qualname = getattr(tool, "__qualname__", None) or ""
    if module_name.startswith(HOST_TOOL_MODULE_PREFIXES):
        return {"kind": "host", "name": name}

    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    root_src = os.path.join(_real(ROOT_PATH), "src")
    if (
        inspect.isfunction(tool)
        and module_file
        and module_name != "__main__"
        and "<" not in qualname
        and _is_within(_real(module_file), root_src)
    ):
        return {"kind": "module", "name": name, "module": module_name, "qualname": qualname}

    # Developer-provided callables that the worker cannot import (scripts, closures,
    # test doubles). They are trusted code, so they run in the parent.
    return {"kind": "host", "name": name}


def _kill(proc: subprocess.Popen, tmpdir: str) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass
    shutil.rmtree(tmpdir, ignore_errors=True)


class SandboxedInterpreter:
    """A persistent, sandboxed Python session (variables survive between ``run`` calls)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        readable_paths: Optional[Iterable[str]] = None,
    ):
        if not sys.platform.startswith("linux"):
            raise SandboxUnavailableError(
                f"Sandboxed code execution needs Linux with Landlock; refusing to run "
                f"model-written code on {sys.platform!r}."
            )
        self._model_path = model_path
        self._tools = dict(tools or {})
        self._readable_paths = [p for p in (readable_paths or []) if isinstance(p, str)]
        self._specs = [_tool_spec(name, tool) for name, tool in self._tools.items()]
        self._host_tools = {s["name"]: self._tools[s["name"]] for s in self._specs if s["kind"] == "host"}
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._finalizer = None
        self._start()

    def _config(self, tmpdir: str) -> Dict[str, Any]:
        grants = _Grants()
        for path in SYSTEM_READ_DIRS:
            grants.add_read_dir(path)
        for path in SYSTEM_READ_FILES:
            grants.add_read_file(path)
        grants.add_sources()
        kept_sys_path = grants.add_python(sys.path)
        model_path = os.path.abspath(self._model_path) if self._model_path else None
        if model_path:
            grants.add_ifc(model_path)
        for path in self._readable_paths:
            grants.add_ifc(path)
        grants.read_write_dirs.append(tmpdir)

        return {
            "sys_path": kept_sys_path,
            "src_path": grants.src,
            "config_values": CONFIG_VALUES,
            "boilerplate": FUNCTION_BOILERPLATE,
            "source_tool_prelude": SOURCE_TOOL_PRELUDE,
            "model_path": self._model_path,
            "tools": self._specs,
            "env_allowlist": list(ENV_ALLOWLIST),
            "read_dirs": grants.read_dirs,
            "read_files": grants.read_files,
            "read_dir_only": grants.read_dir_only,
            "read_write_dirs": grants.read_write_dirs,
            "read_write_files": grants.read_write_files,
        }

    def _start(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="cobbie-sandbox-")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            cwd=os.getcwd(),
            close_fds=True,
        )
        self._proc = proc
        self._finalizer = weakref.finalize(self, _kill, proc, tmpdir)
        try:
            self._send(self._config(tmpdir))
            message = self._serve_until({"ready", "fatal"})
        except Exception as e:
            self.close()
            raise SandboxUnavailableError(f"Sandboxed interpreter failed to start: {e}") from e
        if message is None or message.get("op") != "ready":
            reason = message.get("message") if message else f"worker exited with code {proc.poll()}"
            self.close()
            raise SandboxUnavailableError(f"Sandboxed interpreter could not be set up: {reason}")
        logger.debug(f"Sandboxed interpreter ready with {len(self._specs)} tools")

    def _send(self, message: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self._proc.stdin.flush()

    def _receive(self) -> Optional[Dict[str, Any]]:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            return None
        message = json.loads(line)
        return message if isinstance(message, dict) else None

    def _serve_until(self, ops: set) -> Optional[Dict[str, Any]]:
        """Answer host-tool calls from the worker until it sends one of ``ops``."""
        while True:
            message = self._receive()
            if message is None or message.get("op") in ops:
                return message
            if message.get("op") == "call":
                self._send(self._call_host_tool(message))

    def _call_host_tool(self, message: Dict[str, Any]) -> Dict[str, Any]:
        name = message.get("name")
        tool = self._host_tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return {"op": "call_result", "error": f"Unknown helper function: {name!r}"}
        args = message.get("args") if isinstance(message.get("args"), list) else []
        kwargs = message.get("kwargs") if isinstance(message.get("kwargs"), dict) else {}
        buffer = StringIO()
        try:
            with redirect_stdout(buffer):
                value = tool(*args, **kwargs)
        except Exception as e:
            return {"op": "call_result", "stdout": buffer.getvalue(), "error": f"{type(e).__name__}: {e}"}
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        return {"op": "call_result", "stdout": buffer.getvalue(), "value": value}

    def run(self, python_code: str) -> Dict[str, Any]:
        """
        Run code in the session. Returns a dict with ``status`` ("ok", "syntax_error" or
        "error") and either ``stdout``/``stderr`` or ``message``.
        """
        with self._lock:
            notice = ""
            if self._proc is None or self._proc.poll() is not None:
                self.close()
                self._start()
                notice = "Note: the Python session was restarted; earlier variables are gone.\n"
            try:
                self._send({"op": "exec", "code": python_code})
                result = self._serve_until({"result"})
            except (BrokenPipeError, OSError, ValueError) as e:
                result = None
                logger.error(f"Sandboxed interpreter channel failed: {e}")
            if result is None:
                code = self._proc.poll() if self._proc else None
                self.close()
                return {
                    "status": "error",
                    "message": f"{notice}The sandboxed Python process exited unexpectedly (exit code {code}).",
                }
            if notice and result.get("status") == "ok":
                result["stderr"] = notice + (result.get("stderr") or "")
            return result

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self._proc = None

    def __enter__(self) -> "SandboxedInterpreter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
