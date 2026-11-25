"""
Run model-written Python outside the host process.

Code actions returned by the agents, and helper functions written by them, can be
steered by text inside IFC models and earlier execution output. They are executed in
a child process (`src/util/sandbox_worker.py`) that:

- receives none of the host's environment variables (API keys stay in the host),
- can read Python, the project's `src/` tree and the IFC models it is given, but not
  the project's `.env`, the rest of the project root, or the user's home directory,
- can write only to a private temporary directory (so it cannot add or replace
  helpers in `src/tools/created`),
- cannot open TCP connections or signal/inspect processes outside its sandbox.

Linux uses Landlock (kernel 5.13+); macOS uses sandbox-exec. Where neither is available
the interpreter refuses to start rather than run unconfined.

Helpers from `src/tools/initial` are host code that needs credentials (documentation
lookup); the child calls them over a JSON channel and never receives the key.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import weakref
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, Iterable, List, Optional

from loguru import logger

from src.config import FUNCTION_BOILERPLATE, ROOT_PATH, SRC_PATH
from src.schemas import Result, err, ok
from src.util.sandbox_worker import Channel, ProtocolError

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_worker.py")
HOST_TOOL_MODULE_PREFIX = "src.tools.initial"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

_PASSTHROUGH_ENV = ("PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ")
_LINUX_SYSTEM_READ = (
    "/usr", "/lib", "/lib64", "/lib32", "/libx32", "/bin", "/sbin",
    "/etc", "/opt", "/nix", "/proc", "/sys", "/dev",
)
_MAX_HOST_TOOL_OUTPUT = 1_000_000


class SandboxUnavailableError(RuntimeError):
    """Raised when no sandbox can be set up; code is then not executed at all."""


class SandboxedTool:
    """
    A helper function whose code only runs inside the sandboxed interpreter.

    The host keeps its name, signature and docstring (for prompts) and how the child
    should load it, but never imports or executes it.
    """

    def __init__(
        self,
        name: str,
        signature: str,
        doc: str,
        module: Optional[str] = None,
        attr: Optional[str] = None,
        source: Optional[str] = None,
    ):
        self.__name__ = name
        self.__doc__ = doc
        self.signature_text = signature
        self.module = module
        self.attr = attr
        self.source = source

    def spec(self, name: str) -> dict:
        if self.source is not None:
            return {"name": name, "kind": "source", "source": self.source, "function": self.__name__}
        return {"name": name, "kind": "module", "module": self.module, "attr": self.attr}

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Tool '{self.__name__}' only runs inside the sandboxed interpreter"
        )

    def __repr__(self) -> str:
        return f"<SandboxedTool {self.__name__}{self.signature_text}>"


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _protected_roots() -> List[str]:
    roots = [_real(ROOT_PATH)]
    home = os.path.expanduser("~")
    if home and home != "~":
        roots.append(_real(home))
    return roots


def _sandbox_policy(read_files: Iterable[str], temp_dir: str) -> dict:
    """
    Paths the child may use. A candidate that contains the project root or the home
    directory (and so .env, credentials, other projects) is only listable.
    """
    protected = _protected_roots()
    candidates = [SRC_PATH, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    candidates += [p for p in sys.path if p and os.path.isabs(p)]
    if sys.platform.startswith("linux"):
        candidates += list(_LINUX_SYSTEM_READ)

    read: List[str] = []
    listable: List[str] = []
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        path = _real(candidate)
        if any(_is_within(root, path) for root in protected):
            target = listable
        else:
            target = read
        if path not in target:
            target.append(path)

    files = []
    for path in read_files:
        if path and os.path.isfile(path):
            files.append(_real(path))

    write_files = ["/dev/null"]
    if sys.platform == "darwin":
        write_files.append("/dev/dtracehelper")

    return {
        "read": read + files,
        "list": listable,
        "write": [_real(temp_dir)],
        "write_files": write_files,
        "protected": protected,
    }


def _seatbelt_command(policy: dict) -> List[str]:
    params: Dict[str, str] = {}

    def param(prefix: str, index: int, value: str) -> str:
        key = f"{prefix}_{index}"
        params[key] = value
        return f'(param "{key}")'

    write_rules = [f"(subpath {param('WRITE', i, p)})" for i, p in enumerate(policy["write"])]
    write_rules += [f"(literal {param('WRITE_FILE', i, p)})" for i, p in enumerate(policy["write_files"])]
    protected_rules = [f"(subpath {param('PROTECTED', i, p)})" for i, p in enumerate(policy["protected"])]
    allowed_rules = [f"(subpath {param('READ', i, p)})" for i, p in enumerate(policy["read"])]
    allowed_rules += [f"(literal {param('LIST', i, p)})" for i, p in enumerate(policy["list"])]
    allowed_rules += [f"(subpath {param('WRITE', i, p)})" for i, p in enumerate(policy["write"])]

    profile = "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            f"(deny file-write* (require-not (require-any {' '.join(write_rules)})))",
            "(deny file-read-data (require-all"
            f" (require-any {' '.join(protected_rules)})"
            f" (require-not (require-any {' '.join(allowed_rules)}))))",
            '(deny sysctl-read (sysctl-name "kern.procargs") (sysctl-name "kern.procargs2"))',
        ]
    )
    command = [SANDBOX_EXEC, "-p", profile]
    for key, value in params.items():
        command += ["-D", f"{key}={value}"]
    return command


def _is_host_tool(fn: Callable) -> bool:
    return (getattr(fn, "__module__", None) or "").startswith(HOST_TOOL_MODULE_PREFIX)


def _tool_spec(name: str, fn: Any) -> dict:
    if isinstance(fn, SandboxedTool):
        return fn.spec(name)
    if _is_host_tool(fn):
        return {"name": name, "kind": "host"}
    module = getattr(fn, "__module__", None) or ""
    qualname = getattr(fn, "__qualname__", "") or ""
    if module and module != "__main__" and "<locals>" not in qualname:
        return {"name": name, "kind": "module", "module": module, "attr": qualname}
    source = textwrap.dedent(inspect.getsource(fn))
    return {"name": name, "kind": "source", "source": source, "function": fn.__name__}


def _finalize(proc: Optional[subprocess.Popen], channel: Optional[Channel], temp_dir: str) -> None:
    if channel is not None:
        channel.close()
    if proc is not None and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    shutil.rmtree(temp_dir, ignore_errors=True)


class SandboxedInterpreter:
    """Persistent Python session running in a sandboxed child process."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        extra_read_paths: Iterable[str] = (),
        tool_specs: Iterable[dict] = (),
    ):
        self.model_path = model_path
        self._tools = dict(tools or {})
        self._host_tools = {
            name: fn
            for name, fn in self._tools.items()
            if not isinstance(fn, SandboxedTool) and callable(fn) and _is_host_tool(fn)
        }
        self._specs = [_tool_spec(name, fn) for name, fn in self._tools.items()]
        self._specs += list(tool_specs)
        self._extra_read_paths = [p for p in extra_read_paths if p]
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._channel: Optional[Channel] = None
        self._finalizer = None
        self.tool_info: Dict[str, dict] = {}
        self.failures: Dict[str, str] = {}
        self._start()

    def _start(self) -> None:
        if sys.platform.startswith("linux"):
            platform = "linux"
        elif sys.platform == "darwin":
            platform = "darwin"
            if not os.path.exists(SANDBOX_EXEC):
                raise SandboxUnavailableError(f"{SANDBOX_EXEC} not found")
        else:
            raise SandboxUnavailableError(f"no sandbox available for platform {sys.platform}")

        temp_dir = tempfile.mkdtemp(prefix="cobbie-sandbox-")
        read_files = [self.model_path, *self._extra_read_paths]
        policy = _sandbox_policy([p for p in read_files if p], temp_dir)

        env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
        for key in ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
            env[key] = temp_dir
        env["ROOT_PATH"] = ROOT_PATH
        blank_env = sorted(k for k in os.environ if k not in env)

        parent_read, child_write = os.pipe()
        child_read, parent_write = os.pipe()
        command = [sys.executable, "-I", "-S", "-B", WORKER_PATH, str(child_read), str(child_write)]
        if platform == "darwin":
            command = _seatbelt_command(policy) + command

        proc = None
        channel = Channel(parent_read, parent_write)
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=os.getcwd(),
                pass_fds=(child_read, child_write),
                close_fds=True,
            )
        except BaseException:
            _finalize(None, channel, temp_dir)
            raise
        finally:
            os.close(child_read)
            os.close(child_write)

        self._proc = proc
        self._channel = channel
        self._finalizer = weakref.finalize(self, _finalize, proc, channel, temp_dir)

        sys_path = [ROOT_PATH] + [p for p in sys.path if p and os.path.isabs(p) and p != ROOT_PATH]
        try:
            channel.send(
                {
                    "op": "init",
                    "platform": platform,
                    "policy": policy,
                    "sys_path": sys_path,
                    "blank_env": blank_env,
                    "model_path": self.model_path,
                    "tools": self._specs,
                    "boilerplate": FUNCTION_BOILERPLATE,
                }
            )
            reply = self._receive_until({"ready", "init_error"})
        except (EOFError, OSError, ProtocolError, ValueError) as exc:
            self.close()
            raise SandboxUnavailableError(f"sandboxed interpreter failed to start: {exc}") from exc

        if reply["op"] == "init_error":
            self.close()
            logger.error(f"Sandboxed interpreter failed to start:\n{reply.get('traceback', '')}")
            raise SandboxUnavailableError(str(reply.get("message", "unknown error")))

        tool_info = reply.get("tools")
        failures = reply.get("failures")
        self.tool_info = tool_info if isinstance(tool_info, dict) else {}
        self.failures = failures if isinstance(failures, dict) else {}
        for name, error in self.failures.items():
            logger.warning(f"Tool '{name}' could not be loaded in the sandbox: {error}")

    def _receive_until(self, ops: set) -> dict:
        assert self._channel is not None
        while True:
            message = self._channel.recv()
            op = message.get("op")
            if op in ops:
                return message
            if op == "tool_call":
                self._channel.send(self._call_host_tool(message))
                continue
            raise ProtocolError(f"unexpected message {op!r}")

    def _call_host_tool(self, message: dict) -> dict:
        name = message.get("name")
        args = message.get("args")
        kwargs = message.get("kwargs")
        fn = self._host_tools.get(name) if isinstance(name, str) else None
        if fn is None or not isinstance(args, list) or not isinstance(kwargs, dict):
            return {"op": "tool_result", "stdout": "", "value": None, "error": "unknown tool call"}

        buffer = io.StringIO()
        value: Any = None
        error = None
        try:
            with redirect_stdout(buffer):
                value = fn(*args, **{str(k): v for k, v in kwargs.items()})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        output = buffer.getvalue()[:_MAX_HOST_TOOL_OUTPUT]
        return {"op": "tool_result", "stdout": output, "value": value, "error": error}

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def run(self, python_code: str) -> dict:
        """Execute code in the session; returns the worker's result message."""
        with self._lock:
            restarted = False
            if not self.alive:
                self.close()
                self._start()
                restarted = True
            try:
                assert self._channel is not None
                self._channel.send({"op": "exec", "code": python_code})
                result = self._receive_until({"result"})
            except (EOFError, OSError, ProtocolError, ValueError) as exc:
                code = None
                if self._proc is not None:
                    try:
                        code = self._proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                self.close()
                return {
                    "status": "error",
                    "message": f"sandboxed interpreter stopped ({exc}; exit code {code})",
                }
            if restarted:
                result["restarted"] = True
            return result

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self._proc = None
        self._channel = None
        self._finalizer = None

    def __enter__(self) -> "SandboxedInterpreter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_tool_from_source(function_name: str, code: str) -> Result[SandboxedTool, str]:
    """
    Validate model-written helper source inside the sandbox and return a tool that
    the sandboxed interpreter can load. The host never executes `code`.
    """
    spec_tool = SandboxedTool(function_name, "", "", source=textwrap.dedent(code).strip())
    try:
        with SandboxedInterpreter(tools={function_name: spec_tool}) as sandbox:
            if function_name in sandbox.failures:
                return err(sandbox.failures[function_name])
            info = sandbox.tool_info.get(function_name)
    except SandboxUnavailableError as exc:
        return err(str(exc))
    if not isinstance(info, dict):
        return err(f"Function {function_name} not found in generated code")
    return ok(
        SandboxedTool(
            function_name,
            str(info.get("signature", "(...)")),
            str(info.get("doc", "")),
            source=spec_tool.source,
        )
    )


def describe_tool_modules(package: str, modules: List[str]) -> tuple[Dict[str, SandboxedTool], Dict[str, str]]:
    """
    Import `package.<module>` for each module inside the sandbox and return the public
    functions they define as SandboxedTools, plus import failures by module name.
    """
    spec = {"name": package, "kind": "directory", "package": package, "modules": modules}
    with SandboxedInterpreter(tool_specs=[spec]) as sandbox:
        info, failures = sandbox.tool_info, sandbox.failures

    tools: Dict[str, SandboxedTool] = {}
    for name, meta in info.items():
        if not isinstance(meta, dict) or not str(name).isidentifier():
            continue
        module = str(meta.get("module", ""))
        if not module.startswith(package + "."):
            continue
        tools[name] = SandboxedTool(
            name,
            str(meta.get("signature", "(...)")),
            str(meta.get("doc", "")),
            module=module,
            attr=str(meta.get("attr", name)),
        )
    return tools, {str(k): str(v) for k, v in failures.items()}
