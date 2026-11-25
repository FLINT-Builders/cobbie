"""
Child process that runs model-written Python for the code-executing agents.

`src.util.sandbox` starts this file with `python -I -S -B`, an environment that holds no
credentials, and a private temporary directory as HOME and cwd. Before any project
module, tool or model code is imported, the process confines itself (Landlock on
Linux; on macOS the parent starts it under sandbox-exec) so that it can read code and
IFC models but cannot read the project's .env or the user's home directory, write
outside its temporary directory, open network connections, or inspect the parent.

Only the standard library may be imported at module level: the parent imports
`Channel` from here, and the child must reach the sandbox step without project code.
"""

import ctypes
import inspect
import io
import json
import os
import stat
import struct
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_CHARS = 1_000_000
MAX_DOC_CHARS = 20_000


class ProtocolError(Exception):
    pass


class SandboxError(Exception):
    pass


class Channel:
    """Length-prefixed JSON messages over a pair of pipe file descriptors."""

    def __init__(self, read_fd: int, write_fd: int):
        self._read_fd = read_fd
        self._write_fd = write_fd

    def send(self, message: dict) -> None:
        data = json.dumps(message).encode("utf-8")
        if len(data) > MAX_MESSAGE_BYTES:
            raise ProtocolError("message too large")
        view = memoryview(struct.pack(">Q", len(data)) + data)
        while view:
            written = os.write(self._write_fd, view)
            view = view[written:]

    def recv(self) -> dict:
        (length,) = struct.unpack(">Q", self._read_exact(8))
        if length > MAX_MESSAGE_BYTES:
            raise ProtocolError("message too large")
        message = json.loads(self._read_exact(length).decode("utf-8"))
        if not isinstance(message, dict):
            raise ProtocolError("message is not an object")
        return message

    def _read_exact(self, size: int) -> bytes:
        chunks = []
        while size:
            chunk = os.read(self._read_fd, min(size, 1 << 20))
            if not chunk:
                raise EOFError("channel closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        for fd in (self._read_fd, self._write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


# --------------------------------------------------------------------------
# Landlock (Linux)
# --------------------------------------------------------------------------

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15
_FS_FILE_RIGHTS = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV

_NET_BIND_TCP = 1 << 0
_NET_CONNECT_TCP = 1 << 1
_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
_SCOPE_SIGNAL = 1 << 1


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def apply_landlock(policy: dict) -> None:
    """
    Restrict this process (and anything it later executes) to `policy`.

    policy keys: "read" (read + execute beneath), "list" (directory listing only),
    "write" (full access beneath), "write_files" (read/write of single files).
    Any right not granted by a rule is denied; TCP and signals/abstract sockets
    outside the sandbox are denied when the kernel supports it.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    abi = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        None,
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if abi < 1:
        raise SandboxError(
            f"Landlock is not available on this kernel (errno {ctypes.get_errno()})"
        )

    handled_fs = (1 << 13) - 1
    if abi >= 2:
        handled_fs |= _FS_REFER
    if abi >= 3:
        handled_fs |= _FS_TRUNCATE
    if abi >= 5:
        handled_fs |= _FS_IOCTL_DEV

    attr = _RulesetAttr(handled_fs, 0, 0)
    attr_size = 8
    if abi >= 4:
        attr.handled_access_net = _NET_BIND_TCP | _NET_CONNECT_TCP
        attr_size = 16
    if abi >= 6:
        attr.scoped = _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL
        attr_size = 24

    ruleset_fd = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(attr_size),
        ctypes.c_uint32(0),
    )
    if ruleset_fd < 0:
        raise SandboxError(f"landlock_create_ruleset failed (errno {ctypes.get_errno()})")

    def add_rule(path: str, access: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                access &= _FS_FILE_RIGHTS
            access &= handled_fs
            if not access:
                return
            rule = _PathBeneathAttr(access, fd)
            result = libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
                ctypes.c_int(ruleset_fd),
                ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(rule),
                ctypes.c_uint32(0),
            )
            if result != 0:
                raise SandboxError(
                    f"landlock_add_rule failed for {path} (errno {ctypes.get_errno()})"
                )
        finally:
            os.close(fd)

    try:
        for path in policy.get("read", []):
            add_rule(path, _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR | _FS_IOCTL_DEV)
        for path in policy.get("list", []):
            add_rule(path, _FS_READ_DIR)
        for path in policy.get("write", []):
            add_rule(path, handled_fs)
        for path in policy.get("write_files", []):
            add_rule(path, _FS_READ_FILE | _FS_WRITE_FILE | _FS_TRUNCATE)

        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise SandboxError(f"prctl(NO_NEW_PRIVS) failed (errno {ctypes.get_errno()})")
        if libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF), ctypes.c_int(ruleset_fd), ctypes.c_uint32(0)
        ) != 0:
            raise SandboxError(f"landlock_restrict_self failed (errno {ctypes.get_errno()})")
    finally:
        os.close(ruleset_fd)


# --------------------------------------------------------------------------
# seccomp (Linux)
# --------------------------------------------------------------------------

# Landlock does not mediate connecting to existing Unix sockets (docker.sock, ssh
# agents) and cannot block TCP before kernel 6.7, so socket creation (including via
# io_uring) is refused outright. truncate/chmod/chown are refused because Landlock
# does not cover them on all kernels and they could damage helper files.
_AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}
_DENIED_SYSCALLS = {
    "x86_64": (41, 425, 76, 90, 91, 92, 93, 94, 260, 268, 452),
    "aarch64": (198, 425, 45, 52, 53, 54, 55, 452),
}
_X32_SYSCALL_BIT = 0x40000000
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO_EPERM = 0x00050000 | 1
_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_JGE_K = 0x35
_BPF_RET_K = 0x06


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def apply_seccomp() -> None:
    machine = os.uname().machine
    if machine not in _AUDIT_ARCH:
        raise SandboxError(f"no seccomp filter for architecture {machine!r}")
    denied = _DENIED_SYSCALLS[machine]

    # Layout: load arch, check, load nr, [x32 check], one JEQ per denied syscall,
    # RET ALLOW, RET EPERM. Jump offsets are relative to the next instruction.
    checks = 1 if machine == "x86_64" else 0
    count = 3 + checks + len(denied) + 2
    deny_index = count - 1
    program = [
        (_BPF_LD_W_ABS, 0, 0, 4),
        (_BPF_JEQ_K, 0, deny_index - 2, _AUDIT_ARCH[machine]),
        (_BPF_LD_W_ABS, 0, 0, 0),
    ]
    if checks:
        program.append((_BPF_JGE_K, deny_index - len(program) - 1, 0, _X32_SYSCALL_BIT))
    for nr in denied:
        program.append((_BPF_JEQ_K, deny_index - len(program) - 1, 0, nr))
    program.append((_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW))
    program.append((_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO_EPERM))
    assert len(program) == count

    filters = (_SockFilter * count)(*[_SockFilter(*ins) for ins in program])
    fprog = _SockFprog(count, filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SandboxError(f"prctl(NO_NEW_PRIVS) failed (errno {ctypes.get_errno()})")
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise SandboxError(f"seccomp filter failed (errno {ctypes.get_errno()})")


def verify_confinement(policy: dict) -> None:
    """Refuse to continue if writing into the project root is still possible."""
    for root in policy.get("protected", []):
        probe = os.path.join(root, f".cobbie-sandbox-probe-{os.getpid()}")
        try:
            fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            continue
        os.close(fd)
        os.unlink(probe)
        raise SandboxError(f"sandbox is not enforced: {root} is writable")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"


def _describe(fn) -> dict:
    try:
        signature = str(inspect.signature(fn))
    except (TypeError, ValueError):
        signature = "(...)"
    doc = inspect.getdoc(fn) or ""
    return {"signature": _cap(signature, MAX_DOC_CHARS), "doc": _cap(doc, MAX_DOC_CHARS)}


def _host_tool(channel: Channel, name: str):
    def call_host_tool(*args, **kwargs):
        channel.send({"op": "tool_call", "name": name, "args": list(args), "kwargs": kwargs})
        reply = channel.recv()
        if reply.get("op") != "tool_result":
            raise ProtocolError("unexpected reply to tool call")
        output = reply.get("stdout")
        if isinstance(output, str) and output:
            sys.stdout.write(output)
        if reply.get("error"):
            raise RuntimeError(str(reply["error"]))
        return reply.get("value")

    call_host_tool.__name__ = name
    return call_host_tool


def _load_tools(channel: Channel, specs: list) -> tuple[dict, dict, dict]:
    import importlib

    tools: dict = {}
    info: dict = {}
    failures: dict = {}

    for spec in specs:
        kind = spec.get("kind")
        name = spec.get("name")
        try:
            if kind == "host":
                tools[name] = _host_tool(channel, name)
                continue
            if kind == "module":
                obj = importlib.import_module(spec["module"])
                for part in spec["attr"].split("."):
                    obj = getattr(obj, part)
                tools[name] = obj
                info[name] = {**_describe(obj), "module": spec["module"], "attr": spec["attr"]}
                continue
            if kind == "source":
                from src.util._create_function_from_source_code import (
                    _create_function_from_source_code,
                )

                created = _create_function_from_source_code(
                    code=spec["source"], function_name=spec["function"]
                )
                if created.is_err():
                    failures[name] = str(created.unwrap_err())
                    continue
                tools[name] = created.unwrap()
                info[name] = _describe(tools[name])
                continue
            if kind == "directory":
                package = spec["package"]
                for module_name in spec["modules"]:
                    try:
                        module = importlib.import_module(f"{package}.{module_name}")
                    except BaseException as exc:
                        failures[module_name] = f"{type(exc).__name__}: {exc}"
                        continue
                    for attr, fn in inspect.getmembers(module):
                        if (
                            inspect.isfunction(fn)
                            and fn.__module__ == module.__name__
                            and not attr.startswith("_")
                        ):
                            tools[attr] = fn
                            info[attr] = {
                                **_describe(fn),
                                "module": module.__name__,
                                "attr": attr,
                            }
                continue
            failures[str(name)] = f"unknown tool kind {kind!r}"
        except BaseException as exc:
            failures[str(name)] = f"{type(exc).__name__}: {exc}"

    return tools, info, failures


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _run_code(interpreter, code: str) -> dict:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            compiled = compile(code, "<string>", "exec")
        except SyntaxError as exc:
            return {"status": "syntax_error", "message": str(exc)}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        try:
            interpreter.runcode(compiled)
        except BaseException:
            interpreter.showtraceback()
    return {
        "status": "ok",
        "stdout": _cap(stdout_buffer.getvalue(), MAX_OUTPUT_CHARS),
        "stderr": _cap(stderr_buffer.getvalue(), MAX_OUTPUT_CHARS),
    }


def _neutralize_dotenv() -> None:
    # Project modules call load_dotenv(find_dotenv()) on import; the sandbox denies
    # reading .env, so make those calls no-ops instead of failing the import.
    try:
        import dotenv
    except ImportError:
        return
    setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    setattr(dotenv, "find_dotenv", lambda *args, **kwargs: "")


def main(read_fd: int, write_fd: int) -> int:
    channel = Channel(read_fd, write_fd)
    try:
        init = channel.recv()
    except (EOFError, ProtocolError, ValueError):
        return 1

    try:
        if init.get("op") != "init":
            raise ProtocolError("expected init message")

        sys.path[:] = [p for p in init["sys_path"] if isinstance(p, str)]
        # Withheld variables are present but empty so modules that index os.environ
        # (src.config does) still import.
        for name in init.get("blank_env", []):
            os.environ.setdefault(name, "")

        if init["platform"] == "linux":
            apply_seccomp()
            apply_landlock(init["policy"])
        elif init["platform"] != "darwin":
            raise SandboxError(f"no sandbox for platform {init['platform']!r}")
        verify_confinement(init["policy"])

        _neutralize_dotenv()

        from code import InteractiveInterpreter

        tools, info, failures = _load_tools(channel, init.get("tools", []))
        interpreter_globals: dict = {}
        if init.get("model_path"):
            interpreter_globals["path_ifc_model"] = init["model_path"]
        interpreter_globals.update(tools)
        interpreter = InteractiveInterpreter(interpreter_globals)
        interpreter.runcode(compile(init.get("boilerplate", ""), "<setup>", "exec"))
    except BaseException as exc:
        try:
            channel.send(
                {
                    "op": "init_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": _cap(traceback.format_exc(), MAX_OUTPUT_CHARS),
                }
            )
        except Exception:
            pass
        return 1

    channel.send({"op": "ready", "tools": info, "failures": failures})

    while True:
        try:
            message = channel.recv()
        except (EOFError, ProtocolError, ValueError):
            return 0
        op = message.get("op")
        if op == "exec":
            channel.send({"op": "result", **_run_code(interpreter, str(message.get("code", "")))})
        elif op == "close":
            return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))
