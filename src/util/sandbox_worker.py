"""
Worker process that runs model-written Python for src/util/sandbox.py.

This file is executed as a script (``python -I -S -B sandbox_worker.py``), never
imported. Until the Landlock ruleset and the seccomp filter are in place it must only
use the standard library: no project module, no tool and no user code may run before
the process is restricted.

Protocol: one JSON object per line. The parent writes to this process's stdin and reads
from its original stdout. Python-level output of executed code is captured; anything
written straight to file descriptor 1 is redirected to stderr so it cannot corrupt the
protocol stream.
"""

import code
import ctypes
import importlib
import io
import json
import os
import platform
import struct
import sys
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from functools import reduce

MAX_CAPTURED_CHARS = 2_000_000

# Landlock (include/uapi/linux/landlock.h)
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1

FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13
FS_TRUNCATE = 1 << 14
FS_IOCTL_DEV = 1 << 15

NET_BIND_TCP = 1 << 0
NET_CONNECT_TCP = 1 << 1

SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
SCOPE_SIGNAL = 1 << 1

FILE_ONLY_ACCESS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV
READ_ACCESS = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR

# prctl / seccomp (include/uapi/linux/prctl.h, seccomp.h, filter.h, audit.h)
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
EPERM = 1
BPF_LD_W_ABS = 0x20
BPF_JEQ_K = 0x15
BPF_JGE_K = 0x35
BPF_RET_K = 0x06
X32_SYSCALL_BIT = 0x40000000

# socket, io_uring_setup, io_uring_enter, io_uring_register. io_uring can create
# sockets without going through socket(2), so it is blocked as well.
SECCOMP_ARCHES = {
    "x86_64": (0xC000003E, (41, 425, 426, 427)),
    "aarch64": (0xC00000B7, (198, 425, 426, 427)),
}


class SandboxSetupError(Exception):
    pass


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    return libc


def _check(result: int, what: str) -> int:
    if result < 0:
        err = ctypes.get_errno()
        raise SandboxSetupError(f"{what} failed: {os.strerror(err)} (errno {err})")
    return result


def _fs_access_for_abi(abi: int) -> int:
    access = (1 << 13) - 1
    if abi >= 2:
        access |= FS_REFER
    if abi >= 3:
        access |= FS_TRUNCATE
    if abi >= 5:
        access |= FS_IOCTL_DEV
    return access


def apply_landlock(libc, read_dirs, read_files, read_dir_only, read_write_dirs, read_write_files) -> int:
    abi = libc.syscall(
        ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET),
        None,
        ctypes.c_size_t(0),
        ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
    )
    _check(abi, "Landlock ABI probe (is Landlock enabled in this kernel?)")

    handled_fs = _fs_access_for_abi(abi)
    handled_net = NET_BIND_TCP | NET_CONNECT_TCP if abi >= 4 else 0
    scoped = SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL if abi >= 6 else 0
    if abi >= 6:
        attr = struct.pack("QQQ", handled_fs, handled_net, scoped)
    elif abi >= 4:
        attr = struct.pack("QQ", handled_fs, handled_net)
    else:
        attr = struct.pack("Q", handled_fs)
    attr_buf = ctypes.create_string_buffer(attr, len(attr))

    ruleset_fd = _check(
        libc.syscall(
            ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET),
            attr_buf,
            ctypes.c_size_t(len(attr)),
            ctypes.c_uint32(0),
        ),
        "landlock_create_ruleset",
    )

    def add(path: str, access: int, required: bool) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            if required:
                raise SandboxSetupError(f"cannot open sandbox path {path!r}")
            return
        try:
            if not os.path.isdir(path):
                access &= FILE_ONLY_ACCESS
            access &= handled_fs
            rule = ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)
            _check(
                libc.syscall(
                    ctypes.c_long(SYS_LANDLOCK_ADD_RULE),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                    rule,
                    ctypes.c_uint32(0),
                ),
                f"landlock_add_rule({path!r})",
            )
        finally:
            os.close(fd)

    # Resolves to this process only; other processes' /proc entries (environ, mem) stay hidden.
    add("/proc/self", READ_ACCESS, required=False)
    for path in read_dirs:
        add(path, READ_ACCESS, required=False)
    for path in read_files:
        add(path, FS_READ_FILE, required=False)
    for path in read_dir_only:
        add(path, FS_READ_DIR, required=False)
    for path in read_write_files:
        add(path, FS_READ_FILE | FS_WRITE_FILE, required=False)
    for path in read_write_dirs:
        add(path, handled_fs, required=True)

    _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "prctl(PR_SET_NO_NEW_PRIVS)")
    _check(
        libc.syscall(
            ctypes.c_long(SYS_LANDLOCK_RESTRICT_SELF),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(0),
        ),
        "landlock_restrict_self",
    )
    os.close(ruleset_fd)
    return abi


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def apply_seccomp(libc) -> None:
    machine = platform.machine()
    if machine not in SECCOMP_ARCHES:
        raise SandboxSetupError(f"no seccomp filter for architecture {machine!r}")
    audit_arch, blocked = SECCOMP_ARCHES[machine]

    deny = SECCOMP_RET_ERRNO | EPERM
    program = [(BPF_LD_W_ABS, 0, 0, 4)]  # seccomp_data.arch
    # Any other ABI (e.g. 32-bit compat) is denied outright.
    program.append((BPF_JEQ_K, 1, 0, audit_arch))
    program.append((BPF_RET_K, 0, 0, deny))
    program.append((BPF_LD_W_ABS, 0, 0, 0))  # seccomp_data.nr
    checks = [(BPF_JGE_K, X32_SYSCALL_BIT)] + [(BPF_JEQ_K, nr) for nr in blocked]
    for index, (op, value) in enumerate(checks):
        remaining = len(checks) - index - 1
        # jump to the deny instruction (after the remaining checks and the allow)
        program.append((op, remaining + 1, 0, value))
    program.append((BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    program.append((BPF_RET_K, 0, 0, deny))

    raw = b"".join(struct.pack("HBBI", *insn) for insn in program)
    buf = ctypes.create_string_buffer(raw, len(raw))
    prog = _SockFprog(len(program), ctypes.cast(buf, ctypes.c_void_p))
    _check(
        libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(prog), 0, 0),
        "prctl(PR_SET_SECCOMP)",
    )


def install_project_stubs(config: dict) -> None:
    """Make ``src`` importable without running src/__init__.py, which loads .env and API keys."""
    src_dir = config["src_path"]
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [src_dir]
    src_pkg.__package__ = "src"
    cfg = types.ModuleType("src.config")
    for key, value in config["config_values"].items():
        setattr(cfg, key, value)
        setattr(src_pkg, key, value)
    src_pkg.config = cfg
    sys.modules["src"] = src_pkg
    sys.modules["src.config"] = cfg


class Channel:
    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer

    def send(self, message: dict) -> None:
        self._writer.write(json.dumps(message, default=repr).encode("utf-8") + b"\n")
        self._writer.flush()

    def receive(self) -> dict:
        line = self._reader.readline()
        if not line:
            raise EOFError("parent closed the channel")
        return json.loads(line)


def _make_host_proxy(channel: Channel, name: str):
    def proxy(*args, **kwargs):
        channel.send({"op": "call", "name": name, "args": list(args), "kwargs": kwargs})
        reply = channel.receive()
        if reply.get("stdout"):
            sys.stdout.write(reply["stdout"])
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply.get("value")

    proxy.__name__ = name
    proxy.__qualname__ = name
    return proxy


def _make_broken_tool(name: str, error: str):
    def broken(*args, **kwargs):
        raise RuntimeError(f"Helper function {name!r} could not be loaded: {error}")

    broken.__name__ = name
    return broken


def load_tools(config: dict, channel: Channel) -> dict:
    tools = {}
    source_modules: dict = {}
    for spec in config["tools"]:
        name = spec["name"]
        try:
            if spec["kind"] == "host":
                tools[name] = _make_host_proxy(channel, name)
            elif spec["kind"] == "module":
                module = importlib.import_module(spec["module"])
                tools[name] = reduce(getattr, spec["qualname"].split("."), module)
            elif spec["kind"] == "source":
                key = (spec["module"], spec["source"])
                module = source_modules.get(key)
                if module is None:
                    module = types.ModuleType(spec["module"])
                    module.__file__ = f"<generated helper {spec['module']}>"
                    sys.modules[spec["module"]] = module
                    source_modules[key] = module
                    exec(compile(config["source_tool_prelude"], "<prelude>", "exec"), module.__dict__)
                    exec(compile(spec["source"], module.__file__, "exec"), module.__dict__)
                tools[name] = getattr(module, name)
            else:
                raise ValueError(f"unknown tool kind {spec['kind']!r}")
        except BaseException as e:  # noqa: BLE001 - a broken helper must not stop the worker
            print(f"[sandbox] could not load helper {name!r}: {e!r}", file=sys.stderr)
            tools[name] = _make_broken_tool(name, repr(e))
    return tools


def _cap(text: str) -> str:
    return text if len(text) <= MAX_CAPTURED_CHARS else text[:MAX_CAPTURED_CHARS]


def run_code(interpreter: code.InteractiveInterpreter, python_code: str) -> dict:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            compiled = compile(python_code, "<string>", "exec")
        except SyntaxError as e:
            return {"op": "result", "status": "syntax_error", "message": str(e)}
        except Exception as e:
            return {"op": "result", "status": "error", "message": str(e)}
        try:
            interpreter.runcode(compiled)
        except SystemExit as e:
            stderr_buffer.write(f"SystemExit: {e.code}\n")
        except BaseException:  # noqa: BLE001 - report everything back to the agent
            stderr_buffer.write(traceback.format_exc())
    return {
        "op": "result",
        "status": "ok",
        "stdout": _cap(stdout_buffer.getvalue()),
        "stderr": _cap(stderr_buffer.getvalue()),
    }


def main() -> int:
    reader = os.fdopen(os.dup(0), "rb")
    writer = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    channel = Channel(reader, writer)

    try:
        config = channel.receive()
    except Exception as e:
        print(f"[sandbox] invalid configuration: {e!r}", file=sys.stderr)
        return 2

    try:
        if sys.platform != "linux":
            raise SandboxSetupError(f"unsupported platform {sys.platform!r}")
        for key in list(os.environ):
            if key not in config["env_allowlist"]:
                del os.environ[key]
        libc = _libc()
        apply_landlock(
            libc,
            read_dirs=config["read_dirs"],
            read_files=config["read_files"],
            read_dir_only=config["read_dir_only"],
            read_write_dirs=config["read_write_dirs"],
            read_write_files=config["read_write_files"],
        )
        apply_seccomp(libc)
    except Exception as e:
        channel.send({"op": "fatal", "message": f"{type(e).__name__}: {e}"})
        return 3

    sys.path[:] = config["sys_path"]
    sys.dont_write_bytecode = True
    install_project_stubs(config)

    namespace: dict = {}
    if config.get("model_path"):
        namespace["path_ifc_model"] = config["model_path"]
    namespace.update(load_tools(config, channel))
    interpreter = code.InteractiveInterpreter(namespace)
    interpreter.runcode(compile(config["boilerplate"], "<setup>", "exec"))
    channel.send({"op": "ready"})

    while True:
        try:
            message = channel.receive()
        except EOFError:
            return 0
        if message.get("op") == "exec":
            channel.send(run_code(interpreter, message.get("code", "")))
        elif message.get("op") == "close":
            return 0


if __name__ == "__main__":
    sys.exit(main())
