"""Worker process that runs agent-generated Python code inside an OS sandbox.

Started by src/util/sandbox.py as ``python -I -B _sandbox_worker.py``. It must not
import anything from ``src``: importing src.config would load the .env file and
put provider keys into this process.

Before any untrusted code runs (tool modules included), the worker confines
itself and cannot undo it afterwards:

- Linux: Landlock grants read-only access to the Python installation, the shared
  library directories and the IFC model files the orchestrator named. Nothing is
  writable and TCP bind/connect is denied. A seccomp filter refuses socket
  creation, exec, ptrace, io_uring, namespace and mount calls, and metadata
  changes that Landlock does not cover.
- macOS: the orchestrator starts this process under ``sandbox-exec`` with a
  deny-by-default profile, and the worker refuses to run if it is not sandboxed.

Any other platform, or a kernel without Landlock, makes the worker exit with a
fatal message instead of running code unconfined.

Messages are length-prefixed JSON on the original stdin/stdout pipes.
"""

import builtins
import code
import ctypes
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import linecache
import os
import platform
import resource
import signal
import site
import stat
import struct
import sys
import sysconfig

_HEADER = struct.Struct(">Q")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class Channel:
    def __init__(self, read_fd: int, write_fd: int):
        self.read_fd = read_fd
        self.write_fd = write_fd

    def send(self, message: dict) -> None:
        data = json.dumps(message).encode("utf-8")
        payload = _HEADER.pack(len(data)) + data
        view = memoryview(payload)
        while view:
            written = os.write(self.write_fd, view)
            view = view[written:]

    def _read_exact(self, size: int) -> bytes:
        chunks = []
        while size:
            chunk = os.read(self.read_fd, min(size, 1 << 20))
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def receive(self) -> dict:
        (size,) = _HEADER.unpack(self._read_exact(_HEADER.size))
        if size > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
        return json.loads(self._read_exact(size).decode("utf-8"))


class CappedTextIO(io.TextIOBase):
    """stdout/stderr replacement that keeps at most ``limit`` characters."""

    encoding = "utf-8"

    def __init__(self, limit: int):
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0
        self.dropped = False

    def writable(self) -> bool:
        return True

    def write(self, text) -> int:
        text = str(text)
        room = self.limit - self.size
        if room > 0:
            kept = text[:room]
            self.parts.append(kept)
            self.size += len(kept)
        if len(text) > max(room, 0):
            self.dropped = True
        return len(text)

    def getvalue(self) -> str:
        value = "".join(self.parts)
        if self.dropped:
            value += f"\n...\nWARNING:\nOutput cut at {self.limit} characters"
        return value


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


def apply_limits(limits: dict) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    memory = limits.get("memory_bytes")
    if memory and sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))


# ---------------------------------------------------------------------------
# Linux: Landlock + seccomp
# ---------------------------------------------------------------------------

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_ABI1_ALL = (1 << 13) - 1
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15
_NET_BIND_TCP = 1 << 0
_NET_CONNECT_TCP = 1 << 1
_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
_SCOPE_SIGNAL = 1 << 1

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
        _LIBC.syscall.restype = ctypes.c_long
        _LIBC.prctl.restype = ctypes.c_int
    return _LIBC


def _syscall(number: int, *args) -> int:
    result = _libc().syscall(ctypes.c_long(number), *args)
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return result


def _prctl(option: int, arg2: int, arg3: int = 0, arg4: int = 0, arg5: int = 0) -> None:
    if _libc().prctl(
        ctypes.c_int(option),
        ctypes.c_ulong(arg2),
        ctypes.c_ulong(arg3),
        ctypes.c_ulong(arg4),
        ctypes.c_ulong(arg5),
    ) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _python_read_dirs() -> list[str]:
    prefixes = {
        os.path.realpath(p)
        for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    }
    candidates = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            candidates.add(value)
    candidates.update(site.getsitepackages())
    for entry in sys.path:
        if not entry:
            continue
        real = os.path.realpath(entry)
        if any(_is_within(real, prefix) for prefix in prefixes):
            candidates.add(entry)
    return sorted(candidates)


def _library_read_paths() -> list[str]:
    paths = {
        "/lib",
        "/lib32",
        "/lib64",
        "/usr/lib",
        "/usr/lib32",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/share/zoneinfo",
        "/etc/ld.so.cache",
        "/etc/localtime",
        "/proc/cpuinfo",
        "/proc/meminfo",
        "/proc/stat",
        "/sys/devices/system/cpu",
        "/dev/urandom",
    }
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if entry:
            paths.add(entry)
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
            for line in maps:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    paths.add(os.path.dirname(parts[5].strip()))
    except OSError:
        pass
    return sorted(paths)


def _allowed_read_paths(init: dict) -> list[str]:
    protected = [os.path.realpath(p) for p in init.get("protected_paths", []) if p]
    allowed = []
    for path in _python_read_dirs() + _library_read_paths():
        if not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        # A directory that contains the repository, the .env file or the home
        # directory would expose the secrets this sandbox exists to protect.
        if real == "/" or any(_is_within(p, real) for p in protected):
            continue
        allowed.append(real)
    return allowed


def _add_path_rule(ruleset_fd: int, path: str, access: int) -> None:
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            access &= _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV
        attr = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
        _syscall(
            _SYS_LANDLOCK_ADD_RULE,
            ctypes.c_long(ruleset_fd),
            ctypes.c_long(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_long(0),
        )
    finally:
        os.close(fd)


def _apply_landlock(init: dict) -> None:
    try:
        abi = _syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(None),
            ctypes.c_long(0),
            ctypes.c_long(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError as e:
        raise RuntimeError(f"Landlock is not available on this kernel ({e})") from None

    handled_fs = _FS_ABI1_ALL
    if abi >= 2:
        handled_fs |= _FS_REFER
    if abi >= 3:
        handled_fs |= _FS_TRUNCATE
    if abi >= 5:
        handled_fs |= _FS_IOCTL_DEV
    attr = _RulesetAttr(
        handled_access_fs=handled_fs,
        handled_access_net=(_NET_BIND_TCP | _NET_CONNECT_TCP) if abi >= 4 else 0,
        scoped=(_SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL) if abi >= 6 else 0,
    )
    size = 8 if abi < 4 else 16 if abi < 6 else 24
    ruleset_fd = _syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.c_long(size),
        ctypes.c_long(0),
    )
    try:
        read_access = _FS_READ_FILE | _FS_READ_DIR
        for path in _allowed_read_paths(init):
            _add_path_rule(ruleset_fd, path, read_access)
        for path in init.get("readable_files", []):
            if os.path.isfile(path):
                _add_path_rule(ruleset_fd, path, _FS_READ_FILE)
        _add_path_rule(ruleset_fd, os.devnull, _FS_READ_FILE | _FS_WRITE_FILE)
        _prctl(_PR_SET_NO_NEW_PRIVS, 1)
        _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_long(ruleset_fd), ctypes.c_long(0))
    finally:
        os.close(ruleset_fd)


_AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}

# Syscalls refused with EPERM. Numbers from the kernel's unistd tables.
_DENIED_SYSCALLS = {
    "x86_64": {
        "socket": 41, "socketpair": 53, "execve": 59, "execveat": 322,
        "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311,
        "keyctl": 250, "add_key": 248, "request_key": 249, "bpf": 321,
        "perf_event_open": 298, "userfaultfd": 323, "io_uring_setup": 425,
        "io_uring_enter": 426, "io_uring_register": 427, "mount": 165,
        "umount2": 166, "unshare": 272, "setns": 308, "pivot_root": 155,
        "chroot": 161, "kexec_load": 246, "kexec_file_load": 320,
        "init_module": 175, "finit_module": 313, "open_by_handle_at": 304,
        "name_to_handle_at": 303, "fanotify_init": 300, "mount_setattr": 442,
        "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431,
        "fsmount": 432, "fspick": 433, "pidfd_open": 434, "pidfd_getfd": 438,
        "pidfd_send_signal": 424, "tkill": 200, "truncate": 76,
        "chmod": 90, "fchmod": 91, "fchmodat": 268, "fchmodat2": 452,
        "chown": 92, "fchown": 93, "lchown": 94, "fchownat": 260,
        "setxattr": 188, "lsetxattr": 189, "fsetxattr": 190,
        "removexattr": 197, "lremovexattr": 198, "fremovexattr": 199,
        "utime": 132, "utimes": 235, "futimesat": 261, "utimensat": 280,
    },
    "aarch64": {
        "socket": 198, "socketpair": 199, "execve": 221, "execveat": 281,
        "ptrace": 117, "process_vm_readv": 270, "process_vm_writev": 271,
        "keyctl": 219, "add_key": 217, "request_key": 218, "bpf": 280,
        "perf_event_open": 241, "userfaultfd": 282, "io_uring_setup": 425,
        "io_uring_enter": 426, "io_uring_register": 427, "mount": 40,
        "umount2": 39, "unshare": 97, "setns": 268, "pivot_root": 41,
        "chroot": 51, "kexec_load": 104, "kexec_file_load": 294,
        "init_module": 105, "finit_module": 273, "open_by_handle_at": 265,
        "name_to_handle_at": 264, "fanotify_init": 262, "mount_setattr": 442,
        "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431,
        "fsmount": 432, "fspick": 433, "pidfd_open": 434, "pidfd_getfd": 438,
        "pidfd_send_signal": 424, "tkill": 130, "truncate": 45,
        "fchmod": 52, "fchmodat": 53, "fchmodat2": 452,
        "fchown": 55, "fchownat": 54,
        "setxattr": 5, "lsetxattr": 6, "fsetxattr": 7,
        "removexattr": 14, "lremovexattr": 15, "fremovexattr": 16,
        "utimensat": 88,
    },
}
# kill/tgkill are allowed only when the target is this process.
_SELF_SIGNAL_SYSCALLS = {"x86_64": (62, 234), "aarch64": (129, 131)}

_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_JGE_K = 0x35
_BPF_RET_K = 0x06
_RET_ALLOW = 0x7FFF0000
_RET_ERRNO_EPERM = 0x00050000 | 1
_RET_KILL_PROCESS = 0x80000000


def _seccomp_program(machine: str, pid: int) -> list[tuple[int, int, int, int]]:
    """Return the BPF program as (code, jt, jf, k) tuples."""
    arch = _AUDIT_ARCH[machine]
    denied = sorted(set(_DENIED_SYSCALLS[machine].values()))
    self_signal = _SELF_SIGNAL_SYSCALLS[machine]

    # Built with symbolic jump targets, resolved below.
    program: list[tuple] = [
        (_BPF_LD_W_ABS, None, None, 4),
        (_BPF_JEQ_K, None, "kill", arch),
        (_BPF_LD_W_ABS, None, None, 0),
    ]
    if machine == "x86_64":
        program.append((_BPF_JGE_K, "deny", None, 0x40000000))
    for number in denied:
        program.append((_BPF_JEQ_K, "deny", None, number))
    for number in self_signal:
        program.append((_BPF_JEQ_K, "self_signal", None, number))
    program.append((_BPF_RET_K, None, None, _RET_ALLOW))
    # BPF jumps only go forward, so this block has its own allow return.
    program.append(("label", "self_signal"))
    # The kernel reads pid_t from the low 32 bits of the first argument.
    program.append((_BPF_LD_W_ABS, None, None, 16))
    program.append((_BPF_JEQ_K, None, "deny", pid))
    program.append((_BPF_RET_K, None, None, _RET_ALLOW))
    program.append(("label", "deny"))
    program.append((_BPF_RET_K, None, None, _RET_ERRNO_EPERM))
    program.append(("label", "kill"))
    program.append((_BPF_RET_K, None, None, _RET_KILL_PROCESS))

    labels: dict[str, int] = {}
    position = 0
    for item in program:
        if item[0] == "label":
            labels[item[1]] = position
        else:
            position += 1

    resolved = []
    position = 0
    for item in program:
        if item[0] == "label":
            continue
        code_, jt, jf, k = item

        def offset(target):
            if target is None:
                return 0
            delta = labels[target] - position - 1
            if not 0 <= delta <= 255:
                raise RuntimeError("seccomp jump out of range")
            return delta

        resolved.append((code_, offset(jt), offset(jf), k))
        position += 1
    return resolved


def _apply_seccomp() -> None:
    machine = platform.machine().lower()
    if machine == "arm64":
        machine = "aarch64"
    if machine not in _AUDIT_ARCH:
        raise RuntimeError(f"no seccomp filter for architecture {machine!r}")
    program = _seccomp_program(machine, os.getpid())
    filters = (_SockFilter * len(program))(*[_SockFilter(*ins) for ins in program])
    fprog = _SockFprog(len(program), filters)
    _prctl(_PR_SET_NO_NEW_PRIVS, 1)
    _prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.addressof(fprog))


def _single_threaded() -> bool:
    try:
        return len(os.listdir("/proc/self/task")) == 1
    except OSError:
        return False


def apply_sandbox(init: dict) -> None:
    if sys.platform.startswith("linux"):
        # Landlock and seccomp apply to the calling thread and its future
        # children only, so no other thread may exist yet.
        if not _single_threaded():
            raise RuntimeError("worker must be single-threaded before sandboxing")
        _apply_landlock(init)
        _apply_seccomp()
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None)
        libc.sandbox_check.restype = ctypes.c_int
        libc.sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        if libc.sandbox_check(os.getpid(), None, 0) != 1:
            raise RuntimeError("worker is not running under sandbox-exec")
    else:
        raise RuntimeError(f"no sandbox implementation for platform {sys.platform!r}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class SourceModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve tool modules from source text sent by the orchestrator.

    The ``src`` package is answered only from that source map, so tool code
    cannot reach the real repository modules (src.config among them).
    """

    def __init__(self, modules: dict):
        self.modules = modules
        self.packages = {"src"}
        for name in modules:
            parts = name.split(".")
            for i in range(1, len(parts)):
                self.packages.add(".".join(parts[:i]))
        for name, entry in modules.items():
            lines = entry["source"].splitlines(keepends=True)
            linecache.cache[entry["filename"]] = (
                len(entry["source"]), None, lines, entry["filename"]
            )

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(
                fullname, self, origin=self.modules[fullname]["filename"]
            )
        if fullname in self.packages:
            spec = importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname == "src" or fullname.startswith("src."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        entry = self.modules.get(module.__name__)
        if entry is None:
            return
        module.__file__ = entry["filename"]
        exec(compile(entry["source"], entry["filename"], "exec"), module.__dict__)


def _default_tool_globals() -> dict:
    import math

    import ifcopenshell
    import ifcopenshell.entity_instance
    import ifcopenshell.file
    import ifcopenshell.geom
    import ifcopenshell.util.element
    import ifcopenshell.util.geolocation
    import ifcopenshell.util.placement
    import ifcopenshell.util.shape
    import ifcopenshell.util.system
    import numpy
    import pandas

    return {
        "__builtins__": builtins,
        "__name__": "generated_tool",
        "ifcopenshell": ifcopenshell,
        "ifcopenshell.util.element": ifcopenshell.util.element,
        "ifcopenshell.util.shape": ifcopenshell.util.shape,
        "ifcopenshell.util.placement": ifcopenshell.util.placement,
        "ifcopenshell.util.geolocation": ifcopenshell.util.geolocation,
        "ifcopenshell.util.system": ifcopenshell.util.system,
        "ifcopenshell.geom": ifcopenshell.geom,
        "ifcopenshell.file": ifcopenshell.file,
        "ifcopenshell.entity_instance": ifcopenshell.entity_instance,
        "math": math,
        "numpy": numpy,
        "pandas": pandas,
    }


def _error_wrapped(function, name: str):
    def tool_wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            return f"Error processing the function with the `tool_wrapper` {name}: {str(e)}"

    tool_wrapper.__name__ = name
    tool_wrapper.__doc__ = function.__doc__
    tool_wrapper.__wrapped__ = function  # type: ignore[attr-defined]
    return tool_wrapper


def _host_stub(channel: Channel, name: str):
    def host_tool(*args, **kwargs):
        channel.send(
            {"type": "host_call", "name": name, "args": list(args), "kwargs": kwargs}
        )
        reply = channel.receive()
        if reply.get("type") != "host_result":
            raise RuntimeError("unexpected reply from the orchestrator")
        if reply.get("stdout"):
            sys.stdout.write(str(reply["stdout"]))
        if reply.get("stderr"):
            sys.stderr.write(str(reply["stderr"]))
        if reply.get("ok"):
            return reply.get("value")
        raise RuntimeError(reply.get("error", "host tool failed"))

    host_tool.__name__ = name
    return host_tool


def define_tools(channel: Channel, init: dict, namespace: dict) -> list[str]:
    modules = dict(init.get("modules", {}))
    for spec in init.get("tools", []):
        if spec["kind"] == "source" and spec.get("module"):
            modules.setdefault(
                spec["module"], {"source": spec["source"], "filename": spec["filename"]}
            )
    sys.meta_path.insert(0, SourceModuleFinder(modules))

    warnings = []
    for spec in init.get("tools", []):
        name = spec["name"]
        try:
            if spec["kind"] == "host":
                namespace[name] = _host_stub(channel, name)
            elif spec["kind"] == "module":
                module = importlib.import_module(spec["module"])
                namespace[name] = getattr(module, spec["attr"])
            elif spec["kind"] == "source" and spec.get("module"):
                module = importlib.import_module(spec["module"])
                namespace[name] = getattr(module, spec["function"])
            elif spec["kind"] == "source":
                scope = _default_tool_globals()
                filename = spec.get("filename") or f"<generated tool {name}>"
                linecache.cache[filename] = (
                    len(spec["source"]), None,
                    spec["source"].splitlines(keepends=True), filename,
                )
                exec(compile(spec["source"], filename, "exec"), scope)
                function = scope[spec["function"]]
                if spec.get("wrap_errors"):
                    function = _error_wrapped(function, name)
                namespace[name] = function
            else:
                raise ValueError(f"unknown tool kind {spec['kind']!r}")
        except Exception as e:
            warnings.append(f"Tool {name!r} could not be loaded: {type(e).__name__}: {e}")
    return warnings


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_code(interpreter: code.InteractiveInterpreter, source: str, limit: int) -> dict:
    try:
        compiled = compile(source, "<string>", "exec")
    except SyntaxError as e:
        return {"type": "result", "syntax_error": str(e), "stdout": "", "stderr": ""}
    except Exception as e:
        return {"type": "result", "runtime_error": str(e), "stdout": "", "stderr": ""}

    stdout, stderr = CappedTextIO(limit), CappedTextIO(limit)
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        interpreter.runcode(compiled)
    except SystemExit as e:
        stderr.write(f"SystemExit: {e.code}\n")
    finally:
        sys.stdout, sys.stderr = saved
    return {"type": "result", "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def main() -> int:
    read_fd, write_fd = os.dup(0), os.dup(1)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    os.close(devnull)
    channel = Channel(read_fd, write_fd)

    try:
        init = channel.receive()
    except (EOFError, ValueError):
        return 1

    try:
        apply_limits(init.get("limits", {}))
        apply_sandbox(init)
    except Exception as e:
        channel.send({"type": "fatal", "error": f"{type(e).__name__}: {e}"})
        return 1

    namespace: dict = {"__name__": "__main__", "__builtins__": builtins}
    if init.get("model_path"):
        namespace["path_ifc_model"] = init["model_path"]
    interpreter = code.InteractiveInterpreter(namespace)
    limit = int(init.get("output_limit", 1_000_000))

    try:
        exec(compile(init.get("boilerplate", ""), "<setup>", "exec"), namespace)
    except Exception as e:
        channel.send({"type": "fatal", "error": f"setup failed: {type(e).__name__}: {e}"})
        return 1
    warnings = define_tools(channel, init, namespace)
    channel.send({"type": "ready", "warnings": warnings})

    while True:
        try:
            message = channel.receive()
        except (EOFError, ValueError):
            return 0
        if message.get("type") == "exec":
            channel.send(run_code(interpreter, message.get("code", ""), limit))
        elif message.get("type") == "close":
            return 0


if __name__ == "__main__":
    sys.exit(main())
