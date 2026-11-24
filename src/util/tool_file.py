"""Confined filesystem access for tool source files.

Tool names come from language-model output, so every read, write, and delete
of ``<tools-dir>/<name>.py`` goes through here. The name must be a plain
Python identifier, and the file is opened relative to a descriptor of the
tools directory without following a symlink in the final component, so the
operation cannot land outside that directory.
"""

import keyword
import os
import secrets
import stat
from contextlib import contextmanager
from typing import Iterator


def validate_tool_name(name: object) -> str:
    """Return ``name`` if it is a usable tool identifier, else raise ``ValueError``."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Invalid tool name: {name!r}")
    return name


@contextmanager
def _open_tools_dir(directory: str) -> Iterator[int]:
    dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield dir_fd
    finally:
        os.close(dir_fd)


def read_tool_source(directory: str, name: str) -> str:
    file_name = f"{validate_tool_name(name)}.py"
    with _open_tools_dir(directory) as dir_fd:
        fd = os.open(file_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
        with os.fdopen(fd, "r") as file:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"Tool file is not a regular file: {file_name}")
            return file.read()


def write_tool_source(directory: str, name: str, source: str) -> str:
    """Write ``source`` to the tool file and return its path for logging.

    The content goes to a fresh temporary file that is then renamed over the
    target, so an existing symlink or hard link at the target is replaced
    rather than written through.
    """
    file_name = f"{validate_tool_name(name)}.py"
    tmp_name = f".{file_name}.{secrets.token_hex(8)}.tmp"
    with _open_tools_dir(directory) as dir_fd:
        fd = os.open(
            tmp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o666,
            dir_fd=dir_fd,
        )
        try:
            with os.fdopen(fd, "w") as file:
                file.write(source)
            os.replace(tmp_name, file_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
            raise
    return os.path.join(directory, file_name)


def delete_tool_source(directory: str, name: str) -> bool:
    """Unlink the tool file. Returns False if it did not exist."""
    file_name = f"{validate_tool_name(name)}.py"
    with _open_tools_dir(directory) as dir_fd:
        try:
            os.unlink(file_name, dir_fd=dir_fd)
        except FileNotFoundError:
            return False
    return True
