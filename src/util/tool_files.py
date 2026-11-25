"""Confined filesystem access to tool source files.

A tool name is only ever used as a bare ``<name>.py`` entry of one directory.
Each operation opens that directory once and works relative to its file
descriptor without following symlinks, so a name cannot reach a file
elsewhere through "..", an absolute path, or a symlinked directory or file.
"""

import keyword
import os
import re
import stat
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

FileIdentity = Tuple[int, int]

_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OPEN_FLAGS = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class ToolFileError(Exception):
    """Raised when a tool name or tool file cannot be used safely."""


def is_valid_tool_name(name: object) -> bool:
    """Return True if name is a plain Python identifier usable as a tool file name."""
    return (
        isinstance(name, str)
        and _TOOL_NAME.fullmatch(name) is not None
        and not keyword.iskeyword(name)
    )


def _file_name(tool_name: str) -> str:
    if not is_valid_tool_name(tool_name):
        raise ToolFileError(f"Invalid tool name: {tool_name!r}")
    return f"{tool_name}.py"


@contextmanager
def _open_directory(directory: str) -> Iterator[int]:
    dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        yield dir_fd
    finally:
        os.close(dir_fd)


def _verify(fd: int, expected_identity: Optional[FileIdentity]) -> FileIdentity:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise ToolFileError("Tool path is not a regular file")
    identity = (st.st_dev, st.st_ino)
    if expected_identity is not None and identity != tuple(expected_identity):
        raise ToolFileError("Tool file changed since it was selected")
    return identity


def read_tool_source(
    directory: str,
    tool_name: str,
    expected_identity: Optional[FileIdentity] = None,
) -> Tuple[str, FileIdentity]:
    """
    Read ``<tool_name>.py`` from directory.

    Returns:
        The source code and the identity of the file it was read from.
    """
    file_name = _file_name(tool_name)
    with _open_directory(directory) as dir_fd:
        fd = os.open(file_name, os.O_RDONLY | _OPEN_FLAGS, dir_fd=dir_fd)
        with os.fdopen(fd, "r", encoding="utf-8") as file:
            identity = _verify(file.fileno(), expected_identity)
            return file.read(), identity


def write_tool_source(
    directory: str,
    tool_name: str,
    source: str,
    expected_identity: Optional[FileIdentity] = None,
    create_only: bool = False,
) -> FileIdentity:
    """
    Write ``<tool_name>.py`` in directory.

    Args:
        expected_identity: Only overwrite the existing file with this identity.
        create_only: Refuse to overwrite an existing file.

    Returns:
        The identity of the written file.
    """
    file_name = _file_name(tool_name)
    if expected_identity is not None:
        flags = os.O_WRONLY
    elif create_only:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    else:
        flags = os.O_WRONLY | os.O_CREAT
    with _open_directory(directory) as dir_fd:
        fd = os.open(file_name, flags | _OPEN_FLAGS, 0o644, dir_fd=dir_fd)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            identity = _verify(file.fileno(), expected_identity)
            file.truncate(0)
            file.write(source)
            return identity


def delete_tool_source(
    directory: str,
    tool_name: str,
    expected_identity: Optional[FileIdentity] = None,
) -> bool:
    """
    Delete ``<tool_name>.py`` from directory.

    Args:
        expected_identity: Only delete the file with this identity.

    Returns:
        True if a file was deleted, False if there was none and no identity was expected.
    """
    file_name = _file_name(tool_name)
    with _open_directory(directory) as dir_fd:
        try:
            fd = os.open(file_name, os.O_RDONLY | _OPEN_FLAGS, dir_fd=dir_fd)
        except FileNotFoundError:
            if expected_identity is not None:
                raise ToolFileError("Tool file no longer exists")
            return False
        try:
            _verify(fd, expected_identity)
        finally:
            os.close(fd)
        os.unlink(file_name, dir_fd=dir_fd)
        return True
