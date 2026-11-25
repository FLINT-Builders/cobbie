"""Validation for tool names that are mapped to files in a tool library directory."""

import keyword
import os
import re

# Public, ASCII-only Python identifiers: no path separators, dots or leading underscores,
# so `<name>.py` always names a file directly inside the library directory.
_TOOL_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def is_valid_tool_name(name: object) -> bool:
    """Return True if `name` can be used as a tool name and library file name."""
    return (
        isinstance(name, str)
        and _TOOL_NAME_PATTERN.fullmatch(name) is not None
        and not keyword.iskeyword(name)
    )


def tool_file_path(tool_name: str, directory_path: str) -> str:
    """
    Return the path of the library file for `tool_name` inside `directory_path`.

    Raises:
        ValueError: If `tool_name` is not a valid tool name.
    """
    if not is_valid_tool_name(tool_name):
        raise ValueError(f"Invalid tool name: {tool_name!r}")
    return os.path.join(directory_path, f"{tool_name}.py")
