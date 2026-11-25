"""Structural markers of the Cobbie execution history and a parser for them.

Each executed code action is recorded by `_execute_code_action` as an
`--- Iteration N ---` block with `Code:` and `Result:` sections. The thoughts,
code and printed output inside a block are arbitrary text (printed output can
come straight from an IFC file), so every line in them that would read as one
of those markers is escaped before it is written. Marker lines in a history can
then only come from the recording template, which lets
`extract_executed_code` recover the code that was actually executed.
"""

import ast
import re
import warnings
from typing import List

_ITERATION_MARKER = re.compile(r"--- Iteration \d+ ---")
_SECTION_MARKERS = ("Code:", "Result:")
_ESCAPE_PREFIX = "\\"
_TEMPLATE_INDENT = " " * 8


def _is_marker(line: str) -> bool:
    stripped = line.strip()
    return stripped in _SECTION_MARKERS or _ITERATION_MARKER.fullmatch(stripped) is not None


def escape_history_field(text: str) -> str:
    """Escape lines of `text` that would otherwise read as history markers."""
    return "".join(
        f"{_ESCAPE_PREFIX}{line}" if _is_marker(line) else line
        for line in str(text).splitlines(keepends=True)
    )


def extract_executed_code(execution_history: str) -> List[str]:
    """Return the code sections recorded by the executor, in order."""
    executed: List[str] = []
    in_attempt = False
    code_lines: List[str] | None = None

    for line in execution_history.splitlines():
        if not _is_marker(line):
            if code_lines is not None:
                code_lines.append(line)
            continue

        stripped = line.strip()
        if _ITERATION_MARKER.fullmatch(stripped):
            in_attempt = True
            code_lines = None
        elif stripped == "Code:" and in_attempt and code_lines is None:
            code_lines = []
        elif stripped == "Result:" and code_lines is not None:
            if code_lines and code_lines[0].startswith(_TEMPLATE_INDENT):
                code_lines[0] = code_lines[0][len(_TEMPLATE_INDENT):]
            executed.append("\n".join(code_lines))
            in_attempt = False
            code_lines = None
        else:
            in_attempt = False
            code_lines = None

    return executed


def _calls_function(python_code: str, function_name: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(python_code)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
        for node in ast.walk(tree)
    )


def function_called_in_history(execution_history: str, function_name: str) -> bool:
    """Whether executor-recorded code in the history calls `function_name`."""
    return any(
        _calls_function(code, function_name)
        for code in extract_executed_code(execution_history)
    )
