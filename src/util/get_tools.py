"""Generic tool loading utilities for all tool directories.

Tools in src/tools/created are written by the LLM during training. They are
never imported into this process: each one is parsed and returned as a
sandbox-only tool (see src/util/sandbox.py) that carries the exact source that
was read. By default only created tools whose sha256 is recorded in
REVIEWED_CREATED_TOOLS_PATH are returned; record a review with
scripts/review_created_tools.py.
"""

import hashlib
import importlib
import inspect
import json
import keyword
import os
from typing import Callable, Dict, List, Literal

from loguru import logger

from src.config import (
    CREATED_TOOLS_PATH,
    INITIAL_TOOLS_PATH,
    MANUAL_TOOLS_PATH,
    REVIEWED_CREATED_TOOLS_PATH,
)
from src.util.sandbox import public_functions, source_tool

# Map directory names to their paths
TOOL_PATHS = {
    "initial": INITIAL_TOOLS_PATH,
    "created": CREATED_TOOLS_PATH,
    "manual": MANUAL_TOOLS_PATH,
}


def created_tool_path(function_name: str, directory_path: str | None = None) -> str:
    """Path of the file for a created tool; rejects names that are not identifiers."""
    if (
        not isinstance(function_name, str)
        or not function_name.isidentifier()
        or keyword.iskeyword(function_name)
    ):
        raise ValueError(f"Invalid tool name: {function_name!r}")
    return os.path.join(directory_path or CREATED_TOOLS_PATH, f"{function_name}.py")


def tool_source_digest(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def load_reviewed_tools() -> Dict[str, str]:
    """Map of created tool file name to the sha256 a reviewer approved."""
    try:
        with open(REVIEWED_CREATED_TOOLS_PATH, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.error(f"Could not read {REVIEWED_CREATED_TOOLS_PATH}: {e}")
        return {}
    reviewed = data.get("tools", {}) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in reviewed.items()} if isinstance(reviewed, dict) else {}


def _load_created_tools(
    tool_path: str, allow_deletion: bool, include_unreviewed: bool
) -> Dict[str, Callable]:
    reviewed = load_reviewed_tools()
    fn_dict: Dict[str, Callable] = {}
    pending: List[str] = []

    for file_name in sorted(os.listdir(tool_path)):
        if not file_name.endswith(".py") or file_name == "__init__.py":
            continue
        module_name = file_name[:-3]
        file_path = os.path.join(tool_path, file_name)
        try:
            # Read once; the reviewed digest is checked against, and the sandbox
            # runs, these exact bytes.
            with open(file_path, "rb") as file:
                raw = file.read()
            if reviewed.get(file_name) != tool_source_digest(raw) and not include_unreviewed:
                pending.append(file_name)
                continue
            source = raw.decode("utf-8")
            for name in public_functions(source, file_path):
                fn_dict[name] = source_tool(
                    source,
                    name,
                    module_name=f"src.tools.created.{module_name}",
                    filename=file_path,
                )
        except Exception as e:
            if allow_deletion:
                logger.warning(
                    f"Could not load created tool '{module_name}'. Deleting it. Error: {e}"
                )
                try:
                    os.remove(file_path)
                    logger.info(f"Successfully deleted problematic tool file: {file_path}")
                except OSError as remove_error:
                    logger.error(f"Error deleting file {file_path}: {remove_error}")
            else:
                logger.error(f"Could not load created tool '{module_name}'. Keeping file. Error: {e}")

    if pending:
        logger.warning(
            f"Skipping {len(pending)} created tool(s) that have not been reviewed: "
            f"{', '.join(pending)}. Review them with scripts/review_created_tools.py"
        )
    logger.debug(f"Loaded {len(fn_dict)} tools from created/: {list(fn_dict.keys())}")
    return fn_dict


def get_tools_from_directory(
    directory: Literal["initial", "created", "manual"],
    allow_deletion: bool = False,
    include_unreviewed: bool = False,
) -> Dict[str, Callable]:
    """
    Load tools from specified directory using dynamic module inspection.

    Created tools are parsed, not imported, and only ever run in the sandbox.

    Args:
        directory: Which tool directory to load from ('initial', 'created', or 'manual')
        allow_deletion: If True, delete files that fail to load (created tools only)
        include_unreviewed: Also return created tools that no one has reviewed yet
            (the training loop uses this so it can test the tools it just wrote)

    Returns:
        Dictionary mapping function names to callable functions

    Raises:
        ValueError: If directory is not one of the valid options
    """
    if directory not in TOOL_PATHS:
        raise ValueError(f"Invalid directory '{directory}'. Must be one of: {list(TOOL_PATHS.keys())}")

    tool_path = TOOL_PATHS[directory]

    # Check if directory exists
    if not os.path.exists(tool_path):
        logger.warning(f"Tool directory does not exist: {tool_path}")
        return {}

    if directory == "created":
        return _load_created_tools(tool_path, allow_deletion, include_unreviewed)

    # Get all Python files in the directory (excluding __init__.py)
    try:
        python_files = [
            f[:-3]
            for f in os.listdir(tool_path)
            if f.endswith(".py") and f != "__init__.py"
        ]
    except OSError as e:
        logger.error(f"Could not list files in {tool_path}: {e}")
        return {}

    fn_dict: Dict[str, Callable] = {}

    for module_name in python_files:
        try:
            # Use absolute import path
            module = importlib.import_module(f"src.tools.{directory}.{module_name}")

            # Get all functions from the module
            for name, fn in inspect.getmembers(module):
                # Only include functions that are:
                # - actual functions (not classes, modules, etc.)
                # - defined in this module (not imported)
                # - public (don't start with underscore)
                if (
                    inspect.isfunction(fn)
                    and fn.__module__ == module.__name__
                    and not name.startswith("_")
                ):
                    fn_dict[name] = fn

        except Exception as e:
            logger.error(
                f"Could not import module '{module_name}' from {directory}/. Keeping file. Error: {e}"
            )
            continue

    logger.debug(f"Loaded {len(fn_dict)} tools from {directory}/: {list(fn_dict.keys())}")
    return fn_dict


def get_tools(
    directories: List[Literal["initial", "created", "manual"]],
    allow_created_deletion: bool = False,
    include_unreviewed: bool = False,
) -> Dict[str, Callable]:
    """
    Load tools from multiple directories with duplicate handling.

    Tools are loaded in the order specified. If the same function name appears
    in multiple directories, the version from the later directory wins.

    Args:
        directories: List of directories to load from (processed in order)
        allow_created_deletion: Whether to delete broken created tools
        include_unreviewed: Whether to include created tools no one has reviewed

    Returns:
        Combined dictionary of all tools (later directories override earlier)

    Examples:
        # Load only initial tools
        tools = get_tools(['initial'])

        # Load initial and created (default evaluation behavior)
        tools = get_tools(['initial', 'created'], allow_created_deletion=True)

        # Load all tools, with manual overriding created and initial
        tools = get_tools(['initial', 'created', 'manual'])
    """
    if not directories:
        raise ValueError("At least one directory must be specified")

    fn_dict: Dict[str, Callable] = {}
    tool_origins: Dict[str, str] = {}  # Track which directory each tool came from

    for directory in directories:
        # Determine if deletion is allowed for this directory
        allow_deletion = allow_created_deletion and directory == "created"

        # Load tools from this directory
        dir_tools = get_tools_from_directory(
            directory, allow_deletion=allow_deletion, include_unreviewed=include_unreviewed
        )

        # Check for duplicates and log warnings
        for tool_name, tool_fn in dir_tools.items():
            if tool_name in fn_dict:
                logger.warning(
                    f"Tool '{tool_name}' from '{directory}/' overrides version from '{tool_origins[tool_name]}/'"
                )
            fn_dict[tool_name] = tool_fn
            tool_origins[tool_name] = directory

    logger.info(f"Loaded {len(fn_dict)} total tools from {len(directories)} directories")
    return fn_dict
