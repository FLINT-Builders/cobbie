"""Delete tools from filesystem and database."""

import os

from loguru import logger

from src.config import CREATED_TOOLS_PATH
from src.db.query import delete_tool_from_db
from src.util.tool_file import delete_tool_source, validate_tool_name


def delete_tool(tool_name: str) -> bool:
    """
    Delete tool from filesystem and database.

    Returns:
        True if successful, False otherwise
    """
    try:
        try:
            validate_tool_name(tool_name)
        except ValueError as e:
            # Such a name can never map to a file in the tools directory, so
            # only its stale metadata row is removed.
            logger.error(f"Refusing to delete tool file for {tool_name!r}: {e}")
            delete_tool_from_db(tool_name)
            return False

        if delete_tool_source(CREATED_TOOLS_PATH, tool_name):
            logger.info(f"Deleted tool file: {os.path.join(CREATED_TOOLS_PATH, f'{tool_name}.py')}")

        # Delete metadata
        delete_tool_from_db(tool_name)
        logger.info(f"Deleted tool metadata: {tool_name}")

        return True

    except Exception as e:
        logger.error(f"Error deleting tool {tool_name}: {e}")
        return False
