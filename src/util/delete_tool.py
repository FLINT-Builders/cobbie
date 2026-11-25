"""Delete tools from filesystem and database."""

from typing import Optional

from loguru import logger

from src.config import CREATED_TOOLS_PATH
from src.db.query import delete_tool_from_db
from src.util.tool_files import FileIdentity, delete_tool_source


def delete_tool(tool_name: str, expected_identity: Optional[FileIdentity] = None) -> bool:
    """
    Delete tool from filesystem and database.

    Args:
        tool_name: Name of the created tool
        expected_identity: Only delete the tool file with this identity

    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete file
        if delete_tool_source(CREATED_TOOLS_PATH, tool_name, expected_identity):
            logger.info(f"Deleted tool file: {tool_name}.py")

        # Delete metadata
        delete_tool_from_db(tool_name)
        logger.info(f"Deleted tool metadata: {tool_name}")

        return True

    except Exception as e:
        logger.error(f"Error deleting tool {tool_name!r}: {e}")
        return False
