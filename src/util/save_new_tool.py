from typing import Optional

from loguru import logger

from src.config import CREATED_TOOLS_PATH
from src.db.query import register_new_tool
from src.util.tool_names import tool_file_path

def save_new_tool(function_name: str, function_implementation: str, directory_path:Optional[str] = None, global_question_num: Optional[int] = None, overwrite: bool = True)->bool:
    """
    Save the provided function code as a file named according to the provided function_name to the new_file_path. If no path is provided, will use the default location for created tools from the config file.

    Args:
        function_name: Name of the function to save. Must be a plain Python identifier.
        function_implementation: Source code of the function
        directory_path: Optional custom directory path
        global_question_num: Global question number when tool was created (for tracking)
        overwrite: If False, refuse to replace a tool file that already exists

    Returns:
        True if save succeeded, False otherwise
    """
    tool_directory = directory_path or CREATED_TOOLS_PATH
    try:
        file_path = tool_file_path(function_name, tool_directory)
    except ValueError as e:
        logger.error(f"Refusing to save tool: {e}")
        return False

    try:
        file = open(file=file_path, mode="w" if overwrite else "x")
    except FileExistsError:
        logger.error(f"Refusing to save new tool '{function_name}': {file_path} already exists")
        return False

    with file:
        file.write(function_implementation)
        logger.info(f"New function saved to {file_path}")

        # Register tool in database if global_question_num provided
        if global_question_num is not None:
            register_new_tool(function_name, global_question_num)
            logger.info(f"Tool '{function_name}' registered at question {global_question_num}")

        return True
