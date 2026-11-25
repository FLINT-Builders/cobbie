import os
from typing import Optional

from loguru import logger

from src.config import CREATED_TOOLS_PATH
from src.db.query import register_new_tool
from src.util.tool_names import is_valid_tool_name

def save_new_tool(function_name: str, function_implementation: str, directory_path:Optional[str] = None, global_question_num: Optional[int] = None)->bool:
    """
    Save the provided function code as a file named according to the provided function_name to the new_file_path. If no path is provided, will use the default location for created tools from the config file.

    Args:
        function_name: Name of the function to save
        function_implementation: Source code of the function
        directory_path: Optional custom directory path
        global_question_num: Global question number when tool was created (for tracking)

    Returns:
        True if save succeeded, False otherwise
    """
    if not is_valid_tool_name(function_name):
        logger.error(f"Refusing to save tool with invalid name: {function_name!r}")
        return False

    tool_directory = directory_path or CREATED_TOOLS_PATH
    file_path = os.path.join(tool_directory, f"{function_name}.py")
    with open(file=file_path, mode="w") as file:
        file.write(function_implementation)
        logger.info(f"New function saved to {file_path}")

        # Register tool in database if global_question_num provided
        if global_question_num is not None:
            register_new_tool(function_name, global_question_num)
            logger.info(f"Tool '{function_name}' registered at question {global_question_num}")

        return True
