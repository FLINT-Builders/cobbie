from typing import Optional

from loguru import logger

from src.db.query import register_new_tool
from src.util.get_tools import created_tool_path

def save_new_tool(function_name: str, function_implementation: str, directory_path:Optional[str] = None, global_question_num: Optional[int] = None)->bool:
    """
    Save the provided function code as a file named according to the provided function_name to the new_file_path. If no path is provided, will use the default location for created tools from the config file.

    The saved tool is not reviewed: get_tools skips it until someone records a
    review with scripts/review_created_tools.py, and it only ever runs in the
    sandboxed interpreter.

    Args:
        function_name: Name of the function to save (must be a Python identifier)
        function_implementation: Source code of the function
        directory_path: Optional custom directory path
        global_question_num: Global question number when tool was created (for tracking)

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        file_path = created_tool_path(function_name, directory_path)
    except ValueError as e:
        logger.error(f"Refusing to save tool: {e}")
        return False
    with open(file=file_path, mode="w") as file:
        file.write(function_implementation)
        logger.info(f"New function saved to {file_path} (pending review)")

        # Register tool in database if global_question_num provided
        if global_question_num is not None:
            register_new_tool(function_name, global_question_num)
            logger.info(f"Tool '{function_name}' registered at question {global_question_num}")

        return True
