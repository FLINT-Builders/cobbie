from typing import Optional, Tuple
from src.config import CREATED_TOOLS_PATH
from src.schemas import Result, ok, err, Err
from src.util.tool_files import FileIdentity, read_tool_source


def get_function_code(
    function_name: str,
    path: Optional[str] = None,
    expected_identity: Optional[FileIdentity] = None,
) -> Result[str, str]:
    """
    Retrieve the source code for a dynamically created tool function.

    Args:
        function_name (str): The name of the function to retrieve.
        path (Optional[str], optional): Custom file path to read the source code from.
            If None, defaults to the created tools directory using the function_name.
            Defaults to None.
        expected_identity: When reading from the created tools directory, only
            return the code if the file still has this identity.

    """
    try:
        if path is None:
            code, _ = read_tool_source(
                CREATED_TOOLS_PATH, function_name, expected_identity
            )
            return ok(code)

        with open(file=path, mode="r") as file:
            code = file.read()

        return ok(code)

    except Exception as e:
        return err(error=str(e))


def get_function_code_with_identity(
    function_name: str,
) -> Result[Tuple[str, FileIdentity], str]:
    """
    Retrieve a created tool's source code together with the identity of its file,
    so later steps can refuse to act on a different file.
    """
    try:
        return ok(read_tool_source(CREATED_TOOLS_PATH, function_name))
    except Exception as e:
        return err(error=str(e))


if __name__ == "__main__":
    function_name = "get_interior_doors_count"
    code = get_function_code(function_name=function_name)
    print(code)

    function_name = "check_fire_rating"
    code = get_function_code(function_name)
    print(code)
