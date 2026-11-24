from typing import Optional
from src.config import CREATED_TOOLS_PATH
from src.schemas import Result, ok, err, Err
from src.util.tool_file import read_tool_source


def get_function_code(
    function_name: str,
    path: Optional[str] = None,
) -> Result[str, str]:
    """
    Retrieve the source code for a dynamically created tool function.

    Args:
        function_name (str): The name of the function to retrieve.
        path (Optional[str], optional): Custom file path to read the source code from.
            If None, defaults to the created tools directory using the function_name.
            Defaults to None.

    """
    try:
        if path is None:
            code = read_tool_source(CREATED_TOOLS_PATH, function_name)
        else:
            with open(file=path, mode="r") as file:
                code = file.read()

        return ok(code)

    except Exception as e:
        return err(error=str(e))



if __name__ == "__main__":
    function_name = "get_interior_doors_count"
    code = get_function_code(function_name=function_name)
    print(code)

    function_name = "check_fire_rating"
    code = get_function_code(function_name)
    print(code)
