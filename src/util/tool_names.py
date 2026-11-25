import keyword


def is_valid_tool_name(name: str) -> bool:
    """
    Tool names come from model output and become file names in src/tools/created,
    so only plain Python identifiers are accepted (no path separators or "..").
    """
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
