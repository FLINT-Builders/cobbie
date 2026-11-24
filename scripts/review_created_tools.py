"""
Review LLM-created tools before evaluation runs can use them.

get_tools() only returns a tool from src/tools/created when the sha256 of its
file is recorded in src/tools/reviewed_created_tools.json. This script shows
every created tool that is new or changed since it was reviewed and records
the ones you approve. A changed file needs a new review.

Usage:
    uv run scripts/review_created_tools.py           # Review pending tools one by one
    uv run scripts/review_created_tools.py --list    # Only list pending tools
"""

import argparse
import json
import os

from src.config import CREATED_TOOLS_PATH, REVIEWED_CREATED_TOOLS_PATH
from src.util.get_tools import load_reviewed_tools, tool_source_digest


def pending_tools(reviewed: dict) -> list[tuple[str, bytes]]:
    pending = []
    for file_name in sorted(os.listdir(CREATED_TOOLS_PATH)):
        if not file_name.endswith(".py") or file_name == "__init__.py":
            continue
        with open(os.path.join(CREATED_TOOLS_PATH, file_name), "rb") as file:
            raw = file.read()
        if reviewed.get(file_name) != tool_source_digest(raw):
            pending.append((file_name, raw))
    return pending


def save_reviewed(reviewed: dict) -> None:
    with open(REVIEWED_CREATED_TOOLS_PATH, "w", encoding="utf-8") as file:
        json.dump({"tools": dict(sorted(reviewed.items()))}, file, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="only list tools pending review")
    args = parser.parse_args()

    reviewed = load_reviewed_tools()
    existing = {
        f for f in os.listdir(CREATED_TOOLS_PATH) if f.endswith(".py") and f != "__init__.py"
    }
    reviewed = {name: digest for name, digest in reviewed.items() if name in existing}
    pending = pending_tools(reviewed)

    if not pending:
        print("No created tools are pending review.")
        return
    if args.list:
        for file_name, _ in pending:
            print(file_name)
        return

    for file_name, raw in pending:
        print("=" * 80)
        print(file_name)
        print("=" * 80)
        print(raw.decode("utf-8", errors="replace"))
        answer = input(f"Approve {file_name}? [y/N/q] ").strip().lower()
        if answer == "q":
            break
        if answer == "y":
            reviewed[file_name] = tool_source_digest(raw)
            save_reviewed(reviewed)
            print(f"Recorded review of {file_name}.")


if __name__ == "__main__":
    main()
