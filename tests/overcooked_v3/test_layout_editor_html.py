import ast
import inspect
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EDITOR_PATH = REPO_ROOT / "jaxmarl" / "tools" / "layout_editor_v3.html"

# This worktree is nested under another checkout locally, so force imports to
# resolve against the worktree under test.
sys.path.insert(0, str(REPO_ROOT))

from jaxmarl.environments.overcooked_v3.layouts import (  # noqa: E402
    Layout,
    overcooked_v3_layouts,
)


def _editor_html() -> str:
    return EDITOR_PATH.read_text(encoding="utf-8")


def _extract_js_array(name: str) -> list[str]:
    match = re.search(rf"const {name} = (\[[\s\S]*?\]);", _editor_html())
    assert match, f"Could not find {name} in {EDITOR_PATH}"
    return json.loads(match.group(1))


def _editor_supported_symbols() -> set[str]:
    return set(_extract_js_array("SUPPORTED_LAYOUT_SYMBOLS"))


def _tool_entries() -> list[dict[str, str]]:
    tool_pattern = re.compile(
        r"\{\s*group: \"(?P<group>[^\"]+)\", "
        r"symbol: \"(?P<symbol>(?:\\.|[^\"])*)\", "
        r"name: \"(?P<name>(?:\\.|[^\"])*)\", "
        r"color: [^,]+, text: \"(?P<text>(?:\\.|[^\"])*)\", "
        r"shortcut: \"(?P<shortcut>(?:\\.|[^\"])*)\"\s*\}"
    )
    tools = []
    for match in tool_pattern.finditer(_editor_html()):
        tools.append(
            {
                key: json.loads(f'"{value}"')
                for key, value in match.groupdict().items()
            }
        )
    assert tools, f"Could not parse TOOLS in {EDITOR_PATH}"
    return tools


def _python_parser_symbols() -> set[str]:
    source = textwrap.dedent(inspect.getsource(Layout.from_string))
    tree = ast.parse(source)
    symbols = set("0123456789")
    symbol_maps = {
        "char_to_static_item",
        "item_conveyor_chars",
        "player_conveyor_chars",
        "moving_wall_chars",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if target_names & symbol_maps and isinstance(node.value, ast.Dict):
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        symbols.add(key.value)

        is_char_equality = (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
        )
        if is_char_equality:
            values = [node.left, *node.comparators]
            has_char = any(
                isinstance(value, ast.Name) and value.id == "char"
                for value in values
            )
            if not has_char:
                continue
            for value in values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    symbols.add(value.value)

    return symbols


def test_layout_editor_declares_every_python_layout_parser_symbol():
    parser_symbols = _python_parser_symbols()
    editor_symbols = _editor_supported_symbols()

    assert parser_symbols <= editor_symbols, (
        "layout_editor_v3.html is missing symbols accepted by "
        f"Layout.from_string(): {sorted(parser_symbols - editor_symbols)!r}"
    )


def test_layout_editor_supports_symbols_from_registered_layouts():
    editor_symbols = _editor_supported_symbols()
    missing_by_layout = {}

    for layout_name, layout in overcooked_v3_layouts.items():
        layout_symbols = {
            symbol for symbol in layout.to_string().strip("\n") if symbol != "\n"
        }
        missing = layout_symbols - editor_symbols
        if missing:
            missing_by_layout[layout_name] = sorted(missing)

    assert not missing_by_layout, (
        "Registered Overcooked V3 layouts use symbols that the HTML editor "
        f"does not declare: {missing_by_layout!r}"
    )


def test_layout_editor_tools_have_unique_shortcuts_and_supported_symbols():
    tools = _tool_entries()
    editor_symbols = _editor_supported_symbols()
    tool_symbols = {tool["symbol"] for tool in tools}
    shortcut_counts = Counter(tool["shortcut"] for tool in tools)
    duplicate_shortcuts = {
        shortcut: count
        for shortcut, count in shortcut_counts.items()
        if count > 1
    }

    assert tool_symbols <= editor_symbols, (
        f"TOOLS includes unsupported symbols: {sorted(tool_symbols - editor_symbols)!r}"
    )
    assert not duplicate_shortcuts, (
        f"TOOLS has duplicate shortcuts: {duplicate_shortcuts!r}"
    )
    assert next(tool for tool in tools if tool["symbol"] == "W")["shortcut"] == "w"
    assert next(tool for tool in tools if tool["symbol"] == "w")["shortcut"] == "W"
