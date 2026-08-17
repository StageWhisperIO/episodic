import re
from pathlib import Path

from episodic import cli

README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def _cli_reference_table_text():
    text = README_PATH.read_text(encoding="utf-8")
    start = text.index("## CLI reference")
    end = text.index("\n## ", start + len("## CLI reference"))
    return text[start:end]


def test_readme_cli_reference_table_only_documents_real_subcommands():
    table = _cli_reference_table_text()
    tokens = re.findall(r"`episodic ([a-z][a-z0-9-]*)", table)
    assert tokens, "expected at least one `episodic <subcommand>` token in the CLI reference table"

    known = set(cli.list_commands())
    stale = [token for token in tokens if token not in known]
    assert not stale, (
        f"README's CLI reference table documents subcommands that no longer exist: {stale} "
        f"(known subcommands: {sorted(known)})"
    )


def test_list_commands_matches_build_parser_subcommands():
    parser = cli.build_parser()
    choices = set(parser._subparsers._group_actions[0].choices)
    assert set(cli.list_commands()) == choices
