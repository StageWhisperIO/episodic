from episodic.core import diffparse


def test_parsed_unified_ends_with_newline():
    patch = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    entry = diffparse.parse_unified_diff(patch)[0]
    assert entry["unified"].endswith("\n")
    assert entry["unified"].startswith("diff --git a/mod.py b/mod.py")


def test_parsed_unified_preserves_trailing_blank_context_line():
    patch = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-a\n"
        "+A\n"
        " b\n"
        " \n"
    )
    entry = diffparse.parse_unified_diff(patch)[0]
    lines = entry["unified"].split("\n")
    hunk = lines[lines.index("@@ -1,3 +1,3 @@") + 1:]
    old_lines = sum(1 for line in hunk if line[:1] in (" ", "-"))
    assert old_lines == 3
    assert entry["unified"].endswith(" \n")


def test_join_unified_terminates_each_block_and_ignores_empties():
    blocks = [
        "diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b",
        None,
        "   ",
        "diff --git a/g b/g\n@@ -1 +1 @@\n-c\n+d\n\n",
    ]
    joined = diffparse.join_unified(blocks)
    assert joined.count("diff --git") == 2
    assert "+b\ndiff --git a/g" in joined
    assert joined.endswith("+d\n")


def test_join_unified_empty_input_is_empty_string():
    assert diffparse.join_unified([None, "", "  \n"]) == ""
