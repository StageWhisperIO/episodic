from examples.slugify import slugify, deslugify


def test_spaces():
    assert slugify("hello world") == "hello-world"


def test_punctuation():
    assert slugify("hello, world!") == "hello-world"


def test_mixed_case():
    assert slugify("Hello World") == "hello-world"


def test_leading_trailing_symbols():
    assert slugify("--Hello, World!--") == "hello-world"


def test_collapses_runs():
    assert slugify("a   b___c") == "a-b-c"


def test_deslugify():
    assert deslugify("hello-world") == "Hello World"
