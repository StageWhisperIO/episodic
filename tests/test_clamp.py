import sys

sys.path.insert(0, "examples")
from clamp import clamp


def test_clamp_below_range():
    assert clamp(1, 2, 5) == 2


def test_clamp_in_range():
    assert clamp(3, 2, 5) == 3


def test_clamp_above_range():
    assert clamp(6, 2, 5) == 5
