import pytest

from episodic import trainers
from episodic.trainers.mlx import split_rows


def _row(text):
    return {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": text}]}


def test_mlx_sft_is_registered():
    assert "mlx-sft" in trainers.available()
    assert trainers.get("mlx-sft").consumes == ("sft",)


def test_split_holds_out_a_fraction():
    train, valid = split_rows([_row(str(i)) for i in range(5)], 0.2)
    assert len(valid) == 1 and len(train) == 4
    assert all("messages" in row and set(row) == {"messages"} for row in train + valid)


def test_split_single_row_reuses_it_for_both():
    train, valid = split_rows([_row("only")], 0.2)
    assert train == valid and len(train) == 1


def test_split_empty_raises():
    with pytest.raises(ValueError):
        split_rows([], 0.2)
