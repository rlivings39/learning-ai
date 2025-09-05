import pytest

from . import char_classification


def test_letter_to_index():
    assert char_classification.letter_to_index("a") == 0
    assert char_classification.letter_to_index("g") == 6
