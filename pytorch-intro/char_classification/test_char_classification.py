import os

import pytest

from . import char_classification


def test_letter_to_index():
    assert char_classification.letter_to_index("a") == 0
    assert char_classification.letter_to_index("g") == 6


def test_dataset():
    ds = char_classification.NamesDataset()
    assert len(ds.labels) > 0
    first = ds[0]
    assert len(first) == 4
    assert (len(ds)) == 20074


def test_basic_forward():
    ds = char_classification.NamesDataset()
    rnn = char_classification.CharRNN(
        char_classification.n_letters, 128, len(ds.labels_unique)
    )
    out = rnn(char_classification.line_to_tensor("Albert"))
    label, label_idx = ds.label_from_output(out)
    assert label in ds.labels_unique
    assert label == ds.labels_unique[label_idx]
