import pytest
import BPETokenizer


@pytest.fixture
def tokenizer():
    """Provides a fresh instance of BPETokenizer for each test."""
    return BPETokenizer.BPETokenizer()

def test_pre_tokenize_assignment_example(tokenizer):
    """Verifies the pre-tokenization logic according to the exact example in the assignment:

    Input: "Alice was 7 years old, wasn't she?"
    Output: ["Alice", "was", "7", "years", "old", ",", "wasn", "'", "t", "she",
    "?"]
    """
    text = "Alice was 7 years old, wasn't she?"
    expected = [
        "Alice",
        "was",
        "7",
        "years",
        "old",
        ",",
        "wasn",
        "'",
        "t",
        "she",
        "?",
    ]
    assert tokenizer._pre_tokenize(text) == expected

