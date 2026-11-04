import re
from typing import Dict, List, Tuple


class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer implementation.

    Maintains internal vocabulary and merge rules learned during training.
    """

    def __init__(self):
        self.vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

    def pre_tokenize(self, text: str) -> List[str]:
        """Splits input text into pre-tokens based on assignment specifications:

        - Consecutive sequence of English letters: one pre-token
        - Consecutive sequence of digits: one pre-token
        - Punctuation or special non-whitespace characters: individual
        pre-tokens
        - Whitespace characters are excluded and act as delimiters
        """
        pattern = r"[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]"
        return re.findall(pattern, text)
