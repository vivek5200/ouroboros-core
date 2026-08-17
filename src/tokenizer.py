"""Module 1: AST-aware Tokenizer with Phantom Padding.

Phantom Padding: L_max = 1024 tokens.
All sequences are padded to L_max before entering the attention mechanism.
"""

L_MAX = 1024  # Phantom Padding maximum sequence length


def tokenize(source_code: str) -> list[int]:
    """Tokenize source code into token IDs.

    TODO: Implement AST-aware tokenization.
    """
    raise NotImplementedError("Tokenizer not yet implemented")


def phantom_pad(token_ids: list[int], pad_id: int = 0) -> list[int]:
    """Pad token sequence to L_max using Phantom Padding.

    Args:
        token_ids: Input token IDs.
        pad_id: Padding token ID.

    Returns:
        Padded token sequence of length L_MAX.
    """
    if len(token_ids) > L_MAX:
        return token_ids[:L_MAX]
    return token_ids + [pad_id] * (L_MAX - len(token_ids))
