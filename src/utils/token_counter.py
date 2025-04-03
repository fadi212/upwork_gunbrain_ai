# src/utils/token_counter.py

import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a given text for a specified OpenAI model.

    Args:
        text (str): The text to count tokens for.
        model (str): The OpenAI model name.

    Returns:
        int: Number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if the model is not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
