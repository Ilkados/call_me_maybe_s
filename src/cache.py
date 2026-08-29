import sys
from typing import Dict

_token_cache: Dict[str, list[int]] = {}

try:
    from llm_sdk.llm_sdk import Small_LLM_Model
    model = Small_LLM_Model()
except KeyboardInterrupt:
    print("KeyboardInterrupt Error!", file=sys.stderr)
    sys.exit(0)


def get_tokens_id(text: str) -> list[int]:
    """Return cached token IDs for the given text, encoding it if needed."""
    if text not in _token_cache:
        _token_cache[text] = model.encode(text).tolist()[0]
    return _token_cache[text]
