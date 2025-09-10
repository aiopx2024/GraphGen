from dataclasses import dataclass
from typing import List
import os
import tiktoken

# 设置离线模式环境变量
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/app/cache/tiktoken")
os.environ.setdefault("HF_HOME", "/app/cache/huggingface")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


def get_tokenizer(tokenizer_name: str = "cl100k_base"):
    """
    Get a tokenizer instance by name.

    :param tokenizer_name: tokenizer name, tiktoken encoding name or Hugging Face model name
    :return: tokenizer instance
    """
    # 内网环境优先使用 Hugging Face tokenizer，避免访问外网
    if TRANSFORMERS_AVAILABLE:
        try:
            # 尝试使用 Hugging Face tokenizer
            if tokenizer_name == "cl100k_base":
                # 对于 cl100k_base，使用 GPT-4 compatible tokenizer
                return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            # 如果 Hugging Face 失败，再尝试 tiktoken
            pass
    
    # 如果 Hugging Face 不可用或失败，使用 tiktoken
    try:
        if tokenizer_name in tiktoken.list_encoding_names():
            return tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        # 内网环境下 tiktoken 可能无法访问外网，使用本地 fallback
        if TRANSFORMERS_AVAILABLE:
            print(f"Warning: tiktoken failed to load {tokenizer_name} due to network issues, falling back to GPT-2 tokenizer")
            return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        else:
            raise ValueError(f"Failed to load tokenizer {tokenizer_name}: {e}. Please ensure transformers is installed for offline usage.")

@dataclass
class Tokenizer:
    model_name: str = "cl100k_base"

    def __post_init__(self):
        self.tokenizer = get_tokenizer(self.model_name)

    def encode_string(self, text: str) -> List[int]:
        """
        Encode text to tokens

        :param text
        :return: tokens
        """
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens to text

        :param tokens
        :return: text
        """
        return self.tokenizer.decode(tokens)

    def chunk_by_token_size(
        self, content: str, overlap_token_size=128, max_token_size=1024
    ):
        tokens = self.encode_string(content)
        results = []
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = self.decode_tokens(
                tokens[start : start + max_token_size]
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
        return results
