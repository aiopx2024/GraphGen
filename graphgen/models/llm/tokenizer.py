from dataclasses import dataclass
from typing import List
import os
import re

# 简化的内网版本，去掉复杂的transformers依赖
# 使用基本的字符计数和分割逻辑

class SimpleTokenizer:
    """简化的tokenizer，适用于内网环境，无需复杂的模型下载"""
    
    def __init__(self, chars_per_token: int = 4):
        """
        简单的tokenizer，基于字符数估算token数量
        
        Args:
            chars_per_token: 平均每个token的字符数（中文约2-3，英文约4-5）
        """
        self.chars_per_token = chars_per_token
    
    def encode(self, text: str) -> List[int]:
        """简单编码：将文本转换为字符索引列表"""
        return [ord(c) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """简单解码：将字符索引列表转换回文本"""
        return ''.join(chr(token) for token in tokens if 0 <= token <= 1114111)

def get_tokenizer(tokenizer_name: str = "simple"):
    """
    获取简化的tokenizer实例
    内网版本：使用简单的字符分割，无需下载模型
    """
    print(f"🔧 使用简化tokenizer: {tokenizer_name}")
    return SimpleTokenizer()

@dataclass
class Tokenizer:
    """简化的Tokenizer类，适用于内网环境"""
    model_name: str = "simple"

    def __post_init__(self):
        self.tokenizer = get_tokenizer(self.model_name)
        # 基于字符的简单token估算
        self.chars_per_token = 4  # 平均每个token 4个字符

    def encode_string(self, text: str) -> List[int]:
        """
        将文本编码为token列表（简化版：基于字符数估算）
        """
        # 简单的token估算：字符数除以平均每token字符数
        estimated_tokens = len(text) // self.chars_per_token
        return list(range(estimated_tokens))  # 返回索引列表

    def decode_tokens(self, tokens: List[int]) -> str:
        """
        将token列表解码为文本（简化版：直接返回原文本的前N个字符）
        """
        # 由于我们使用简化逻辑，这里返回空字符串
        # 在实际使用中，分块时会直接使用原文本
        return ""

    def chunk_by_token_size(
        self, content: str, overlap_token_size=128, max_token_size=1024
    ):
        """
        按token大小分割文本（简化版：基于字符数）
        
        Args:
            content: 要分割的文本
            overlap_token_size: 重叠的token数量
            max_token_size: 最大token数量
        
        Returns:
            分块结果列表
        """
        # 将token大小转换为字符大小
        max_chars = max_token_size * self.chars_per_token
        overlap_chars = overlap_token_size * self.chars_per_token
        
        results = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # 计算当前块的结束位置
            end = min(start + max_chars, len(content))
            
            # 提取文本块
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                # 估算token数量
                estimated_tokens = len(chunk_content) // self.chars_per_token + 1
                
                results.append({
                    "tokens": min(max_token_size, estimated_tokens),
                    "content": chunk_content,
                    "chunk_order_index": chunk_index,
                })
                
                chunk_index += 1
            
            # 计算下一个块的起始位置（考虑重叠）
            if end >= len(content):
                break
            
            start = end - overlap_chars
            if start <= 0:
                start = end
        
        return results
