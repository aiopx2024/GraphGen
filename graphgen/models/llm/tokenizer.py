from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import re
import warnings

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

        # 语言感知的token比例
        self.lang_ratios = {
            'zh': 1.8,   # 中文：1.8字符/token
            'en': 4.2,   # 英文：4.2字符/token
            'mixed': 2.8 # 混合文本：2.8字符/token
        }

    def detect_language_ratio(self, text: str) -> float:
        """检测文本语言比例，动态调整token估算"""
        if not text.strip():
            return self.lang_ratios['mixed']

        # 计算中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())

        if total_chars == 0:
            return self.lang_ratios['mixed']

        zh_ratio = chinese_chars / total_chars

        # 根据中文比例选择适当的估算比例
        if zh_ratio > 0.7:
            return self.lang_ratios['zh']
        elif zh_ratio < 0.3:
            return self.lang_ratios['en']
        else:
            # 混合文本：根据实际比例加权平均
            return (self.lang_ratios['zh'] * zh_ratio +
                   self.lang_ratios['en'] * (1 - zh_ratio))

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
    """改进的Tokenizer类，支持语言感知的切分策略"""
    model_name: str = "simple"

    def __post_init__(self):
        self.tokenizer = get_tokenizer(self.model_name)

        # 语义边界标记
        self.sentence_boundaries = {
            'zh': ['。', '！', '？', '；'],
            'en': ['.', '!', '?', ';'],
            'common': ['\n\n', '\n', '\t']
        }

        # 段落边界标记
        self.paragraph_boundaries = ['\n\n', '\r\n\r\n']

    def estimate_tokens(self, text: str, language_aware: bool = True) -> int:
        """
        估算文本的token数量

        Args:
            text: 要估算的文本
            language_aware: 是否启用语言感知估算

        Returns:
            估算的token数量
        """
        if not text.strip():
            return 0

        if language_aware:
            chars_per_token = self.tokenizer.detect_language_ratio(text)
        else:
            chars_per_token = 4.0  # 默认估算

        return max(1, int(len(text) / chars_per_token))

    def encode_string(self, text: str) -> List[int]:
        """
        将文本编码为token列表（简化版：基于字符数估算）
        """
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int]) -> str:
        """
        将token列表解码为文本
        """
        return self.tokenizer.decode(tokens)

    def split_by_semantic_boundaries(self, text: str, boundary_markers: List[str] = None) -> List[str]:
        """
        按语义边界分割文本

        Args:
            text: 要分割的文本
            boundary_markers: 自定义边界标记

        Returns:
            分割后的文本段列表
        """
        if not text.strip():
            return []

        # 使用默认边界标记或自定义标记
        if boundary_markers is None:
            markers = (
                self.sentence_boundaries['zh'] +
                self.sentence_boundaries['en'] +
                self.sentence_boundaries['common']
            )
        else:
            markers = boundary_markers

        # 首先按段落分割
        paragraphs = re.split(r'\n\s*\n', text.strip())

        segments = []
        for para in paragraphs:
            if not para.strip():
                continue

            # 按句子分割段落
            # 构建正则表达式模式
            pattern = '|'.join(re.escape(marker) for marker in markers if marker not in ['\n\n', '\n', '\t'])
            if pattern:
                sentences = re.split(f'({pattern})', para)
                current_sentence = ""

                for part in sentences:
                    if part.strip():
                        current_sentence += part
                        if any(part.endswith(marker) for marker in markers):
                            if current_sentence.strip():
                                segments.append(current_sentence.strip())
                                current_sentence = ""

                # 添加剩余部分
                if current_sentence.strip():
                    segments.append(current_sentence.strip())
            else:
                # 如果没有找到句子边界，直接添加段落
                segments.append(para.strip())

        return [seg for seg in segments if seg.strip()]

    def chunk_by_token_size(
        self,
        content: str,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
        strategy: str = "semantic",
        preserve_boundaries: bool = True,
        min_chunk_size: int = 100,
        language_aware: bool = True,
        boundary_markers: List[str] = None
    ) -> List[Dict]:
        """
        改进的文本分块方法，支持多种策略

        Args:
            content: 要分割的文本
            overlap_token_size: 重叠的token数量
            max_token_size: 最大token数量
            strategy: 切分策略 ("simple", "semantic", "hierarchical")
            preserve_boundaries: 是否保持语义边界
            min_chunk_size: 最小chunk大小
            language_aware: 是否启用语言感知
            boundary_markers: 自定义边界标记

        Returns:
            分块结果列表
        """
        if not content.strip():
            return []

        if strategy == "simple":
            return self._simple_chunk(content, overlap_token_size, max_token_size, language_aware)
        elif strategy == "semantic":
            return self._semantic_chunk(
                content, overlap_token_size, max_token_size,
                preserve_boundaries, min_chunk_size, language_aware, boundary_markers
            )
        elif strategy == "hierarchical":
            return self._hierarchical_chunk(
                content, overlap_token_size, max_token_size,
                preserve_boundaries, min_chunk_size, language_aware, boundary_markers
            )
        else:
            warnings.warn(f"Unknown strategy '{strategy}', falling back to 'semantic'")
            return self._semantic_chunk(
                content, overlap_token_size, max_token_size,
                preserve_boundaries, min_chunk_size, language_aware, boundary_markers
            )

    def _simple_chunk(self, content: str, overlap_token_size: int, max_token_size: int, language_aware: bool) -> List[Dict]:
        """简单滑动窗口切分"""
        if language_aware:
            chars_per_token = self.tokenizer.detect_language_ratio(content)
        else:
            chars_per_token = 4.0

        max_chars = int(max_token_size * chars_per_token)
        overlap_chars = int(overlap_token_size * chars_per_token)

        results = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + max_chars, len(content))
            chunk_content = content[start:end].strip()

            if chunk_content:
                estimated_tokens = self.estimate_tokens(chunk_content, language_aware)
                results.append({
                    "tokens": min(max_token_size, estimated_tokens),
                    "content": chunk_content,
                    "chunk_order_index": chunk_index,
                })
                chunk_index += 1

            if end >= len(content):
                break
            start = max(end - overlap_chars, start + 1)

        return results

    def _semantic_chunk(self, content: str, overlap_token_size: int, max_token_size: int,
                       preserve_boundaries: bool, min_chunk_size: int, language_aware: bool,
                       boundary_markers: List[str]) -> List[Dict]:
        """基于语义边界的切分"""
        if preserve_boundaries:
            segments = self.split_by_semantic_boundaries(content, boundary_markers)
        else:
            segments = [content]

        results = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for segment in segments:
            segment_tokens = self.estimate_tokens(segment, language_aware)

            # 如果单个段落就超过最大限制，需要强制分割
            if segment_tokens > max_token_size:
                # 先处理当前累积的chunk
                if current_chunk and current_tokens >= min_chunk_size:
                    chunk_content = ' '.join(current_chunk)
                    results.append({
                        "tokens": current_tokens,
                        "content": chunk_content,
                        "chunk_order_index": chunk_index,
                    })
                    chunk_index += 1

                # 对超大段落进行强制分割
                large_chunks = self._simple_chunk(segment, overlap_token_size, max_token_size, language_aware)
                for chunk in large_chunks:
                    chunk["chunk_order_index"] = chunk_index
                    results.append(chunk)
                    chunk_index += 1

                current_chunk = []
                current_tokens = 0
                continue

            # 检查加入当前段落是否超过限制
            if current_tokens + segment_tokens > max_token_size:
                if current_chunk and current_tokens >= min_chunk_size:
                    chunk_content = ' '.join(current_chunk)
                    results.append({
                        "tokens": current_tokens,
                        "content": chunk_content,
                        "chunk_order_index": chunk_index,
                    })
                    chunk_index += 1

                # 处理重叠：从当前chunk的末尾开始新chunk
                if overlap_token_size > 0 and current_chunk:
                    overlap_segments = []
                    overlap_tokens = 0

                    # 从后往前选择段落作为重叠内容
                    for i in range(len(current_chunk) - 1, -1, -1):
                        seg_tokens = self.estimate_tokens(current_chunk[i], language_aware)
                        if overlap_tokens + seg_tokens <= overlap_token_size:
                            overlap_segments.insert(0, current_chunk[i])
                            overlap_tokens += seg_tokens
                        else:
                            break

                    current_chunk = overlap_segments
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(segment)
            current_tokens += segment_tokens

        # 处理剩余内容
        if current_chunk and current_tokens >= min_chunk_size:
            chunk_content = ' '.join(current_chunk)
            results.append({
                "tokens": current_tokens,
                "content": chunk_content,
                "chunk_order_index": chunk_index,
            })

        return results

    def _hierarchical_chunk(self, content: str, overlap_token_size: int, max_token_size: int,
                          preserve_boundaries: bool, min_chunk_size: int, language_aware: bool,
                          boundary_markers: List[str]) -> List[Dict]:
        """层次化切分：文档->段落->句子"""
        # 首先按段落分割
        paragraphs = re.split(r'\n\s*\n', content.strip())

        results = []
        chunk_index = 0

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self.estimate_tokens(para, language_aware)

            if para_tokens <= max_token_size:
                # 段落适合单个chunk
                if para_tokens >= min_chunk_size:
                    results.append({
                        "tokens": para_tokens,
                        "content": para.strip(),
                        "chunk_order_index": chunk_index,
                    })
                    chunk_index += 1
            else:
                # 段落需要进一步分割
                para_chunks = self._semantic_chunk(
                    para, overlap_token_size, max_token_size,
                    preserve_boundaries, min_chunk_size, language_aware, boundary_markers
                )
                for chunk in para_chunks:
                    chunk["chunk_order_index"] = chunk_index
                    results.append(chunk)
                    chunk_index += 1

        return results
