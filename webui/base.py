from dataclasses import dataclass
from typing import Any

@dataclass
class GraphGenParams:
    """
    GraphGen parameters
    """
    if_trainee_model: bool
    input_file: str
    tokenizer: str
    qa_form: str
    bidirectional: bool
    expand_method: str
    max_extra_edges: int
    max_tokens: int
    max_depth: int
    edge_sampling: str
    isolated_node_strategy: str
    loss_strategy: str
    synthesizer_url: str
    synthesizer_model: str
    trainee_model: str
    api_key: str
    chunk_size: int
    # 新增的chunking参数
    chunk_overlap_size: int = 128
    chunking_strategy: str = "semantic"
    preserve_boundaries: bool = True
    min_chunk_size: int = 100
    language_aware: bool = True
    rpm: int = 1000
    tpm: int = 50000
    quiz_samples: int = 2
    trainee_url: str = ""
    trainee_api_key: str = ""
    token_counter: Any = None
