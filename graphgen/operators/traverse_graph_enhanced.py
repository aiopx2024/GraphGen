# Enhanced version with source tracking
import asyncio
from graphgen.operators.traverse_graph import *

async def _extract_source_metadata(_process_nodes: list, _process_edges: list, text_chunks_storage):
    """Extract source metadata for traceability"""
    # 收集所有source_id
    source_ids = []
    
    for node in _process_nodes:
        if "source_id" in node:
            source_ids.extend(node["source_id"].split("<SEP>"))
    
    for edge in _process_edges:
        if "source_id" in edge[2]:
            source_ids.extend(edge[2]["source_id"].split("<SEP>"))
    
    # 去重
    unique_source_ids = list(set(source_ids))
    
    # 获取chunk内容
    chunk_contents = await text_chunks_storage.get_by_ids(unique_source_ids)
    
    # 构建溯源信息
    source_metadata = {
        "source_chunks": unique_source_ids,
        "chunk_contents": {k: v["content"] if isinstance(v, dict) else v for k, v in chunk_contents.items()},
        "doc_ids": [chunk_contents[chunk_id].get("full_doc_id", "") for chunk_id in unique_source_ids if chunk_id in chunk_contents],
        "entities_used": [node["node_id"] for node in _process_nodes],
        "relations_used": [(edge[0], edge[1], edge[2]["description"]) for edge in _process_edges]
    }
    
    return source_metadata

async def traverse_graph_by_edge_enhanced(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """Enhanced version with full traceability"""
    
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_batch(_process_batch: tuple) -> dict:
        async with semaphore:
            try:
                # ... 原有逻辑保持不变 ...
                # 生成QA对
                qas = []  # 这里是原有的QA生成逻辑
                
                # 添加溯源信息
                source_metadata = await _extract_source_metadata(
                    _process_batch[0], _process_batch[1], text_chunks_storage
                )
                
                final_results = {}
                for qa in qas:
                    qa_enhanced = {
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "loss": qa.get("loss", 0),
                        # 新增：完整的溯源信息
                        "metadata": {
                            "qa_type": "multi_hop",
                            "generation_method": traverse_strategy.qa_form,
                            "source_tracing": source_metadata,
                            "subgraph_info": {
                                "nodes_count": len(_process_batch[0]),
                                "edges_count": len(_process_batch[1]),
                                "max_depth": traverse_strategy.max_depth
                            }
                        }
                    }
                    final_results[compute_content_hash(qa["question"])] = qa_enhanced
                
                return final_results
            
            except Exception as e:
                logger.error("Error in enhanced processing: %s", e)
                return {}
    
    # ... 其余逻辑与原函数相同 ...