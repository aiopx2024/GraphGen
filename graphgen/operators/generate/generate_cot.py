import asyncio
from typing import Dict, List, Tuple

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import CommunityDetector, NetworkXStorage, OpenAIModel
from graphgen.templates import COT_GENERATION_PROMPT, COT_TEMPLATE_DESIGN_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger

# LLM调用位置跟踪：记录每个调用位置是否已经输出过详细日志
llm_call_tracker = set()


def log_llm_call_once(call_location: str, prompt: str, context: str = None, is_before: bool = True):
    """
    为LLM调用记录日志，每个位置只在第一次调用时输出详细信息
    
    Args:
        call_location (str): 调用位置标识，如 'cot_generation'
        prompt (str): 发送给LLM的提示词
        context (str, optional): LLM返回的结果
        is_before (bool): True表示调用前，False表示调用后
    """
    call_key = f"{call_location}_{'before' if is_before else 'after'}"
    
    if call_key not in llm_call_tracker:
        llm_call_tracker.add(call_key)
        
        if is_before:
            logger.info("=== LLM调用前 [%s] ===", call_location)
            logger.info("Prompt: %s", prompt)
        else:
            logger.info("=== LLM调用后 [%s] ===", call_location)
            if context:
                logger.info("Context: %s", context)


async def generate_cot(
    graph_storage: NetworkXStorage,
    synthesizer_llm_client: OpenAIModel,
    method_params: Dict = None,
):
    method = method_params.get("method", "leiden")
    detector = CommunityDetector(
        graph_storage=graph_storage, method=method, method_params=method_params
    )

    results = await detector.detect_communities()

    # Convert results to a format suitable for summarization
    communities = {}
    for node, community_id in results.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    if not communities:
        return {}

    semaphore = asyncio.Semaphore(value=1000)

    async def _generate_from_single_community(
        c_id: int, nodes: List[str]
    ) -> Tuple[int, Tuple[str, str, str, Dict]]:
        """Summarize a single community."""
        async with semaphore:
            entities: List[str] = []
            relationships: List[str] = []
            source_chunks = set()
            doc_ids = set()
            entities_used = []
            relations_used = []

            for n in nodes:
                node_data = await graph_storage.get_node(n)
                if node_data is not None:
                    entities.append(f"({n}: {node_data.get('description')})")
                    entities_used.append(n)
                    
                    # 收集溯源信息
                    if "source_id" in node_data:
                        source_chunks.update(node_data["source_id"].split("<SEP>"))

                edges = await graph_storage.get_node_edges(n)
                for edge in edges:
                    target = edge[1]
                    if target in nodes:
                        edge_data = await graph_storage.get_edge(n, target)
                        relationships.append(
                            f"({n}) - [{edge_data['description']}] -> ({target})"
                        )
                        relations_used.append((n, target, edge_data['description']))
                        
                        # 收集边的溯源信息
                        if "source_id" in edge_data:
                            source_chunks.update(edge_data["source_id"].split("<SEP>"))

            entities_str = "\n".join(entities)
            relationships_str = "\n".join(relationships)

            language = (
                "English"
                if detect_main_language(entities_str + relationships_str) == "en"
                else "Chinese"
            )

            prompt = COT_TEMPLATE_DESIGN_PROMPT[language]["TEMPLATE"].format(
                entities=entities_str,
                relationships=relationships_str,
            )

            # 调用前记录日志（仅第一次）
            log_llm_call_once("cot_template_design", prompt, is_before=True)
            
            cot_template = await synthesizer_llm_client.generate_answer(prompt)
            
            # 调用后记录日志（仅第一次）
            log_llm_call_once("cot_template_design", prompt, cot_template, is_before=False)

            if "问题：" in cot_template and "推理路径设计：" in cot_template:
                question = cot_template.split("问题：")[1].split("推理路径设计：")[0].strip()
                reasoning_path = cot_template.split("推理路径设计：")[1].strip()
            elif (
                "Question:" in cot_template and "Reasoning-Path Design:" in cot_template
            ):
                question = (
                    cot_template.split("Question:")[1]
                    .split("Reasoning-Path Design:")[0]
                    .strip()
                )
                reasoning_path = cot_template.split("Reasoning-Path Design:")[1].strip()
            else:
                raise ValueError("COT template format is incorrect.")

            prompt = COT_GENERATION_PROMPT[language]["TEMPLATE"].format(
                entities=entities_str,
                relationships=relationships_str,
                question=question,
                reasoning_template=reasoning_path,
            )

            # 调用前记录日志（仅第一次）
            log_llm_call_once("cot_answer_generation", prompt, is_before=True)
            
            cot_answer = await synthesizer_llm_client.generate_answer(prompt)
            
            # 调用后记录日志（仅第一次）
            log_llm_call_once("cot_answer_generation", prompt, cot_answer, is_before=False)

            # 构建溯源信息
            source_metadata = {
                "source_chunks": list(source_chunks),
                "chunk_contents": {},  # CoT基于社区检测，不直接映射到具体chunks
                "doc_ids": list(doc_ids),
                "entities_used": entities_used,
                "relations_used": relations_used
            }

            return c_id, (question, reasoning_path, cot_answer, source_metadata)

    cid_nodes = list(communities.items())

    results: Dict = {}
    async for coro in tqdm_async(
        asyncio.as_completed(
            [_generate_from_single_community(cid, nodes) for cid, nodes in cid_nodes]
        ),
        total=len(cid_nodes),
        desc="[Generating COT] Generating CoT data from communities",
        unit="community",
    ):
        cid, (q, r, a, source_metadata) = await coro
        results[compute_content_hash(q)] = {
            "question": q,
            "reasoning_path": r,
            "answer": a,
            "metadata": {
                "qa_type": "cot",
                "generation_method": "community_detection",
                "community_id": cid,
                "source_tracing": source_metadata,
                "subgraph_info": {
                    "nodes_count": len(communities[cid]),
                    "edges_count": len([r for r in source_metadata["relations_used"]]),
                    "max_depth": 1  # CoT基于社区，深度为1
                }
            }
        }

    return results
