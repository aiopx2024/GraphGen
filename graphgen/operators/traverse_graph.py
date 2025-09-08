import asyncio
from asyncio.locks import Semaphore

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import (
    JsonKVStorage,
    NetworkXStorage,
    OpenAIModel,
    Tokenizer,
    TraverseStrategy,
)
from graphgen.operators.kg.split_kg import get_batches_with_strategy
from graphgen.templates import (
    ANSWER_REPHRASING_PROMPT,
    MULTI_HOP_GENERATION_PROMPT,
    QUESTION_GENERATION_PROMPT,
)
from graphgen.utils import compute_content_hash, detect_main_language, logger

# LLM调用位置跟踪：记录每个调用位置是否已经输出过详细日志
llm_call_tracker = set()


def log_llm_call_once(call_location: str, prompt: str, context: str = None, is_before: bool = True):
    """
    为LLM调用记录日志，每个位置只在第一次调用时输出详细信息
    
    Args:
        call_location (str): 调用位置标识，如 'multi_hop_generation'
        prompt (str): 发送给LLM的提示词
        context (str, optional): LLM返回的结果
        is_before (bool): True表示调用前，False表示调用后
    """
    call_key = f"{call_location}_{'before' if is_before else 'after'}"
    
    if call_key not in llm_call_tracker:
        llm_call_tracker.add(call_key)
        
        if is_before:
            logger.info("=== LLM调用前 [%s] ===", call_location)
            logger.info("Prompt: %s", prompt)  # 输出完整prompt
        else:
            logger.info("=== LLM调用后 [%s] ===", call_location)
            if context:
                logger.info("Context: %s", context)  # 输出完整context


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
    
    # 获取chunk内容 - get_by_ids返回的是list
    chunk_contents_list = await text_chunks_storage.get_by_ids(unique_source_ids)
    chunk_contents = {}
    doc_ids = []
    
    # 将list转换为dict并提取doc_ids
    for i, chunk_id in enumerate(unique_source_ids):
        if i < len(chunk_contents_list) and chunk_contents_list[i] is not None:
            content = chunk_contents_list[i]
            chunk_contents[chunk_id] = content["content"] if isinstance(content, dict) and "content" in content else str(content)
            
            # 提取doc_id
            if isinstance(content, dict) and "full_doc_id" in content:
                doc_ids.append(content["full_doc_id"])
    
    # 构建溯源信息
    source_metadata = {
        "source_chunks": unique_source_ids,
        "chunk_contents": chunk_contents,
        "doc_ids": list(set(doc_ids)),
        "entities_used": [node["node_id"] for node in _process_nodes],
        "relations_used": [(edge[0], edge[1], edge[2]["description"]) for edge in _process_edges]
    }
    
    return source_metadata


async def _pre_tokenize(
    graph_storage: NetworkXStorage, tokenizer: Tokenizer, edges: list, nodes: list
) -> tuple:

    sem = asyncio.Semaphore(1000)

    async def handle_edge(edge: tuple) -> tuple:
        async with sem:
            if "length" not in edge[2]:
                edge[2]["length"] = len(
                    await asyncio.get_event_loop().run_in_executor(
                        None, tokenizer.encode_string, edge[2]["description"]
                    )
                )
            return edge

    async def handle_node(node: dict) -> dict:
        async with sem:
            if "length" not in node[1]:
                node[1]["length"] = len(
                    await asyncio.get_event_loop().run_in_executor(
                        None, tokenizer.encode_string, node[1]["description"]
                    )
                )
            return node

    new_edges = []
    new_nodes = []

    for result in tqdm_async(
        asyncio.as_completed([handle_edge(edge) for edge in edges]),
        total=len(edges),
        desc="Pre-tokenizing edges",
    ):
        new_edge = await result
        await graph_storage.update_edge(new_edge[0], new_edge[1], new_edge[2])
        new_edges.append(new_edge)

    for result in tqdm_async(
        asyncio.as_completed([handle_node(node) for node in nodes]),
        total=len(nodes),
        desc="Pre-tokenizing nodes",
    ):
        new_node = await result
        await graph_storage.update_node(new_node[0], new_node[1])
        new_nodes.append(new_node)

    await graph_storage.index_done_callback()
    return new_edges, new_nodes


async def _construct_rephrasing_prompt(
    _process_nodes: list,
    _process_edges: list,
    text_chunks_storage: JsonKVStorage,
    add_context: bool = False,
) -> str:
    entities = [
        f"{_process_node['node_id']}: {_process_node['description']}"
        for _process_node in _process_nodes
    ]
    relations = [
        f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
        for _process_edge in _process_edges
    ]

    entities_str = "\n".join(
        [f"{index + 1}. {entity}" for index, entity in enumerate(entities)]
    )
    relations_str = "\n".join(
        [f"{index + 1}. {relation}" for index, relation in enumerate(relations)]
    )
    language = (
        "Chinese"
        if detect_main_language(entities_str + relations_str) == "zh"
        else "English"
    )

    if add_context:
        original_ids = [
            node["source_id"].split("<SEP>")[0] for node in _process_nodes
        ] + [edge[2]["source_id"].split("<SEP>")[0] for edge in _process_edges]

        original_ids = list(set(original_ids))
        original_text = await text_chunks_storage.get_by_ids(original_ids)
        original_text = "\n".join(
            [
                f"{index + 1}. {text['content']}"
                for index, text in enumerate(original_text)
            ]
        )

        prompt = ANSWER_REPHRASING_PROMPT[language]["CONTEXT_TEMPLATE"].format(
            language=language,
            original_text=original_text,
            entities=entities_str,
            relationships=relations_str,
        )
        return prompt

    prompt = ANSWER_REPHRASING_PROMPT[language]["TEMPLATE"].format(
        language=language, entities=entities_str, relationships=relations_str
    )
    return prompt


def get_average_loss(batch: tuple, loss_strategy: str) -> float:
    try:
        if loss_strategy == "only_edge":
            return sum(edge[2]["loss"] for edge in batch[1]) / len(batch[1])
        if loss_strategy == "both":
            return sum(edge[2]["loss"] for edge in batch[1]) + sum(
                node["loss"] for node in batch[0]
            ) / (len(batch[0]) + len(batch[1]))
        raise ValueError("Invalid loss strategy")
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error calculating average loss: %s", e)
        return -1.0


def _post_process_synthetic_data(data):
    block = data.split("\n\n")
    qas = []
    for line in block:
        if "Question:" in line and "Answer:" in line:
            question = line.split("Question:")[1].split("Answer:")[0].strip()
            answer = line.split("Answer:")[1].strip()
            qas.append({"question": question, "answer": answer})
        elif "问题：" in line and "答案：" in line:
            question = line.split("问题：")[1].split("答案：")[0].strip()
            answer = line.split("答案：")[1].strip()
            qas.append({"question": question, "answer": answer})
        elif "问题:" in line and "回答:" in line:
            question = line.split("问题:")[1].split("回答:")[0].strip()
            answer = line.split("回答:")[1].strip()
            qas.append({"question": question, "answer": answer})
    return qas


async def traverse_graph_by_edge(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """
    Traverse the graph

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param traverse_strategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_nodes_and_edges(
        _process_nodes: list,
        _process_edges: list,
    ) -> str:
        prompt = await _construct_rephrasing_prompt(
            _process_nodes, _process_edges, text_chunks_storage, add_context=False
        )
        
        # 调用前记录日志（仅第一次）
        log_llm_call_once("text_rephrasing", prompt, is_before=True)
        
        context = await llm_client.generate_answer(prompt)
        
        # 调用后记录日志（仅第一次）
        log_llm_call_once("text_rephrasing", prompt, context, is_before=False)

        # post-process the context
        if context.startswith("Rephrased Text:"):
            context = context[len("Rephrased Text:") :].strip()
        elif context.startswith("重述文本:"):
            context = context[len("重述文本:") :].strip()

        return context

    async def _process_single_batch(
        _process_batch: tuple, question_type: str = "single"
    ) -> dict:
        async with semaphore:
            context = await _process_nodes_and_edges(
                _process_batch[0],
                _process_batch[1],
            )

            language = "Chinese" if detect_main_language(context) == "zh" else "English"
            pre_length = sum(node["length"] for node in _process_batch[0]) + sum(
                edge[2]["length"] for edge in _process_batch[1]
            )

            if question_type == "single":
                prompt_text = QUESTION_GENERATION_PROMPT[language]["SINGLE_TEMPLATE"].format(
                    answer=context
                )
                
                # 调用前记录日志（仅第一次）
                log_llm_call_once("single_question_generation", prompt_text, is_before=True)
                
                question = await llm_client.generate_answer(prompt_text)
                
                # 调用后记录日志（仅第一次）
                log_llm_call_once("single_question_generation", prompt_text, question, is_before=False)
                if question.startswith("Question:"):
                    question = question[len("Question:") :].strip()
                elif question.startswith("问题："):
                    question = question[len("问题：") :].strip()

                logger.info(
                    "%d nodes and %d edges processed",
                    len(_process_batch[0]),
                    len(_process_batch[1]),
                )
                logger.info("Pre-length: %s", pre_length)
                logger.info("Question: %s", question)
                logger.info("Answer: %s", context)
                
                # 提取溯源信息
                source_metadata = await _extract_source_metadata(
                    _process_batch[0], _process_batch[1], text_chunks_storage
                )

                return {
                    compute_content_hash(context): {
                        "question": question,
                        "answer": context,
                        "loss": get_average_loss(
                            _process_batch, traverse_strategy.loss_strategy
                        ),
                        # 新增：完整的溯源信息
                        "metadata": {
                            "qa_type": "single",
                            "generation_method": traverse_strategy.qa_form,
                            "source_tracing": source_metadata,
                            "subgraph_info": {
                                "nodes_count": len(_process_batch[0]),
                                "edges_count": len(_process_batch[1]),
                                "max_depth": getattr(traverse_strategy, 'max_depth', 1)
                            }
                        }
                    }
                }

            prompt_text = QUESTION_GENERATION_PROMPT[language]["MULTI_TEMPLATE"].format(
                doc=context
            )
            
            # 调用前记录日志（仅第一次）
            log_llm_call_once("multi_question_generation", prompt_text, is_before=True)
            
            content = await llm_client.generate_answer(prompt_text)
            
            # 调用后记录日志（仅第一次）
            log_llm_call_once("multi_question_generation", prompt_text, content, is_before=False)
            qas = _post_process_synthetic_data(content)

            if len(qas) == 0:
                print(content)
                logger.error(
                    "Error occurred while processing batch, question or answer is None"
                )
                return {}

            final_results = {}
            logger.info(
                "%d nodes and %d edges processed",
                len(_process_batch[0]),
                len(_process_batch[1]),
            )
            logger.info("Pre-length: %s", pre_length)
            
            # 提取溯源信息
            source_metadata = await _extract_source_metadata(
                _process_batch[0], _process_batch[1], text_chunks_storage
            )
            
            for qa in qas:
                logger.info("Question: %s", qa["question"])
                logger.info("Answer: %s", qa["answer"])
                final_results[compute_content_hash(qa["question"])] = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "loss": get_average_loss(
                        _process_batch, traverse_strategy.loss_strategy
                    ),
                    # 新增：完整的溯源信息
                    "metadata": {
                        "qa_type": "multi_hop",
                        "generation_method": traverse_strategy.qa_form,
                        "source_tracing": source_metadata,
                        "subgraph_info": {
                            "nodes_count": len(_process_batch[0]),
                            "edges_count": len(_process_batch[1]),
                            "max_depth": getattr(traverse_strategy, 'max_depth', 1)
                        }
                    }
                }
            return final_results

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    processing_batches = await get_batches_with_strategy(
        nodes, edges, graph_storage, traverse_strategy
    )

    for result in tqdm_async(
        asyncio.as_completed(
            [_process_single_batch(batch) for batch in processing_batches]
        ),
        total=len(processing_batches),
        desc="[4/4]Generating QAs",
    ):
        try:
            if progress_bar is not None:
                progress_bar(
                    len(results) / len(processing_batches), desc="[4/4]Generating QAs"
                )
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[4/4]Generating QAs")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while generating QA: %s", e)

    return results


async def traverse_graph_atomically(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """
    Traverse the graph atomicly

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param traverse_strategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """
    assert traverse_strategy.qa_form == "atomic"

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _generate_question(node_or_edge: tuple):
        if len(node_or_edge) == 2:
            des = node_or_edge[0] + ": " + node_or_edge[1]["description"]
            loss = node_or_edge[1]["loss"]
            source_id = node_or_edge[1].get("source_id", "")
            is_node = True
            entity_name = node_or_edge[0]
        else:
            des = node_or_edge[2]["description"]
            loss = node_or_edge[2]["loss"]
            source_id = node_or_edge[2].get("source_id", "")
            is_node = False
            entity_name = f"{node_or_edge[0]}-{node_or_edge[1]}"

        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(des) == "zh" else "English"

                # 调用前记录日志（仅第一次）
                log_llm_call_once("atomic_generation", 
                    QUESTION_GENERATION_PROMPT[language]["SINGLE_QA_TEMPLATE"].format(doc=des), 
                    is_before=True)
                
                qa = await llm_client.generate_answer(
                    QUESTION_GENERATION_PROMPT[language]["SINGLE_QA_TEMPLATE"].format(
                        doc=des
                    )
                )
                
                # 调用后记录日志（仅第一次）
                log_llm_call_once("atomic_generation", 
                    QUESTION_GENERATION_PROMPT[language]["SINGLE_QA_TEMPLATE"].format(doc=des), 
                    qa, is_before=False)

                if "Question:" in qa and "Answer:" in qa:
                    question = qa.split("Question:")[1].split("Answer:")[0].strip()
                    answer = qa.split("Answer:")[1].strip()
                elif "问题：" in qa and "答案：" in qa:
                    question = qa.split("问题：")[1].split("答案：")[0].strip()
                    answer = qa.split("答案：")[1].strip()
                else:
                    return {}

                question = question.strip('"')
                answer = answer.strip('"')
                
                # 获取chunk内容用于溯源
                source_chunks = source_id.split("<SEP>") if source_id else []
                chunk_contents_list = await text_chunks_storage.get_by_ids(source_chunks)
                chunk_contents = {}
                doc_ids = []
                
                # 将list转换为dict并提取doc_ids
                for i, chunk_id in enumerate(source_chunks):
                    if i < len(chunk_contents_list) and chunk_contents_list[i] is not None:
                        content = chunk_contents_list[i]
                        chunk_contents[chunk_id] = content["content"] if isinstance(content, dict) and "content" in content else str(content)
                        
                        # 提取doc_id
                        if isinstance(content, dict) and "full_doc_id" in content:
                            doc_ids.append(content["full_doc_id"])

                logger.info("Question: %s", question)
                logger.info("Answer: %s", answer)
                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": answer,
                        "loss": loss,
                        # 新增：完整的溯源信息
                        "metadata": {
                            "qa_type": "atomic",
                            "generation_method": "atomic",
                            "source_tracing": {
                                "source_chunks": source_chunks,
                                "chunk_contents": chunk_contents,
                                "doc_ids": list(set(doc_ids)),
                                "entities_used": [entity_name] if is_node else [],
                                "relations_used": [] if is_node else [(node_or_edge[0], node_or_edge[1], des)]
                            },
                            "subgraph_info": {
                                "nodes_count": 1 if is_node else 0,
                                "edges_count": 0 if is_node else 1,
                                "max_depth": 1
                            }
                        }
                    }
                }
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error occurred while generating question: %s", e)
                return {}

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    tasks = []
    for node in nodes:
        if "<SEP>" in node[1]["description"]:
            description_list = node[1]["description"].split("<SEP>")
            for item in description_list:
                tasks.append((node[0], {
                    "description": item, 
                    "loss": node[1]["loss"],
                    "source_id": node[1].get("source_id", "")
                }))
        else:
            tasks.append((node[0], node[1]))
    for edge in edges:
        if "<SEP>" in edge[2]["description"]:
            description_list = edge[2]["description"].split("<SEP>")
            for item in description_list:
                tasks.append(
                    (edge[0], edge[1], {
                        "description": item, 
                        "loss": edge[2]["loss"],
                        "source_id": edge[2].get("source_id", "")
                    })
                )
        else:
            tasks.append((edge[0], edge[1], edge[2]))

    for result in tqdm_async(
        asyncio.as_completed([_generate_question(task) for task in tasks]),
        total=len(tasks),
        desc="[4/4]Generating QAs",
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(tasks), desc="[4/4]Generating QAs")
            results.update(await result)
            if progress_bar is not None and len(results) == len(tasks):
                progress_bar(1, desc="[4/4]Generating QAs")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while generating QA: %s", e)
    return results


async def traverse_graph_for_multi_hop(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """
    基于知识图谱进行多跳推理问答对生成
    
    这是GraphGen框架中最核心的函数之一，负责生成需要多步推理的复杂问答对。
    与atomic和aggregated类型不同，multi_hop问答需要在多个实体和关系间进行跳跃式推理。
    
    工作流程：
    1. 获取图中的所有节点和边，进行预处理和Token化
    2. 根据遍历策略将图划分成多个处理批次（子图）
    3. 对每个批次并发生成多跳推理问答对
    4. 提取问答对并添加完整的溯源元数据
    
    Args:
        llm_client (OpenAIModel): OpenAI语言模型客户端，用于生成问答对
        tokenizer (Tokenizer): 分词器，用于Token计数和文本处理
        graph_storage (NetworkXStorage): 图存储后端，包含实体和关系信息
        traverse_strategy (TraverseStrategy): 图遍历策略，定义如何划分和处理子图
        text_chunks_storage (JsonKVStorage): 文本块存储，用于溯源信息提取
        progress_bar (gr.Progress, optional): Gradio进度条，用于Web界面显示进度
        max_concurrent (int): 最大并发数，控制同时处理的批次数量，默认1000
        
    Returns:
        dict: 生成的问答对字典，格式为：
            {
                "问题内容哈希": {
                    "question": "多跳推理问题",
                    "answer": "需要跨多个实体推理的答案", 
                    "loss": 平均损失值,
                    "metadata": {
                        "qa_type": "multi_hop",
                        "generation_method": 生成方法,
                        "source_tracing": 溯源信息,
                        "subgraph_info": 子图统计信息
                    }
                }
            }
    """
    # 初始化并发控制信号量，限制同时处理的批次数量，避免过度消耗资源
    semaphore: Semaphore = asyncio.Semaphore(max_concurrent)

    # 存储最终生成的问答对结果
    results = {}
    
    # 从图存储中获取所有边和节点数据
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    # 对图数据进行预处理：添加token计数、过滤无效数据等
    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    # 根据遍历策略将节点和边组织成处理批次（子图）
    # 每个批次包含相关联的节点和边，用于生成一个多跳推理问答对
    processing_batches = await get_batches_with_strategy(
        nodes, edges, graph_storage, traverse_strategy
    )

    async def _process_single_batch(_process_batch: tuple) -> dict:
        """
        处理单个批次（子图），生成一个多跳推理问答对
        
        此内部函数是整个多跳推理生成的核心逻辑：
        1. 从子图中提取实体和关系信息
        2. 根据语言选择合适的提示词模板
        3. 调用LLM生成问答对
        4. 解析和清洗生成的结果
        5. 添加完整的元数据和溯源信息
        
        Args:
            _process_batch (tuple): 包含节点和边的元组，格式: (nodes, edges)
            
        Returns:
            dict: 包含问题、答案和元数据的字典，失败时返回空字典
        """
        async with semaphore:  # 控制并发数量，防止过多请求同时进行
            try:
                # 检测输入文本的主要语言，以选择合适的提示词模板
                # 支持中文和英文两种语言的问答对生成
                language = (
                    "Chinese"
                    if detect_main_language(_process_batch[0][0]["description"]) == "zh"
                    else "English"
                )

                # 解析批次数据：获取节点和边信息
                _process_nodes = _process_batch[0]  # 子图中的所有节点（实体）
                _process_edges = _process_batch[1]  # 子图中的所有边（关系）

                # 构建实体列表：将节点ID和描述组合成易读格式
                # 格式："node_id: description"
                entities = [
                    f"{_process_node['node_id']}: {_process_node['description']}"
                    for _process_node in _process_nodes
                ]

                # 构建关系列表：将边的起点、终点和关系描述组合
                # 格式："source_node -- target_node: relationship_description"
                relations = [
                    f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
                    for _process_edge in _process_edges
                ]

                # 将实体列表格式化为编号列表，供LLM更好理解
                entities_str = "\n".join(
                    [f"{index + 1}. {entity}" for index, entity in enumerate(entities)]
                )
                # 将关系列表格式化为编号列表，供LLM更好理解
                relations_str = "\n".join(
                    [
                        f"{index + 1}. {relation}"
                        for index, relation in enumerate(relations)
                    ]
                )

                # 根据检测到的语言选择对应的提示词模板
                # MULTI_HOP_GENERATION_PROMPT包含中英文两个版本的多跳推理提示词
                prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
                    entities=entities_str, relationships=relations_str
                )

                # 调用前记录日志（仅第一次）
                log_llm_call_once("multi_hop_generation", prompt, is_before=True)
                
                # 调用LLM生成多跳推理问答对
                # 这是整个流程中最关键的一步，会消耗最多时间和资源
                context = await llm_client.generate_answer(prompt)
                
                # 调用后记录日志（仅第一次）
                log_llm_call_once("multi_hop_generation", prompt, context, is_before=False)

                # 解析LLM返回的内容，提取问题和答案
                # 支持中英文两种格式的输出解析
                if "Question:" in context and "Answer:" in context:
                    # 英文格式：Question: ... Answer: ...
                    question = context.split("Question:")[1].split("Answer:")[0].strip()
                    answer = context.split("Answer:")[1].strip()
                elif "问题：" in context and "答案：" in context:
                    # 中文格式：问题：... 答案：...
                    question = context.split("问题：")[1].split("答案：")[0].strip()
                    answer = context.split("答案：")[1].strip()
                else:
                    # 如果LLM返回的内容格式不正确，跳过该批次
                    return {}

                # 清理问题和答案文本：移除多余的引号
                question = question.strip('"')
                answer = answer.strip('"')

                # 记录生成的问答对，用于调试和追踪
                logger.info("Question: %s", question)
                logger.info("Answer: %s", answer)
                
                # 从子图中提取溯源信息，用于追踪问答对的数据来源
                source_metadata = await _extract_source_metadata(
                    _process_nodes, _process_edges, text_chunks_storage
                )

                # 构建最终的问答对结果，包含丰富的元数据
                return {
                    compute_content_hash(question): {  # 使用问题内容的哈希值作为唯一标识符
                        "question": question,
                        "answer": answer,
                        # 计算该批次的平均损失值，用于评估问答对质量
                        "loss": get_average_loss(
                            _process_batch, traverse_strategy.loss_strategy
                        ),
                        # 元数据：包含完整的溯源和统计信息
                        "metadata": {
                            "qa_type": "multi_hop",  # 问答对类型标识
                            "generation_method": traverse_strategy.qa_form,  # 生成方法
                            "source_tracing": source_metadata,  # 数据溯源信息
                            "subgraph_info": {  # 子图统计信息
                                "nodes_count": len(_process_nodes),  # 节点数量
                                "edges_count": len(_process_edges),  # 边数量
                                "max_depth": getattr(traverse_strategy, 'max_depth', 1)  # 最大推理深度
                            }
                        }
                    }
                }

            except Exception as e:  # pylint: disable=broad-except
                # 在处理单个批次时发生异常，记录错误信息但不中断整个流程
                logger.error("Error occurred while processing batch: %s", e)
                return {}  # 返回空字典，该批次将被跳过

    # 并发处理所有批次，并显示进度
    # 使用asyncio.as_completed确保任务完成后立即处理结果，提高效率
    async for result in tqdm_async(
        asyncio.as_completed(
            [_process_single_batch(batch) for batch in processing_batches]
        ),
        total=len(processing_batches),
        desc="[4/4]Generating QAs",  # 进度条描述：第4步共4步 - 生成问答对
    ):
        try:
            # 更新Web界面的进度条（如果存在）
            if progress_bar is not None:
                progress_bar(
                    len(results) / len(processing_batches), desc="[4/4]Generating QAs"
                )
            
            # 将此批次的结果合并到总结果中
            # await result 是因为result本身是一个coroutine对象
            results.update(await result)
            
            # 当所有批次处理完成时，将进度条设置为100%
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[4/4]Generating QAs")
        except Exception as e:  # pylint: disable=broad-except
            # 记录处理过程中的任何异常，但不中断整个生成流程
            logger.error("Error occurred while generating QA: %s", e)
    
    # 返回包含所有生成的多跳推理问答对的字典
    # 每个问答对都包含完整的元数据和溯源信息
    return results
