import asyncio
from collections import defaultdict

from tqdm.asyncio import tqdm as tqdm_async
from graphgen.models import JsonKVStorage, OpenAIModel, NetworkXStorage
from graphgen.utils import logger, detect_main_language
from graphgen.templates import DESCRIPTION_REPHRASING_PROMPT

# LLM调用位置跟踪：记录每个调用位置是否已经输出过详细日志
llm_call_tracker = set()


def log_llm_call_once(call_location: str, prompt: str, context: str = None, is_before: bool = True):
    """
    为LLM调用记录日志，每个位置只在第一次调用时输出详细信息
    
    Args:
        call_location (str): 调用位置标识，如 'quiz_generation'
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


async def quiz(
        synth_llm_client: OpenAIModel,
        graph_storage: NetworkXStorage,
        rephrase_storage: JsonKVStorage,
        max_samples: int = 1,
        max_concurrent: int = 1000) -> JsonKVStorage:
    """
    Get all edges and quiz them

    :param synth_llm_client: generate statements
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param max_samples: max samples for each edge
    :param max_concurrent: max concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_quiz(
        des: str,
        prompt: str,
        gt: str
    ):
        async with semaphore:
            try:
                # 如果在rephrase_storage中已经存在，直接取出
                descriptions = await rephrase_storage.get_by_id(des)
                if descriptions:
                    return None

                # 调用前记录日志（仅第一次）
                log_llm_call_once("quiz_generation", prompt, is_before=True)
                
                new_description = await synth_llm_client.generate_answer(
                    prompt,
                    temperature=1
                )
                
                # 调用后记录日志（仅第一次）
                log_llm_call_once("quiz_generation", prompt, new_description, is_before=False)
                return  {des: [(new_description, gt)]}

            except Exception as e: # pylint: disable=broad-except
                logger.error("Error when quizzing description %s: %s", des, e)
                return None


    edges = await graph_storage.get_all_edges()
    nodes = await graph_storage.get_all_nodes()

    results = defaultdict(list)
    tasks = []
    for edge in edges:
        edge_data = edge[2]

        description = edge_data["description"]
        language = "English" if detect_main_language(description) == "en" else "Chinese"

        results[description] = [(description, 'yes')]

        for i in range(max_samples):
            if i > 0:
                tasks.append(
                    _process_single_quiz(description,
                                          DESCRIPTION_REPHRASING_PROMPT[language]['TEMPLATE'].format(
                                              input_sentence=description), 'yes')
                )
            tasks.append(_process_single_quiz(description,
                                              DESCRIPTION_REPHRASING_PROMPT[language]['ANTI_TEMPLATE'].format(
                                                  input_sentence=description), 'no'))

    for node in nodes:
        node_data = node[1]
        description = node_data["description"]
        language = "English" if detect_main_language(description) == "en" else "Chinese"

        results[description] = [(description, 'yes')]

        for i in range(max_samples):
            if i > 0:
                tasks.append(
                    _process_single_quiz(description,
                                          DESCRIPTION_REPHRASING_PROMPT[language]['TEMPLATE'].format(
                                              input_sentence=description), 'yes')
                )
            tasks.append(_process_single_quiz(description,
                                              DESCRIPTION_REPHRASING_PROMPT[language]['ANTI_TEMPLATE'].format(
                                                  input_sentence=description), 'no'))

    for result in tqdm_async(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Quizzing descriptions"
    ):
        new_result = await result
        if new_result:
            for key, value in new_result.items():
                results[key].extend(value)

    for key, value in results.items():
        results[key] = list(set(value))
        await rephrase_storage.upsert({key: results[key]})


    return rephrase_storage
