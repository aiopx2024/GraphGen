from typing import List

from graphgen.models import Chunk, OpenAIModel
from graphgen.templates import COREFERENCE_RESOLUTION_PROMPT
from graphgen.utils import detect_main_language, logger

# LLM调用位置跟踪：记录每个调用位置是否已经输出过详细日志
llm_call_tracker = set()


def log_llm_call_once(call_location: str, prompt: str, context: str = None, is_before: bool = True):
    """
    为LLM调用记录日志，每个位置只在第一次调用时输出详细信息
    
    Args:
        call_location (str): 调用位置标识，如 'coreference_resolution'
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


async def resolute_coreference(
    llm_client: OpenAIModel, chunks: List[Chunk]
) -> List[Chunk]:
    """
    Resolute conference

    :param llm_client: LLM model
    :param chunks: List of chunks
    :return: List of chunks
    """

    if len(chunks) == 0:
        return chunks

    results = [chunks[0]]

    for _, chunk in enumerate(chunks[1:]):
        language = detect_main_language(chunk.content)
        prompt = COREFERENCE_RESOLUTION_PROMPT[language].format(
            reference=results[0].content, input_sentence=chunk.content
        )
        
        # 调用前记录日志（仅第一次）
        log_llm_call_once("coreference_resolution", prompt, is_before=True)
        
        result = await llm_client.generate_answer(prompt)
        
        # 调用后记录日志（仅第一次）
        log_llm_call_once("coreference_resolution", prompt, result, is_before=False)
        
        results.append(Chunk(id=chunk.id, content=result))

    return results
