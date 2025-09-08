import asyncio

from .log import logger


def create_event_loop() -> asyncio.AbstractEventLoop:
    """
    确保始终有可用的事件循环
    
    这个函数解决了在不同环境下运行异步代码的问题：
    
    场景1 - 命令行工具：
        # 没有事件循环，需要创建
        python -m graphgen.generate --config atomic_config.yaml
    
    场景2 - Jupyter Notebook：
        # 已有事件循环，但可能已关闭
        graphgen = GraphGen(config)
        graphgen.insert()  # 需要检查并重用或创建
    
    场景3 - Web应用：
        # 在Gradio中，可能有多个线程和事件循环
        await graph_gen.async_insert_data(data, data_type)
    
    如果当前事件循环已关闭或不存在，它会创建一个新的事件循环并将其设置为当前事件循环。

    Returns:
        asyncio.AbstractEventLoop: 当前或新创建的事件循环
    """
    try:
        # 尝试获取当前事件循环
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # 如果没有事件循环存在或已关闭，创建一个新的
        logger.info("在主线程中创建新的事件循环")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop
