import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Union, cast

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from .models import (
    Chunk,
    JsonKVStorage,
    JsonListStorage,
    NetworkXStorage,
    OpenAIModel,
    Tokenizer,
    TraverseStrategy,
)
from .models.storage.base_storage import StorageNameSpace
from .operators import (
    extract_kg,
    generate_cot,
    judge_statement,
    quiz,
    search_all,
    traverse_graph_atomically,
    traverse_graph_by_edge,
    traverse_graph_for_multi_hop,
)
from .utils import (
    compute_content_hash,
    create_event_loop,
    format_generation_results,
    logger,
    read_file,
)

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class GraphGen:
    unique_id: int = int(time.time())
    working_dir: str = os.path.join(sys_path, "cache")
    config: Dict = field(default_factory=dict)

    # llm
    tokenizer_instance: Tokenizer = None
    synthesizer_llm_client: OpenAIModel = None
    trainee_llm_client: OpenAIModel = None

    # text chunking
    # TODO: make it configurable
    chunk_size: int = 1024
    chunk_overlap_size: int = 100

    # search
    search_config: dict = field(
        default_factory=lambda: {"enabled": False, "search_types": ["wikipedia"]}
    )

    # traversal
    traverse_strategy: TraverseStrategy = None

    # webui
    progress_bar: gr.Progress = None

    def __post_init__(self):
        """
        GraphGen类的初始化后处理
        注意：这里会从环境变量中读取API Key，如果环境变量未设置会抛出断言错误
        这就是为什么在webui中需要先设置环境变量的原因
        """
        # 初始化分词器
        self.tokenizer_instance: Tokenizer = Tokenizer(
            model_name=self.config["tokenizer"]
        )
        # 初始化生成器LLM客户端（从环境变量读取）
        self.synthesizer_llm_client: OpenAIModel = OpenAIModel(
            model_name=os.getenv("SYNTHESIZER_MODEL"),
            api_key=os.getenv("SYNTHESIZER_API_KEY"),
            base_url=os.getenv("SYNTHESIZER_BASE_URL"),
            tokenizer_instance=self.tokenizer_instance,
        )
        # 初始化学生模型LLM客户端（从环境变量读取）
        self.trainee_llm_client: OpenAIModel = OpenAIModel(
            model_name=os.getenv("TRAINEE_MODEL"),
            api_key=os.getenv("TRAINEE_API_KEY"),
            base_url=os.getenv("TRAINEE_BASE_URL"),
            tokenizer_instance=self.tokenizer_instance,
        )
        self.search_config = self.config["search"]

        if "traverse_strategy" in self.config:
            self.traverse_strategy = TraverseStrategy(
                **self.config["traverse_strategy"]
            )

        # 初始化各种存储实例
        self.full_docs_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="full_docs"
        )
        self.text_chunks_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="text_chunks"
        )
        self.graph_storage: NetworkXStorage = NetworkXStorage(
            self.working_dir, namespace="graph"
        )
        self.search_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="search"
        )
        self.rephrase_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="rephrase"
        )
        self.qa_storage: JsonListStorage = JsonListStorage(
            os.path.join(self.working_dir, "data", "graphgen", str(self.unique_id)),
            namespace=f"qa-{self.unique_id}",
        )

    async def async_split_chunks(
        self, data: List[Union[List, Dict]], data_type: str
    ) -> dict:
        """
        将输入数据分割为文本块
        支持raw和chunked两种输入格式
        
        Args:
            data: 输入数据，可以是原始文本或已分块的数据
            data_type: 数据类型，'raw' 或 'chunked'
        
        Returns:
            分块后的数据字典，以chunk_id为键
        """
        # TODO: configurable whether to use coreference resolution
        if len(data) == 0:
            return {}

        inserting_chunks = {}
        if data_type == "raw":
            assert isinstance(data, list) and isinstance(data[0], dict)
            # 为每个文档计算哈希值
            new_docs = {
                compute_content_hash(doc["content"], prefix="doc-"): {
                    "content": doc["content"]
                }
                for doc in data
            }
            _add_doc_keys = await self.full_docs_storage.filter_keys(
                list(new_docs.keys())
            )
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs) == 0:
                logger.warning("All docs are already in the storage")
                return {}
            logger.info("[New Docs] inserting %d docs", len(new_docs))

            cur_index = 1
            doc_number = len(new_docs)
            async for doc_key, doc in tqdm_async(
                new_docs.items(), desc="[1/4]Chunking documents", unit="doc"
            ):
                # 按Token大小分割文档
                chunks = {
                    compute_content_hash(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in self.tokenizer_instance.chunk_by_token_size(
                        doc["content"], self.chunk_overlap_size, self.chunk_size
                    )
                }
                inserting_chunks.update(chunks)

                if self.progress_bar is not None:
                    self.progress_bar(cur_index / doc_number, f"Chunking {doc_key}")
                    cur_index += 1

            _add_chunk_keys = await self.text_chunks_storage.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
        elif data_type == "chunked":
            # 处理已分块的数据
            assert isinstance(data, list) and isinstance(data[0], list)
            new_docs = {
                compute_content_hash("".join(chunk["content"]), prefix="doc-"): {
                    "content": "".join(chunk["content"])
                }
                for doc in data
                for chunk in doc
            }
            _add_doc_keys = await self.full_docs_storage.filter_keys(
                list(new_docs.keys())
            )
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs) == 0:
                logger.warning("All docs are already in the storage")
                return {}
            logger.info("[New Docs] inserting %d docs", len(new_docs))
            async for doc in tqdm_async(
                data, desc="[1/4]Chunking documents", unit="doc"
            ):
                doc_str = "".join([chunk["content"] for chunk in doc])
                for chunk in doc:
                    chunk_key = compute_content_hash(chunk["content"], prefix="chunk-")
                    inserting_chunks[chunk_key] = {
                        **chunk,
                        "full_doc_id": compute_content_hash(doc_str, prefix="doc-"),
                    }
            _add_chunk_keys = await self.text_chunks_storage.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # 保存文档和分块数据
        await self.full_docs_storage.upsert(new_docs)
        await self.text_chunks_storage.upsert(inserting_chunks)

        return inserting_chunks

    def insert(self):
        """
        同步插入数据的包装方法
        
        这个方法是 async_insert() 的同步版本，主要用于：
        1. 命令行工具调用
        2. 脚本化处理
        3. 不支持异步的环境
        
        内部通过创建事件循环来运行异步的 async_insert() 方法
        相比 insert_data()，这个方法从配置文件中读取输入文件路径
        """
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert())

    def insert_data(self, data, data_type):
        """
        直接插入数据，主webui使用
        
        Args:
            data: 要插入的数据
            data_type: 数据类型，'raw' 或 'chunked'
        """
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert_data(data, data_type))

    async def async_insert_data(self, data, data_type):
        """
        异步插入数据到知识图谱
        这是webui调用的主要数据处理流程
        
        Args:
            data: 要插入的数据
            data_type: 数据类型，'raw' 或 'chunked'
        """
        # 首先将数据分块
        inserting_chunks = await self.async_split_chunks(data, data_type)

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return
        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))

        # 进行实体和关系抽取，构建知识图谱
        logger.info("[Entity and Relation Extraction]...")
        _add_entities_and_relations = await extract_kg(
            llm_client=self.synthesizer_llm_client,
            kg_instance=self.graph_storage,
            tokenizer_instance=self.tokenizer_instance,
            chunks=[
                Chunk(id=k, content=v["content"]) for k, v in inserting_chunks.items()
            ],
            progress_bar=self.progress_bar,
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted")
            return

        await self._insert_done()

    async def async_insert(self):
        """
        异步插入数据到知识图谱（从配置文件读取输入）
        
        这是传统的数据插入方法，主要用于：
        1. 命令行工具传入配置文件
        2. 批处理任务
        3. 脚本化执行
        
        与 insert_data() 的区别：
        - 这个方法从 self.config["input_file"] 读取文件路径
        - insert_data() 直接接收数据参数，主webui使用
        
        数据处理流程：
        1. 从配置中获取输入文件路径和数据类型
        2. 读取文件内容
        3. 分块处理
        4. 抽取实体和关系
        5. 构建知识图谱
        """
        input_file = self.config["input_file"]
        data_type = self.config["input_data_type"]
        data = read_file(input_file)

        inserting_chunks = await self.async_split_chunks(data, data_type)

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return
        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))

        logger.info("[Entity and Relation Extraction]...")
        _add_entities_and_relations = await extract_kg(
            llm_client=self.synthesizer_llm_client,
            kg_instance=self.graph_storage,
            tokenizer_instance=self.tokenizer_instance,
            chunks=[
                Chunk(id=k, content=v["content"]) for k, v in inserting_chunks.items()
            ],
            progress_bar=self.progress_bar,
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted")
            return

        await self._insert_done()

    async def _insert_done(self):
        """
        数据插入完成后的清理工作
        
        这个私有方法在数据插入完成后调用，负责：
        1. 对所有存储实例执行索引完成回调
        2. 确保数据持久化存储
        3. 更新缓存和索引
        
        涉及的存储实例：
        - full_docs_storage: 完整文档存储
        - text_chunks_storage: 文本分块存储
        - graph_storage: 知识图谱存储
        - search_storage: 搜索结果存储
        """
        tasks = []
        for storage_instance in [
            self.full_docs_storage,
            self.text_chunks_storage,
            self.graph_storage,
            self.search_storage,
        ]:
            if storage_instance is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_instance).index_done_callback())
        await asyncio.gather(*tasks)

    def search(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_search())

    async def async_search(self):
        logger.info(
            "Search is %s", "enabled" if self.search_config["enabled"] else "disabled"
        )
        if self.search_config["enabled"]:
            logger.info(
                "[Search] %s ...", ", ".join(self.search_config["search_types"])
            )
            all_nodes = await self.graph_storage.get_all_nodes()
            all_nodes_names = [node[0] for node in all_nodes]
            new_search_entities = await self.full_docs_storage.filter_keys(
                all_nodes_names
            )
            logger.info(
                "[Search] Found %d entities to search", len(new_search_entities)
            )
            _add_search_data = await search_all(
                search_types=self.search_config["search_types"],
                search_entities=new_search_entities,
            )
            if _add_search_data:
                await self.search_storage.upsert(_add_search_data)
                logger.info("[Search] %d entities searched", len(_add_search_data))

                # Format search results for inserting
                search_results = []
                for _, search_data in _add_search_data.items():
                    search_results.extend(
                        [
                            {"content": search_data[key]}
                            for key in list(search_data.keys())
                        ]
                    )
                # TODO: fix insert after search
                await self.async_insert()

    def quiz(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_quiz())

    def quiz_with_samples(self, max_samples):
        """
        使用指定样本数进行测验（webui使用）
        
        这个方法为webui界面专门设计，允许动态指定Quiz样本数
        而不是使用配置文件中的固定值
        
        Args:
            max_samples: 最大测验样本数
        """
        loop = create_event_loop()
        loop.run_until_complete(self.async_quiz_with_samples(max_samples))

    async def async_quiz_with_samples(self, max_samples):
        await quiz(
            self.synthesizer_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            max_samples,
        )
        await self.rephrase_storage.index_done_callback()

    async def async_quiz(self):
        max_samples = self.config["quiz_and_judge_strategy"]["quiz_samples"]
        await quiz(
            self.synthesizer_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            max_samples,
        )
        await self.rephrase_storage.index_done_callback()

    def judge(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_judge())

    def judge_statements(self, skip=False):
        """
        判断语句质量（webui使用，支持跳过选项）
        
        这个方法为webui界面设计，允许在不使用学生模型时跳过语句判断
        这样可以减少API调用次数和成本
        
        Args:
            skip: 是否跳过语句判断，默认False
        """
        if skip:
            return
        loop = create_event_loop()
        loop.run_until_complete(self.async_judge())

    async def async_judge(self):
        re_judge = self.config["quiz_and_judge_strategy"]["re_judge"]
        _update_relations = await judge_statement(
            self.trainee_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            re_judge,
        )
        await _update_relations.index_done_callback()

    def traverse(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_traverse())

    def traverse_with_strategy(self, traverse_strategy, output_data_type):
        """
        使用指定策略和输出类型遍历图谱（webui使用）
        
        这个方法为webui界面设计，允许动态指定：
        1. 遍历策略（双向、扩展方法、采样策略等）
        2. 输出数据类型（atomic/multi_hop/aggregated）
        
        支持的输出类型：
        - atomic: 原子性问答对，直接从单个实体生成
        - multi_hop: 多跳推理问答对，需要跨多个实体
        - aggregated: 聚合型问答对，基于关系连接
        
        Args:
            traverse_strategy: 遍历策略对象
            output_data_type: 输出数据类型
        """
        self.traverse_strategy = traverse_strategy
        loop = create_event_loop()
        loop.run_until_complete(self.async_traverse_with_strategy(output_data_type))

    async def async_traverse_with_strategy(self, output_data_type):
        if output_data_type == "atomic":
            results = await traverse_graph_atomically(
                self.synthesizer_llm_client,
                self.tokenizer_instance,
                self.graph_storage,
                self.traverse_strategy,
                self.text_chunks_storage,
                self.progress_bar,
            )
        elif output_data_type == "multi_hop":
            results = await traverse_graph_for_multi_hop(
                self.synthesizer_llm_client,
                self.tokenizer_instance,
                self.graph_storage,
                self.traverse_strategy,
                self.text_chunks_storage,
                self.progress_bar,
            )
        elif output_data_type == "aggregated":
            results = await traverse_graph_by_edge(
                self.synthesizer_llm_client,
                self.tokenizer_instance,
                self.graph_storage,
                self.traverse_strategy,
                self.text_chunks_storage,
                self.progress_bar,
            )
        else:
            raise ValueError(f"Unknown qa_form: {output_data_type}")

        # Use default format for webui
        results = format_generation_results(
            results, output_data_format="ChatML"
        )

        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()

    async def async_traverse(self):
        output_data_type = self.config["output_data_type"]

        if output_data_type == "atomic":
            results = await traverse_graph_atomically(
                self.synthesizer_llm_client,
                self.tokenizer_instance,
                self.graph_storage,
                self.traverse_strategy,
                self.text_chunks_storage,
                self.progress_bar,
            )
        elif output_data_type == "multi_hop":
            results = await traverse_graph_for_multi_hop(
                self.synthesizer_llm_client,
                self.tokenizer_instance,
                self.graph_storage,
                self.traverse_strategy,
                self.text_chunks_storage,
                self.progress_bar,
            )
        elif output_data_type == "aggregated":
            results = await traverse_graph_by_edge(
                self.synthesizer_llm_client,
                self.tokenizer_instance,
                self.graph_storage,
                self.traverse_strategy,
                self.text_chunks_storage,
                self.progress_bar,
            )
        else:
            raise ValueError(f"Unknown qa_form: {output_data_type}")

        results = format_generation_results(
            results, output_data_format=self.config["output_data_format"]
        )

        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()

    def generate_reasoning(self, method_params):
        loop = create_event_loop()
        loop.run_until_complete(self.async_generate_reasoning(method_params))

    async def async_generate_reasoning(self, method_params):
        results = await generate_cot(
            self.graph_storage,
            self.synthesizer_llm_client,
            method_params=method_params,
        )

        results = format_generation_results(
            results, output_data_format=self.config["output_data_format"]
        )

        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()

    def clear(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_clear())

    async def async_clear(self):
        await self.full_docs_storage.drop()
        await self.text_chunks_storage.drop()
        await self.search_storage.drop()
        await self.graph_storage.clear()
        await self.rephrase_storage.drop()
        await self.qa_storage.drop()

        logger.info("All caches are cleared")
