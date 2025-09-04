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
        self.tokenizer_instance: Tokenizer = Tokenizer(
            model_name=self.config["tokenizer"]
        )
        self.synthesizer_llm_client: OpenAIModel = OpenAIModel(
            model_name=os.getenv("SYNTHESIZER_MODEL"),
            api_key=os.getenv("SYNTHESIZER_API_KEY"),
            base_url=os.getenv("SYNTHESIZER_BASE_URL"),
            tokenizer_instance=self.tokenizer_instance,
        )
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
        # TODO: configurable whether to use coreference resolution
        if len(data) == 0:
            return {}

        inserting_chunks = {}
        if data_type == "raw":
            assert isinstance(data, list) and isinstance(data[0], dict)
            # compute hash for each document
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

        await self.full_docs_storage.upsert(new_docs)
        await self.text_chunks_storage.upsert(inserting_chunks)

        return inserting_chunks

    def insert(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert())

    def insert_data(self, data, data_type):
        """Insert data directly for webui usage"""
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert_data(data, data_type))

    async def async_insert_data(self, data, data_type):
        """Insert chunks into the graph from provided data"""
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

    async def async_insert(self):
        """
        insert chunks into the graph
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
        """Quiz with specific max samples for webui usage"""
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
        """Judge statements with skip option for webui usage"""
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
        """Traverse with specific strategy and output type for webui usage"""
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
