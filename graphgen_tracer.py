"""
GraphGen 溯源工具 - 实现问答对到原文的反向追踪
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SourceTracing:
    """溯源信息数据类"""
    qa_id: str
    question: str
    answer: str
    source_chunks: List[str]  # chunk-{hash} 列表
    chunk_contents: Dict[str, str]  # chunk_id -> content
    doc_ids: List[str]  # doc-{hash} 列表
    entities_used: List[str]  # 使用的实体
    relations_used: List[Tuple[str, str, str]]  # 使用的关系
    subgraph_depth: int
    confidence_score: float

class GraphGenTracer:
    """GraphGen 溯源追踪器"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.graph_storage = None
        self.text_chunks_storage = None
        self.full_docs_storage = None
    
    def load_storages(self):
        """加载存储实例"""
        from graphgen.models import NetworkXStorage, JsonKVStorage
        
        self.graph_storage = NetworkXStorage(self.cache_dir, namespace="graph")
        self.text_chunks_storage = JsonKVStorage(self.cache_dir, namespace="text_chunks")
        self.full_docs_storage = JsonKVStorage(self.cache_dir, namespace="full_docs")
    
    async def trace_qa_to_source(self, qa_id: str, qa_data: Dict) -> Optional[SourceTracing]:
        """
        从QA对追踪到原始文档
        
        Args:
            qa_id: QA对的ID
            qa_data: QA数据，包含question, answer, metadata等
        
        Returns:
            SourceTracing对象，包含完整的溯源信息
        """
        if self.graph_storage is None:
            self.load_storages()
        
        # 如果QA数据中已包含metadata，直接使用
        if "metadata" in qa_data and "source_tracing" in qa_data["metadata"]:
            source_info = qa_data["metadata"]["source_tracing"]
            return SourceTracing(
                qa_id=qa_id,
                question=qa_data["question"],
                answer=qa_data["answer"],
                source_chunks=source_info["source_chunks"],
                chunk_contents=source_info["chunk_contents"],
                doc_ids=source_info["doc_ids"],
                entities_used=source_info["entities_used"],
                relations_used=source_info["relations_used"],
                subgraph_depth=qa_data["metadata"]["subgraph_info"]["max_depth"],
                confidence_score=1.0 - qa_data.get("loss", 0)
            )
        
        # 如果没有metadata，通过文本匹配尝试追踪
        return await self._trace_by_content_matching(qa_id, qa_data)
    
    async def _trace_by_content_matching(self, qa_id: str, qa_data: Dict) -> Optional[SourceTracing]:
        """通过内容匹配进行追踪（备用方案）"""
        question = qa_data["question"]
        answer = qa_data["answer"]
        
        # 获取所有节点和边
        nodes = list(await self.graph_storage.get_all_nodes())
        edges = list(await self.graph_storage.get_all_edges())
        
        # 寻找与问答内容最相关的实体和关系
        relevant_entities = []
        relevant_relations = []
        source_chunks = set()
        
        for node_id, node_data in nodes:
            if self._text_similarity(question + " " + answer, node_data["description"]) > 0.3:
                relevant_entities.append(node_id)
                source_chunks.update(node_data["source_id"].split("<SEP>"))
        
        for edge in edges:
            if self._text_similarity(question + " " + answer, edge[2]["description"]) > 0.3:
                relevant_relations.append((edge[0], edge[1], edge[2]["description"]))
                source_chunks.update(edge[2]["source_id"].split("<SEP>"))
        
        # 获取chunk内容
        chunk_contents_list = await self.text_chunks_storage.get_by_ids(list(source_chunks))
        chunk_contents = {chunk_id: content for chunk_id, content in zip(source_chunks, chunk_contents_list) if content is not None}
        
        # 获取文档ID
        doc_ids = []
        for chunk_id, content in chunk_contents.items():
            if isinstance(content, dict) and "full_doc_id" in content:
                doc_ids.append(content["full_doc_id"])
            elif hasattr(content, 'get'):
                doc_id = content.get("full_doc_id", "")
                if doc_id:
                    doc_ids.append(doc_id)
        
        return SourceTracing(
            qa_id=qa_id,
            question=question,
            answer=answer,
            source_chunks=list(source_chunks),
            chunk_contents={k: (v["content"] if isinstance(v, dict) and "content" in v else str(v)) 
                          for k, v in chunk_contents.items() if v is not None},
            doc_ids=list(set(doc_ids)),
            entities_used=relevant_entities,
            relations_used=relevant_relations,
            subgraph_depth=1,
            confidence_score=0.8  # 默认置信度
        )
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def generate_tracing_report(self, qa_file_path: str, output_path: str = None):
        """
        为整个QA文件生成溯源报告
        
        Args:
            qa_file_path: QA文件路径
            output_path: 输出路径，默认在原文件同目录生成
        """
        if output_path is None:
            output_path = qa_file_path.replace('.json', '_tracing_report.json')
        
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # 处理不同格式的QA数据
        if isinstance(qa_data, list):
            # ChatML或列表格式
            qa_dict = {}
            for i, item in enumerate(qa_data):
                if "messages" in item:
                    # ChatML格式
                    user_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "user"), "")
                    assistant_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "assistant"), "")
                    qa_dict[f"qa_{i}"] = {
                        "question": user_msg,
                        "answer": assistant_msg,
                        "loss": 0.0,
                        "metadata": item.get("metadata", {})
                    }
                else:
                    # 普通列表格式
                    qa_dict[f"qa_{i}"] = item
            qa_data = qa_dict
        
        tracing_report = []
        
        for qa_id, qa_info in qa_data.items():
            tracing = await self.trace_qa_to_source(qa_id, qa_info)
            if tracing:
                tracing_report.append({
                    "qa_id": qa_id,
                    "question": tracing.question,
                    "answer": tracing.answer,
                    "source_analysis": {
                        "source_chunks_count": len(tracing.source_chunks),
                        "source_documents_count": len(tracing.doc_ids),
                        "entities_involved": len(tracing.entities_used),
                        "relations_involved": len(tracing.relations_used),
                        "confidence_score": tracing.confidence_score
                    },
                    "detailed_tracing": {
                        "source_chunks": tracing.source_chunks,
                        "chunk_contents_preview": {
                            k: v[:200] + "..." if len(v) > 200 else v 
                            for k, v in tracing.chunk_contents.items()
                        },
                        "entities_used": tracing.entities_used,
                        "relations_used": tracing.relations_used,
                        "document_ids": tracing.doc_ids
                    }
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "generation_time": str(datetime.now()),
                "qa_file": qa_file_path,
                "tracing_data": tracing_report
            }, f, ensure_ascii=False, indent=2)
        
        print(f"溯源报告已生成：{output_path}")
        return output_path
    
    async def fact_check_qa(self, qa_id: str, qa_data: Dict) -> Dict:
        """
        对QA进行事实性检查
        
        Returns:
            事实性检查结果，包含原文匹配度、逻辑一致性等
        """
        tracing = await self.trace_qa_to_source(qa_id, qa_data)
        if not tracing:
            return {"status": "无法追踪", "confidence": 0.0}
        
        # 计算答案与原文的匹配度
        answer = qa_data["answer"]
        source_text = " ".join(tracing.chunk_contents.values())
        
        matching_score = self._text_similarity(answer, source_text)
        
        return {
            "status": "可追踪",
            "confidence": tracing.confidence_score,
            "source_matching_score": matching_score,
            "source_chunks_used": len(tracing.source_chunks),
            "entities_involved": len(tracing.entities_used),
            "fact_check_level": (
                "高" if matching_score > 0.7 else
                "中" if matching_score > 0.4 else
                "低"
            ),
            "source_preview": source_text[:300] + "..." if len(source_text) > 300 else source_text
        }

# 使用示例
async def main():
    tracer = GraphGenTracer("/mnt/d/git/GraphGen/cache")
    
    # 为最新生成的QA文件生成溯源报告
    import glob
    qa_files = glob.glob("/mnt/d/git/GraphGen/cache/data/graphgen/*/qa-*.json")
    if qa_files:
        latest_file = max(qa_files, key=os.path.getctime)
        print(f"正在为 {latest_file} 生成溯源报告...")
        report_path = await tracer.generate_tracing_report(latest_file)
        print(f"报告生成完成：{report_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())