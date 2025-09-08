#!/usr/bin/env python3
"""
GraphGen批量处理工具
用于批量处理大量txt文件，生成知识图谱和各种类型的语料对

功能：
1. 批量处理多个txt文件
2. 生成GraphML知识图谱文件
3. 生成不同类型的QA语料对（atomic, aggregated, multi_hop, cot）
4. 支持自定义配置和输出目录
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import yaml
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from graphgen.graphgen import GraphGen
from graphgen.utils import logger, set_logger


def prepare_txt_data(txt_files: List[str], chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    准备txt文件数据，转换为GraphGen可处理的格式
    
    Args:
        txt_files: txt文件路径列表
        chunk_size: 文本分块大小
    
    Returns:
        处理后的数据列表
    """
    data = []
    
    for txt_file in txt_files:
        print(f"处理文件: {txt_file}")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 如果内容过长，进行分块
        if len(content) > chunk_size:
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            for i, chunk in enumerate(chunks):
                data.append({
                    "content": chunk.strip(),
                    "source_file": txt_file,
                    "chunk_id": i
                })
        else:
            data.append({
                "content": content,
                "source_file": txt_file,
                "chunk_id": 0
            })
    
    return data


def create_config(output_type: str, input_file: str, config_template: str = None) -> Dict[str, Any]:
    """
    创建GraphGen配置
    
    Args:
        output_type: 输出类型 (atomic, aggregated, multi_hop, cot)
        input_file: 输入文件路径
        config_template: 配置模板路径
    
    Returns:
        配置字典
    """
    # 基础配置
    config = {
        "input_data_type": "raw",
        "input_file": input_file,
        "output_data_type": output_type,
        "output_data_format": "ChatML",
        "tokenizer": "cl100k_base",
        "search": {
            "enabled": False,
            "search_types": ["wikipedia"]
        },
        "quiz_and_judge_strategy": {
            "enabled": True,
            "quiz_samples": 2,
            "re_judge": False
        }
    }
    
    # 根据输出类型设置遍历策略
    if output_type == "atomic":
        config["traverse_strategy"] = {
            "bidirectional": True,
            "edge_sampling": "max_loss",
            "expand_method": "max_tokens",
            "isolated_node_strategy": "ignore",
            "max_depth": 2,
            "max_tokens": 128,
            "loss_strategy": "only_edge"
        }
    elif output_type == "aggregated":
        config["traverse_strategy"] = {
            "bidirectional": True,
            "edge_sampling": "max_loss",
            "expand_method": "max_width",
            "isolated_node_strategy": "ignore",
            "max_depth": 5,
            "max_extra_edges": 20,
            "max_tokens": 256,
            "loss_strategy": "only_edge"
        }
    elif output_type == "multi_hop":
        config["traverse_strategy"] = {
            "bidirectional": True,
            "edge_sampling": "max_loss",
            "expand_method": "max_tokens",
            "isolated_node_strategy": "ignore",
            "max_depth": 3,
            "max_tokens": 512,
            "loss_strategy": "both"
        }
    elif output_type == "cot":
        config["method_params"] = {
            "method": "leiden",
            "num_communities": 10,
            "max_samples_per_community": 5
        }
    
    return config


def process_single_batch(txt_files: List[str], output_dir: str, output_types: List[str], 
                        chunk_size: int = 512, enable_trainee: bool = True):
    """
    处理单个批次的文件
    
    Args:
        txt_files: txt文件路径列表
        output_dir: 输出目录
        output_types: 输出类型列表
        chunk_size: 文本分块大小
        enable_trainee: 是否启用trainee模型
    """
    # 创建输出目录
    batch_dir = os.path.join(output_dir, f"batch_{int(time.time())}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # 准备数据
    print("准备输入数据...")
    data = prepare_txt_data(txt_files, chunk_size)
    
    # 保存原始数据为jsonl格式
    input_file = os.path.join(batch_dir, "input_data.jsonl")
    with open(input_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存输入数据到: {input_file}")
    print(f"总共处理了 {len(data)} 个文本块")
    
    # 为每种输出类型生成语料
    for output_type in output_types:
        print(f"\n开始生成 {output_type} 类型的语料...")
        
        try:
            # 创建配置
            config = create_config(output_type, input_file)
            
            # 禁用trainee相关功能（如果不需要）
            if not enable_trainee:
                config["quiz_and_judge_strategy"]["enabled"] = False
                if "traverse_strategy" in config:
                    config["traverse_strategy"]["edge_sampling"] = "random"
            
            # 创建输出目录
            type_output_dir = os.path.join(batch_dir, output_type)
            os.makedirs(type_output_dir, exist_ok=True)
            
            # 设置日志
            log_file = os.path.join(type_output_dir, f"graphgen_{output_type}.log")
            set_logger(log_file, if_stream=True)
            
            # 初始化GraphGen
            graph_gen = GraphGen(config=config, working_dir=type_output_dir)
            
            # 执行生成流程
            if output_type == "cot":
                # CoT生成流程
                graph_gen.insert()
                if config["search"]["enabled"]:
                    graph_gen.search()
                graph_gen.generate_reasoning(method_params=config["method_params"])
            else:
                # 常规QA生成流程
                graph_gen.insert()
                
                if config["search"]["enabled"]:
                    graph_gen.search()
                
                if config["quiz_and_judge_strategy"]["enabled"]:
                    graph_gen.quiz()
                    graph_gen.judge_statements()
                else:
                    graph_gen.traverse_strategy.edge_sampling = "random"
                
                graph_gen.traverse()
            
            # 保存GraphML文件
            graphml_path = os.path.join(type_output_dir, f"knowledge_graph_{output_type}.graphml")
            await_graph = graph_gen.graph_storage.get_graph()
            if await_graph:
                import asyncio
                loop = asyncio.get_event_loop()
                graph = loop.run_until_complete(await_graph)
                import networkx as nx
                nx.write_graphml(graph, graphml_path)
                print(f"GraphML已保存到: {graphml_path}")
            
            # 保存配置文件
            config_path = os.path.join(type_output_dir, f"config_{output_type}.yaml")
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ {output_type} 类型语料生成完成")
            print(f"输出目录: {type_output_dir}")
            
        except Exception as e:
            print(f"❌ {output_type} 类型语料生成失败: {str(e)}")
            logger.error(f"Error generating {output_type}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="GraphGen批量处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 处理单个目录下的所有txt文件
  python batch_process.py --input-dir /path/to/txt/files --output-dir results
  
  # 处理指定的txt文件，生成特定类型的语料
  python batch_process.py --input-files file1.txt file2.txt --output-dir results --types atomic aggregated
  
  # 禁用trainee模型（仅使用synthesizer）
  python batch_process.py --input-dir /path/to/txt/files --output-dir results --no-trainee
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', help='包含txt文件的目录路径')
    input_group.add_argument('--input-files', nargs='+', help='指定的txt文件路径列表')
    
    # 输出选项
    parser.add_argument('--output-dir', required=True, help='输出目录路径')
    parser.add_argument('--types', nargs='+', 
                       choices=['atomic', 'aggregated', 'multi_hop', 'cot'],
                       default=['atomic', 'aggregated', 'multi_hop'],
                       help='要生成的语料类型')
    
    # 处理选项
    parser.add_argument('--chunk-size', type=int, default=512, help='文本分块大小')
    parser.add_argument('--batch-size', type=int, default=10, help='每批次处理的文件数量')
    parser.add_argument('--no-trainee', action='store_true', help='禁用trainee模型')
    
    args = parser.parse_args()
    
    # 检查环境变量
    required_vars = ["SYNTHESIZER_MODEL", "SYNTHESIZER_BASE_URL", "SYNTHESIZER_API_KEY"]
    if not args.no_trainee:
        required_vars.extend(["TRAINEE_MODEL", "TRAINEE_BASE_URL", "TRAINEE_API_KEY"])
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ 缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请设置环境变量或使用.env文件")
        return
    
    # 获取txt文件列表
    if args.input_dir:
        txt_files = list(Path(args.input_dir).glob("*.txt"))
        txt_files = [str(f) for f in txt_files]
    else:
        txt_files = args.input_files
    
    if not txt_files:
        print("❌ 未找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    print(f"将生成的语料类型: {', '.join(args.types)}")
    
    # 分批处理
    for i in range(0, len(txt_files), args.batch_size):
        batch_files = txt_files[i:i + args.batch_size]
        print(f"\n处理批次 {i//args.batch_size + 1}/{(len(txt_files)-1)//args.batch_size + 1}")
        print(f"包含文件: {len(batch_files)} 个")
        
        process_single_batch(
            batch_files, 
            args.output_dir, 
            args.types,
            args.chunk_size,
            not args.no_trainee
        )
    
    print(f"\n🎉 批量处理完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()