"""
GraphGen命令行数据生成工具

这是GraphGen框架的主要命令行入口，提供完整的数据生成流程：
1. 文档预处理和知识图谱构建
2. 可选的外部搜索增强
3. 知识缺口识别（Quiz和Judge）
4. 多种策略的问答对生成

支持的输出类型：
- atomic: 原子性问答对，基于单个实体或关系
- aggregated: 聚合型问答对，基于边连接的实体组合
- multi_hop: 多跳推理问答对，需要跨多个实体推理
- cot: 思维链（Chain-of-Thought）推理数据

用法示例：
    python -m graphgen.generate --config_file configs/atomic_config.yaml --output_dir cache
    python -m graphgen.generate --config_file configs/cot_config.yaml --output_dir output
"""
import argparse
import os
import time
from importlib.resources import files

import yaml
from dotenv import load_dotenv

from .graphgen import GraphGen
from .utils import logger, set_logger

# 获取当前模块的绝对路径，用作默认输出目录
sys_path = os.path.abspath(os.path.dirname(__file__))

# 加载环境变量（API Key等敏感信息）
# 这是GraphGen项目的标准做法，确保跨平台兼容性
load_dotenv()


def set_working_dir(folder):
    """
    创建工作目录及必要的子目录结构
    
    GraphGen需要一个标准化的目录结构来存储：
    - data/graphgen/: 生成的问答对数据
    - logs/: 日志文件，用于调试和监控
    
    Args:
        folder: 工作目录路径
    """
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "data", "graphgen"), exist_ok=True)
    os.makedirs(os.path.join(folder, "logs"), exist_ok=True)


def save_config(config_path, global_config):
    """
    保存配置文件到输出目录
    
    在数据生成完成后，保存使用的配置参数，便于：
    1. 重现实验结果
    2. 调试和优化参数
    3. 版本控制和审计
    
    Args:
        config_path: 配置文件保存路径
        global_config: 配置字典，包含所有生成参数
    """
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.dump(
            global_config, config_file, default_flow_style=False, allow_unicode=True
        )


def main():
    """
    GraphGen命令行工具的主入口函数
    
    执行完整的数据生成流程：
    1. 解析命令行参数
    2. 加载配置文件
    3. 初始化GraphGen实例
    4. 执行数据处理流程
    5. 保存结果和配置
    
    支持两种主要流程：
    - 常规QA生成： insert -> search -> quiz -> judge -> traverse
    - CoT生成： insert -> search -> generate_reasoning
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="GraphGen: 基于知识图谱引导的合成数据生成框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 生成原子性问答对
  python -m graphgen.generate --config_file configs/atomic_config.yaml --output_dir cache
  
  # 生成多跳推理数据
  python -m graphgen.generate --config_file configs/multi_hop_config.yaml --output_dir output
  
  # 生成思维链数据
  python -m graphgen.generate --config_file configs/cot_config.yaml --output_dir results
        """
    )
    parser.add_argument(
        "--config_file",
        help="GraphGen配置文件路径，包含生成策略和参数设置",
        default=files("graphgen").joinpath("configs", "aggregated_config.yaml"),
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="输出目录，用于存储生成的数据、日志和配置文件",
        default=sys_path,
        required=True,
        type=str,
    )

    args = parser.parse_args()

    # 设置工作目录并创建必要的子目录
    working_dir = args.output_dir
    set_working_dir(working_dir)

    # 加载配置文件
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 获取输出数据类型和生成唯一ID
    output_data_type = config["output_data_type"]
    unique_id = int(time.time())  # 使用时间戳作为唯一标识符
    
    # 设置日志系统，同时输出到文件和控制台
    set_logger(
        os.path.join(
            working_dir, "logs", f"graphgen_{output_data_type}_{unique_id}.log"
        ),
        if_stream=True,  # 启用控制台输出
    )
    logger.info(
        "GraphGen with unique ID %s logging to %s",
        unique_id,
        os.path.join(
            working_dir, "logs", f"graphgen_{output_data_type}_{unique_id}.log"
        ),
    )

    # 初始化GraphGen实例
    # unique_id用于区分不同的生成任务，避免数据冲突
    graph_gen = GraphGen(working_dir=working_dir, unique_id=unique_id, config=config)

    # 步骤1: 数据插入 - 构建知识图谱
    # 从配置文件指定的输入文件中抽取实体和关系
    graph_gen.insert()

    # 步骤2（可选）: 外部搜索增强
    # 使用Google、Bing、Wikipedia等搜索引擎补充知识图谱信息
    if config["search"]["enabled"]:
        graph_gen.search()

    # 根据输出数据类型选择不同的生成策略
    if output_data_type in ["atomic", "aggregated", "multi_hop"]:
        # 传统问答对生成流程
        
        # 步骤3（可选）: 知识缺口识别
        if "quiz_and_judge_strategy" in config and config[
            "quiz_and_judge_strategy"
        ].get("enabled", False):
            # Quiz: 生成测试问题，评估训练模型对知识点的掌握程度
            graph_gen.quiz()
            # Judge: 评判知识声明的可信度，优先生成低置信度的数据
            graph_gen.judge()
        else:
            logger.warning(
                "Quiz and Judge strategy is disabled. Edge sampling falls back to random."
            )
            # 如果未启用知识缺口识别，则使用随机采样
            graph_gen.traverse_strategy.edge_sampling = "random"
        
        # 步骤4: 图遍历生成问答对
        # 根据配置的策略遍历知识图谱，生成相应类型的问答对
        graph_gen.traverse()
        
    elif output_data_type == "cot":
        # CoT（思维链）生成流程
        # 使用Leiden算法进行社区检测，生成复杂推理过程
        graph_gen.generate_reasoning(method_params=config["method_params"])
        
    else:
        raise ValueError(f"Unsupported output data type: {output_data_type}")

    # 步骤5: 保存结果和配置
    # 生成的数据保存在 working_dir/data/graphgen/unique_id/ 目录下
    output_path = os.path.join(working_dir, "data", "graphgen", str(unique_id))
    
    # 保存使用的配置文件，便于重现实验
    save_config(os.path.join(output_path, f"config-{unique_id}.yaml"), config)
    
    logger.info("GraphGen completed successfully. Data saved to %s", output_path)
    logger.info(
        "Generated files:"
        "\n  - QA pairs: %s/qa-%s.json"
        "\n  - Configuration: %s/config-%s.yaml"
        "\n  - Log file: %s/logs/graphgen_%s_%s.log",
        output_path, unique_id,
        output_path, unique_id, 
        working_dir, output_data_type, unique_id
    )


if __name__ == "__main__":
    main()
