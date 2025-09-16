# CLAUDE.md

本文件为Claude Code (claude.ai/code) 在处理此代码库时提供指导。

## 项目概述

GraphGen是一个由知识图谱引导的合成数据生成框架。它通过使用知识图谱构建、社区检测和多跳推理来生成针对知识缺口的高质量问答对，从而增强大语言模型的监督微调。

## 常用开发命令

### 环境设置
```bash
# 使用uv创建环境（推荐）
uv venv --python 3.10
uv pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件设置SYNTHESIZER_MODEL, SYNTHESIZER_BASE_URL, SYNTHESIZER_API_KEY等
```

### 推荐执行环境

**首选：WSL Ubuntu环境**
- 更好的UTF-8编码支持，避免中文字符显示问题
- 标准的Linux命令行工具生态
- 更稳定的Python包管理

```bash
# 在WSL中执行命令的标准方式
wsl -d Ubuntu-24.04 -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate GraphGen && cd /mnt/d/git/GraphGen && python script.py"

# 如果已在WSL环境中
cd /mnt/d/git/GraphGen
python script.py
```

**备选：PowerShell环境**
```bash
# 在PowerShell中使用conda环境
powershell -Command "& { conda activate 'D:\git\GraphGen\.conda\GraphGen'; cd 'D:\git\GraphGen'; python script.py }"
```

### 运行和测试
```bash
# Web UI界面
python -m webui.app

# CLI数据生成
python -m graphgen.generate --config_file graphgen/configs/cot_config.yaml --output_dir cache/

# 运行不同类型的生成脚本
bash scripts/generate/generate_cot.sh        # 思维链问答对
bash scripts/generate/generate_atomic.sh     # 基础原子问答对
bash scripts/generate/generate_aggregated.sh # 复杂集成问答对
bash scripts/generate/generate_multi_hop.sh  # 多跳推理问答对
```

### 代码质量
```bash
# 代码格式化
black .
isort .

# 运行测试（如果存在）
python -m pytest tests/ -v  # 标准测试
python test_script.py       # 项目特定测试脚本

# 测试改进的chunking系统
python test_chunking_standalone.py
```

### 环境说明
- **conda环境路径**：`D:\git\GraphGen\.conda\GraphGen`
- **WSL路径映射**：`/mnt/d/git/GraphGen`
- **推荐开发环境**：WSL Ubuntu-24.04 + conda GraphGen环境
- **编码处理**：WSL环境下UTF-8支持更完善，避免中文显示问题

## 架构

## 核心架构和组件

### 主要模块结构

1. **GraphGen主类** (`graphgen/graphgen.py`)
   - 核心生成管道的协调器，处理完整的数据生成流程
   - 支持四种输出格式：原子(atomic)、聚合(aggregated)、多跳(multi-hop)、思维链(cot)
   - 通过YAML配置文件(`graphgen/configs/`)进行参数控制

2. **知识图谱处理** (`graphgen/operators/kg/`)
   - `extract_kg.py`: 从输入文本提取实体和关系构建知识图谱
   - `split_kg.py`, `merge_kg.py`: 知识图谱分割和合并操作
   - 使用NetworkX作为图数据结构的基础

3. **图遍历和生成** (`graphgen/operators/`)
   - `traverse_graph.py`: 知识图谱遍历核心逻辑
   - `quiz.py`: 问答对生成逻辑
   - `generate/generate_cot.py`: 思维链问答生成

4. **模型和存储** (`graphgen/models/`)
   - `llm/openai_model.py`: LLM API客户端，支持synthesizer和trainee模型
   - `storage/`: 数据存储抽象，支持JSON和NetworkX格式
   - `community/community_detector.py`: 使用Leiden算法进行社区检测

5. **Web界面** (`webui/`)
   - 基于Gradio的交互式UI，支持多语言
   - 实时进度跟踪和结果可视化

### 数据生成流程

1. **输入处理**: 原始文本 → 文本分块 → 实体关系提取
2. **知识图谱构建**: 使用NetworkX构建细粒度知识图谱
3. **社区检测**: Leiden算法识别相关实体群组
4. **知识缺口分析**: 通过trainee模型校准误差识别薄弱知识点
5. **QA生成**: 针对缺口生成多样化问答对
6. **格式输出**: 支持Alpaca、ShareGPT、ChatML等格式

### 配置系统

每种生成类型对应独立的YAML配置文件：
- `cot_config.yaml`: 思维链推理配置
- `atomic_config.yaml`: 基础事实问答配置
- `aggregated_config.yaml`: 复杂知识集成配置
- `multi_hop_config.yaml`: 多步推理配置

关键配置参数：
- `input_data_type`: raw/chunked输入模式
- `output_data_format`: 输出格式选择
- `search.enabled`: 是否启用外部搜索后端
- `method_params`: 社区检测和采样参数

### 输出结构

生成数据保存至`cache/data/graphgen/`目录，包含：
- 问答对数据文件(JSON格式)
- 知识图谱结构和统计信息
- 生成过程的元数据和质量指标
- 溯源信息用于事实验证

## 开发注意事项

### 代码约定
- 使用black和isort进行代码格式化，配置见pyproject.toml
- 模块化设计：operators处理核心算法，models提供数据抽象
- 异步处理：大部分生成操作使用asyncio提高效率
- 配置驱动：所有生成参数通过YAML文件控制

### 关键入口点
- **CLI**: `python -m graphgen.generate`
- **Web UI**: `python -m webui.app`
- **库调用**: 从`graphgen.graphgen`导入`GraphGen`类

### 常见开发任务
- 修改生成逻辑：主要在`graphgen/operators/`目录
- 调整模型接口：编辑`graphgen/models/llm/`相关文件
- 更新UI：修改`webui/`目录下的组件
- 配置调优：编辑`graphgen/configs/`下的YAML文件
- 添加新的搜索后端：扩展`graphgen/models/search/`模块
