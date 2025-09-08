# CLAUDE.md

本文件为Claude Code (claude.ai/code) 在处理此代码库时提供指导。

## 项目概述

GraphGen是一个由知识图谱引导的合成数据生成框架。它通过使用知识图谱构建、社区检测和多跳推理来生成针对知识缺口的高质量问答对，从而增强大语言模型的监督微调。

## 开发环境配置命令

### 环境设置
```bash
# 克隆和设置
git clone --depth=1 https://github.com/open-sciencelab/GraphGen
cd GraphGen

# 使用uv创建环境（推荐）
uv venv --python 3.10
uv pip install -r requirements.txt

# 替代方案：标准pip
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 配置
```bash
# 复制并配置环境变量
cp .env.example .env
# 编辑.env文件设置：
# - SYNTHESIZER_MODEL, SYNTHESIZER_BASE_URL, SYNTHESIZER_API_KEY
# - TRAINEE_MODEL, TRAINEE_BASE_URL, TRAINEE_API_KEY
```

### 运行应用程序

#### Web UI界面
```bash
python -m webui.app
```

#### CLI数据生成
```bash
# 从源码运行
python -m graphgen.generate --config_file graphgen/configs/cot_config.yaml --output_dir cache/

# 从PyPI安装运行
uv pip install graphgen
graphg --output_dir cache
```

#### 生成脚本
```bash
# 不同数据格式
bash scripts/generate/generate_cot.sh        # 思维链问答对
bash scripts/generate/generate_atomic.sh     # 基础原子问答对
bash scripts/generate/generate_aggregated.sh # 复杂集成问答对
bash scripts/generate/generate_multihop.sh   # 多跳推理问答对
```

#### Docker
```bash
docker build -t graphgen .
docker run -p 7860:7860 graphgen
```

### 开发工具
```bash
# 代码格式化（在pyproject.toml中配置）
black .
isort .

# 预提交钩子
pre-commit install
pre-commit run --all-files
```

## 架构

### 核心组件

1. **知识图谱构建** (`graphgen/operators/`)
   - 从输入文本中提取实体和关系
   - 使用NetworkX构建细粒度知识图谱
   - 使用Leiden算法进行社区检测

2. **数据生成管道** (`graphgen/graphgen.py`)
   - [GraphGen](file://d:\git\GraphGen\graphgen\graphgen.py#L41-L394)类协调整个管道
   - 支持多种输出格式：原子、聚合、多跳、思维链
   - 通过`graphgen/configs/`中的YAML文件进行配置

3. **模型接口** (`graphgen/models/`)
   - [OpenAIModel](file://d:\git\GraphGen\graphgen\models\llm\openai_model.py#L43-L154)：用于合成和训练的LLM客户端
   - [Tokenizer](file://d:\git\GraphGen\graphgen\models\llm\tokenizer.py#L30-L72)：令牌计数和文本处理
   - 知识图谱和生成数据的存储抽象

4. **Web界面** (`webui/`)
   - 基于Gradio的交互式数据生成UI
   - 通过translation.json支持多语言
   - 实时进度跟踪和可视化

### 数据流

1. **输入处理**：原始文本 → 分块 → 知识图谱
2. **缺口分析**：使用训练模型通过校准误差识别知识缺口
3. **社区检测**：使用Leiden算法对相关实体进行分组
4. **问答生成**：生成针对已识别缺口的多样化问答对
5. **输出格式化**：支持Alpaca、ShareGPT、ChatML格式

### 配置

每种生成类型在`graphgen/configs/`中都有自己的YAML配置：
- [cot_config.yaml](file://d:\git\GraphGen\graphgen\configs\cot_config.yaml)：思维链推理
- [atomic_config.yaml](file://d:\git\GraphGen\graphgen\configs\atomic_config.yaml)：基础事实问答
- [aggregated_config.yaml](file://d:\git\GraphGen\graphgen\configs\aggregated_config.yaml)：复杂集成知识
- [multi_hop_config.yaml](file://d:\git\GraphGen\graphgen\configs\multi_hop_config.yaml)：多步推理

关键配置参数：
- `input_data_type`：原始或分块输入
- `output_data_format`：Alpaca、Sharegpt、ChatML
- `search.enabled`：启用网络搜索后端（Google、Bing、Wikipedia、UniProt）
- [method_params](file://d:\git\GraphGen\graphgen\models\community\community_detector.py#L13-L13)：社区检测和采样参数

### 入口点

- **CLI**：`graphgen.generate:main`（安装为[graphgen](file://d:\git\GraphGen\graphgen\graphgen.py#L0-L395)命令）
- **Web UI**：`webui.app:main`
- **库**：从[graphgen.graphgen](file://d:\git\GraphGen\graphgen\graphgen.py#L0-L395)导入[GraphGen](file://d:\git\GraphGen\graphgen\graphgen.py#L41-L394)类

### 输出结构

生成的数据保存到`cache/data/graphgen/`，包含以下元数据：
- 令牌计数和生成统计信息
- 知识图谱结构和社区
- 问答对质量指标和校准分数
