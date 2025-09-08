# GraphGen 配置文件目录

本目录包含 GraphGen 框架的各种配置文件，用于不同类型的数据生成任务。

## 🔧 配置文件列表

### 1. **atomic_config.yaml** - 原子性问答对生成
- **用途**: 生成基于单个实体或关系的简单直接问答对
- **特点**: 问题聚焦单一知识点，答案直接明确
- **适用场景**: 基础知识理解训练、事实性问答
- **输出格式**: Alpaca (指令微调格式)

### 2. **aggregated_config.yaml** - 聚合型问答对生成
- **用途**: 生成基于边连接的实体组合的聚合型问答对
- **特点**: 整合多个相关实体的信息，形成综合性回答
- **适用场景**: 复合信息理解、关系推理训练
- **输出格式**: ChatML (聊天标记语言)

### 3. **multi_hop_config.yaml** - 多跳推理问答对生成
- **用途**: 生成需要跨多个实体进行推理的复杂问答对
- **特点**: 需要多步推理，考验模型的逻辑推理能力
- **适用场景**: 复杂推理训练、逻辑思维能力提升
- **输出格式**: ChatML (聊天标记语言)

### 4. **cot_config.yaml** - 思维链(CoT)推理数据生成
- **用途**: 生成具有复杂推理过程的思维链数据
- **特点**: 使用Leiden算法进行社区检测，生成基于社区的推理链条
- **适用场景**: 复杂推理训练、思维过程展示
- **输出格式**: ShareGPT (对话格式)

## 📋 配置文件结构说明

每个配置文件都包含以下主要部分：

### 输入配置
- `input_data_type`: 输入数据类型 (raw/chunked)
- `input_file`: 输入文件路径

### 输出配置
- `output_data_type`: 输出数据类型 (atomic/aggregated/multi_hop/cot)
- `output_data_format`: 输出格式 (Alpaca/ShareGPT/ChatML)

### 核心配置
- `tokenizer`: 分词器设置
- `search`: 外部搜索配置
- `quiz_and_judge_strategy`: 知识缺口识别策略
- `traverse_strategy`: 图遍历策略 (除CoT外)
- `method_params`: 方法参数 (仅CoT)

## 🚀 使用方法

```bash
# 使用原子性配置生成数据
python -m graphgen.generate --config_file graphgen/configs/atomic_config.yaml --output_dir cache

# 使用多跳推理配置生成数据
python -m graphgen.generate --config_file graphgen/configs/multi_hop_config.yaml --output_dir results

# 使用CoT配置生成思维链数据
python -m graphgen.generate --config_file graphgen/configs/cot_config.yaml --output_dir cot_data
```

## ⚙️ 自定义配置

你可以根据需要修改这些配置文件，或创建新的配置文件：

1. **复制现有配置**: 选择最接近需求的配置文件作为模板
2. **调整参数**: 根据数据特点和训练目标调整相关参数
3. **验证配置**: 使用小数据集测试配置的有效性
4. **批量生成**: 使用验证过的配置进行大规模数据生成

## 🔍 参数调优建议

- **max_depth**: 控制推理深度，越大生成的问答对越复杂
- **max_extra_edges**: 控制每个方向的边数，影响信息丰富度
- **quiz_samples**: 控制知识缺口检测的精度
- **edge_sampling**: 选择合适的采样策略以平衡质量和多样性
