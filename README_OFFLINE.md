# GraphGen 离线部署指南

本指南介绍如何在内网环境中部署GraphGen，解决tokenizer下载和网络依赖问题。

## 🎯 解决的问题

- ✅ **Tokenizer下载失败**：预下载GPT-2、tiktoken等tokenizer到镜像中
- ✅ **NLTK数据缺失**：预下载punkt、stopwords等NLTK数据  
- ✅ **网络依赖**：支持完全离线运行，无需外网访问
- ✅ **模型兼容性**：智能fallback机制，确保tokenizer可用

## 🚀 快速开始

### 1. 配置环境变量

```bash
# 复制环境模板
cp .env.template .env

# 编辑 .env 文件，填入您的API配置
# Windows: notepad .env
# Linux/macOS: vim .env
```

关键配置项：
```env
SYNTHESIZER_API_KEY=sk-your-api-key-here
SYNTHESIZER_BASE_URL=https://api.siliconflow.cn/v1
SYNTHESIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
GRAPHGEN_OFFLINE_MODE=true
```

### 2. 构建并启动服务

```bash
# 构建离线镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 3. 访问应用

打开浏览器访问：http://localhost:7860

## 🔧 详细配置

### 离线模式环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `GRAPHGEN_OFFLINE_MODE` | 启用完全离线模式 | `true` |
| `HF_DATASETS_OFFLINE` | Hugging Face数据集离线模式 | `1` |
| `TRANSFORMERS_OFFLINE` | Transformers模型离线模式 | `1` |
| `TIKTOKEN_CACHE_DIR` | tiktoken缓存目录 | `/app/cache/tiktoken` |
| `HF_HOME` | Hugging Face缓存根目录 | `/app/cache/huggingface` |
| `NLTK_DATA` | NLTK数据目录 | `/app/resources/nltk_data` |

### 内网API服务配置示例

#### 本地Ollama服务
```env
SYNTHESIZER_BASE_URL=http://host.docker.internal:11434/v1
SYNTHESIZER_MODEL=qwen2.5:7b-instruct
SYNTHESIZER_API_KEY=ollama
```

#### 内网vLLM服务
```env
SYNTHESIZER_BASE_URL=http://your-vllm-server:8000/v1
SYNTHESIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
SYNTHESIZER_API_KEY=dummy
```

#### 内网API网关
```env
SYNTHESIZER_BASE_URL=http://your-internal-api.company.com/v1
SYNTHESIZER_MODEL=your-model-name
SYNTHESIZER_API_KEY=your-internal-api-key
```

## 📦 预下载的组件

离线镜像包含以下预下载组件：

### Tokenizer模型
- `gpt2` - GPT-2 tokenizer (cl100k_base fallback)
- `OpenAssistant/reward-model-deberta-v3-large-v2` - 奖励模型tokenizer
- `MingZhong/unieval-sum` - 统一评估tokenizer
- `BAAI/IndustryCorpus2_DataRater` - 数据评分tokenizer

### Tiktoken编码
- `cl100k_base` - GPT-4兼容编码
- `p50k_base` - GPT-3.5兼容编码
- `r50k_base` - GPT-3兼容编码

### NLTK数据
- `punkt` - 句子分割
- `punkt_tab` - 新版句子分割
- `stopwords` - 停用词列表

## 🔧 管理命令

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看日志
docker-compose logs -f

# 重新构建并启动
docker-compose up -d --build

# 重启服务
docker-compose restart

# 查看服务状态
docker-compose ps
```

## 🔍 故障排除

### 1. 容器启动失败

```bash
# 查看容器日志
docker logs graphgen-offline

# 检查容器状态
docker ps -a
```

### 2. API连接问题

检查环境变量配置：
```bash
# 进入容器检查环境
docker exec -it graphgen-offline env | grep -E "(API_KEY|BASE_URL)"
```

### 3. Tokenizer加载失败

```bash
# 验证离线缓存
docker exec -it graphgen-offline python /app/validate_offline_cache.py
```

### 4. 网络连接测试

```bash
# 测试API连接
docker exec -it graphgen-offline curl -s $SYNTHESIZER_BASE_URL/models
```

## 📋 系统要求

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **内存**: 建议4GB+
- **存储**: 建议10GB+可用空间
- **网络**: 构建时需要外网（运行时可完全内网）

## 🔒 安全注意事项

1. **API密钥安全**：不要在代码中硬编码API密钥
2. **网络隔离**：容器默认只暴露7860端口
3. **文件权限**：容器使用非root用户运行
4. **日志安全**：避免在日志中暴露敏感信息

## 💡 性能优化

### 内存优化
```yaml
# docker-compose.offline.yml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

### 存储优化
```bash
# 定期清理未使用的Docker对象
docker system prune -f

# 清理旧镜像
docker image prune -f
```

## 📞 技术支持

如果遇到问题，请：

1. 查看容器日志：`docker logs graphgen-offline`
2. 检查环境配置：确认API密钥和Base URL正确
3. 验证网络连接：测试API服务可访问性
4. 提交Issue：包含完整的错误日志和环境信息

## 🔄 更新指南

更新离线版本：

```bash
# 1. 拉取最新代码
git pull

# 2. 重新构建并启动服务
docker-compose up -d --build
```