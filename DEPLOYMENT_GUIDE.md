# GraphGen 内网离线部署完整指南

## 概述

本指南提供了在实验室内网环境中部署GraphGen的完整解决方案，包括Docker镜像构建、传输和部署等所有步骤。

## 前置要求

### 外网环境（构建阶段）
- Docker Desktop 或 Docker Engine
- Git
- 网络连接（用于下载依赖）

### 内网环境（部署阶段）
- Docker Engine
- 至少4GB内存
- 足够的存储空间（建议10GB+）
- 内网LLM API服务（如本地部署的模型服务）

## 构建阶段（外网环境）

### 1. 准备代码
```bash
git clone <your-repo>
cd GraphGen
```

### 2. 构建离线镜像

#### Windows环境：
```powershell
# 设置执行策略（如需要）
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 运行构建脚本
.\build-offline-image.ps1
```

#### Linux/Mac环境：
```bash
# 给脚本执行权限
chmod +x build-offline-image.sh

# 运行构建脚本
./build-offline-image.sh
```

### 3. 验证构建结果
构建完成后应生成：
- `graphgen-offline-image.tar` - Docker镜像文件
- `deployment-guide.md` - 部署指南
- `docker-compose.yml` - Docker Compose配置

## 传输阶段

将以下文件传输到内网服务器：
```
graphgen-offline-image.tar      # Docker镜像（主要文件）
docker-compose.yml             # Docker Compose配置
deployment-guide.md            # 部署指南
.env.example                   # 环境变量模板
```

## 部署阶段（内网环境）

### 1. 加载Docker镜像
```bash
# 加载镜像
docker load -i graphgen-offline-image.tar

# 验证镜像
docker images graphgen:offline-latest
```

### 2. 配置环境变量
创建 `.env` 文件：
```env
# LLM API配置（关键：使用内网地址）
SYNTHESIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
SYNTHESIZER_BASE_URL=http://your-internal-llm-api:8000/v1
SYNTHESIZER_API_KEY=your_internal_api_key

TRAINEE_MODEL=Qwen/Qwen2.5-7B-Instruct
TRAINEE_BASE_URL=http://your-internal-llm-api:8000/v1
TRAINEE_API_KEY=your_internal_api_key

# API调用限制
RPM=1000
TPM=50000
```

### 3. 创建数据目录
```bash
mkdir -p cache logs data
chmod 755 cache logs data
```

### 4. 启动服务

#### 方式1：使用Docker Compose（推荐）
```bash
docker-compose up -d
```

#### 方式2：直接使用Docker
```bash
docker run -d \
  --name graphgen-app \
  --env-file .env \
  -p 7860:7860 \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  graphgen:offline-latest
```

### 5. 验证部署
```bash
# 检查容器状态
docker ps

# 查看日志
docker logs graphgen-app

# 健康检查
curl http://localhost:7860/
```

### 6. 访问应用
打开浏览器访问：`http://服务器IP:7860`

## 常见问题排查

### 1. 容器启动失败
```bash
# 查看详细日志
docker logs --details graphgen-app

# 检查环境变量
docker exec graphgen-app env | grep -E "(SYNTHESIZER|TRAINEE)"
```

### 2. API连接问题
- 检查内网LLM服务是否正常运行
- 验证API地址和端口可达性
- 确认API Key有效性

### 3. 权限问题
```bash
# 修复目录权限
sudo chown -R 1000:1000 cache logs data
```

### 4. 内存不足
- 检查系统内存使用情况
- 考虑调整Docker内存限制

## 性能优化建议

### 1. 资源配置
```yaml
# docker-compose.yml 中添加资源限制
services:
  graphgen:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### 2. 数据持久化
确保重要数据挂载到宿主机：
- `/app/cache` - 缓存数据
- `/app/logs` - 日志文件
- `/app/data` - 生成的数据

### 3. 网络优化
- 使用内网高速网络连接
- 考虑配置负载均衡（如有多个实例）

## 安全考虑

1. **API Key管理**：使用内网专用的API Key
2. **网络隔离**：确保容器只能访问必要的内网服务
3. **日志管理**：定期清理和轮转日志文件
4. **更新策略**：建立安全的镜像更新流程

## 监控和维护

### 1. 日志监控
```bash
# 实时查看日志
docker logs -f graphgen-app

# 定期清理日志
docker logs graphgen-app --since 7d > recent.log
```

### 2. 健康检查
```bash
# 定期健康检查
curl -f http://localhost:7860/ || echo "Service down"
```

### 3. 备份策略
```bash
# 备份重要数据
tar -czf graphgen-backup-$(date +%Y%m%d).tar.gz cache data logs
```

## 故障恢复

### 1. 容器重启
```bash
docker restart graphgen-app
```

### 2. 镜像重新部署
```bash
docker stop graphgen-app
docker rm graphgen-app
docker-compose up -d
```

### 3. 数据恢复
```bash
# 从备份恢复
tar -xzf graphgen-backup-YYYYMMDD.tar.gz
```

---

## 技术支持

如遇到问题，请检查：
1. Docker和容器日志
2. 网络连通性
3. 资源使用情况
4. 环境变量配置

建议在部署前在测试环境中验证完整流程。