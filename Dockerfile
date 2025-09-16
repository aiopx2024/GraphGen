# GraphGen 内网部署 Dockerfile
# 简化版本，无需下载复杂的tokenizer模型

FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 安装系统依赖（很少变化，可以充分利用缓存）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 先复制requirements文件（依赖变化时才重新安装）
COPY requirements.txt .

# 安装Python依赖（requirements.txt不变时使用缓存）
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 创建基本目录结构
RUN mkdir -p /app/cache /app/logs

# 预下载NLTK数据（保留这个，因为比较简单且有用）
RUN python -c "\
import os; \
import nltk; \
os.environ['NLTK_DATA'] = '/app/resources/nltk_data'; \
os.makedirs('/app/resources/nltk_data', exist_ok=True); \
nltk.download('punkt', download_dir='/app/resources/nltk_data', quiet=True); \
nltk.download('punkt_tab', download_dir='/app/resources/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/app/resources/nltk_data', quiet=True); \
print('✅ NLTK数据下载完成')\
"

# 设置NLTK数据路径
ENV NLTK_DATA=/app/resources/nltk_data

# 创建非-root用户
RUN useradd -m -u 1000 graphgen

# === 以下是应用代码相关，代码变化时才重新执行 ===

# 复制应用代码（放在最后，代码变化时不影响上面的缓存）
COPY --chown=graphgen:graphgen . .

# 设置目录权限
RUN chown -R graphgen:graphgen /app/cache /app/logs

# 默认环境变量
ENV SYNTHESIZER_MODEL="Qwen/Qwen2.5-7B-Instruct"
ENV SYNTHESIZER_BASE_URL="https://api.siliconflow.cn/v1"
ENV SYNTHESIZER_API_KEY=""
ENV TRAINEE_MODEL="Qwen/Qwen2.5-7B-Instruct"
ENV TRAINEE_BASE_URL="https://api.siliconflow.cn/v1"
ENV TRAINEE_API_KEY=""
ENV RPM="1000"
ENV TPM="50000"

# 暴露端口
EXPOSE 7860

# 切换到非root用户
USER graphgen

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 启动命令
CMD ["python", "-m", "webui.app"]