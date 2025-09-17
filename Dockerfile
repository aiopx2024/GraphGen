# GraphGen 开发/调试版 Dockerfile
# 包含完整调试工具，优化分层构建

FROM python:3.10-slim

# ===== Layer 1: 环境变量（几乎不变） =====
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ===== Layer 2: 工作目录（几乎不变） =====
WORKDIR /app

# ===== Layer 3: 系统依赖 - 基础工具（很少变化） =====
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # 基础工具
    curl \
    wget \
    ca-certificates \
    # 编辑器
    vim \
    nano \
    less \
    # 进程管理
    procps \
    lsof \
    # 文本处理
    jq \
    # 开发工具
    git \
    # 用户管理（为后续sudo准备）
    sudo \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # 配置vim
    echo "set number\nset expandtab\nset tabstop=4\nset shiftwidth=4\nset encoding=utf-8" > /etc/vim/vimrc.local

# ===== Layer 4: 系统依赖 - 扩展工具（偶尔变化） =====
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # 进程监控
    htop \
    # 网络工具
    iputils-ping \
    net-tools \
    telnet \
    dnsutils \
    # 系统调试
    strace \
    tree \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ===== Layer 5: Python依赖（偶尔变化） =====
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # 开发工具
    pip install ipython ipdb

# ===== Layer 6: NLTK数据（很少变化） =====
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

# ===== Layer 7: 用户和权限（很少变化） =====
RUN mkdir -p /app/cache /app/logs && \
    useradd -m -u 1000 graphgen && \
    # 给开发用户sudo权限
    echo "graphgen ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R graphgen:graphgen /app/cache /app/logs

# ===== Layer 8: 环境配置（很少变化） =====
ENV NLTK_DATA=/app/resources/nltk_data
ENV SYNTHESIZER_MODEL="Qwen/Qwen2.5-7B-Instruct"
ENV SYNTHESIZER_BASE_URL="https://api.siliconflow.cn/v1"
ENV SYNTHESIZER_API_KEY=""
ENV TRAINEE_MODEL="Qwen/Qwen2.5-7B-Instruct"
ENV TRAINEE_BASE_URL="https://api.siliconflow.cn/v1"
ENV TRAINEE_API_KEY=""
ENV RPM="1000"
ENV TPM="50000"

# ===== Layer 9: 应用代码（经常变化，放在最后！） =====
COPY --chown=graphgen:graphgen . .

# ===== Layer 10: 用户配置（很少变化） =====
USER graphgen

# 设置bash别名，方便使用
RUN echo "alias ll='ls -la'" >> ~/.bashrc && \
    echo "alias ..='cd ..'" >> ~/.bashrc && \
    echo "alias py='python'" >> ~/.bashrc && \
    echo "alias ipy='ipython'" >> ~/.bashrc && \
    echo "alias gs='git status'" >> ~/.bashrc && \
    echo "alias gd='git diff'" >> ~/.bashrc && \
    echo "set -o vi" >> ~/.bashrc && \
    echo "PS1='\[\033[01;32m\]graphgen@dev\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]$ '" >> ~/.bashrc

# ===== Layer 11: 运行配置（很少变化） =====
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 启动命令（可以被覆盖进入bash）
CMD ["python", "-m", "webui.app"]
