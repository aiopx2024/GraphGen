#!/bin/bash
# 检查Docker构建上下文大小的脚本

echo "========== Docker构建上下文分析 =========="
echo

echo "1. 当前目录总大小:"
du -sh . 2>/dev/null || echo "无法计算总大小"

echo
echo "2. .dockerignore排除的主要目录大小:"

# 检查被排除的目录
excluded_dirs=(".conda" ".venv" ".git" "cache" "__pycache__" "GraphGen-Simple-Deploy")

for dir in "${excluded_dirs[@]}"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "   $dir: $size"
    else
        echo "   $dir: 不存在"
    fi
done

echo
echo "3. 被排除的大文件:"
find . -name "*.tar" -o -name "*.tar.gz" -o -name "*.log" | head -10 | while read file; do
    if [ -f "$file" ]; then
        size=$(du -sh "$file" 2>/dev/null | cut -f1)
        echo "   $file: $size"
    fi
done

echo
echo "4. 预估构建上下文优化效果:"
echo "   优化前: 包含所有文件(包括虚拟环境、缓存等)"
echo "   优化后: 仅包含必要的源代码文件"
echo "   预计减少: 数百MB到数GB(取决于虚拟环境大小)"

echo
echo "5. .dockerignore配置验证:"
if [ -f ".dockerignore" ]; then
    echo "   ✅ .dockerignore文件已存在"
    echo "   排除规则数量: $(grep -v '^#' .dockerignore | grep -v '^$' | wc -l)"
else
    echo "   ❌ .dockerignore文件不存在"
fi

echo
echo "========== 分析完成 =========="