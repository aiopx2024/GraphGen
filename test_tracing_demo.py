#!/usr/bin/env python3
"""
GraphGen 溯源功能演示
展示如何使用增强的溯源机制追踪QA到原文的完整映射
"""

import asyncio
import os
import glob
from graphgen_tracer import GraphGenTracer

async def demo_tracing():
    """演示溯源功能"""
    print("🔍 GraphGen 溯源功能演示")
    print("=" * 50)
    
    # 初始化溯源器
    tracer = GraphGenTracer("/mnt/d/git/GraphGen/cache")
    
    # 查找最新的QA文件
    qa_files = glob.glob("/mnt/d/git/GraphGen/cache/data/graphgen/*/qa-*.json")
    if not qa_files:
        print("❌ 未找到QA文件，请先生成一些问答对")
        return
    
    latest_file = max(qa_files, key=os.path.getctime)
    print(f"📂 找到最新QA文件: {latest_file}")
    
    # 生成溯源报告
    print("📊 正在生成溯源报告...")
    report_path = await tracer.generate_tracing_report(latest_file)
    
    # 输出结果
    print(f"✅ 溯源报告已生成: {report_path}")
    print("\n🎯 溯源报告包含以下信息:")
    print("   • 每个QA对的源文档映射")
    print("   • 原始文本块内容")
    print("   • 相关实体和关系")
    print("   • 置信度评分")
    print("   • 子图深度信息")

if __name__ == "__main__":
    asyncio.run(demo_tracing())