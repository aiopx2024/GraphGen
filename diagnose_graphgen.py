#!/usr/bin/env python3
"""
GraphGen诊断工具
用于检查GraphGen运行状态和输出文件
"""

import os
import json
import glob
from pathlib import Path

def check_cache_directory():
    """检查cache目录中的文件"""
    cache_dir = "d:/git/GraphGen/cache"
    
    print("🔍 检查cache目录...")
    print(f"Cache目录: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print("❌ cache目录不存在")
        return
    
    # 检查数据目录
    data_dir = os.path.join(cache_dir, "data", "graphgen")
    if os.path.exists(data_dir):
        print(f"✅ 找到数据目录: {data_dir}")
        
        # 查找所有子目录
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if subdirs:
            print(f"📁 找到 {len(subdirs)} 个数据子目录:")
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(data_dir, subdir)
                print(f"  📂 {subdir}")
                
                # 检查QA文件
                qa_files = glob.glob(os.path.join(subdir_path, "qa-*.json"))
                if qa_files:
                    for qa_file in qa_files:
                        file_size = os.path.getsize(qa_file)
                        print(f"    📄 {os.path.basename(qa_file)} ({file_size} bytes)")
                        
                        # 检查文件内容
                        if file_size > 0:
                            try:
                                with open(qa_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                if isinstance(data, list):
                                    print(f"      ✅ 包含 {len(data)} 条QA对")
                                    if data:
                                        print(f"      📝 示例: {data[0].get('question', 'N/A')[:50]}...")
                                else:
                                    print(f"      📊 数据类型: {type(data)}")
                            except Exception as e:
                                print(f"      ❌ 读取错误: {e}")
                        else:
                            print("      ⚠️  文件为空")
                else:
                    print("    ❌ 未找到QA文件")
        else:
            print("❌ 数据目录为空")
    else:
        print("❌ 未找到数据目录")
    
    # 检查日志文件
    log_dir = os.path.join(cache_dir, "logs")
    if os.path.exists(log_dir):
        print(f"\n📋 检查日志目录: {log_dir}")
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"📄 最新日志: {os.path.basename(latest_log)}")
            
            # 读取最后几行日志
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print("📜 最后几行日志:")
                        for line in lines[-5:]:
                            print(f"    {line.strip()}")
                    else:
                        print("⚠️  日志文件为空")
            except Exception as e:
                print(f"❌ 读取日志错误: {e}")
        else:
            print("❌ 未找到日志文件")
    
    # 检查graph文件
    graph_files = glob.glob(os.path.join(cache_dir, "*.graphml"))
    if graph_files:
        print(f"\n🔗 找到GraphML文件:")
        for graph_file in graph_files:
            file_size = os.path.getsize(graph_file)
            print(f"  📄 {os.path.basename(graph_file)} ({file_size} bytes)")
    else:
        print("\n❌ 未找到GraphML文件")


def check_temp_files():
    """检查临时文件"""
    import tempfile
    temp_dir = tempfile.gettempdir()
    print(f"\n🔄 检查临时目录: {temp_dir}")
    
    # 查找GraphGen相关的临时文件
    temp_files = glob.glob(os.path.join(temp_dir, "tmp*.jsonl"))
    if temp_files:
        print(f"📄 找到 {len(temp_files)} 个临时jsonl文件:")
        for temp_file in sorted(temp_files, key=os.path.getctime, reverse=True)[:5]:
            file_size = os.path.getsize(temp_file)
            print(f"  📄 {os.path.basename(temp_file)} ({file_size} bytes)")
            
            if file_size > 0:
                try:
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"    ✅ 包含 {len(data)} 条记录")
                    else:
                        print(f"    📊 数据类型: {type(data)}")
                except Exception as e:
                    print(f"    ❌ 读取错误: {e}")
    else:
        print("❌ 未找到临时jsonl文件")


def main():
    print("🚀 GraphGen 诊断工具")
    print("=" * 50)
    
    check_cache_directory()
    check_temp_files()
    
    print(f"\n{'='*50}")
    print("💡 诊断建议:")
    print("1. 如果QA文件为空，可能是遍历策略问题")
    print("2. 如果没有输出文件，检查API连接和权限")
    print("3. 查看最新日志了解详细错误信息")
    print("4. 确认输入数据格式正确")

if __name__ == "__main__":
    main()