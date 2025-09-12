#!/usr/bin/env python3
"""
简化版溯源工具 - 针对现有QA格式进行分析
"""

import json
import glob
import os
from datetime import datetime

def analyze_qa_file(qa_file_path: str):
    """分析QA文件的溯源信息"""
    
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"📂 分析文件: {qa_file_path}")
    print(f"📊 QA数据格式: {'ChatML列表' if isinstance(qa_data, list) else '字典'}")
    print(f"📝 总QA对数量: {len(qa_data)}")
    
    # 如果是ChatML格式
    if isinstance(qa_data, list) and qa_data and "messages" in qa_data[0]:
        print("\n🔍 ChatML格式分析:")
        print("   • 这些QA对目前缺少完整的溯源metadata")
        print("   • 需要使用增强的traverse_graph.py来生成带溯源信息的QA")
        
        # 显示前几个QA对示例
        print("\n📄 QA示例:")
        for i, item in enumerate(qa_data[:3]):
            user_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "user"), "")
            assistant_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "assistant"), "")
            print(f"   Q{i+1}: {user_msg[:100]}...")
            print(f"   A{i+1}: {assistant_msg[:100]}...")
            print()
    
    # 检查是否有metadata
    has_metadata = False
    if isinstance(qa_data, dict):
        for qa_id, qa_info in qa_data.items():
            if "metadata" in qa_info and "source_tracing" in qa_info["metadata"]:
                has_metadata = True
                break
    
    if has_metadata:
        print("✅ 发现完整的溯源信息！")
    else:
        print("❌ 未发现溯源metadata，需要重新生成QA以获得完整追踪信息")
    
    return {
        "file_path": qa_file_path,
        "qa_count": len(qa_data),
        "format": "ChatML" if isinstance(qa_data, list) else "dict",
        "has_metadata": has_metadata,
        "analysis_time": str(datetime.now())
    }

def main():
    print("🔍 GraphGen 溯源分析工具")
    print("=" * 50)
    
    # 查找QA文件
    qa_files = glob.glob("cache/data/graphgen/*/qa-*.json")
    if not qa_files:
        print("❌ 未找到QA文件")
        return
    
    latest_file = max(qa_files, key=os.path.getctime)
    
    # 分析文件
    result = analyze_qa_file(latest_file)
    
    # 输出建议
    print("\n💡 溯源实现建议:")
    print("1. 📝 当前代码已增强traverse_graph.py，添加了完整的溯源支持")
    print("2. 🔄 重新生成QA对时将自动包含metadata.source_tracing信息")
    print("3. 🎯 溯源映射包含: 原文档→文本块→实体关系→子图→QA")
    print("4. ✅ 新的QA格式将支持事实性核验")
    
    # 保存分析报告
    report_path = latest_file.replace('.json', '_analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 分析报告已保存: {report_path}")

if __name__ == "__main__":
    main()