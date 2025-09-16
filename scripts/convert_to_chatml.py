#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将GraphGen生成的QA对转换为ChatML格式的脚本
该脚本会读取输入的JSON文件，去除metadata属性，并保存为ChatML格式
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any


def convert_to_chatml_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将QA对数据转换为ChatML格式
    
    Args:
        data: 原始QA对数据
        
    Returns:
        转换后的ChatML格式数据
    """
    chatml_data = []
    
    for item in data:
        # 创建新的ChatML格式项，只保留messages字段
        chatml_item = {
            "messages": item["messages"]
        }
        chatml_data.append(chatml_item)
    
    return chatml_data


def process_file(input_file: str, output_file: str = None):
    """
    处理单个文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则自动生成
    """
    print(f"正在处理文件: {input_file}")
    sys.stdout.flush()
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        sys.stdout.flush()
        return False
    
    try:
        # 读取输入文件
        print(f"尝试读取文件: {input_file}")
        sys.stdout.flush()
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取 {len(data)} 条记录")
        sys.stdout.flush()
        
        # 转换为ChatML格式
        chatml_data = convert_to_chatml_format(data)
        
        # 确定输出文件路径
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_chatml.json"
        
        print(f"输出文件路径: {output_file}")
        sys.stdout.flush()
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"创建输出目录: {output_dir}")
            sys.stdout.flush()
            os.makedirs(output_dir)
        
        # 保存转换后的数据
        print(f"正在写入文件: {output_file}")
        sys.stdout.flush()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chatml_data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成: {input_file} -> {output_file}")
        sys.stdout.flush()
        print(f"共处理 {len(chatml_data)} 条记录")
        sys.stdout.flush()
        return True
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False


def main():
    parser = argparse.ArgumentParser(description='将GraphGen生成的QA对转换为ChatML格式')
    parser.add_argument('input', help='输入JSON文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    print(f"输入文件: {args.input}")
    sys.stdout.flush()
    print(f"输出文件: {args.output}")
    sys.stdout.flush()
    
    success = process_file(args.input, args.output)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()