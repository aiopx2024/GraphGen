#!/usr/bin/env python3
"""
测试DataFrame修复的脚本
模拟WebUI中可能出现的DataFrame错误情况
"""

import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_empty_dataframe_handling():
    """测试空DataFrame的处理"""
    print("=== 测试空DataFrame处理 ===")
    
    # 模拟空DataFrame情况
    empty_df = pd.DataFrame()
    print(f"空DataFrame长度: {len(empty_df)}")
    
    # 测试修复后的逻辑
    total_tokens = 1000
    
    try:
        if empty_df is None or len(empty_df) == 0:
            # 创建默认的DataFrame数据
            _update_data = [
                ["未计算", "未计算", str(total_tokens)]
            ]
            print("✅ 使用默认数据创建DataFrame")
        else:
            # 使用现有的token计数数据
            _update_data = [
                [empty_df.iloc[0, 0], empty_df.iloc[0, 1], str(total_tokens)]
            ]
            print("使用现有数据")
            
        new_df = pd.DataFrame(_update_data, columns=["Source Text Token Count", "Expected Token Usage", "Token Used"])
        print(f"✅ 成功创建DataFrame: {new_df.values.tolist()}")
        
    except Exception as e:
        print(f"❌ DataFrame操作失败: {str(e)}")

def test_normal_dataframe_handling():
    """测试正常DataFrame的处理"""
    print("\n=== 测试正常DataFrame处理 ===")
    
    # 模拟正常DataFrame情况
    normal_df = pd.DataFrame([["1000", "50000", "N/A"]], 
                           columns=["Source Text Token Count", "Expected Token Usage", "Token Used"])
    print(f"正常DataFrame: {normal_df.values.tolist()}")
    
    total_tokens = 1500
    
    try:
        if normal_df is None or len(normal_df) == 0:
            _update_data = [
                ["未计算", "未计算", str(total_tokens)]
            ]
            print("使用默认数据")
        else:
            # 使用现有的token计数数据
            _update_data = [
                [normal_df.iloc[0, 0], normal_df.iloc[0, 1], str(total_tokens)]
            ]
            print("✅ 使用现有数据")
            
        new_df = pd.DataFrame(_update_data, columns=["Source Text Token Count", "Expected Token Usage", "Token Used"])
        print(f"✅ 成功创建DataFrame: {new_df.values.tolist()}")
        
    except Exception as e:
        print(f"❌ DataFrame操作失败: {str(e)}")

def test_count_tokens_function():
    """测试count_tokens函数"""
    print("\n=== 测试count_tokens函数 ===")
    
    # 导入count_tokens函数
    try:
        from webui.count_tokens import count_tokens
        
        # 测试没有文件的情况
        result = count_tokens(None, "cl100k_base", pd.DataFrame())
        print(f"✅ 无文件情况处理: {result.values.tolist()}")
        
        # 测试空DataFrame的情况
        result = count_tokens(None, "cl100k_base", None)
        print(f"✅ 空DataFrame情况处理: {result.values.tolist()}")
        
    except ImportError as e:
        print(f"❌ 无法导入count_tokens函数: {str(e)}")
    except Exception as e:
        print(f"❌ count_tokens测试失败: {str(e)}")

if __name__ == "__main__":
    print("GraphGen DataFrame修复测试")
    print("=" * 50)
    
    test_empty_dataframe_handling()
    test_normal_dataframe_handling() 
    test_count_tokens_function()
    
    print("\n" + "=" * 50)
    print("测试完成！")