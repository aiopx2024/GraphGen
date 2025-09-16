#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

# 读取测试文件
print("Reading test file...")
with open('test_qa.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} records")

# 转换格式
chatml_data = []
for item in data:
    chatml_item = {
        "messages": item["messages"]
    }
    chatml_data.append(chatml_item)

# 保存转换后的数据
with open('test_qa_chatml.json', 'w', encoding='utf-8') as f:
    json.dump(chatml_data, f, ensure_ascii=False, indent=2)

print("Conversion completed!")
print(f"Processed {len(chatml_data)} records")