# QA对转换脚本

这个目录包含用于将GraphGen生成的QA对转换为不同格式的脚本。

## 脚本说明

### convert_to_chatml.py

将GraphGen生成的QA对转换为ChatML格式，去除metadata属性。

#### 使用方法

```bash
python convert_to_chatml.py input.json -o output.json
```

#### 参数说明

- `input`: 输入的JSON文件路径
- `-o, --output`: 输出文件路径（可选，默认为输入文件名加`_chatml.json`后缀）

#### 示例

```bash
# 转换文件并指定输出路径
python convert_to_chatml.py qa_pairs.json -o qa_pairs_chatml.json

# 转换文件并使用默认输出路径
python convert_to_chatml.py qa_pairs.json
# 输出文件将为 qa_pairs_chatml.json
```

#### 输出格式

转换后的文件将只保留[messages](file:///D:/git/GraphGen/baselines/EntiGraph/entigraph_utils/prompt_utils.py#L22-L22)字段，去除所有[metadata](file:///D:/git/GraphGen/graphgen/models/storage/json_storage.py#L34-L34)信息，符合ChatML格式要求。