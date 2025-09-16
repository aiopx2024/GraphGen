import json


def read_file(input_file: str) -> list:
    """
    Read data from a file based on the specified data type.
    :param input_file
    :return:
    """

    if input_file.endswith(".jsonl"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_file.endswith(".txt"):
        # txt文件作为单个完整文档处理，让GraphGen的tokenizer进行智能切分
        # 不要按行分割，这会破坏语义完整性
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        data = [{"content": content}]
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    return data
