from openai import OpenAI
import gradio as gr
from openai import RateLimitError

def test_api_connection(api_base, api_key, model_name):
    """
    测试API连接，但不在遇到速率限制错误时失败
    
    这个函数的关键修改：
    - 区分处理RateLimitError和其他API错误
    - 遇到速率限制时给出警告而不是错误，允许用户继续操作
    
    Args:
        api_base: API基础URL
        api_key: API密钥
        model_name: 模型名称
    """
    client = OpenAI(api_key=api_key, base_url=api_base)
    try:
        # 发送测试请求，只生成1个Token以减少成本
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        if not response.choices or not response.choices[0].message:
            raise gr.Error(f"{model_name}: API返回无效响应")
        gr.Info(f"{model_name}: API连接成功")
    except RateLimitError as e:
        # 不在速率限制时失败，只是警告用户
        gr.Warning(f"{model_name}: API速率限制已达到，但配置有效。您可以继续进行生成。")
    except Exception as e:
        # 其他错误仍然会导致失败
        raise gr.Error(f"{model_name}: API连接失败: {str(e)}")
