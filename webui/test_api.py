from openai import OpenAI
import gradio as gr
from openai import RateLimitError

def test_api_connection(api_base, api_key, model_name):
    """
    测试API连接，但不在遇到速率限制错误时失败
    
    这个函数的关键修改：
    - 区分处理RateLimitError和其他API错误
    - 遇到速率限制时给出警告而不是错误，允许用户继续操作
    - 特别处理401认证错误，提供更详细的错误信息
    
    Args:
        api_base: API基础URL
        api_key: API密钥
        model_name: 模型名称
    """
    # 检查基本参数
    if not api_base or not api_key or not model_name:
        raise gr.Error(f"{model_name}: API配置不完整，请检查URL、API Key和模型名称")
    
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
        error_msg = str(e)
        if "401" in error_msg or "Invalid token" in error_msg or "Unauthorized" in error_msg:
            raise gr.Error(f"{model_name}: API认证失败 - 请检查：\n1. API Key是否正确\n2. API Key是否有效且未过期\n3. 内网LLM服务是否需要不同的认证方式\n错误详情: {error_msg}")
        elif "404" in error_msg or "Not Found" in error_msg:
            raise gr.Error(f"{model_name}: API端点未找到 - 请检查：\n1. API基础URL是否正确\n2. 模型名称是否正确\n3. 内网LLM服务是否正常运行\n错误详情: {error_msg}")
        elif "Connection" in error_msg or "timeout" in error_msg:
            raise gr.Error(f"{model_name}: API连接失败 - 请检查：\n1. 网络连接是否正常\n2. 内网LLM服务地址是否可访问\n3. 防火墙设置是否正确\n错误详情: {error_msg}")
        else:
            # 其他错误仍然会导致失败
            raise gr.Error(f"{model_name}: API连接失败: {error_msg}")
