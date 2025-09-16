import json
import json
import os
import sys
import tempfile

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

# 从.env文件中加载环境变量
# 这是解决Windows下环境变量不自动加载问题的关键步骤
load_dotenv()

from webui.base import GraphGenParams
from webui.cache_utils import cleanup_workspace, setup_workspace
from webui.count_tokens import count_tokens
from webui.i18n import Translate
from webui.i18n import gettext as _
from webui.test_api import test_api_connection

# pylint: disable=wrong-import-position
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from graphgen.graphgen import GraphGen
from graphgen.models import OpenAIModel, Tokenizer, TraverseStrategy
from graphgen.models.llm.limitter import RPM, TPM
from graphgen.utils import set_logger

css = """
.center-row {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""


def init_graph_gen(config: dict, env: dict) -> GraphGen:
    """
    初始化GraphGen实例
    解决环境变量加载和初始化参数配置问题
    
    Args:
        config: 配置字典，包含分词器等参数
        env: 环境变量字典，包含API Key等敏感信息
    
    Returns:
        初始化完成的GraphGen实例
    """
    # 临时设置环境变量以便GraphGen初始化
    # 这确保了GraphGen.__post_init__不会因为缺少API Key而失败
    for key, value in env.items():
        if value:  # 只设置非空值
            os.environ[key] = str(value)
    
    # Set up working directory
    log_file, working_dir = setup_workspace(os.path.join(root_dir, "cache"))

    set_logger(log_file, if_stream=False)
    
    # 为GraphGen初始化创建最小配置
    graph_config = {
        "tokenizer": config.get("tokenizer", "cl100k_base"),
        "search": {"enabled": False},  # 为webui禁用搜索功能
        "input_file": config.get("input_file", ""),
        "input_data_type": "raw",  # 默认数据类型
        "quiz_and_judge_strategy": {
            "enabled": config.get("if_trainee_model", False),
            "quiz_samples": config.get("quiz_samples", 2),
            "re_judge": False
        }
    }
    
    graph_gen = GraphGen(config=graph_config, working_dir=working_dir)

    # Set up LLM clients
    graph_gen.synthesizer_llm_client = OpenAIModel(
        model_name=env.get("SYNTHESIZER_MODEL", ""),
        base_url=env.get("SYNTHESIZER_BASE_URL", ""),
        api_key=env.get("SYNTHESIZER_API_KEY", ""),
        request_limit=True,
        rpm=RPM(env.get("RPM", 1000)),
        tpm=TPM(env.get("TPM", 50000)),
    )

    graph_gen.trainee_llm_client = OpenAIModel(
        model_name=env.get("TRAINEE_MODEL", ""),
        base_url=env.get("TRAINEE_BASE_URL", ""),
        api_key=env.get("TRAINEE_API_KEY", ""),
        request_limit=True,
        rpm=RPM(env.get("RPM", 1000)),
        tpm=TPM(env.get("TPM", 50000)),
    )

    graph_gen.tokenizer_instance = Tokenizer(config.get("tokenizer", "cl100k_base"))

    strategy_config = config.get("traverse_strategy", {})
    graph_gen.traverse_strategy = TraverseStrategy(
        qa_form=strategy_config.get("qa_form"),
        expand_method=strategy_config.get("expand_method"),
        bidirectional=strategy_config.get("bidirectional"),
        max_extra_edges=strategy_config.get("max_extra_edges"),
        max_tokens=strategy_config.get("max_tokens"),
        max_depth=strategy_config.get("max_depth"),
        edge_sampling=strategy_config.get("edge_sampling"),
        isolated_node_strategy=strategy_config.get("isolated_node_strategy"),
        loss_strategy=str(strategy_config.get("loss_strategy")),
    )

    return graph_gen


# pylint: disable=too-many-statements
def run_graphgen(params, progress=gr.Progress()):
    def sum_tokens(client):
        return sum(u["total_tokens"] for u in client.token_usage)

    config = {
        "if_trainee_model": params.if_trainee_model,
        "input_file": params.input_file,
        "tokenizer": params.tokenizer,
        "quiz_samples": params.quiz_samples,
        "traverse_strategy": {
            "qa_form": params.qa_form,
            "bidirectional": params.bidirectional,
            "expand_method": params.expand_method,
            "max_extra_edges": params.max_extra_edges,
            "max_tokens": params.max_tokens,
            "max_depth": params.max_depth,
            "edge_sampling": params.edge_sampling,
            "isolated_node_strategy": params.isolated_node_strategy,
            "loss_strategy": params.loss_strategy,
        },
        "chunk_size": params.chunk_size,  # 保持向后兼容
        "chunking": {
            "chunk_size": params.chunk_size,
            "overlap_size": params.chunk_overlap_size,
            "strategy": params.chunking_strategy,
            "preserve_boundaries": params.preserve_boundaries,
            "min_chunk_size": params.min_chunk_size,
            "language_aware": params.language_aware,
            "boundary_markers": ["\u3002", "\uff01", "\uff1f", ".", "!", "?", "\n\n"]
        },
    }

    env = {
        "SYNTHESIZER_BASE_URL": params.synthesizer_url,
        "SYNTHESIZER_MODEL": params.synthesizer_model,
        "TRAINEE_BASE_URL": params.trainee_url,
        "TRAINEE_MODEL": params.trainee_model,
        "SYNTHESIZER_API_KEY": params.api_key,
        "TRAINEE_API_KEY": params.trainee_api_key,
        "RPM": params.rpm,
        "TPM": params.tpm,
    }

    # Test API connection
    test_api_connection(
        env["SYNTHESIZER_BASE_URL"],
        env["SYNTHESIZER_API_KEY"],
        env["SYNTHESIZER_MODEL"],
    )
    if config["if_trainee_model"]:
        test_api_connection(
            env["TRAINEE_BASE_URL"], env["TRAINEE_API_KEY"], env["TRAINEE_MODEL"]
        )

    # Initialize GraphGen
    graph_gen = init_graph_gen(config, env)
    graph_gen.clear()

    graph_gen.progress_bar = progress

    try:
        # Load input data
        file = config["input_file"]
        if isinstance(file, list):
            file = file[0] if file else None

        # 检查文件是否为空
        if not file:
            raise gr.Error("请上传文件后再运行GraphGen")

        data = []

        if file.endswith(".jsonl"):
            data_type = "raw"
            with open(file, "r", encoding="utf-8") as f:
                data.extend(json.loads(line) for line in f)
        elif file.endswith(".json"):
            data_type = "chunked"
            with open(file, "r", encoding="utf-8") as f:
                data.extend(json.load(f))
        elif file.endswith(".txt"):
            # txt文件作为单个完整文档处理，让GraphGen的tokenizer进行智能切分
            data_type = "raw"
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            # 不要在这里预切分！保持文档完整性，交给GraphGen处理
            data.extend([{"content": content}])
        else:
            raise ValueError(f"Unsupported file type: {file}")

        # Process the data
        graph_gen.insert_data(data, data_type)

        if config["if_trainee_model"]:
            # Generate quiz
            graph_gen.quiz_with_samples(config["quiz_samples"])

            # Judge statements
            graph_gen.judge_statements()
        else:
            graph_gen.traverse_strategy.edge_sampling = "random"
            # Skip judge statements
            graph_gen.judge_statements(skip=True)

        # Traverse graph
        graph_gen.traverse_with_strategy(graph_gen.traverse_strategy, config["traverse_strategy"]["qa_form"])

        # Save output
        output_data = graph_gen.qa_storage.data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmpfile:
            json.dump(output_data, tmpfile, ensure_ascii=False)
            output_file = tmpfile.name

        synthesizer_tokens = sum_tokens(graph_gen.synthesizer_llm_client)
        trainee_tokens = (
            sum_tokens(graph_gen.trainee_llm_client)
            if config["if_trainee_model"]
            else 0
        )
        total_tokens = synthesizer_tokens + trainee_tokens

        data_frame = params.token_counter
        try:
            # 检查DataFrame是否为空或没有数据
            if data_frame is None or len(data_frame) == 0:
                # 创建默认的DataFrame数据
                _update_data = [
                    ["未计算", "未计算", str(total_tokens)]
                ]
            else:
                # 使用现有的token计数数据
                _update_data = [
                    [data_frame.iloc[0, 0], data_frame.iloc[0, 1], str(total_tokens)]
                ]
            new_df = pd.DataFrame(_update_data, columns=["Source Text Token Count", "Expected Token Usage", "Token Used"])
            data_frame = new_df

        except Exception as e:
            # 如果出现任何错误，创建默认数据
            _update_data = [
                ["计算失败", "计算失败", str(total_tokens)]
            ]
            new_df = pd.DataFrame(_update_data, columns=["Source Text Token Count", "Expected Token Usage", "Token Used"])
            data_frame = new_df
            print(f"[WARNING] DataFrame操作失败，使用默认值: {str(e)}")

        return output_file, gr.DataFrame(
            label="Token Stats",
            headers=["Source Text Token Count", "Expected Token Usage", "Token Used"],
            datatype="str",
            interactive=False,
            value=data_frame,
            visible=True,
            wrap=True,
        )

    except Exception as e:  # pylint: disable=broad-except
        # Only cleanup on error
        cleanup_workspace(graph_gen.working_dir)
        raise gr.Error(f"Error occurred: {str(e)}")

    # Note: 不自动清理工作目录，保留生成的数据供用户查看
    # 如需清理，可手动删除 cache 目录下的子文件夹


with gr.Blocks(title="KGMentor Demo", theme=gr.themes.Glass(), css=css) as demo:
    # Header (logo removed)
    lang_btn = gr.Radio(
        choices=[
            ("简体中文", "zh"),
        ],
        value="zh",
        # label=_("Language"),
        render=False,
        container=False,
        elem_classes=["center-row"],
    )

    # Version badges and links removed
    with Translate(
        os.path.join(root_dir, "webui", "translation.json"),
        lang_btn,
        placeholder_langs=["zh"],
        persistant=False,  # True to save the language setting in the browser. Requires gradio >= 5.6.0
    ):
        lang_btn.render()

        gr.Markdown(
            value="# "
            + _("Title")
            + "\n\n"
            + _("### [GraphGen](https://github.com/open-sciencelab/GraphGen) ")
            + _("Intro")
        )

        if_trainee_model = gr.Checkbox(
            label=_("Use Trainee Model"), value=False, interactive=True
        )

        with gr.Accordion(label=_("Model Config"), open=False):
            synthesizer_url = gr.Textbox(
                label="Synthesizer URL",
                value="https://api.siliconflow.cn/v1",
                info=_("Synthesizer URL Info"),
                interactive=True,
            )
            synthesizer_model = gr.Textbox(
                label="Synthesizer Model",
                value="Qwen/Qwen2.5-7B-Instruct",
                info=_("Synthesizer Model Info"),
                interactive=True,
            )
            trainee_url = gr.Textbox(
                label="Trainee URL",
                value="https://api.siliconflow.cn/v1",
                info=_("Trainee URL Info"),
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            trainee_model = gr.Textbox(
                label="Trainee Model",
                value="Qwen/Qwen2.5-7B-Instruct",
                info=_("Trainee Model Info"),
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            trainee_api_key = gr.Textbox(
                label=_("SiliconFlow Token for Trainee Model"),
                type="password",
                value="",
                info="https://cloud.siliconflow.cn/account/ak",
                visible=if_trainee_model.value is True,
            )

        with gr.Accordion(label=_("Generation Config"), open=False):
            chunk_size = gr.Slider(
                label="Chunk Size",
                minimum=256,
                maximum=4096,
                value=512,
                step=256,
                interactive=True,
            )
            chunk_overlap_size = gr.Slider(
                label="Chunk Overlap Size",
                minimum=50,
                maximum=512,
                value=128,
                step=50,
                interactive=True,
            )
            chunking_strategy = gr.Radio(
                choices=["simple", "semantic", "hierarchical"],
                label="Chunking Strategy",
                value="semantic",
                interactive=True,
                info="Simple: 简单滑动窗口; Semantic: 语义感知; Hierarchical: 层次化"
            )
            preserve_boundaries = gr.Checkbox(
                label="Preserve Boundaries",
                value=True,
                interactive=True,
                info="是否保持语义边界（句子、段落）"
            )
            min_chunk_size = gr.Slider(
                label="Min Chunk Size",
                minimum=50,
                maximum=500,
                value=100,
                step=25,
                interactive=True,
                info="最小chunk大小，避免产生过小的片段"
            )
            language_aware = gr.Checkbox(
                label="Language Aware",
                value=True,
                interactive=True,
                info="是否启用语言感知的token估算"
            )
            tokenizer = gr.Textbox(
                label="Tokenizer", value="cl100k_base", interactive=True
            )
            qa_form = gr.Radio(
                choices=["atomic", "multi_hop", "aggregated"],
                label="QA Form",
                value="aggregated",
                interactive=True,
            )
            quiz_samples = gr.Number(
                label="Quiz Samples",
                value=2,
                minimum=1,
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            bidirectional = gr.Checkbox(
                label="Bidirectional", value=True, interactive=True
            )

            expand_method = gr.Radio(
                choices=["max_width", "max_tokens"],
                label="Expand Method",
                value="max_tokens",
                interactive=True,
            )
            max_extra_edges = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                label="Max Extra Edges",
                step=1,
                interactive=True,
                visible=expand_method.value == "max_width",
            )
            max_tokens = gr.Slider(
                minimum=64,
                maximum=1024,
                value=256,
                label="Max Tokens",
                step=64,
                interactive=True,
                visible=(expand_method.value != "max_width"),
            )

            max_depth = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                label="Max Depth",
                step=1,
                interactive=True,
            )
            edge_sampling = gr.Radio(
                choices=["max_loss", "min_loss", "random"],
                label="Edge Sampling",
                value="max_loss",
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            isolated_node_strategy = gr.Radio(
                choices=["add", "ignore"],
                label="Isolated Node Strategy",
                value="ignore",
                interactive=True,
            )
            loss_strategy = gr.Radio(
                choices=["only_edge", "both"],
                label="Loss Strategy",
                value="only_edge",
                interactive=True,
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                api_key = gr.Textbox(
                    label=_("SiliconFlow Token"),
                    type="password",
                    value="",
                    info="https://cloud.siliconflow.cn/account/ak",
                )
            with gr.Column(scale=1):
                test_connection_btn = gr.Button(_("Test Connection"))

        with gr.Blocks():
            with gr.Row(equal_height=True):
                with gr.Column():
                    rpm = gr.Slider(
                        label="RPM",
                        minimum=10,
                        maximum=10000,
                        value=1000,
                        step=100,
                        interactive=True,
                        visible=True,
                    )
                with gr.Column():
                    tpm = gr.Slider(
                        label="TPM",
                        minimum=5000,
                        maximum=5000000,
                        value=50000,
                        step=1000,
                        interactive=True,
                        visible=True,
                    )

        with gr.Blocks():
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    upload_file = gr.File(
                        label=_("Upload File"),
                        file_count="single",
                        file_types=[".txt", ".json", ".jsonl"],
                        interactive=True,
                    )
                    examples_dir = os.path.join(root_dir, "webui", "examples")
                    gr.Examples(
                        examples=[
                            [os.path.join(examples_dir, "txt_demo.txt")],
                            [os.path.join(examples_dir, "raw_demo.jsonl")],
                            [os.path.join(examples_dir, "chunked_demo.json")],
                        ],
                        inputs=upload_file,
                        label=_("Example Files"),
                        examples_per_page=3,
                    )
                with gr.Column(scale=1):
                    output = gr.File(
                        label="Output(See Github FAQ)",
                        file_count="single",
                        interactive=False,
                    )

        with gr.Blocks():
            token_counter = gr.DataFrame(
                label="Token Stats",
                headers=[
                    "Source Text Token Count",
                    "Estimated Token Usage", 
                    "Token Used",
                ],
                datatype="str",
                interactive=False,
                visible=False,
                wrap=True,
                # 提供默认数据以避免空 DataFrame 错误
                value=[["0", "0", "N/A"]]
            )

        submit_btn = gr.Button(_("开始合成"))

        # Test Connection
        test_connection_btn.click(
            test_api_connection,
            inputs=[synthesizer_url, api_key, synthesizer_model],
            outputs=[],
        )

        if if_trainee_model.value:
            test_connection_btn.click(
                test_api_connection,
                inputs=[trainee_url, api_key, trainee_model],
                outputs=[],
            )

        expand_method.change(
            lambda method: (
                gr.update(visible=method == "max_width"),
                gr.update(visible=method != "max_width"),
            ),
            inputs=expand_method,
            outputs=[max_extra_edges, max_tokens],
        )

        if_trainee_model.change(
            lambda use_trainee: [gr.update(visible=use_trainee)] * 5,
            inputs=if_trainee_model,
            outputs=[
                trainee_url,
                trainee_model,
                quiz_samples,
                edge_sampling,
                trainee_api_key,
            ],
        )

        upload_file.change(
            lambda x: (gr.update(visible=True)),
            inputs=[upload_file],
            outputs=[token_counter],
        ).then(
            count_tokens,
            inputs=[upload_file, tokenizer, token_counter],
            outputs=[token_counter],
        )

        # run GraphGen
        submit_btn.click(
            lambda x: (gr.update(visible=False)),
            inputs=[token_counter],
            outputs=[token_counter],
        )

        submit_btn.click(
            lambda *args: run_graphgen(
                GraphGenParams(
                    if_trainee_model=args[0],
                    input_file=args[1],
                    tokenizer=args[2],
                    qa_form=args[3],
                    bidirectional=args[4],
                    expand_method=args[5],
                    max_extra_edges=args[6],
                    max_tokens=args[7],
                    max_depth=args[8],
                    edge_sampling=args[9],
                    isolated_node_strategy=args[10],
                    loss_strategy=args[11],
                    synthesizer_url=args[12],
                    synthesizer_model=args[13],
                    trainee_model=args[14],
                    api_key=args[15],
                    chunk_size=args[16],
                    chunk_overlap_size=args[17],
                    chunking_strategy=args[18],
                    preserve_boundaries=args[19],
                    min_chunk_size=args[20],
                    language_aware=args[21],
                    rpm=args[22],
                    tpm=args[23],
                    quiz_samples=args[24],
                    trainee_url=args[25],
                    trainee_api_key=args[26],
                    token_counter=args[27],
                )
            ),
            inputs=[
                if_trainee_model,
                upload_file,
                tokenizer,
                qa_form,
                bidirectional,
                expand_method,
                max_extra_edges,
                max_tokens,
                max_depth,
                edge_sampling,
                isolated_node_strategy,
                loss_strategy,
                synthesizer_url,
                synthesizer_model,
                trainee_model,
                api_key,
                chunk_size,
                chunk_overlap_size,
                chunking_strategy,
                preserve_boundaries,
                min_chunk_size,
                language_aware,
                rpm,
                tpm,
                quiz_samples,
                trainee_url,
                trainee_api_key,
                token_counter,
            ],
            outputs=[output, token_counter],
        )


if __name__ == "__main__":
    demo.queue(api_open=False, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0")