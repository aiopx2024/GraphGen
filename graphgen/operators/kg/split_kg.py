import random
from collections import defaultdict
from typing import Any

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import NetworkXStorage, TraverseStrategy
from graphgen.utils import logger


async def _get_node_info(
    node_id: str,
    graph_storage: NetworkXStorage,
) -> dict:
    """
    Get node info

    :param node_id: node id
    :param graph_storage: graph storage instance
    :return: node info
    """
    node_data = await graph_storage.get_node(node_id)
    return {"node_id": node_id, **node_data}


def _get_level_n_edges_by_max_width(
    edge_adj_list: dict,
    node_dict: dict,
    edges: list,
    nodes,
    src_edge: tuple,
    max_depth: int,
    bidirectional: bool,
    max_extra_edges: int,
    edge_sampling: str,
    loss_strategy: str = "only_edge",
) -> list:
    """
    Get level n edges for an edge.
    n is decided by max_depth in traverse_strategy

    :param edge_adj_list
    :param node_dict
    :param edges
    :param nodes
    :param src_edge
    :param max_depth
    :param bidirectional
    :param max_extra_edges
    :param edge_sampling
    :return: level n edges
    """
    src_id, tgt_id, _ = src_edge

    level_n_edges = []

    start_nodes = {tgt_id} if not bidirectional else {src_id, tgt_id}

    while max_depth > 0 and max_extra_edges > 0:
        max_depth -= 1

        candidate_edges = [
            edges[edge_id]
            for node in start_nodes
            for edge_id in edge_adj_list[node]
            if not edges[edge_id][2].get("visited", False)
        ]

        if not candidate_edges:
            break

        if len(candidate_edges) >= max_extra_edges:
            if loss_strategy == "both":
                er_tuples = [
                    ([nodes[node_dict[edge[0]]], nodes[node_dict[edge[1]]]], edge)
                    for edge in candidate_edges
                ]
                candidate_edges = _sort_tuples(er_tuples, edge_sampling)[
                    :max_extra_edges
                ]
            elif loss_strategy == "only_edge":
                candidate_edges = _sort_edges(candidate_edges, edge_sampling)[
                    :max_extra_edges
                ]
            else:
                raise ValueError(f"Invalid loss strategy: {loss_strategy}")

            for edge in candidate_edges:
                level_n_edges.append(edge)
                edge[2]["visited"] = True
            break

        max_extra_edges -= len(candidate_edges)
        new_start_nodes = set()

        for edge in candidate_edges:
            level_n_edges.append(edge)
            edge[2]["visited"] = True

            if not edge[0] in start_nodes:
                new_start_nodes.add(edge[0])
            if not edge[1] in start_nodes:
                new_start_nodes.add(edge[1])

        start_nodes = new_start_nodes

    return level_n_edges


def _get_level_n_edges_by_max_tokens(
    edge_adj_list: dict,
    node_dict: dict,
    edges: list,
    nodes: list,
    src_edge: tuple,
    max_depth: int,
    bidirectional: bool,
    max_tokens: int,
    edge_sampling: str,
    loss_strategy: str = "only_edge",
) -> list:
    """
    Get level n edges for an edge.
    n is decided by max_depth in traverse_strategy.

    :param edge_adj_list
    :param node_dict
    :param edges
    :param nodes
    :param src_edge
    :param max_depth
    :param bidirectional
    :param max_tokens
    :param edge_sampling
    :return: level n edges
    """
    src_id, tgt_id, src_edge_data = src_edge

    max_tokens -= (
        src_edge_data["length"]
        + nodes[node_dict[src_id]][1]["length"]
        + nodes[node_dict[tgt_id]][1]["length"]
    )

    level_n_edges = []

    start_nodes = {tgt_id} if not bidirectional else {src_id, tgt_id}
    temp_nodes = {src_id, tgt_id}

    while max_depth > 0 and max_tokens > 0:
        max_depth -= 1

        candidate_edges = [
            edges[edge_id]
            for node in start_nodes
            for edge_id in edge_adj_list[node]
            if not edges[edge_id][2].get("visited", False)
        ]

        if not candidate_edges:
            break

        if loss_strategy == "both":
            er_tuples = [
                ([nodes[node_dict[edge[0]]], nodes[node_dict[edge[1]]]], edge)
                for edge in candidate_edges
            ]
            candidate_edges = _sort_tuples(er_tuples, edge_sampling)
        elif loss_strategy == "only_edge":
            candidate_edges = _sort_edges(candidate_edges, edge_sampling)
        else:
            raise ValueError(f"Invalid loss strategy: {loss_strategy}")

        for edge in candidate_edges:
            max_tokens -= edge[2]["length"]
            if not edge[0] in temp_nodes:
                max_tokens -= nodes[node_dict[edge[0]]][1]["length"]
            if not edge[1] in temp_nodes:
                max_tokens -= nodes[node_dict[edge[1]]][1]["length"]

            if max_tokens < 0:
                return level_n_edges

            level_n_edges.append(edge)
            edge[2]["visited"] = True
            temp_nodes.add(edge[0])
            temp_nodes.add(edge[1])

        new_start_nodes = set()
        for edge in candidate_edges:
            if not edge[0] in start_nodes:
                new_start_nodes.add(edge[0])
            if not edge[1] in start_nodes:
                new_start_nodes.add(edge[1])

        start_nodes = new_start_nodes

    return level_n_edges


def _sort_tuples(er_tuples: list, edge_sampling: str) -> list:
    """
    Sort edges with edge sampling strategy

    :param er_tuples: [(nodes:list, edge:tuple)]
    :param edge_sampling: edge sampling strategy (random, min_loss, max_loss)
    :return: sorted edges
    """
    if edge_sampling == "random":
        er_tuples = random.sample(er_tuples, len(er_tuples))
    elif edge_sampling == "min_loss":
        er_tuples = sorted(
            er_tuples,
            key=lambda x: sum(node[1]["loss"] for node in x[0]) + x[1][2]["loss"],
        )
    elif edge_sampling == "max_loss":
        er_tuples = sorted(
            er_tuples,
            key=lambda x: sum(node[1]["loss"] for node in x[0]) + x[1][2]["loss"],
            reverse=True,
        )
    else:
        raise ValueError(f"Invalid edge sampling: {edge_sampling}")
    edges = [edge for _, edge in er_tuples]
    return edges


def _sort_edges(edges: list, edge_sampling: str) -> list:
    """
    Sort edges with edge sampling strategy

    :param edges: total edges
    :param edge_sampling: edge sampling strategy (random, min_loss, max_loss)
    :return: sorted edges
    """
    if edge_sampling == "random":
        random.shuffle(edges)
    elif edge_sampling == "min_loss":
        edges = sorted(edges, key=lambda x: x[2]["loss"])
    elif edge_sampling == "max_loss":
        edges = sorted(edges, key=lambda x: x[2]["loss"], reverse=True)
    else:
        raise ValueError(f"Invalid edge sampling: {edge_sampling}")
    return edges


async def get_batches_with_strategy(  # pylint: disable=too-many-branches
    nodes: list,
    edges: list,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
):
    """
    根据遍历策略将知识图谱划分成处理批次（子图）
    
    这是GraphGen中图遍历的核心函数，负责将完整的知识图谱按照指定策略划分成多个
    小的子图（批次），每个批次将用于生成一个问答对。此函数是连接知识图谱构建
    和问答对生成的关键桥梁。
    
    主要功能：
    1. 根据expand_method（扩展方法）选择不同的批次划分策略
    2. 根据max_depth（最大深度）控制每个批次的推理复杂度
    3. 根据edge_sampling（边采样）策略优化批次质量
    4. 处理孤立节点，确保图的完整性
    5. 去重节点和边，避免重复处理
    
    Args:
        nodes (list): 知识图谱中的所有节点列表，每个节点包含实体信息
        edges (list): 知识图谱中的所有边列表，每个边表示实体间的关系
        graph_storage (NetworkXStorage): 图存储后端，用于访问节点详细信息
        traverse_strategy (TraverseStrategy): 遍历策略配置，包含所有划分参数
        
    Returns:
        list: 处理批次列表，每个批次包含(nodes, edges)元组，代表一个子图
        
    遍历策略说明：
    - max_width: 基于最大宽度的扩展，控制每个批次包含的边数上限
    - max_tokens: 基于Token数量的扩展，控制每个批次的文本长度
    - bidirectional: 是否进行双向扩展（从起点和终点同时扩展）
    - edge_sampling: 边采样策略（random/loss_asc/loss_desc）
    - isolated_node_strategy: 孤立节点处理策略（add/ignore）
    """
    # 解析遍历策略参数，确定批次划分方式
    expand_method = traverse_strategy.expand_method
    if expand_method == "max_width":
        # 按最大宽度扩展：以边数为限制，控制每个批次包含的关系数量
        logger.info("Using max width strategy")
    elif expand_method == "max_tokens":
        # 按最大Token数扩展：以文本长度为限制，控制每个批次的内容量
        logger.info("Using max tokens strategy")
    else:
        raise ValueError(f"Invalid expand method: {expand_method}")

    # 获取其他遍历参数
    max_depth = traverse_strategy.max_depth  # 最大推理深度，决定多跳推理的复杂度
    edge_sampling = traverse_strategy.edge_sampling  # 边采样策略，影响问答对质量

    # 初始化数据结构：为图遍历做准备
    edge_adj_list: dict[Any, Any] = defaultdict(list)  # 邻接表：存储每个节点连接的边
    node_dict = {}  # 节点字典：将节点名映射到索引，加速查找
    processing_batches = []  # 存储最终生成的所有批次

    # 节点信息缓存：避免重复查询相同节点的详细信息
    node_cache = {}

    async def get_cached_node_info(node_id: str) -> dict:
        """
        获取节点详细信息（带缓存）
        
        为了提高性能，避免对相同节点重复查询数据库。
        特别在处理大型知识图谱时，这种缓存机制可以显著提升速度。
        
        Args:
            node_id (str): 节点的唯一标识符
            
        Returns:
            dict: 包含节点完整信息的字典
        """
        if node_id not in node_cache:
            node_cache[node_id] = await _get_node_info(node_id, graph_storage)
        return node_cache[node_id]

    # 构建节点索引字典：将节点名映射到在列表中的位置
    for i, (node_name, _) in enumerate(nodes):
        node_dict[node_name] = i

    # 根据损失策略对边进行排序，优先处理高质量或低质量的边
    if traverse_strategy.loss_strategy == "both":
        # 同时考虑边和节点的损失值，进行综合排序
        er_tuples = [
            ([nodes[node_dict[edge[0]]], nodes[node_dict[edge[1]]]], edge)
            for edge in edges
        ]
        edges = _sort_tuples(er_tuples, edge_sampling)
    elif traverse_strategy.loss_strategy == "only_edge":
        # 仅考虑边的损失值进行排序
        edges = _sort_edges(edges, edge_sampling)
    else:
        raise ValueError(f"Invalid loss strategy: {traverse_strategy.loss_strategy}")

    # 构建邻接表：为每个节点记录其相关边的索引
    # 这个数据结构支持快速查找与指定节点相关的所有边
    for i, (src, tgt, _) in enumerate(edges):
        edge_adj_list[src].append(i)  # 起点节点的相关边
        edge_adj_list[tgt].append(i)  # 终点节点的相关边

    # 遍历所有边，为每个边生成一个批次（子图）
    # 每个批次以一个边为核心，然后向周围扩展相关节点和边
    for edge in tqdm_async(edges, desc="Preparing batches"):
        # 检查边是否已经被处理过，避免重复处理
        if "visited" in edge[2] and edge[2]["visited"]:
            continue

        # 标记边为已访问，防止后续重复处理
        edge[2]["visited"] = True

        # 初始化当前批次的节点和边列表
        _process_nodes = []  # 存储当前批次的所有节点
        _process_edges = []  # 存储当前批次的所有边

        # 获取当前边的起点和终点节点ID
        src_id = edge[0]
        tgt_id = edge[1]

        # 将当前边的两个节点添加到批次中（作为初始节点）
        _process_nodes.extend(
            [await get_cached_node_info(src_id), await get_cached_node_info(tgt_id)]
        )
        _process_edges.append(edge)  # 将当前边添加到批次中

        # 根据扩展方法选择不同的子图扩展策略
        if expand_method == "max_width":
            # 使用最大宽度策略：以边数为限制进行多层扩展
            level_n_edges = _get_level_n_edges_by_max_width(
                edge_adj_list,
                node_dict,
                edges,
                nodes,
                edge,
                max_depth,
                traverse_strategy.bidirectional,
                traverse_strategy.max_extra_edges,
                edge_sampling,
                traverse_strategy.loss_strategy,
            )
        else:
            # 使用最大Token数策略：以文本长度为限制进行多层扩展
            level_n_edges = _get_level_n_edges_by_max_tokens(
                edge_adj_list,
                node_dict,
                edges,
                nodes,
                edge,
                max_depth,
                traverse_strategy.bidirectional,
                traverse_strategy.max_tokens,
                edge_sampling,
                traverse_strategy.loss_strategy,
            )

        # 将扩展得到的边和相关节点添加到当前批次中
        # 这些边组成了以初始边为中心的多跳推理网络
        for _edge in level_n_edges:
            _process_nodes.append(await get_cached_node_info(_edge[0]))  # 添加边的起点
            _process_nodes.append(await get_cached_node_info(_edge[1]))  # 添加边的终点
            _process_edges.append(_edge)  # 添加边本身

        # 对批次中的节点和边进行去重处理
        # 去重节点：使用node_id作为唯一标识，保持最后一个出现的节点信息
        _process_nodes = list(
            {node["node_id"]: node for node in _process_nodes}.values()
        )
        # 去重边：使用(起点, 终点)作为唯一标识，保持最后一个出现的边信息
        _process_edges = list(
            {(edge[0], edge[1]): edge for edge in _process_edges}.values()
        )

        # 将处理完成的批次（子图）添加到结果列表中
        processing_batches.append((_process_nodes, _process_edges))

    # 记录生成的批次数量，用于统计和调试
    logger.info("Processing batches: %d", len(processing_batches))

    # 处理孤立节点（没有任何边连接的节点）
    isolated_node_strategy = traverse_strategy.isolated_node_strategy
    if isolated_node_strategy == "add":
        # 将孤立节点作为单独的批次添加，用于生成简单的单实体问答对
        processing_batches = await _add_isolated_nodes(
            nodes, processing_batches, graph_storage
        )
        logger.info(
            "Processing batches after adding isolated nodes: %d",
            len(processing_batches),
        )

    # 返回所有处理批次，每个批次包含一个子图的节点和边
    # 这些批次将被用于后续的问答对生成
    return processing_batches


async def _add_isolated_nodes(
    nodes: list,
    processing_batches: list,
    graph_storage: NetworkXStorage,
) -> list:
    visited_nodes = set()
    for _process_nodes, _process_edges in processing_batches:
        for node in _process_nodes:
            visited_nodes.add(node["node_id"])
    for node in nodes:
        if node[0] not in visited_nodes:
            _process_nodes = [await _get_node_info(node[0], graph_storage)]
            processing_batches.append((_process_nodes, []))
    return processing_batches
