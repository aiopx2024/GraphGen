#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphML知识图谱可视化工具
创建交互式网页可视化
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Tuple

def load_graphml(file_path: str) -> nx.Graph:
    """加载GraphML文件"""
    try:
        G = nx.read_graphml(file_path)
        print(f"✅ 成功加载图谱: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        return G
    except Exception as e:
        print(f"❌ 加载GraphML文件失败: {e}")
        return None

def analyze_graph(G: nx.Graph) -> Dict:
    """分析图谱结构"""
    print("\n📊 图谱结构分析:")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    print(f"图密度: {nx.density(G):.4f}")
    print(f"连通分量: {nx.number_connected_components(G)}")
    
    # 实体类型统计
    entity_types = {}
    for node in G.nodes():
        entity_type = G.nodes[node].get('entity_type', 'UNKNOWN')
        if isinstance(entity_type, str):
            entity_type = entity_type.strip('"')
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print("\n🏷️ 实体类型分布:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")
    
    # 重要节点分析
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\n🌟 最重要的5个节点:")
    for node, centrality in top_nodes:
        clean_name = str(node).strip('"')[:30]
        print(f"  {clean_name}: 连接{G.degree(node)}个节点")
    
    return {
        'entity_types': entity_types,
        'top_nodes': top_nodes,
        'stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'components': nx.number_connected_components(G)
        }
    }

def create_3d_visualization(G: nx.Graph, output_file: str = "knowledge_graph_3d.html"):
    """创建3D交互式可视化"""
    print("\n🎨 创建3D可视化...")
    
    # 计算3D布局
    pos = nx.spring_layout(G, dim=3, k=1.5, iterations=50, seed=42)
    
    # 节点坐标
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]
    
    # 边坐标
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # 颜色映射
    color_map = {
        'WORK': '#FF6B6B',        # 红色 - 作品/品种
        'LOCATION': '#4ECDC4',    # 青色 - 地点
        'DATE': '#45B7D1',       # 蓝色 - 日期
        'TECHNOLOGY': '#96CEB4',  # 绿色 - 技术
        'CONCEPT': '#FFEAA7',     # 黄色 - 概念
        'CHEMICAL': '#DDA0DD',    # 紫色 - 化学物质
        'GENE': '#FFB347',        # 橙色 - 基因
        'ORGANIZATION': '#F8C8DC', # 粉色 - 机构
        'NATURE': '#98FB98',      # 浅绿 - 性质
        'EVENT': '#F0E68C',       # 卡其色 - 事件
        'ORGANISM': '#DEB887',    # 棕色 - 生物
        'MISSION': '#B0C4DE',     # 浅蓝 - 任务
        'UNKNOWN': '#D3D3D3'      # 灰色 - 未知
    }
    
    # 准备节点数据
    node_info = []
    node_colors = []
    node_sizes = []
    node_symbols = []
    
    for node in G.nodes():
        # 获取节点属性
        entity_type = G.nodes[node].get('entity_type', 'UNKNOWN')
        if isinstance(entity_type, str):
            entity_type = entity_type.strip('"')
        
        description = G.nodes[node].get('description', '')
        if isinstance(description, str):
            description = description.strip('"')
        
        source_id = G.nodes[node].get('source_id', '')
        degree = G.degree(node)
        
        # 清理节点名称
        clean_name = str(node).strip('"')
        
        # 构建hover信息
        hover_text = f"""
        <b>节点:</b> {clean_name}<br>
        <b>类型:</b> {entity_type}<br>
        <b>连接数:</b> {degree}<br>
        <b>描述:</b> {description[:100]}{'...' if len(description) > 100 else ''}<br>
        <b>来源:</b> {source_id}
        """
        node_info.append(hover_text)
        
        # 设置颜色和大小
        node_colors.append(color_map.get(entity_type, '#D3D3D3'))
        node_sizes.append(max(degree * 3 + 8, 8))  # 根据度数设置大小
        
        # 根据类型设置符号 (只使用3D支持的符号)
        symbol_map = {
            'WORK': 'diamond', 'LOCATION': 'square', 'DATE': 'circle',
            'TECHNOLOGY': 'cross', 'CONCEPT': 'diamond-open', 'CHEMICAL': 'square-open',
            'GENE': 'x', 'ORGANIZATION': 'circle-open'
        }
        node_symbols.append(symbol_map.get(entity_type, 'circle'))
    
    # 创建边轨迹
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        mode='lines',
        name='关系'
    )
    
    # 创建节点轨迹
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        hovertext=node_info,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.8,
            line=dict(width=1, color='rgba(50,50,50,0.5)'),
            symbol=node_symbols
        ),
        name='实体'
    )
    
    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # 设置布局
    fig.update_layout(
        title={
            'text': '知识图谱 3D 可视化',
            'x': 0.5,
            'font': {'size': 20}
        },
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # 添加说明文本
    fig.add_annotation(
        text=f"节点: {G.number_of_nodes()} | 边: {G.number_of_edges()} | 密度: {nx.density(G):.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # 保存HTML文件
    plot(fig, filename=output_file, auto_open=False)
    print(f"✅ 3D可视化已保存: {output_file}")

def create_2d_network_visualization(G: nx.Graph, output_file: str = "knowledge_graph_2d.html"):
    """创建2D网络可视化"""
    print("\n🌐 创建2D网络可视化...")
    
    # 计算2D布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 节点坐标
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # 边坐标
    edge_x, edge_y = [], []
    edge_info = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # 获取边的描述
        edge_data = G.get_edge_data(edge[0], edge[1])
        description = edge_data.get('description', '') if edge_data else ''
        edge_info.append(f"{edge[0]} → {edge[1]}: {description}")
    
    # 颜色映射（同3D版本）
    color_map = {
        'WORK': '#FF6B6B', 'LOCATION': '#4ECDC4', 'DATE': '#45B7D1',
        'TECHNOLOGY': '#96CEB4', 'CONCEPT': '#FFEAA7', 'CHEMICAL': '#DDA0DD',
        'GENE': '#FFB347', 'ORGANIZATION': '#F8C8DC', 'NATURE': '#98FB98',
        'EVENT': '#F0E68C', 'ORGANISM': '#DEB887', 'MISSION': '#B0C4DE',
        'UNKNOWN': '#D3D3D3'
    }
    
    # 准备节点数据
    node_info = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        entity_type = G.nodes[node].get('entity_type', 'UNKNOWN')
        if isinstance(entity_type, str):
            entity_type = entity_type.strip('"')
        
        description = G.nodes[node].get('description', '')
        if isinstance(description, str):
            description = description.strip('"')
        
        degree = G.degree(node)
        clean_name = str(node).strip('"')
        
        hover_text = f"<b>{clean_name}</b><br>类型: {entity_type}<br>连接: {degree}<br>{description[:80]}{'...' if len(description) > 80 else ''}"
        node_info.append(hover_text)
        node_colors.append(color_map.get(entity_type, '#D3D3D3'))
        node_sizes.append(max(degree * 5 + 10, 10))
    
    # 创建图形
    fig = go.Figure()
    
    # 添加边
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines',
        name='关系',
        showlegend=False
    ))
    
    # 添加节点
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_info,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.8,
            line=dict(width=1, color='rgba(50,50,50,0.5)')
        ),
        text=[str(node).strip('"')[:10] for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=8, color="white"),
        name='实体'
    ))
    
    # 设置布局
    fig.update_layout(
        title={
            'text': '知识图谱 2D 网络可视化',
            'font': {'size': 16}
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text=f"节点: {G.number_of_nodes()} | 边: {G.number_of_edges()}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    # 保存文件
    plot(fig, filename=output_file, auto_open=False)
    print(f"✅ 2D可视化已保存: {output_file}")

def create_statistics_dashboard(G: nx.Graph, analysis: Dict, output_file: str = "graph_statistics.html"):
    """创建统计信息仪表板"""
    print("\n📈 创建统计仪表板...")
    
    from plotly.subplots import make_subplots
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('实体类型分布', '节点度分布', '连通分量大小', '关键统计指标'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # 1. 实体类型分布饼图
    entity_types = analysis['entity_types']
    fig.add_trace(
        go.Pie(
            labels=list(entity_types.keys()),
            values=list(entity_types.values()),
            name="实体类型"
        ),
        row=1, col=1
    )
    
    # 2. 节点度分布直方图
    degrees = [G.degree(node) for node in G.nodes()]
    fig.add_trace(
        go.Histogram(
            x=degrees,
            nbinsx=20,
            name="度分布"
        ),
        row=1, col=2
    )
    
    # 3. 连通分量大小
    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]
    fig.add_trace(
        go.Bar(
            x=[f"分量{i+1}" for i in range(len(component_sizes))],
            y=component_sizes,
            name="连通分量"
        ),
        row=2, col=1
    )
    
    # 4. 关键指标
    stats = analysis['stats']
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=stats['density'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={"text": "图密度"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 0.9}}
        ),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(
        title_text="知识图谱统计分析仪表板",
        showlegend=False,
        height=800
    )
    
    # 保存文件
    plot(fig, filename=output_file, auto_open=False)
    print(f"✅ 统计仪表板已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='GraphML知识图谱可视化工具')
    parser.add_argument('--input', '-i', default='/mnt/d/git/GraphGen/cache/graph.graphml',
                       help='GraphML输入文件路径')
    parser.add_argument('--output-dir', '-o', default='/mnt/d/git/GraphGen/cache',
                       help='输出目录')
    parser.add_argument('--open-browser', action='store_true',
                       help='生成后自动打开浏览器')
    
    args = parser.parse_args()
    
    print("🚀 GraphML知识图谱可视化工具")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return
    
    # 加载图谱
    G = load_graphml(args.input)
    if G is None:
        return
    
    # 分析图谱
    analysis = analyze_graph(G)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成可视化文件
    output_files = []
    
    # 3D可视化
    output_3d = os.path.join(args.output_dir, "knowledge_graph_3d.html")
    create_3d_visualization(G, output_3d)
    output_files.append(output_3d)
    
    # 2D可视化
    output_2d = os.path.join(args.output_dir, "knowledge_graph_2d.html")
    create_2d_network_visualization(G, output_2d)
    output_files.append(output_2d)
    
    # 统计仪表板
    output_stats = os.path.join(args.output_dir, "graph_statistics.html")
    create_statistics_dashboard(G, analysis, output_stats)
    output_files.append(output_stats)
    
    print(f"\n🎉 可视化完成! 生成了 {len(output_files)} 个HTML文件:")
    for file in output_files:
        print(f"  📄 {file}")
    
    # 自动打开浏览器
    if args.open_browser:
        import webbrowser
        print("\n🌐 正在打开浏览器...")
        for file in output_files:
            webbrowser.open(f"file://{os.path.abspath(file)}")

if __name__ == "__main__":
    main()