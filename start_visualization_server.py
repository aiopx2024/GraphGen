#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动本地HTTP服务器查看知识图谱可视化
"""

import http.server
import socketserver
import webbrowser
import os
import time
import threading

def start_server(port=8080, directory="cache"):
    """启动HTTP服务器"""
    os.chdir(directory)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"🌐 服务器启动成功!")
        print(f"📍 服务地址: http://localhost:{port}")
        print(f"📁 服务目录: {os.getcwd()}")
        print("\n📊 可用的可视化文件:")
        print(f"  🔗 3D可视化: http://localhost:{port}/knowledge_graph_3d.html")
        print(f"  🔗 2D可视化: http://localhost:{port}/knowledge_graph_2d.html") 
        print(f"  🔗 统计分析: http://localhost:{port}/graph_statistics.html")
        print(f"  🔗 原始GraphML: http://localhost:{port}/graph.graphml")
        print("\n💡 提示: 按 Ctrl+C 停止服务器")
        
        # 延迟3秒后自动打开浏览器
        def open_browser():
            time.sleep(3)
            print("\n🚀 正在打开浏览器...")
            try:
                # 打开3D可视化作为默认页面
                webbrowser.open(f"http://localhost:{port}/knowledge_graph_3d.html")
            except Exception as e:
                print(f"⚠️ 无法自动打开浏览器: {e}")
                print("请手动访问上述链接")
        
        # 在后台线程中打开浏览器
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 服务器已停止")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='启动知识图谱可视化服务器')
    parser.add_argument('--port', '-p', type=int, default=8080, help='服务器端口 (默认: 8080)')
    parser.add_argument('--directory', '-d', default='cache', help='服务目录 (默认: cache)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"❌ 目录不存在: {args.directory}")
        exit(1)
    
    print("🚀 知识图谱可视化服务器")
    print("=" * 50)
    
    start_server(args.port, args.directory)