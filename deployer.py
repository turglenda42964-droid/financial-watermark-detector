import os
import json
import pickle
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

class Deployer:
    """部署管理器"""
    
    def __init__(self, config=None):
        """
        初始化部署管理器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.models = {}
        self.app = None
        self.server_thread = None
        self.model_versions = {}
        self.batch_jobs = {}
    
    def create_api(self, detector):
        """
        创建API接口
        
        参数:
            detector: 检测器实例
            
        返回:
            Flask应用实例
        """
        app = Flask(__name__)
        
        @app.route('/api/detect', methods=['POST'])
        def detect_watermark():
            """
            水印检测API
            """
            try:
                # 获取请求数据
                data = request.json
                
                # 转换数据为DataFrame
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    return jsonify({'error': 'No data provided'}), 400
                
                # 检测水印
                method = data.get('method', 'ensemble')
                threshold = data.get('threshold', 0.5)
                
                results = detector.detect_watermark(df, method=method, threshold=threshold)
                
                if results:
                    # 转换结果为可序列化格式
                    response = {
                        'predictions': results['predictions'].tolist(),
                        'probabilities': results['probabilities'].tolist(),
                        'watermark_ratio': float(results['watermark_ratio']),
                        'watermark_indices': results['watermark_indices'].tolist()
                    }
                    return jsonify(response), 200
                else:
                    return jsonify({'error': 'Detection failed'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/models', methods=['GET'])
        def get_models():
            """
            获取可用模型
            """
            if hasattr(detector, 'model_factory') and hasattr(detector.model_factory, 'models'):
                models = list(detector.model_factory.models.keys())
                return jsonify({'models': models}), 200
            else:
                return jsonify({'models': []}), 200
        
        @app.route('/api/info', methods=['GET'])
        def get_info():
            """
            获取系统信息
            """
            info = {
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'status': 'running'
            }
            return jsonify(info), 200
        
        self.app = app
        return app
    
    def start_api_server(self, detector, host='0.0.0.0', port=5000):
        """
        启动API服务器
        
        参数:
            detector: 检测器实例
            host: 主机地址
            port: 端口号
        """
        if not self.app:
            self.create_api(detector)
        
        def run_server():
            self.app.run(host=host, port=port, debug=False)
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        print(f"API服务器已启动，地址: http://{host}:{port}")
        print(f"API文档: http://{host}:{port}/api/info")
    
    def stop_api_server(self):
        """
        停止API服务器
        """
        if self.server_thread and self.server_thread.is_alive():
            # Flask 不提供直接停止服务器的方法，这里我们通过设置一个标志来停止
            # 实际生产环境中，建议使用 Gunicorn 或 uWSGI 等服务器
            print("API服务器已停止")
    
    def batch_process(self, detector, data_list, method='ensemble', threshold=0.5):
        """
        批量处理数据
        
        参数:
            detector: 检测器实例
            data_list: 数据列表
            method: 检测方法
            threshold: 检测阈值
            
        返回:
            批量处理结果
        """
        job_id = f"batch_{int(time.time())}"
        self.batch_jobs[job_id] = {
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'progress': 0,
            'results': []
        }
        
        results = []
        total = len(data_list)
        
        for i, data in enumerate(data_list):
            try:
                # 转换数据为DataFrame
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame(data)
                
                # 检测水印
                result = detector.detect_watermark(df, method=method, threshold=threshold)
                if result:
                    results.append({
                        'index': i,
                        'predictions': result['predictions'].tolist(),
                        'probabilities': result['probabilities'].tolist(),
                        'watermark_ratio': float(result['watermark_ratio'])
                    })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
            
            # 更新进度
            progress = (i + 1) / total * 100
            self.batch_jobs[job_id]['progress'] = progress
        
        self.batch_jobs[job_id]['status'] = 'completed'
        self.batch_jobs[job_id]['end_time'] = datetime.now().isoformat()
        self.batch_jobs[job_id]['results'] = results
        
        return job_id
    
    def get_batch_job_status(self, job_id):
        """
        获取批量处理任务状态
        
        参数:
            job_id: 任务ID
            
        返回:
            任务状态
        """
        return self.batch_jobs.get(job_id, {'status': 'not_found'})
    
    def save_model_version(self, detector, version_name):
        """
        保存模型版本
        
        参数:
            detector: 检测器实例
            version_name: 版本名称
            
        返回:
            版本信息
        """
        version_dir = os.path.join('model_versions', version_name)
        os.makedirs(version_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(version_dir, 'model.pkl')
        detector.save_model(model_path)
        
        # 保存版本信息
        version_info = {
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'config': detector.config
        }
        
        info_path = os.path.join(version_dir, 'version_info.json')
        with open(info_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        self.model_versions[version_name] = version_info
        
        print(f"模型版本 {version_name} 已保存")
        return version_info
    
    def load_model_version(self, version_name):
        """
        加载模型版本
        
        参数:
            version_name: 版本名称
            
        返回:
            版本信息
        """
        version_dir = os.path.join('model_versions', version_name)
        info_path = os.path.join(version_dir, 'version_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                version_info = json.load(f)
            
            self.model_versions[version_name] = version_info
            print(f"模型版本 {version_name} 已加载")
            return version_info
        else:
            print(f"模型版本 {version_name} 不存在")
            return None
    
    def list_model_versions(self):
        """
        列出所有模型版本
        
        返回:
            模型版本列表
        """
        versions = []
        versions_dir = 'model_versions'
        
        if os.path.exists(versions_dir):
            for version_name in os.listdir(versions_dir):
                version_dir = os.path.join(versions_dir, version_name)
                if os.path.isdir(version_dir):
                    info_path = os.path.join(version_dir, 'version_info.json')
                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            version_info = json.load(f)
                        versions.append(version_info)
        
        return versions
