import os
import pandas as pd
import numpy as np

class DataLoader:
    """数据加载器"""
    
    def __init__(self, config=None):
        """
        初始化数据加载器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
    
    def load_data(self, data_path, label_col='watermark', time_col=None):
        """
        加载数据
        
        参数:
            data_path: 数据文件路径
            label_col: 标签列名
            time_col: 时间列名
            
        返回:
            data: 加载的数据
            X: 特征数据
            y: 标签数据
        """
        print(f"正在加载数据: {data_path}")
        
        # 根据文件扩展名选择加载方式
        file_ext = os.path.splitext(data_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                data = pd.read_csv(data_path)
            elif file_ext == '.pkl' or file_ext == '.pickle':
                data = pd.read_pickle(data_path)
            elif file_ext == '.parquet':
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            print("将生成示例数据...")
            return self._generate_sample_data()
        
        print(f"数据加载成功，形状: {data.shape}")
        print(f"数据列: {list(data.columns)}")
        
        # 分离特征和标签
        if label_col in data.columns:
            X = data.drop(columns=[label_col])
            y = data[label_col]
            print(f"水印样本数: {sum(y)}/{len(y)} ({(sum(y)/len(y))*100:.2f}%)")
        else:
            X = data
            y = None
            print("警告: 未找到标签列，将进行无监督检测")
        
        return data, X, y
    
    def load_data_from_dataframe(self, dataframe, label_col='watermark'):
        """
        从DataFrame加载数据
        
        参数:
            dataframe: DataFrame数据
            label_col: 标签列名
            
        返回:
            data: 加载的数据
            X: 特征数据
            y: 标签数据
        """
        data = dataframe.copy()
        
        # 分离特征和标签
        if label_col in data.columns:
            X = data.drop(columns=[label_col])
            y = data[label_col]
            print(f"水印样本数: {sum(y)}/{len(y)} ({(sum(y)/len(y))*100:.2f}%)")
        else:
            X = data
            y = None
            print("警告: 未找到标签列，将进行无监督检测")
        
        return data, X, y
    
    def _generate_sample_data(self):
        """
        生成示例数据
        
        返回:
            data: 生成的示例数据
            X: 特征数据
            y: 标签数据
        """
        print("生成示例数据...")
        n_samples = 1000
        n_features = 50
        
        # 生成示例数据
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # 添加水印模式（示例）
        watermark_pattern = np.sin(np.linspace(0, 4*np.pi, n_features)) * 0.3
        n_watermark = n_samples // 2
        
        # 为一半样本添加水印
        for i in range(n_watermark):
            X[i] += watermark_pattern + np.random.randn(n_features) * 0.1
        
        # 创建标签
        y = np.zeros(n_samples)
        y[:n_watermark] = 1
        
        # 创建DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_cols)
        data['watermark'] = y
        
        print(f"生成示例数据: {n_samples}个样本, {n_features}个特征")
        print(f"水印样本: {n_watermark}/{n_samples}")
        
        return data, data[feature_cols], data['watermark']
