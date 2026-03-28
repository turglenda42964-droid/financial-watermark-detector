import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats

class Monitor:
    """监控器"""
    
    def __init__(self, config=None):
        """
        初始化监控器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.performance_history = {}
        self.data_drift_history = {}
        self.ab_test_results = {}
        self.metrics_dir = 'monitor_metrics'
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def monitor_performance(self, model_name, y_true, y_pred, y_pred_proba=None):
        """
        监控模型性能
        
        参数:
            model_name: 模型名称
            y_true: 真实标签
            y_pred: 预测结果
            y_pred_proba: 预测概率
            
        返回:
            性能指标
        """
        timestamp = datetime.now().isoformat()
        
        # 计算性能指标
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # 保存性能历史
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        self.performance_history[model_name].append(metrics)
        
        # 保存到文件
        self._save_metrics(f'{model_name}_performance.json', self.performance_history[model_name])
        
        print(f"模型 {model_name} 性能监控完成")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"  AUC分数: {metrics['auc']:.4f}")
        
        return metrics
    
    def detect_data_drift(self, reference_data, current_data, feature_names=None):
        """
        检测数据漂移
        
        参数:
            reference_data: 参考数据
            current_data: 当前数据
            feature_names: 特征名称
            
        返回:
            漂移检测结果
        """
        timestamp = datetime.now().isoformat()
        
        # 确保数据是numpy数组
        if isinstance(reference_data, pd.DataFrame):
            reference_data = reference_data.values
        if isinstance(current_data, pd.DataFrame):
            current_data = current_data.values
        
        # 计算每个特征的漂移
        drift_results = []
        num_features = reference_data.shape[1]
        
        for i in range(num_features):
            feature_name = feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'
            
            # 使用Kolmogorov-Smirnov测试检测分布漂移
            try:
                stat, p_value = stats.kstest(reference_data[:, i], current_data[:, i])
                is_drifted = p_value < 0.05
                
                drift_results.append({
                    'feature': feature_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'is_drifted': is_drifted
                })
            except Exception as e:
                drift_results.append({
                    'feature': feature_name,
                    'error': str(e)
                })
        
        # 计算整体漂移
        try:
            # 使用Mann-Whitney U测试检测整体漂移
            stat, p_value = stats.mannwhitneyu(reference_data.flatten(), current_data.flatten())
            overall_drift = p_value < 0.05
        except Exception as e:
            overall_drift = False
            p_value = None
            stat = None
        
        result = {
            'timestamp': timestamp,
            'overall_drift': overall_drift,
            'overall_p_value': p_value,
            'overall_statistic': stat,
            'feature_drifts': drift_results
        }
        
        # 保存漂移历史
        self.data_drift_history[timestamp] = result
        
        # 保存到文件
        self._save_metrics('data_drift.json', self.data_drift_history)
        
        print("数据漂移检测完成")
        print(f"  整体漂移: {'是' if overall_drift else '否'}")
        if p_value is not None:
            print(f"  p值: {p_value:.4f}")
        
        # 打印漂移的特征
        drifted_features = [f['feature'] for f in drift_results if f.get('is_drifted', False)]
        if drifted_features:
            print(f"  漂移的特征: {drifted_features}")
        else:
            print("  没有检测到特征漂移")
        
        return result
    
    def run_ab_test(self, model_a, model_b, X, y, test_name=None):
        """
        运行A/B测试
        
        参数:
            model_a: 模型A
            model_b: 模型B
            X: 测试数据
            y: 真实标签
            test_name: 测试名称
            
        返回:
            A/B测试结果
        """
        timestamp = datetime.now().isoformat()
        test_name = test_name or f'ab_test_{int(datetime.now().timestamp())}'
        
        # 模型A预测
        y_pred_a = model_a.predict(X)
        y_pred_proba_a = model_a.predict_proba(X)[:, 1] if hasattr(model_a, 'predict_proba') else None
        
        # 模型B预测
        y_pred_b = model_b.predict(X)
        y_pred_proba_b = model_b.predict_proba(X)[:, 1] if hasattr(model_b, 'predict_proba') else None
        
        # 计算模型A的性能
        metrics_a = {
            'accuracy': accuracy_score(y, y_pred_a),
            'precision': precision_score(y, y_pred_a, zero_division=0),
            'recall': recall_score(y, y_pred_a, zero_division=0),
            'f1': f1_score(y, y_pred_a, zero_division=0)
        }
        
        if y_pred_proba_a is not None and len(np.unique(y)) > 1:
            metrics_a['auc'] = roc_auc_score(y, y_pred_proba_a)
        
        # 计算模型B的性能
        metrics_b = {
            'accuracy': accuracy_score(y, y_pred_b),
            'precision': precision_score(y, y_pred_b, zero_division=0),
            'recall': recall_score(y, y_pred_b, zero_division=0),
            'f1': f1_score(y, y_pred_b, zero_division=0)
        }
        
        if y_pred_proba_b is not None and len(np.unique(y)) > 1:
            metrics_b['auc'] = roc_auc_score(y, y_pred_proba_b)
        
        # 计算性能差异
        performance_diff = {}
        for metric in metrics_a:
            if metric in metrics_b:
                performance_diff[metric] = metrics_b[metric] - metrics_a[metric]
        
        # 确定获胜模型
        if metrics_b['f1'] > metrics_a['f1']:
            winner = 'model_b'
        elif metrics_b['f1'] < metrics_a['f1']:
            winner = 'model_a'
        else:
            winner = 'tie'
        
        result = {
            'test_name': test_name,
            'timestamp': timestamp,
            'model_a_metrics': metrics_a,
            'model_b_metrics': metrics_b,
            'performance_diff': performance_diff,
            'winner': winner
        }
        
        # 保存A/B测试结果
        self.ab_test_results[test_name] = result
        
        # 保存到文件
        self._save_metrics('ab_test_results.json', self.ab_test_results)
        
        print(f"A/B测试 {test_name} 完成")
        print("模型A性能:")
        for metric, value in metrics_a.items():
            print(f"  {metric}: {value:.4f}")
        print("模型B性能:")
        for metric, value in metrics_b.items():
            print(f"  {metric}: {value:.4f}")
        print("性能差异:")
        for metric, diff in performance_diff.items():
            print(f"  {metric}: {diff:+.4f}")
        print(f"获胜模型: {winner}")
        
        return result
    
    def get_performance_history(self, model_name=None):
        """
        获取性能历史
        
        参数:
            model_name: 模型名称
            
        返回:
            性能历史
        """
        if model_name:
            return self.performance_history.get(model_name, [])
        return self.performance_history
    
    def get_data_drift_history(self):
        """
        获取数据漂移历史
        
        返回:
            数据漂移历史
        """
        return self.data_drift_history
    
    def get_ab_test_results(self, test_name=None):
        """
        获取A/B测试结果
        
        参数:
            test_name: 测试名称
            
        返回:
            A/B测试结果
        """
        if test_name:
            return self.ab_test_results.get(test_name, {})
        return self.ab_test_results
    
    def _save_metrics(self, filename, data):
        """
        保存指标到文件
        
        参数:
            filename: 文件名
            data: 数据
        """
        file_path = os.path.join(self.metrics_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_metrics(self, filename):
        """
        从文件加载指标
        
        参数:
            filename: 文件名
            
        返回:
            加载的数据
        """
        file_path = os.path.join(self.metrics_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
