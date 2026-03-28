#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主检测器模块

整合所有功能，提供统一的检测接口
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .model_factory import ModelFactory
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class DetectionResult:
    """检测结果类"""
    
    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化检测结果
        
        Args:
            predictions: 预测标签
            probabilities: 预测概率
            feature_importance: 特征重要性
            metadata: 元数据
        """
        self.predictions = predictions
        self.probabilities = probabilities
        self.feature_importance = feature_importance or {}
        self.metadata = metadata or {}
    
    def summary(self) -> str:
        """生成结果摘要"""
        watermark_count = np.sum(self.predictions)
        total_count = len(self.predictions)
        watermark_ratio = watermark_count / total_count
        
        avg_confidence = np.mean(np.max(self.probabilities, axis=1))
        
        summary = []
        summary.append("=" * 60)
        summary.append("检测结果摘要")
        summary.append("=" * 60)
        summary.append(f"总样本数: {total_count}")
        summary.append(f"检测到水印: {watermark_count} ({watermark_ratio*100:.2f}%)")
        summary.append(f"平均置信度: {avg_confidence:.4f}")
        summary.append("=" * 60)
        
        return "\n".join(summary)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        df = pd.DataFrame({
            'prediction': self.predictions,
            'probability_no_watermark': self.probabilities[:, 0],
            'probability_watermark': self.probabilities[:, 1],
        })
        return df
    
    def save(self, output_dir: Union[str, Path]) -> None:
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果
        df = self.to_dataframe()
        df.to_csv(output_dir / 'predictions.csv', index=False)
        
        # 保存摘要
        with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write(self.summary())
        
        logger.info(f"结果已保存到: {output_dir}")


class WatermarkDetector:
    """水印检测器主类"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """
        初始化检测器
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        if config_path:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            self.config = self._default_config()
        
        # 初始化组件
        self.data_loader = self._init_data_loader()
        self.feature_extractor = self._init_feature_extractor()
        self.model_factory = self._init_model_factory()
        self.evaluator = self._init_evaluator()
        
        self.is_trained = False
        
        logger.info("水印检测器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'data': {
                'time_column': 'timestamp',
                'value_column': 'close',
                'label_column': 'watermark_label',
            },
            'features': {
                'window_size': 60,
                'step_size': 10,
                'feature_groups': [
                    'statistical', 'time_domain', 'frequency_domain',
                    'complexity', 'watermark_specific'
                ],
            },
            'model': {
                'type': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                },
            },
            'detection': {
                'threshold': 0.5,
                'batch_size': 1000,
            },
            'output': {
                'save_features': True,
                'save_model': True,
            },
        }
    
    def _init_data_loader(self) -> DataLoader:
        """初始化数据加载器"""
        data_config = self.config.get('data', {})
        return DataLoader(
            time_column=data_config.get('time_column', 'timestamp'),
        )
    
    def _init_feature_extractor(self) -> FeatureExtractor:
        """初始化特征提取器"""
        feature_config = self.config.get('features', {})
        return FeatureExtractor(
            window_size=feature_config.get('window_size', 60),
            step_size=feature_config.get('step_size', 10),
            feature_groups=feature_config.get('feature_groups'),
        )
    
    def _init_model_factory(self) -> ModelFactory:
        """初始化模型工厂"""
        model_config = self.config.get('model', {})
        return ModelFactory.from_config(model_config)
    
    def _init_evaluator(self) -> Evaluator:
        """初始化评估器"""
        return Evaluator(output_dir='results')
    
    def train(
        self,
        data_path: Union[str, Path, pd.DataFrame],
        validation_split: float = 0.2,
    ) -> 'WatermarkDetector':
        """
        训练模型
        
        Args:
            data_path: 数据路径或DataFrame
            validation_split: 验证集比例
            
        Returns:
            self
        """
        logger.info("开始训练模型...")
        
        # 加载数据
        if isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            df = self.data_loader.load(data_path)
        
        # 提取特征
        value_column = self.config['data'].get('value_column', 'close')
        features_df = self.feature_extractor.extract(df, target_column=value_column)
        
        # 准备训练数据
        label_column = self.config['data'].get('label_column', 'watermark_label')
        
        # 从原始数据中获取标签
        feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
        X = features_df[feature_cols].values
        
        # 为每个窗口分配标签（使用窗口中点的标签）
        y = []
        for _, row in features_df.iterrows():
            window_data = df.loc[row['window_start']:row['window_end']]
            if label_column in window_data.columns:
                label = window_data[label_column].mode()[0]  # 使用众数
            else:
                label = 0  # 默认无水印
            y.append(label)
        
        y = np.array(y)
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # 标准化
        X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
        X_val_scaled = self.feature_extractor.scaler.transform(X_val)
        
        # 训练模型
        self.model_factory.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # 验证
        y_val_pred = self.model_factory.predict(X_val_scaled)
        y_val_proba = self.model_factory.predict_proba(X_val_scaled)
        
        metrics = self.evaluator.evaluate(
            y_val, y_val_pred, y_val_proba,
            model_name=self.config['model'].get('type', 'model')
        )
        
        # 打印报告
        report = self.evaluator.generate_report(metrics)
        print(report)
        
        # 保存模型
        if self.config.get('output', {}).get('save_model', True):
            model_path = Path('models') / f"{self.config['model'].get('type', 'model')}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_factory.save(model_path)
        
        logger.info("模型训练完成")
        
        return self
    
    def detect(
        self,
        data_path: Union[str, Path, pd.DataFrame],
    ) -> DetectionResult:
        """
        执行检测
        
        Args:
            data_path: 数据路径或DataFrame
            
        Returns:
            检测结果
        """
        if not self.is_trained:
            logger.warning("模型尚未训练，检测结果可能不准确")
        
        logger.info("开始检测水印...")
        
        # 加载数据
        if isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            df = self.data_loader.load(data_path)
        
        # 提取特征
        value_column = self.config['data'].get('value_column', 'close')
        features_df = self.feature_extractor.extract(df, target_column=value_column)
        
        feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
        X = features_df[feature_cols].values
        
        # 标准化
        X_scaled = self.feature_extractor.scaler.transform(X)
        
        # 预测
        predictions = self.model_factory.predict(X_scaled)
        probabilities = self.model_factory.predict_proba(X_scaled)
        
        # 获取特征重要性
        try:
            feature_importance = self.model_factory.get_feature_importance(feature_cols)
        except Exception:
            feature_importance = {}
        
        # 构建结果
        result = DetectionResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_importance=feature_importance,
            metadata={
                'window_starts': features_df['window_start'].tolist(),
                'window_ends': features_df['window_end'].tolist(),
            },
        )
        
        logger.info("检测完成")
        print(result.summary())
        
        return result
    
    def evaluate(
        self,
        data_path: Union[str, Path, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            data_path: 数据路径或DataFrame
            
        Returns:
            评估指标
        """
        logger.info("开始评估模型...")
        
        # 加载数据
        if isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            df = self.data_loader.load(data_path)
        
        # 提取特征
        value_column = self.config['data'].get('value_column', 'close')
        features_df = self.feature_extractor.extract(df, target_column=value_column)
        
        feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
        X = features_df[feature_cols].values
        
        # 获取标签
        label_column = self.config['data'].get('label_column', 'watermark_label')
        y = []
        for _, row in features_df.iterrows():
            window_data = df.loc[row['window_start']:row['window_end']]
            if label_column in window_data.columns:
                label = window_data[label_column].mode()[0]
            else:
                label = 0
            y.append(label)
        
        y = np.array(y)
        
        # 标准化
        X_scaled = self.feature_extractor.scaler.transform(X)
        
        # 预测
        y_pred = self.model_factory.predict(X_scaled)
        y_proba = self.model_factory.predict_proba(X_scaled)
        
        # 评估
        metrics = self.evaluator.evaluate(
            y, y_pred, y_proba,
            model_name=self.config['model'].get('type', 'model')
        )
        
        # 生成可视化
        self.evaluator.plot_confusion_matrix(y, y_pred)
        self.evaluator.plot_roc_curve(y, y_proba)
        self.evaluator.plot_precision_recall_curve(y, y_proba)
        
        if self.model_factory.get_feature_importance():
            self.evaluator.plot_feature_importance(
                self.model_factory.get_feature_importance(feature_cols)
            )
        
        # 保存报告
        self.evaluator.save_report(metrics)
        
        logger.info("评估完成")
        
        return metrics
    
    def save_results(
        self,
        result: DetectionResult,
        output_dir: Union[str, Path],
    ) -> None:
        """
        保存检测结果
        
        Args:
            result: 检测结果
            output_dir: 输出目录
        """
        result.save(output_dir)
    
    def load_model(self, model_path: Union[str, Path]) -> 'WatermarkDetector':
        """
        加载预训练模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            self
        """
        self.model_factory.load(model_path)
        self.is_trained = True
        logger.info(f"模型已从 {model_path} 加载")
        return self
