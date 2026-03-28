#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工厂模块

统一管理各种机器学习模型的创建、训练和预测
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类"""
    
    # 支持的模型类型
    SUPPORTED_MODELS = [
        'xgboost',
        'lightgbm',
        'random_forest',
        'gradient_boosting',
        'logistic_regression',
        'svm',
    ]
    
    def __init__(self, model_type: str = 'xgboost', model_params: Optional[Dict[str, Any]] = None):
        """
        初始化模型工厂
        
        Args:
            model_type: 模型类型
            model_params: 模型参数
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.is_trained = False
        
        logger.info(f"初始化模型工厂，模型类型: {model_type}")
    
    def create_model(self) -> Any:
        """
        创建模型实例
        
        Returns:
            模型实例
        """
        logger.info(f"创建模型: {self.model_type}")
        
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBClassifier(**default_params)
            
        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            default_params.update(self.model_params)
            self.model = lgb.LGBMClassifier(**default_params)
            
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)
            
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            }
            default_params.update(self.model_params)
            self.model = GradientBoostingClassifier(**default_params)
            
        elif self.model_type == 'logistic_regression':
            default_params = {
                'max_iter': 1000,
                'random_state': 42,
                'n_jobs': -1,
            }
            default_params.update(self.model_params)
            self.model = LogisticRegression(**default_params)
            
        elif self.model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42,
            }
            default_params.update(self.model_params)
            self.model = SVC(**default_params, probability=True)
        
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'ModelFactory':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            **fit_params: 额外的训练参数
            
        Returns:
            self
        """
        if self.model is None:
            self.create_model()
        
        logger.info(f"开始训练模型，样本数: {len(X)}, 特征数: {X.shape[1]}")
        
        self.model.fit(X, y, **fit_params)
        self.is_trained = True
        
        logger.info("模型训练完成")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测类别
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        
        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            logger.warning("当前模型不支持特征重要性")
            return {}
        
        # 构建字典
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        
        # 按重要性排序
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def save(self, path: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法保存")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和元数据
        save_dict = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'is_trained': self.is_trained,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"模型已保存到: {path}")
    
    def load(self, path: Union[str, Path]) -> 'ModelFactory':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            self
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.model_type = save_dict['model_type']
        self.model_params = save_dict['model_params']
        self.is_trained = save_dict['is_trained']
        
        logger.info(f"模型已从 {path} 加载")
        
        return self
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModelFactory':
        """
        从配置创建模型工厂
        
        Args:
            config: 配置字典
            
        Returns:
            ModelFactory实例
        """
        model_type = config.get('type', 'xgboost')
        model_params = config.get('params', {})
        
        return cls(model_type=model_type, model_params=model_params)


class EnsembleModel:
    """集成模型"""
    
    def __init__(
        self,
        models: Optional[List[ModelFactory]] = None,
        weights: Optional[List[float]] = None,
        voting: str = 'soft',
    ):
        """
        初始化集成模型
        
        Args:
            models: 模型列表
            weights: 模型权重
            voting: 投票方式，'hard' 或 'soft'
        """
        self.models = models or []
        self.weights = weights
        self.voting = voting
        
        if weights is not None and len(weights) != len(models):
            raise ValueError("权重数量必须与模型数量相同")
    
    def add_model(self, model: ModelFactory, weight: float = 1.0) -> None:
        """添加模型"""
        self.models.append(model)
        
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.models:
            raise RuntimeError("没有可用的模型")
        
        if self.voting == 'hard':
            # 硬投票
            predictions = np.array([model.predict(X) for model in self.models])
            
            if self.weights:
                # 加权投票
                result = []
                for i in range(X.shape[0]):
                    votes = predictions[:, i]
                    weighted_votes = {}
                    for vote, weight in zip(votes, self.weights):
                        weighted_votes[vote] = weighted_votes.get(vote, 0) + weight
                    result.append(max(weighted_votes, key=weighted_votes.get))
                return np.array(result)
            else:
                # 简单多数投票
                from scipy import stats
                return stats.mode(predictions, axis=0)[0].flatten()
        
        else:
            # 软投票
            probas = np.array([model.predict_proba(X) for model in self.models])
            
            if self.weights:
                weights = np.array(self.weights).reshape(-1, 1, 1)
                weighted_probas = probas * weights
                avg_probas = np.sum(weighted_probas, axis=0) / np.sum(self.weights)
            else:
                avg_probas = np.mean(probas, axis=0)
            
            return np.argmax(avg_probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.models:
            raise RuntimeError("没有可用的模型")
        
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        if self.weights:
            weights = np.array(self.weights).reshape(-1, 1, 1)
            weighted_probas = probas * weights
            return np.sum(weighted_probas, axis=0) / np.sum(self.weights)
        else:
            return np.mean(probas, axis=0)
