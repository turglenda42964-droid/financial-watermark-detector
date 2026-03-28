#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估器模块

用于评估模型性能和检测效果
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path('results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            model_name: 模型名称
            
        Returns:
            评估指标字典
        """
        logger.info(f"评估模型: {model_name}")
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # 如果有概率预测，计算AUC
        if y_proba is not None:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba_positive)
            except ValueError:
                metrics['auc'] = None
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # 计算特异性
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'] + 1e-10)
        
        logger.info(f"评估结果 - 准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            
        Returns:
            报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append(f"模型评估报告: {metrics['model_name']}")
        report.append("=" * 60)
        report.append("")
        
        # 主要指标
        report.append("主要指标:")
        report.append(f"  准确率 (Accuracy):    {metrics['accuracy']:.4f}")
        report.append(f"  精确率 (Precision):   {metrics['precision']:.4f}")
        report.append(f"  召回率 (Recall):      {metrics['recall']:.4f}")
        report.append(f"  F1 分数:              {metrics['f1']:.4f}")
        report.append(f"  特异性 (Specificity): {metrics['specificity']:.4f}")
        
        if 'auc' in metrics and metrics['auc'] is not None:
            report.append(f"  AUC:                  {metrics['auc']:.4f}")
        
        report.append("")
        
        # 混淆矩阵
        report.append("混淆矩阵:")
        report.append(f"  真阴性 (TN): {metrics['tn']}")
        report.append(f"  假阳性 (FP): {metrics['fp']}")
        report.append(f"  假阴性 (FN): {metrics['fn']}")
        report.append(f"  真阳性 (TP): {metrics['tp']}")
        report.append("")
        
        # 详细分类报告
        report.append("详细分类报告:")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Watermark', 'Watermark'],
            yticklabels=['No Watermark', 'Watermark'],
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        else:
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
        auc = roc_auc_score(y_true, y_proba_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存到: {save_path}")
        else:
            plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR曲线已保存到: {save_path}")
        else:
            plt.savefig(self.output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制特征重要性
        
        Args:
            importance_dict: 特征重要性字典
            top_n: 显示前N个特征
            save_path: 保存路径
        """
        features = list(importance_dict.keys())[:top_n]
        importances = list(importance_dict.values())[:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存到: {save_path}")
        else:
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def compare_models(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            save_path: 保存路径
            
        Returns:
            比较结果DataFrame
        """
        if not self.metrics_history:
            logger.warning("没有可比较的模型")
            return pd.DataFrame()
        
        # 提取关键指标
        comparison_data = []
        for metrics in self.metrics_history:
            comparison_data.append({
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'Specificity': metrics['specificity'],
                'AUC': metrics.get('auc', None),
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'AUC']
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in df.columns:
                df_plot = df.dropna(subset=[metric])
                if not df_plot.empty:
                    axes[idx].bar(df_plot['Model'], df_plot[metric])
                    axes[idx].set_title(metric)
                    axes[idx].set_ylim(0, 1)
                    axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return df
    
    def save_report(self, metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        保存评估报告到文件
        
        Args:
            metrics: 评估指标
            save_path: 保存路径
        """
        report = self.generate_report(metrics)
        
        if save_path is None:
            save_path = self.output_dir / f"{metrics['model_name']}_report.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"评估报告已保存到: {save_path}")
