#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
"""

from typing import Optional, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_time_series(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Time Series",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制时序图
    
    Args:
        df: 数据框
        columns: 要绘制的列
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        ax.plot(df.index, df[col], label=col, alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_ohlc(
    df: pd.DataFrame,
    title: str = "OHLC Chart",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制K线图
    
    Args:
        df: 包含OHLC数据的DataFrame
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算颜色
    colors = ['green' if close >= open else 'red'
              for open, close in zip(df['open'], df['close'])]
    
    # 绘制K线
    for i, (idx, row) in enumerate(df.iterrows()):
        # 实体
        height = abs(row['close'] - row['open'])
        bottom = min(row['close'], row['open'])
        ax.bar(i, height, bottom=bottom, color=colors[i], width=0.8)
        
        # 影线
        ax.plot([i, i], [row['low'], row['high']], color=colors[i], linewidth=1)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.set_xticks(range(0, len(df), max(1, len(df) // 10)))
    ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d') 
                        for i in range(0, len(df), max(1, len(df) // 10))],
                       rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_feature_distribution(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制特征分布图
    
    Args:
        df: 数据框
        columns: 要绘制的列
        figsize: 图表大小
        save_path: 保存路径
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:12]
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    # 隐藏多余的子图
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制相关性矩阵
    
    Args:
        df: 数据框
        figsize: 图表大小
        save_path: 保存路径
    """
    # 计算相关性
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.8},
    )
    
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_watermark_detection_result(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    value_column: str = 'close',
    figsize: tuple = (15, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制水印检测结果
    
    Args:
        df: 原始数据
        predictions: 预测结果
        probabilities: 预测概率
        value_column: 数值列名
        figsize: 图表大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 上图：价格和检测结果
    ax1 = axes[0]
    ax1.plot(df.index, df[value_column], label='Price', alpha=0.7)
    
    # 标记检测到的水印位置
    watermark_indices = np.where(predictions == 1)[0]
    if len(watermark_indices) > 0:
        ax1.scatter(
            df.index[watermark_indices],
            df[value_column].iloc[watermark_indices],
            color='red',
            label='Watermark Detected',
            s=50,
            zorder=5,
        )
    
    ax1.set_ylabel('Price')
    ax1.set_title('Watermark Detection Result')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：置信度
    ax2 = axes[1]
    ax2.fill_between(
        range(len(probabilities)),
        probabilities[:, 1],
        alpha=0.5,
        label='Watermark Probability',
    )
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: dict,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        figsize: 图表大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 损失曲线
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
