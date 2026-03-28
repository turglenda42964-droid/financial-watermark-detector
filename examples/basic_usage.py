#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础使用示例

展示如何使用金融时序数据水印检测系统的基本功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.detector import WatermarkDetector
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.model_factory import ModelFactory
from src.evaluator import Evaluator


def example_1_basic_detection():
    """示例1: 基础检测流程"""
    print("=" * 60)
    print("示例1: 基础检测流程")
    print("=" * 60)
    
    # 1. 创建检测器（使用默认配置）
    detector = WatermarkDetector()
    
    # 2. 训练模型
    print("\n1. 训练模型...")
    detector.train('data/sample/sample_data.csv', validation_split=0.3)
    
    # 3. 执行检测
    print("\n2. 执行检测...")
    result = detector.detect('data/sample/sample_data.csv')
    
    # 4. 查看结果
    print("\n3. 检测结果:")
    print(result.summary())
    
    # 5. 保存结果
    result.save('results/example1')
    print("\n4. 结果已保存到 results/example1")


def example_2_custom_config():
    """示例2: 使用自定义配置"""
    print("\n" + "=" * 60)
    print("示例2: 使用自定义配置")
    print("=" * 60)
    
    # 自定义配置
    config = {
        'data': {
            'time_column': 'timestamp',
            'value_column': 'close',
            'label_column': 'watermark_label',
        },
        'features': {
            'window_size': 5,
            'step_size': 1,
            'feature_groups': ['statistical', 'time_domain', 'watermark_specific'],
        },
        'model': {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
            }
        },
        'output': {
            'save_model': True,
        }
    }
    
    # 创建检测器
    detector = WatermarkDetector(config=config)
    
    # 训练
    print("\n使用LightGBM训练模型...")
    detector.train('data/sample/sample_data.csv')
    
    # 评估
    print("\n评估模型...")
    metrics = detector.evaluate('data/sample/sample_data.csv')
    
    print(f"\n准确率: {metrics['accuracy']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")


def example_3_feature_extraction():
    """示例3: 特征提取"""
    print("\n" + "=" * 60)
    print("示例3: 特征提取")
    print("=" * 60)
    
    # 加载数据
    loader = DataLoader(time_column='timestamp')
    df = loader.load('data/sample/sample_data.csv')
    
    print(f"\n原始数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 创建特征提取器
    extractor = FeatureExtractor(
        window_size=5,
        step_size=1,
        feature_groups=['statistical', 'time_domain']
    )
    
    # 提取特征
    print("\n提取特征...")
    features_df = extractor.extract(df, target_column='close')
    
    print(f"\n特征数据形状: {features_df.shape}")
    print(f"\n前5个特征:")
    feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
    for col in feature_cols[:5]:
        print(f"  - {col}")
    
    # 查看特征统计
    print("\n特征统计:")
    print(features_df[feature_cols].describe())


def example_4_model_comparison():
    """示例4: 模型对比"""
    print("\n" + "=" * 60)
    print("示例4: 模型对比")
    print("=" * 60)
    
    # 准备数据
    loader = DataLoader(time_column='timestamp')
    df = loader.load('data/sample/sample_data.csv')
    
    extractor = FeatureExtractor(
        window_size=5,
        step_size=1,
        feature_groups=['statistical', 'time_domain']
    )
    
    features_df = extractor.extract(df, target_column='close')
    feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
    X = features_df[feature_cols].values
    
    # 获取标签
    y = []
    for _, row in features_df.iterrows():
        window_data = df.loc[row['window_start']:row['window_end']]
        if 'watermark_label' in window_data.columns:
            label = window_data['watermark_label'].mode()[0]
        else:
            label = 0
        y.append(label)
    y = np.array(y)
    
    # 对比不同模型
    from sklearn.model_selection import cross_val_score
    
    models = {
        'Random Forest': ModelFactory('random_forest', {'n_estimators': 50}),
        'XGBoost': ModelFactory('xgboost', {'n_estimators': 50}),
        'LightGBM': ModelFactory('lightgbm', {'n_estimators': 50}),
    }
    
    print("\n模型对比结果:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 60)
    
    for name, model_factory in models.items():
        model_factory.fit(X, y)
        
        # 交叉验证
        scores_acc = cross_val_score(model_factory.model, X, y, cv=3, scoring='accuracy')
        scores_f1 = cross_val_score(model_factory.model, X, y, cv=3, scoring='f1')
        
        print(f"{name:<20} {scores_acc.mean():.4f}       {scores_f1.mean():.4f}")


def example_5_load_and_predict():
    """示例5: 加载模型并预测"""
    print("\n" + "=" * 60)
    print("示例5: 加载模型并预测")
    print("=" * 60)
    
    # 先训练并保存模型
    print("\n1. 训练并保存模型...")
    detector = WatermarkDetector()
    detector.train('data/sample/sample_data.csv')
    
    # 保存模型
    import os
    os.makedirs('models', exist_ok=True)
    detector.model_factory.save('models/example_model.pkl')
    print("模型已保存到 models/example_model.pkl")
    
    # 创建新检测器并加载模型
    print("\n2. 加载模型...")
    new_detector = WatermarkDetector()
    new_detector.load_model('models/example_model.pkl')
    
    # 使用加载的模型进行预测
    print("\n3. 使用加载的模型进行预测...")
    result = new_detector.detect('data/sample/sample_data.csv')
    print(result.summary())


def example_6_batch_processing():
    """示例6: 批量处理"""
    print("\n" + "=" * 60)
    print("示例6: 批量处理")
    print("=" * 60)
    
    # 加载目录中的所有数据
    print("\n1. 加载数据目录...")
    loader = DataLoader(time_column='timestamp')
    
    # 注意：这里假设sample目录中有多个文件
    # 实际使用时请确保目录中有多个CSV文件
    try:
        df = loader.load_directory('data/sample', pattern='*.csv')
        print(f"加载了 {len(df)} 条数据")
    except Exception as e:
        print(f"加载目录失败: {e}")
        print("使用单个文件代替...")
        df = loader.load('data/sample/sample_data.csv')
    
    # 批量处理
    print("\n2. 批量处理...")
    detector = WatermarkDetector()
    detector.train(df, validation_split=0.3)
    result = detector.detect(df)
    
    print("\n3. 批量处理结果:")
    print(result.summary())


if __name__ == '__main__':
    print("金融时序数据水印检测系统 - 使用示例")
    print("=" * 60)
    
    # 运行所有示例
    try:
        example_1_basic_detection()
    except Exception as e:
        print(f"示例1失败: {e}")
    
    try:
        example_2_custom_config()
    except Exception as e:
        print(f"示例2失败: {e}")
    
    try:
        example_3_feature_extraction()
    except Exception as e:
        print(f"示例3失败: {e}")
    
    try:
        example_4_model_comparison()
    except Exception as e:
        print(f"示例4失败: {e}")
    
    try:
        example_5_load_and_predict()
    except Exception as e:
        print(f"示例5失败: {e}")
    
    try:
        example_6_batch_processing()
    except Exception as e:
        print(f"示例6失败: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
