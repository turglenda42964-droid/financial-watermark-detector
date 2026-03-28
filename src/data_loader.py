#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器模块

负责加载和预处理金融时序数据
"""

from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """金融时序数据加载器"""
    
    def __init__(
        self,
        time_column: str = "timestamp",
        value_columns: Optional[List[str]] = None,
        freq: Optional[str] = None,
    ):
        """
        初始化数据加载器
        
        Args:
            time_column: 时间列名称
            value_columns: 数值列名称列表，None表示自动检测
            freq: 数据频率，如 '1min', '1H', '1D'
        """
        self.time_column = time_column
        self.value_columns = value_columns
        self.freq = freq
        
    def load(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            path: 数据文件路径
            **kwargs: 额外的读取参数
            
        Returns:
            加载的DataFrame
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        logger.info(f"正在加载数据: {path}")
        
        # 根据文件扩展名选择加载方式
        if path.suffix == '.csv':
            df = self._load_csv(path, **kwargs)
        elif path.suffix in ['.xlsx', '.xls']:
            df = self._load_excel(path, **kwargs)
        elif path.suffix == '.parquet':
            df = self._load_parquet(path, **kwargs)
        elif path.suffix == '.pkl':
            df = self._load_pickle(path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        logger.info(f"数据加载完成，形状: {df.shape}")
        
        # 预处理
        df = self._preprocess(df)
        
        return df
    
    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        加载目录中的所有数据文件
        
        Args:
            directory: 数据目录路径
            pattern: 文件匹配模式
            **kwargs: 额外的读取参数
            
        Returns:
            合并后的DataFrame
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        files = list(directory.glob(pattern))
        
        if not files:
            logger.warning(f"在 {directory} 中未找到匹配 {pattern} 的文件")
            return pd.DataFrame()
        
        logger.info(f"找到 {len(files)} 个数据文件")
        
        dfs = []
        for file in files:
            try:
                df = self.load(file, **kwargs)
                dfs.append(df)
            except Exception as e:
                logger.error(f"加载文件 {file} 失败: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"合并后数据形状: {combined.shape}")
        
        return combined
    
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        default_kwargs = {
            'parse_dates': [self.time_column] if self.time_column else True,
            'index_col': self.time_column if self.time_column else None,
        }
        default_kwargs.update(kwargs)
        return pd.read_csv(path, **default_kwargs)
    
    def _load_excel(self, path: Path, **kwargs) -> pd.DataFrame:
        """加载Excel文件"""
        default_kwargs = {
            'parse_dates': [self.time_column] if self.time_column else True,
        }
        default_kwargs.update(kwargs)
        return pd.read_excel(path, **default_kwargs)
    
    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        """加载Parquet文件"""
        return pd.read_parquet(path, **kwargs)
    
    def _load_pickle(self, path: Path, **kwargs) -> pd.DataFrame:
        """加载Pickle文件"""
        return pd.read_pickle(path, **kwargs)
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 确保时间索引
        if self.time_column and self.time_column in df.columns:
            df[self.time_column] = pd.to_datetime(df[self.time_column])
            df = df.set_index(self.time_column)
        
        # 排序
        df = df.sort_index()
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 重采样（如果指定了频率）
        if self.freq:
            df = self._resample(df, self.freq)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 前向填充
        df = df.ffill()
        
        # 后向填充剩余缺失值
        df = df.bfill()
        
        # 删除仍有缺失值的行
        df = df.dropna()
        
        return df
    
    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """重采样数据"""
        logger.info(f"重采样到频率: {freq}")
        
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # OHLCV重采样
        resampled = pd.DataFrame()
        
        if 'open' in df.columns:
            resampled['open'] = df['open'].resample(freq).first()
        if 'high' in df.columns:
            resampled['high'] = df['high'].resample(freq).max()
        if 'low' in df.columns:
            resampled['low'] = df['low'].resample(freq).min()
        if 'close' in df.columns:
            resampled['close'] = df['close'].resample(freq).last()
        if 'volume' in df.columns:
            resampled['volume'] = df['volume'].resample(freq).sum()
        
        # 其他数值列取均值
        for col in numeric_cols:
            if col not in resampled.columns:
                resampled[col] = df[col].resample(freq).mean()
        
        return resampled.dropna()
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        shuffle: bool = False,
    ) -> tuple:
        """
        划分训练集和测试集
        
        Args:
            df: 输入数据
            test_size: 测试集比例
            shuffle: 是否打乱数据（时序数据通常不打乱）
            
        Returns:
            (train_df, test_df) 元组
        """
        if shuffle:
            df = df.sample(frac=1, random_state=42)
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"数据划分: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")
        
        return train_df, test_df
