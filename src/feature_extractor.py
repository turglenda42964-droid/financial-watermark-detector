#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取器模块

从金融时序数据中提取多维度特征用于水印检测
"""

from typing import List, Optional, Dict, Any, Union
import logging

import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """金融时序数据特征提取器"""
    
    def __init__(
        self,
        window_size: int = 60,
        step_size: int = 10,
        feature_groups: Optional[List[str]] = None,
    ):
        """
        初始化特征提取器
        
        Args:
            window_size: 滑动窗口大小
            step_size: 滑动步长
            feature_groups: 启用的特征组，可选: ['statistical', 'time_domain', 
                          'frequency_domain', 'complexity', 'watermark_specific']
        """
        self.window_size = window_size
        self.step_size = step_size
        self.feature_groups = feature_groups or [
            'statistical', 'time_domain', 'frequency_domain', 
            'complexity', 'watermark_specific'
        ]
        
        self.scaler = StandardScaler()
        
    def extract(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
    ) -> pd.DataFrame:
        """
        提取特征
        
        Args:
            df: 输入数据框
            target_column: 目标列名（用于提取特征的价格列）
            
        Returns:
            特征数据框
        """
        logger.info(f"开始提取特征，窗口大小: {self.window_size}, 步长: {self.step_size}")
        
        features_list = []
        
        # 滑动窗口提取特征
        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            window = df.iloc[i:i + self.window_size]
            
            feature_dict = {}
            feature_dict['window_start'] = window.index[0]
            feature_dict['window_end'] = window.index[-1]
            
            # 提取各组特征
            if 'statistical' in self.feature_groups:
                feature_dict.update(self._extract_statistical_features(window, target_column))
            
            if 'time_domain' in self.feature_groups:
                feature_dict.update(self._extract_time_domain_features(window, target_column))
            
            if 'frequency_domain' in self.feature_groups:
                feature_dict.update(self._extract_frequency_domain_features(window, target_column))
            
            if 'complexity' in self.feature_groups:
                feature_dict.update(self._extract_complexity_features(window, target_column))
            
            if 'watermark_specific' in self.feature_groups:
                feature_dict.update(self._extract_watermark_specific_features(window, target_column))
            
            features_list.append(feature_dict)
        
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"特征提取完成，特征数量: {len(features_df.columns) - 2}, 样本数量: {len(features_df)}")
        
        return features_df
    
    def _extract_statistical_features(
        self,
        window: pd.DataFrame,
        column: str,
    ) -> Dict[str, float]:
        """提取统计特征"""
        values = window[column].values
        
        features = {
            'stat_mean': np.mean(values),
            'stat_std': np.std(values),
            'stat_var': np.var(values),
            'stat_min': np.min(values),
            'stat_max': np.max(values),
            'stat_range': np.max(values) - np.min(values),
            'stat_median': np.median(values),
            'stat_skewness': stats.skew(values),
            'stat_kurtosis': stats.kurtosis(values),
            'stat_percentile_25': np.percentile(values, 25),
            'stat_percentile_75': np.percentile(values, 75),
            'stat_iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'stat_cv': np.std(values) / (np.mean(values) + 1e-10),  # 变异系数
        }
        
        return features
    
    def _extract_time_domain_features(
        self,
        window: pd.DataFrame,
        column: str,
    ) -> Dict[str, float]:
        """提取时域特征"""
        values = window[column].values
        
        # 计算收益率
        returns = np.diff(values) / (values[:-1] + 1e-10)
        
        features = {
            'td_return_mean': np.mean(returns),
            'td_return_std': np.std(returns),
            'td_return_skew': stats.skew(returns),
            'td_return_kurt': stats.kurtosis(returns),
            'td_positive_ratio': np.sum(returns > 0) / len(returns),
            'td_max_drawdown': self._calculate_max_drawdown(values),
        }
        
        # 趋势特征
        x = np.arange(len(values))
        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        features['td_trend_slope'] = slope
        features['td_trend_r2'] = r_value ** 2
        
        # 自相关特征
        features['td_autocorr_lag1'] = pd.Series(values).autocorr(lag=1) or 0
        features['td_autocorr_lag5'] = pd.Series(values).autocorr(lag=5) or 0
        
        return features
    
    def _extract_frequency_domain_features(
        self,
        window: pd.DataFrame,
        column: str,
    ) -> Dict[str, float]:
        """提取频域特征"""
        values = window[column].values
        n = len(values)
        
        # FFT
        fft_values = fft(values)
        fft_magnitude = np.abs(fft_values[:n//2])
        freqs = fftfreq(n, d=1)[:n//2]
        
        # 功率谱密度
        freqs_psd, psd = welch(values, fs=1.0, nperseg=min(n, 256))
        
        features = {
            'fd_fft_mean': np.mean(fft_magnitude),
            'fd_fft_std': np.std(fft_magnitude),
            'fd_fft_max': np.max(fft_magnitude),
            'fd_fft_entropy': self._calculate_entropy(fft_magnitude),
            'fd_psd_mean': np.mean(psd),
            'fd_psd_std': np.std(psd),
            'fd_psd_max': np.max(psd),
            'fd_spectral_centroid': np.sum(freqs_psd * psd) / (np.sum(psd) + 1e-10),
            'fd_spectral_rolloff': self._calculate_rolloff(freqs_psd, psd, 0.85),
        }
        
        # 峰值频率
        peaks, _ = find_peaks(fft_magnitude, height=np.max(fft_magnitude) * 0.1)
        features['fd_peak_count'] = len(peaks)
        if len(peaks) > 0:
            features['fd_peak_freq_mean'] = np.mean(freqs[peaks])
            features['fd_peak_mag_mean'] = np.mean(fft_magnitude[peaks])
        else:
            features['fd_peak_freq_mean'] = 0
            features['fd_peak_mag_mean'] = 0
        
        return features
    
    def _extract_complexity_features(
        self,
        window: pd.DataFrame,
        column: str,
    ) -> Dict[str, float]:
        """提取复杂度特征"""
        values = window[column].values
        
        features = {
            'complexity_approx_entropy': self._approximate_entropy(values, m=2, r=0.2),
            'complexity_sample_entropy': self._sample_entropy(values, m=2, r=0.2),
            'complexity_higuchi_fd': self._higuchi_fractal_dimension(values),
            'complexity_katz_fd': self._katz_fractal_dimension(values),
        }
        
        return features
    
    def _extract_watermark_specific_features(
        self,
        window: pd.DataFrame,
        column: str,
    ) -> Dict[str, float]:
        """提取水印特定特征"""
        values = window[column].values
        
        # 最低有效位分析
        lsb_pattern = self._analyze_lsb(values)
        
        # 差分分析
        diff = np.diff(values)
        diff_pattern = self._analyze_difference_pattern(diff)
        
        features = {
            'wm_lsb_uniformity': lsb_pattern['uniformity'],
            'wm_lsb_entropy': lsb_pattern['entropy'],
            'wm_diff_mean': np.mean(diff),
            'wm_diff_std': np.std(diff),
            'wm_diff_pattern_regularity': diff_pattern['regularity'],
            'wm_periodicity_score': self._detect_periodicity(values),
            'wm_anomaly_score': self._detect_anomalies(values),
        }
        
        return features
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown)
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """计算香农熵"""
        values = np.abs(values)
        values = values / (np.sum(values) + 1e-10)
        return -np.sum(values * np.log2(values + 1e-10))
    
    def _calculate_rolloff(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        threshold: float = 0.85,
    ) -> float:
        """计算频谱滚降点"""
        cumulative = np.cumsum(psd)
        threshold_value = threshold * cumulative[-1]
        rolloff_idx = np.where(cumulative >= threshold_value)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        return 0
    
    def _approximate_entropy(self, values: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """近似熵"""
        N = len(values)
        r = r * np.std(values)
        
        def _phi(m):
            x = np.array([values[i:i+m] for i in range(N - m + 1)])
            C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (N - m + 1)
            return np.sum(np.log(C + 1e-10)) / (N - m + 1)
        
        return _phi(m) - _phi(m + 1)
    
    def _sample_entropy(self, values: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """样本熵"""
        N = len(values)
        r = r * np.std(values)
        
        def _count_matches(m):
            count = 0
            for i in range(N - m):
                for j in range(i + 1, N - m):
                    if np.max(np.abs(values[i:i+m] - values[j:j+m])) <= r:
                        count += 1
            return count
        
        B = _count_matches(m)
        A = _count_matches(m + 1)
        
        return -np.log((A + 1e-10) / (B + 1e-10))
    
    def _higuchi_fractal_dimension(self, values: np.ndarray, k_max: int = 10) -> float:
        """Higuchi分形维数"""
        N = len(values)
        L = []
        x = np.arange(1, N + 1)
        
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                Lmk = np.sum(np.abs(np.diff(values[idx]))) * (N - 1) / (len(idx) * k)
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        
        L = np.array(L)
        k_values = np.arange(1, k_max + 1)
        
        # 线性回归求斜率
        slope, _, _, _, _ = stats.linregress(np.log(k_values), np.log(L + 1e-10))
        return -slope
    
    def _katz_fractal_dimension(self, values: np.ndarray) -> float:
        """Katz分形维数"""
        N = len(values)
        L = np.sum(np.abs(np.diff(values)))
        d = np.max(np.abs(values - values[0]))
        
        if L == 0 or d == 0:
            return 0
        
        return np.log(N) / (np.log(N) + np.log(d / L))
    
    def _analyze_lsb(self, values: np.ndarray) -> Dict[str, float]:
        """分析最低有效位"""
        # 提取小数部分
        fractional = np.modf(values)[0]
        lsb = (fractional * 100).astype(int) % 10
        
        unique, counts = np.unique(lsb, return_counts=True)
        probs = counts / len(lsb)
        
        return {
            'uniformity': np.std(probs),
            'entropy': -np.sum(probs * np.log2(probs + 1e-10)),
        }
    
    def _analyze_difference_pattern(self, diff: np.ndarray) -> Dict[str, float]:
        """分析差分模式"""
        # 计算差分的周期性
        autocorr = np.correlate(diff, diff, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 找到第一个峰值（排除lag=0）
        peaks, _ = find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
        
        if len(peaks) > 0:
            regularity = autocorr[peaks[0] + 1] / autocorr[0]
        else:
            regularity = 0
        
        return {'regularity': regularity}
    
    def _detect_periodicity(self, values: np.ndarray) -> float:
        """检测周期性"""
        fft_vals = np.abs(fft(values))
        freqs = fftfreq(len(values))
        
        # 排除直流分量
        positive_freqs = freqs[1:len(freqs)//2]
        positive_fft = fft_vals[1:len(fft_vals)//2]
        
        if len(positive_fft) == 0:
            return 0
        
        # 计算频谱的峰值集中度
        peak_idx = np.argmax(positive_fft)
        peak_ratio = positive_fft[peak_idx] / (np.sum(positive_fft) + 1e-10)
        
        return peak_ratio
    
    def _detect_anomalies(self, values: np.ndarray) -> float:
        """检测异常值"""
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0
        
        z_scores = np.abs((values - mean) / std)
        anomaly_ratio = np.sum(z_scores > 3) / len(values)
        
        return anomaly_ratio
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'close') -> np.ndarray:
        """拟合并转换数据"""
        features_df = self.extract(df, target_column)
        
        # 排除非特征列
        feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
        X = features_df[feature_cols].values
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def transform(self, df: pd.DataFrame, target_column: str = 'close') -> np.ndarray:
        """转换数据（使用已拟合的scaler）"""
        features_df = self.extract(df, target_column)
        
        feature_cols = [c for c in features_df.columns if c not in ['window_start', 'window_end']]
        X = features_df[feature_cols].values
        
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
