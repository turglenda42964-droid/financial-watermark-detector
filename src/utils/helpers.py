#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辅助函数模块
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        verbose: 是否启用详细输出
        
    Returns:
        日志记录器
    """
    if verbose:
        level = logging.DEBUG
    
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    保存JSON文件
    
    Args:
        data: 数据字典
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        path: 文件路径
        
    Returns:
        数据字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        初始化进度追踪器
        
        Args:
            total: 总任务数
            description: 任务描述
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        """更新进度"""
        self.current += n
        self._print_progress()
    
    def _print_progress(self) -> None:
        """打印进度"""
        percentage = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "N/A"
        
        print(
            f"\r{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - ETA: {eta_str}",
            end='',
            flush=True
        )
        
        if self.current >= self.total:
            print()  # 换行


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        是否验证通过
    """
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必需的键: {key}")
    return True


def chunk_list(lst: list, chunk_size: int):
    """
    将列表分块
    
    Args:
        lst: 列表
        chunk_size: 块大小
        
    Yields:
        块
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
