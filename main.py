#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融时序数据水印检测系统 - 主程序入口
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.detector import WatermarkDetector
from src.utils.helpers import setup_logger


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='金融时序数据水印检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --config configs/default_config.yaml --input data/sample/test.csv
  python main.py --train --config configs/default_config.yaml --input data/raw/
  python main.py --evaluate --config configs/default_config.yaml --input data/processed/
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径 (默认: configs/default_config.yaml)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入数据路径 (文件或目录)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/',
        help='输出结果目录 (默认: results/)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='训练模式'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='评估模式'
    )
    
    parser.add_argument(
        '--detect',
        action='store_true',
        help='检测模式 (默认)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(verbose=args.verbose)
    logger.info("=" * 60)
    logger.info("金融时序数据水印检测系统启动")
    logger.info("=" * 60)
    
    # 加载配置
    try:
        config = load_config(args.config)
        logger.info(f"配置文件加载成功: {args.config}")
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        sys.exit(1)
    
    # 初始化检测器
    try:
        detector = WatermarkDetector(config)
        logger.info("检测器初始化成功")
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        sys.exit(1)
    
    # 执行相应模式
    try:
        if args.train:
            logger.info("进入训练模式...")
            detector.train(args.input)
            logger.info("训练完成")
            
        elif args.evaluate:
            logger.info("进入评估模式...")
            results = detector.evaluate(args.input)
            detector.save_results(results, args.output)
            logger.info("评估完成")
            
        else:
            logger.info("进入检测模式...")
            results = detector.detect(args.input)
            detector.save_results(results, args.output)
            logger.info("检测完成")
            
    except Exception as e:
        logger.error(f"执行过程中出现错误: {e}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("程序执行完毕")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
