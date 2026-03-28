#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
运行所有测试用例并生成测试报告
"""

import unittest
import sys
from pathlib import Path


def run_tests():
    """运行所有测试"""
    # 发现所有测试
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / 'tests'
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回退出码
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
