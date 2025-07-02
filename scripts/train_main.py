#!/usr/bin/env python3
# scripts/train_main.py
"""
主训练脚本 - 调用modules中的训练模块
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.train_refactored import main

if __name__ == "__main__":
    main()