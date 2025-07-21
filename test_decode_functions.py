#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.decode import decode_mabsa, decode_mate, decode_mner

def test_decode_functions():
    """测试解码函数是否能正确处理各种输入格式"""
    print("=== 测试解码函数 ===")
    
    # 测试 MABSA
    print("\n--- 测试 MABSA ---")
    
    # 正常情况
    mabsa_labels = [0, 1, 2, 0, 5, 6, 0]
    try:
        result = decode_mabsa(mabsa_labels)
        print(f"✅ 正常输入: {result}")
    except Exception as e:
        print(f"❌ 正常输入失败: {e}")
    
    # 嵌套列表情况
    mabsa_nested = [[0, 1, 2], [0, 5, 6, 0]]
    try:
        result = decode_mabsa(mabsa_nested)
        print(f"✅ 嵌套列表输入: {result}")
    except Exception as e:
        print(f"❌ 嵌套列表输入失败: {e}")
    
    # 测试 MATE
    print("\n--- 测试 MATE ---")
    
    # 正常情况
    mate_labels = [0, 1, 2, 0, 1, 0]
    try:
        result = decode_mate(mate_labels)
        print(f"✅ 正常输入: {result}")
    except Exception as e:
        print(f"❌ 正常输入失败: {e}")
    
    # 嵌套列表情况
    mate_nested = [[0, 1, 2], [0, 1, 0]]
    try:
        result = decode_mate(mate_nested)
        print(f"✅ 嵌套列表输入: {result}")
    except Exception as e:
        print(f"❌ 嵌套列表输入失败: {e}")
    
    # 测试 MNER
    print("\n--- 测试 MNER ---")
    
    # 正常情况
    mner_labels = [0, 1, 2, 0, 3, 4, 0]
    try:
        result = decode_mner(mner_labels)
        print(f"✅ 正常输入: {result}")
    except Exception as e:
        print(f"❌ 正常输入失败: {e}")
    
    # 嵌套列表情况
    mner_nested = [[0, 1, 2], [0, 3, 4, 0]]
    try:
        result = decode_mner(mner_nested)
        print(f"✅ 嵌套列表输入: {result}")
    except Exception as e:
        print(f"❌ 嵌套列表输入失败: {e}")
    
    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    test_decode_functions() 