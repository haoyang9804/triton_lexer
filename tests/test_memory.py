#!/usr/bin/env python3
"""
测试 Triton 内存模拟器的功能
演示新实现的 load 和 store 操作以及内存管理
"""

import numpy as np
import sys
import os

# 添加父目录到路径，以便导入 mtriton 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mtriton.tl import tl, MemorySpace, TensorPointer, _global_memory_space

def test_basic_memory_operations():
    """测试基本的内存操作"""
    print("=== 测试基本内存操作 ===")
    
    # 创建测试数据
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    print(f"原始数据:\n{data}")
    
    # 测试基本的 load 和 store
    loaded_data = tl.load(data)
    print(f"加载的数据:\n{loaded_data}")
    
    # 测试带掩码的加载
    mask = np.array([[True, False, True, False], [False, True, False, True]])
    masked_load = tl.load(data, mask=mask, other=-1)
    print(f"带掩码的加载结果:\n{masked_load}")
    
    # 测试存储
    target = np.zeros_like(data)
    tl.store(target, data * 2)
    print(f"存储后的数据:\n{target}")
    
    # 测试带掩码的存储
    tl.store(target, 99, mask=mask)
    print(f"带掩码存储后的数据:\n{target}")

def test_tensor_pointer():
    """测试 TensorPointer 功能"""
    print("\n=== 测试 TensorPointer 功能 ===")
    
    # 创建内存空间并分配内存
    memory_space = MemorySpace()
    
    # 分配一个 4x4 的 float32 数组
    shape = (4, 4)
    addr = memory_space.allocate(16, np.float32)
    
    # 初始化数据
    base_data = memory_space.get_memory(addr)
    base_data[:] = np.arange(16, dtype=np.float32)
    print(f"基础内存数据:\n{base_data.reshape(4, 4)}")
    
    # 创建指向 2x2 块的指针
    ptr = TensorPointer(
        base_addr=addr,
        shape=(2, 2),
        dtype=np.float32,
        strides=(4, 1),  # 4x4 矩阵的步长
        offset=5 * np.float32().itemsize,  # 从索引 5 开始（第1行第1列）
        memory_space=memory_space
    )
    
    # 通过指针加载数据
    block_data = tl.load(ptr)
    print(f"通过指针加载的 2x2 块:\n{block_data}")
    
    # 通过指针存储数据
    new_values = np.array([[100, 101], [102, 103]], dtype=np.float32)
    tl.store(ptr, new_values)
    print(f"存储新值后的完整内存:\n{base_data.reshape(4, 4)}")

def test_block_pointer():
    """测试块指针功能"""
    print("\n=== 测试块指针功能 ===")
    
    # 创建一个大的数组
    base_array = np.arange(24, dtype=np.float32).reshape(4, 6)
    print(f"基础数组:\n{base_array}")
    
    # 创建块指针
    block_ptr = tl.make_block_ptr(
        base=base_array,
        shape=(4, 6),           # 原始形状
        strides=(6, 1),         # 步长
        offsets=(1, 2),         # 偏移：从第1行第2列开始
        block_shape=(2, 3),     # 块形状：2x3
        order=(0, 1)            # 维度顺序
    )
    
    # 通过块指针加载数据
    block_data = tl.load(block_ptr)
    print(f"通过块指针加载的数据:\n{block_data}")
    
    # 推进指针
    advanced_ptr = tl.advance(block_ptr, (1, 1))
    advanced_data = tl.load(advanced_ptr)
    print(f"推进指针后加载的数据:\n{advanced_data}")

def test_mask_operations():
    """测试掩码操作"""
    print("\n=== 测试掩码操作 ===")
    
    # 创建测试数据
    data = np.random.randn(4, 4).astype(np.float32)
    print(f"原始数据:\n{data}")
    
    # 创建掩码：只选择正值
    mask = data > 0
    print(f"掩码 (data > 0):\n{mask}")
    
    # 使用掩码加载，负值替换为 -999
    masked_load = tl.load(data, mask=mask, other=-999)
    print(f"掩码加载结果:\n{masked_load}")
    
    # 创建目标数组并使用掩码存储
    target = np.zeros_like(data)
    tl.store(target, 1.0, mask=mask)
    print(f"掩码存储结果（正值位置设为1）:\n{target}")

def test_cache_hints():
    """测试缓存提示"""
    print("\n=== 测试缓存提示 ===")
    
    data = np.arange(10, dtype=np.float32)
    
    # 测试不同的缓存模式
    cache_modes = [".cg", ".ca", ".cs", ".cv"]
    
    for cache_mode in cache_modes:
        result = tl.load(data, cache=cache_mode)
        print(f"使用缓存模式 {cache_mode}: {result[:5]}...")  # 只显示前5个元素
    
    # 测试无效的缓存模式（应该产生警告）
    try:
        result = tl.load(data, cache=".invalid")
        print("无效缓存模式测试完成")
    except Exception as e:
        print(f"缓存模式错误: {e}")

def test_boundary_conditions():
    """测试边界条件"""
    print("\n=== 测试边界条件 ===")
    
    # 测试越界访问
    memory_space = MemorySpace()
    addr = memory_space.allocate(10, np.float32)
    
    # 创建一个会越界的指针
    ptr = TensorPointer(
        base_addr=addr,
        shape=(5,),
        dtype=np.float32,
        offset=8 * np.float32().itemsize,  # 偏移到接近末尾
        memory_space=memory_space
    )
    
    try:
        # 这应该抛出越界错误
        tl.load(ptr)
        print("错误：越界访问没有被检测到！")
    except RuntimeError as e:
        print(f"正确检测到越界访问: {e}")

if __name__ == "__main__":
    print("Triton 内存模拟器测试")
    print("=" * 50)
    
    test_basic_memory_operations()
    test_tensor_pointer()
    test_block_pointer()
    test_mask_operations()
    test_cache_hints()
    test_boundary_conditions()
    
    print("\n测试完成！") 