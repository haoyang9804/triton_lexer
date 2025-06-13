import numpy as np
from typing import TypeVar, Callable, Any, Union, List, Tuple, Optional, Sequence, Dict
import functools
import random
import math
import threading
import warnings

T = TypeVar('T')
ArrayLike = Union[np.ndarray, List, Tuple]

def get_dtype_itemsize(dtype) -> int:
    """获取数据类型的 itemsize"""
    try:
        # 如果是 numpy 类型类（如 np.float32）
        if isinstance(dtype, type) and issubclass(dtype, np.generic):
            return dtype().itemsize
        # 如果已经是 dtype 实例
        elif hasattr(dtype, 'itemsize'):
            return int(dtype.itemsize)
        # 其他情况，转换为 dtype
        else:
            return np.dtype(dtype).itemsize
    except Exception:
        # 默认情况，使用 float32 的 itemsize
        return 4

class MemorySpace:
    """模拟 Triton 的内存空间"""
    def __init__(self):
        self.global_memory: Dict[int, np.ndarray] = {}
        self.shared_memory: Dict[int, np.ndarray] = {}
        self.next_addr = 0x1000  # 起始地址
        
    def allocate(self, size: int, dtype: np.dtype = np.float32, memory_type: str = "global") -> int:
        """分配内存并返回地址"""
        addr = self.next_addr
        itemsize = get_dtype_itemsize(dtype)
        self.next_addr += size * itemsize
        
        if memory_type == "global":
            self.global_memory[addr] = np.zeros(size, dtype=dtype)
        elif memory_type == "shared":
            self.shared_memory[addr] = np.zeros(size, dtype=dtype)
        
        return addr
    
    def get_memory(self, addr: int, memory_type: str = "global") -> Optional[np.ndarray]:
        """根据地址获取内存"""
        if memory_type == "global":
            return self.global_memory.get(addr)
        elif memory_type == "shared":
            return self.shared_memory.get(addr)
        return None

class TensorPointer:
    """模拟 Triton 的张量指针"""
    def __init__(self, base_addr: int, shape: Tuple[int, ...], dtype: np.dtype, 
                 strides: Optional[Tuple[int, ...]] = None, offset: int = 0, 
                 memory_space: Optional[MemorySpace] = None, memory_type: str = "global"):
        self.base_addr = base_addr
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.memory_space = memory_space or _global_memory_space
        self.memory_type = memory_type
        
        if strides is None:
            # 计算 C 风格的步长
            self.strides = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
        else:
            self.strides = strides
    
    def get_effective_address(self, indices: Optional[Tuple[int, ...]] = None) -> int:
        """计算有效地址"""
        addr = self.base_addr + self.offset
        if indices:
            itemsize = get_dtype_itemsize(self.dtype)
            for i, (idx, stride) in enumerate(zip(indices, self.strides)):
                addr += idx * stride * itemsize
        return addr
    
    def advance_offset(self, offsets: Sequence[int]):
        """推进指针偏移"""
        total_offset = 0
        for offset, stride in zip(offsets, self.strides):
            total_offset += offset * stride
        itemsize = get_dtype_itemsize(self.dtype)
        self.offset += total_offset * itemsize
        return self
    
    def get_data(self) -> Optional[np.ndarray]:
        """获取指针指向的数据"""
        return self.memory_space.get_memory(self.base_addr, self.memory_type)

# 全局内存空间实例
_global_memory_space = MemorySpace()

class tl:
    # Basic types
    int1 = np.int8
    int8 = np.int8
    int16 = np.int16
    int32 = np.int32
    int64 = np.int64
    uint32 = np.uint32
    float16 = np.float16
    float32 = np.float32
    bfloat16 = np.float16
    float8e4nv = np.float32  # Simulation
    float8e5 = np.float32    # Simulation

    @staticmethod
    def constexpr(func: Callable[..., T]) -> Callable[..., T]:
        """
        Simulate tl.constexpr decorator.
        Used in Triton to mark compile-time constants.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def program_id(axis: int) -> int:
        """Returns the ID of the current program instance on the given axis"""
        return 0  # Simplified implementation

    @staticmethod
    def num_programs(axis: int) -> int:
        """Returns the number of program instances launched on the given axis"""
        return 1  # Simplified implementation

    # 内存/指针操作
    @staticmethod
    def load(ptr: Union[ArrayLike, TensorPointer], mask: Optional[ArrayLike] = None, 
             other: Optional[Any] = None, cache: Optional[str] = None, 
             evict: Optional[str] = None, volatile: bool = False) -> np.ndarray:
        """从指针定义的内存位置加载数据张量"""
        # 处理缓存提示
        if cache and cache not in [".cg", ".ca", ".cs", ".cv"]:
            warnings.warn(f"Unknown cache mode: {cache}, ignoring")
        
        if evict and evict not in [".lu", ".wb"]:
            warnings.warn(f"Unknown evict mode: {evict}, ignoring")
        
        # 处理 TensorPointer
        if isinstance(ptr, TensorPointer):
            base_data = ptr.get_data()
            if base_data is None:
                raise RuntimeError(f"Invalid memory access at address {ptr.base_addr}")
            
            # 使用步长和偏移计算正确的内存访问
            itemsize = get_dtype_itemsize(ptr.dtype)
            base_offset = ptr.offset // itemsize
            
            # 创建结果数组
            result = np.zeros(ptr.shape, dtype=ptr.dtype)
            
            # 根据步长从内存中获取数据
            for indices in np.ndindex(ptr.shape):
                # 计算线性索引
                linear_idx = base_offset
                for i, (idx, stride) in enumerate(zip(indices, ptr.strides)):
                    linear_idx += idx * stride
                
                # 检查边界
                if linear_idx >= base_data.size:
                    raise RuntimeError(f"Memory access out of bounds: trying to access index {linear_idx}, but array size is {base_data.size}")
                
                # 从内存获取值
                result[indices] = base_data.flat[linear_idx]
        
        # 处理常规数组
        elif isinstance(ptr, np.ndarray):
            result = ptr.copy()
        else:
            result = np.asarray(ptr)
        
        # 应用掩码
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            # 确保掩码形状兼容
            if mask.shape != result.shape:
                if mask.size == result.size:
                    mask = mask.reshape(result.shape)
                else:
                    mask = np.broadcast_to(mask, result.shape)
            
            if other is not None:
                other_array = np.asarray(other)
                if other_array.shape != result.shape:
                    other_array = np.broadcast_to(other_array, result.shape)
                result = np.where(mask, result, other_array)
            else:
                # 对于被掩码的元素，使用零值
                result = np.where(mask, result, np.zeros_like(result))
        
        # 模拟缓存行为（在实际实现中会影响性能）
        if cache == ".cg":  # Cache at global level
            pass  # 在模拟中无实际效果
        elif cache == ".ca":  # Cache at all levels
            pass
        elif cache == ".cs":  # Cache streaming
            pass
        elif cache == ".cv":  # Cache volatile
            pass
        
        return result

    @staticmethod
    def store(ptr: Union[ArrayLike, TensorPointer], value: Any, 
              mask: Optional[ArrayLike] = None, cache: Optional[str] = None, 
              evict: Optional[str] = None) -> None:
        """将数据张量存储到指针定义的内存位置"""
        # 处理缓存提示
        if cache and cache not in [".cg", ".ca", ".cs", ".cv", ".wb"]:
            warnings.warn(f"Unknown cache mode: {cache}, ignoring")
        
        if evict and evict not in [".lu", ".wb"]:
            warnings.warn(f"Unknown evict mode: {evict}, ignoring")
        
        value_array = np.asarray(value)
        
        # 处理 TensorPointer
        if isinstance(ptr, TensorPointer):
            base_data = ptr.get_data()
            if base_data is None:
                raise RuntimeError(f"Invalid memory access at address {ptr.base_addr}")
            
            # 使用步长和偏移计算正确的内存访问
            itemsize = get_dtype_itemsize(ptr.dtype)
            base_offset = ptr.offset // itemsize
            
            # 确保值的形状匹配
            if value_array.shape != ptr.shape:
                if value_array.size == 1:
                    value_array = np.full(ptr.shape, value_array.item(), dtype=ptr.dtype)
                else:
                    value_array = np.broadcast_to(value_array, ptr.shape)
            
            # 根据步长向内存存储数据
            for indices in np.ndindex(ptr.shape):
                # 计算线性索引
                linear_idx = base_offset
                for i, (idx, stride) in enumerate(zip(indices, ptr.strides)):
                    linear_idx += idx * stride
                
                # 检查边界
                if linear_idx >= base_data.size:
                    raise RuntimeError(f"Memory access out of bounds: trying to store at index {linear_idx}, but array size is {base_data.size}")
                
                # 应用掩码
                if mask is not None:
                    mask = np.asarray(mask, dtype=bool)
                    if mask.shape != ptr.shape:
                        if mask.size == int(np.prod(ptr.shape)):
                            mask = mask.reshape(ptr.shape)
                        else:
                            mask = np.broadcast_to(mask, ptr.shape)
                    
                    # 只在掩码为 True 时存储
                    if mask[indices]:
                        base_data.flat[linear_idx] = value_array[indices]
                else:
                    # 直接存储值
                    base_data.flat[linear_idx] = value_array[indices]
        
        # 处理常规数组
        elif isinstance(ptr, np.ndarray):
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != ptr.shape:
                    if mask.size == ptr.size:
                        mask = mask.reshape(ptr.shape)
                    else:
                        mask = np.broadcast_to(mask, ptr.shape)
                
                if value_array.shape != ptr.shape:
                    if value_array.size == 1:
                        value_array = np.full(ptr.shape, value_array.item())
                    else:
                        value_array = np.broadcast_to(value_array, ptr.shape)
                
                ptr[mask] = value_array[mask]
            else:
                if value_array.shape != ptr.shape:
                    if value_array.size == 1:
                        ptr.fill(value_array.item())
                    else:
                        ptr[:] = np.broadcast_to(value_array, ptr.shape)
                else:
                    ptr[:] = value_array
        else:
            raise TypeError("ptr must be a numpy array or TensorPointer")

    @staticmethod 
    def make_block_ptr(base: Union[np.ndarray, int], shape: Tuple[int, ...], 
                      strides: Tuple[int, ...], offsets: Tuple[int, ...], 
                      block_shape: Tuple[int, ...], order: Tuple[int, ...]) -> TensorPointer:
        """返回指向父张量中块的指针"""
        if isinstance(base, np.ndarray):
            # 为现有数组创建指针
            addr = _global_memory_space.allocate(base.size, base.dtype)
            base_data = _global_memory_space.get_memory(addr)
            base_data[:] = base.flat
        else:
            addr = base  # 假设 base 是地址
        
        # 计算偏移
        total_offset = 0
        for offset, stride in zip(offsets, strides):
            total_offset += offset * stride
        
        return TensorPointer(
            base_addr=addr,
            shape=block_shape,
            dtype=base.dtype if isinstance(base, np.ndarray) else np.float32,
            strides=strides,
            offset=total_offset,
            memory_space=_global_memory_space
        )

    @staticmethod
    def advance(ptr: Union[ArrayLike, TensorPointer], offsets: Sequence[int]) -> Union[np.ndarray, TensorPointer]:
        """推进张量指针的偏移量"""
        if isinstance(ptr, TensorPointer):
            new_ptr = TensorPointer(
                base_addr=ptr.base_addr,
                shape=ptr.shape,
                dtype=ptr.dtype,
                strides=ptr.strides,
                offset=ptr.offset,
                memory_space=ptr.memory_space,
                memory_type=ptr.memory_type
            )
            return new_ptr.advance_offset(offsets)
        else:
            return np.asarray(ptr)  # 简化实现

    @staticmethod
    def make_tensor_descriptor(base: Any, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> dict:
        """创建张量描述符对象"""
        return {"base": base, "shape": shape, "strides": strides}

    # Creation operations
    @staticmethod 
    def arange(start: int, end: int, step: int = 1, dtype: np.dtype = np.int32) -> np.ndarray:
        """Returns consecutive values in the half-open interval [start, end)"""
        return np.arange(start, end, step, dtype=dtype)

    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Returns a zero-filled tensor of given shape and data type"""
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def full(shape: Tuple[int, ...], value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
        """Returns a tensor filled with scalar value of given shape and data type"""
        return np.full(shape, value, dtype=dtype)

    @staticmethod
    def zeros_like(x: ArrayLike) -> np.ndarray:
        """Returns a zero tensor with same shape and type as the given tensor"""
        return np.zeros_like(x)

    @staticmethod
    def cast(x: ArrayLike, dtype: np.dtype) -> np.ndarray:
        """Casts tensor to given data type"""
        return np.asarray(x).astype(dtype)

    @staticmethod
    def cat(lhs: ArrayLike, rhs: ArrayLike) -> np.ndarray:
        """Concatenates given blocks"""
        return np.concatenate([lhs, rhs], axis=-1)

    # Shape operations
    @staticmethod
    def reshape(x: ArrayLike, shape: Tuple[int, ...]) -> np.ndarray:
        """Returns tensor with same elements but different shape"""
        return np.reshape(x, shape)

    @staticmethod
    def view(x: ArrayLike, shape: Tuple[int, ...]) -> np.ndarray:
        """Returns tensor with same elements but different shape"""
        return np.reshape(x, shape)

    @staticmethod
    def broadcast_to(x: ArrayLike, shape: Tuple[int, ...]) -> np.ndarray:
        """Attempts to broadcast given tensor to new shape"""
        return np.broadcast_to(x, shape)

    @staticmethod
    def expand_dims(x: ArrayLike, axis: int) -> np.ndarray:
        """Expands tensor shape by inserting new length-1 dimension"""
        return np.expand_dims(x, axis)

    @staticmethod
    def flip(x: ArrayLike, axis: int) -> np.ndarray:
        """Flips tensor x along dimension axis"""
        return np.flip(x, axis)

    @staticmethod
    def permute(x: ArrayLike, dims: Tuple[int, ...]) -> np.ndarray:
        """Permutes dimensions of tensor"""
        return np.transpose(x, dims)

    @staticmethod
    def trans(x: ArrayLike, order: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Permutes dimensions of tensor"""
        if order is None:
            return np.transpose(x)
        return np.transpose(x, order)

    @staticmethod
    def split(x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Splits tensor into two parts along last dimension"""
        return np.split(x, 2, axis=-1)

    @staticmethod
    def join(lhs: ArrayLike, rhs: ArrayLike) -> np.ndarray:
        """Joins given tensors in new minor dimension"""
        return np.stack([lhs, rhs], axis=-1)

    @staticmethod
    def interleave(lhs: ArrayLike, rhs: ArrayLike) -> np.ndarray:
        """Interleaves values of two tensors along last dimension"""
        result = np.empty((*lhs.shape[:-1], lhs.shape[-1] * 2), dtype=lhs.dtype)
        result[..., ::2] = lhs
        result[..., 1::2] = rhs
        return result

    @staticmethod
    def ravel(x: ArrayLike) -> np.ndarray:
        """Returns contiguous flattened view of x"""
        return np.ravel(x)

    # Linear algebra operations
    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike, allow_tf32: bool = True, 
            input_precision: Optional[str] = None, max_num_imprecise_acc: Optional[int] = None) -> np.ndarray:
        """Returns matrix product of two blocks"""
        return np.dot(a, b)

    # Indexing operations
    @staticmethod
    def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Returns element tensor from x or y based on condition"""
        return np.where(condition, x, y)

    @staticmethod
    def gather(x: ArrayLike, indices: ArrayLike, axis: int = 0) -> np.ndarray:
        """Gathers from tensor along given dimension"""
        return np.take_along_axis(x, indices, axis)

    # Mathematical operations
    @staticmethod
    def abs(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise absolute value of x"""
        return np.abs(x)

    @staticmethod
    def ceil(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise ceiling of x"""
        return np.ceil(x)

    @staticmethod
    def floor(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise floor of x"""
        return np.floor(x)

    @staticmethod
    def clamp(x: ArrayLike, min_val: Any, max_val: Any) -> np.ndarray:
        """Limits input tensor x within range [min_val, max_val]"""
        return np.clip(x, min_val, max_val)

    @staticmethod
    def cos(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise cosine of x"""
        return np.cos(x)

    @staticmethod
    def sin(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise sine of x"""
        return np.sin(x)

    @staticmethod
    def exp(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise exponential of x"""
        return np.exp(x)

    @staticmethod
    def exp2(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise exponential (base 2) of x"""
        return np.exp2(x)

    @staticmethod
    def log(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise natural logarithm of x"""
        return np.log(x)

    @staticmethod
    def log2(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise logarithm (base 2) of x"""
        return np.log2(x)

    @staticmethod
    def sqrt(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise square root of x"""
        return np.sqrt(x)

    @staticmethod
    def rsqrt(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise reciprocal square root of x"""
        return 1 / np.sqrt(x)

    @staticmethod
    def maximum(x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Calculates element-wise maximum of x and y"""
        return np.maximum(x, y)

    @staticmethod
    def minimum(x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Calculates element-wise minimum of x and y"""
        return np.minimum(x, y)

    @staticmethod
    def sigmoid(x: ArrayLike) -> np.ndarray:
        """Calculates element-wise sigmoid of x"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
        """Calculates element-wise softmax of x"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def fma(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        """Calculates element-wise fused multiply-add of x, y, and z"""
        return x * y + z

    @staticmethod
    def cdiv(x: int, div: int) -> int:
        """Calculates element-wise ceiling division of x by div"""
        return (x + div - 1) // div

    @staticmethod
    def fdiv(x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Calculates element-wise fast division of x by y"""
        """计算 x 和 y 的元素级快速除法"""
        return x / y

    @staticmethod
    def erf(x: ArrayLike) -> np.ndarray:
        """计算 x 的元素级误差函数"""
        return np.vectorize(math.erf)(x)

    @staticmethod
    def umulhi(x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """计算两个整数的 2N 位乘积的最高 N 位"""
        return np.right_shift(np.multiply(x.astype(np.uint64), y.astype(np.uint64)), 32).astype(x.dtype)

    # 约简操作
    @staticmethod
    def max(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的最大值"""
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的最小值"""
        return np.min(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def sum(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的和"""
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def argmax(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的最大索引"""
        return np.argmax(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def argmin(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的最小索引"""
        return np.argmin(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce(combine_fn: Callable, x: ArrayLike, axis: int = 0) -> np.ndarray:
        """将 combine_fn 应用于输入张量沿提供的轴的所有元素"""
        return np.apply_along_axis(lambda arr: functools.reduce(combine_fn, arr), axis, x)

    @staticmethod
    def xor_sum(x: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的异或和"""
        return np.bitwise_xor.reduce(x, axis=axis)

    # 扫描/排序操作
    @staticmethod
    def cumsum(x: ArrayLike, axis: int = 0) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的累积和"""
        return np.cumsum(x, axis=axis)

    @staticmethod
    def cumprod(x: ArrayLike, axis: int = 0) -> np.ndarray:
        """返回输入张量沿提供的轴的所有元素的累积乘积"""
        return np.cumprod(x, axis=axis)

    @staticmethod
    def associative_scan(combine_fn: Callable, x: ArrayLike, axis: int = 0) -> np.ndarray:
        """将 combine_fn 应用于输入张量沿提供的轴的每个元素"""
        return np.apply_along_axis(lambda arr: np.array([functools.reduce(combine_fn, arr[:i+1]) for i in range(len(arr))]), axis, x)

    @staticmethod
    def histogram(x: ArrayLike, num_bins: Optional[int] = None) -> np.ndarray:
        """基于输入张量计算直方图"""
        if num_bins is None:
            num_bins = int(np.max(x)) + 1
        return np.histogram(x, bins=num_bins, range=(0, num_bins))[0]

    @staticmethod
    def sort(x: ArrayLike, axis: int = -1, descending: bool = False) -> np.ndarray:
        """对张量进行排序"""
        result = np.sort(x, axis=axis)
        if descending:
            result = np.flip(result, axis=axis)
        return result

    # 原子操作
    @staticmethod
    def atomic_add(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子加法"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] += value
        else:
            ptr[:] += value
        return old_value

    @staticmethod
    def atomic_cas(ptr: ArrayLike, cmp: Any, val: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子比较交换"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        condition = (old_value == cmp)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            condition = condition & mask
        ptr[condition] = val
        return old_value

    @staticmethod
    def atomic_xchg(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子交换"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] = value
        else:
            ptr[:] = value
        return old_value

    @staticmethod
    def atomic_max(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子最大值操作"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] = np.maximum(ptr[mask], value)
        else:
            ptr[:] = np.maximum(ptr, value)
        return old_value

    @staticmethod
    def atomic_min(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子最小值操作"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] = np.minimum(ptr[mask], value)
        else:
            ptr[:] = np.minimum(ptr, value)
        return old_value

    @staticmethod
    def atomic_and(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子逻辑与"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] = np.bitwise_and(ptr[mask], value)
        else:
            ptr[:] = np.bitwise_and(ptr, value)
        return old_value

    @staticmethod
    def atomic_or(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子逻辑或"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] = np.bitwise_or(ptr[mask], value)
        else:
            ptr[:] = np.bitwise_or(ptr, value)
        return old_value

    @staticmethod
    def atomic_xor(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None) -> np.ndarray:
        """在指针指定的内存位置执行原子逻辑异或"""
        old_value = ptr.copy() if isinstance(ptr, np.ndarray) else np.asarray(ptr)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ptr[mask] = np.bitwise_xor(ptr[mask], value)
        else:
            ptr[:] = np.bitwise_xor(ptr, value)
        return old_value

    # 随机数生成
    @staticmethod
    def rand(shape: Tuple[int, ...], seed: int, offset: ArrayLike) -> np.ndarray:
        """给定种子标量和偏移块，返回 U(0,1) 中的随机 float32 块"""
        np.random.seed(seed)
        return np.random.random(shape).astype(np.float32)

    @staticmethod
    def randn(shape: Tuple[int, ...], seed: int, offset: ArrayLike) -> np.ndarray:
        """给定种子标量和偏移块，返回 N(0,1) 中的随机 float32 块"""
        np.random.seed(seed)
        return np.random.normal(0, 1, shape).astype(np.float32)

    @staticmethod
    def randint(shape: Tuple[int, ...], seed: int, offset: ArrayLike) -> np.ndarray:
        """给定种子标量和偏移块，返回随机 int32 的单个块"""
        np.random.seed(seed)
        return np.random.randint(0, 2**31, shape, dtype=np.int32)

    @staticmethod
    def randint4x(shape: Tuple[int, ...], seed: int, offset: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """给定种子标量和偏移块，返回四个随机 int32 块"""
        np.random.seed(seed)
        return tuple(np.random.randint(0, 2**31, shape, dtype=np.int32) for _ in range(4))

    # 迭代器
    @staticmethod
    def range(*args) -> range:
        """永远向上计数的迭代器"""
        return range(*args)

    @staticmethod
    def static_range(*args) -> range:
        """永远向上计数的迭代器（编译时）"""
        return range(*args)

    # 编译器提示操作
    @staticmethod
    def multiple_of(input_tensor: ArrayLike, value: int) -> None:
        """让编译器知道输入中的值都是 value 的倍数"""
        if isinstance(input_tensor, np.ndarray):
            assert np.all(input_tensor % value == 0), f"Not all values are multiples of {value}"
        elif isinstance(input_tensor, (list, tuple)):
            assert all(x % value == 0 for x in input_tensor), f"Not all values are multiples of {value}"

    @staticmethod
    def max_contiguous(input_tensor: ArrayLike, value: int) -> None:
        """让编译器知道输入中前 value 个值是连续的"""
        if isinstance(input_tensor, np.ndarray):
            assert input_tensor.size <= value, f"Tensor size {input_tensor.size} exceeds max contiguous {value}"
        elif isinstance(input_tensor, (list, tuple)):
            assert len(input_tensor) <= value, f"Length {len(input_tensor)} exceeds max contiguous {value}"

    @staticmethod
    def max_constancy(input_tensor: ArrayLike, value: int) -> None:
        """让编译器知道输入中前 value 个值是常量"""
        pass  # 编译时提示，运行时无操作

    @staticmethod
    def assume(condition: bool) -> None:
        """允许编译器假设条件为真"""
        if not condition:
            raise RuntimeError("Assumption violated")

    @staticmethod
    def debug_barrier() -> None:
        """插入屏障以同步块中的所有线程"""
        # 在 CPU 模拟中使用线程同步
        barrier = threading.Barrier(1)
        barrier.wait()

    # 调试操作
    @staticmethod
    def static_print(*args: Any) -> None:
        """在编译时打印值"""
        print("STATIC:", *args)

    @staticmethod
    def static_assert(condition: bool, message: str = "") -> None:
        """在编译时断言条件"""
        assert condition, f"Static assertion failed: {message}"

    @staticmethod
    def device_print(*args: Any) -> None:
        """从设备在运行时打印值"""
        print("DEVICE:", *args)

    @staticmethod
    def device_assert(condition: bool, message: str = "") -> None:
        """从设备在运行时断言条件"""
        assert condition, f"Device assertion failed: {message}"

    # 内联汇编
    @staticmethod
    def inline_asm_elementwise(asm: str, constraints: str, args: Sequence[Any], 
                              dtype: Any, is_pure: bool = True, pack: int = 1) -> Any:
        """在张量上执行内联汇编"""
        # 简化实现：直接返回第一个参数
        return args[0] if args else None

    # 新增函数以匹配 triton_keywords.txt
    @staticmethod
    def philox(seed: int, offset: int) -> float:
        """Philox 随机数生成器"""
        np.random.seed(seed + offset)
        return np.random.random()

    @staticmethod
    def tensor(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """创建未初始化的张量"""
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def swizzle2d(x: ArrayLike, size_i: int, size_j: int, size_g: int) -> np.ndarray:
        """2D swizzle 操作"""
        return np.asarray(x)  # 简化实现

    # 新增的分组和函数别名
    class math:
        @staticmethod
        def exp(x: ArrayLike) -> np.ndarray:
            return tl.exp(x)

        @staticmethod
        def exp2(x: ArrayLike) -> np.ndarray:
            return tl.exp2(x)

        @staticmethod
        def log(x: ArrayLike) -> np.ndarray:
            return tl.log(x)

        @staticmethod
        def log2(x: ArrayLike) -> np.ndarray:
            return tl.log2(x)

        @staticmethod
        def sqrt(x: ArrayLike) -> np.ndarray:
            return tl.sqrt(x)

        @staticmethod
        def rsqrt(x: ArrayLike) -> np.ndarray:
            return tl.rsqrt(x)

        @staticmethod
        def pow(x: ArrayLike, y: ArrayLike) -> np.ndarray:
            return np.power(x, y)

        @staticmethod
        def max(x: ArrayLike, y: ArrayLike) -> np.ndarray:
            return tl.maximum(x, y)

        @staticmethod
        def min(x: ArrayLike, y: ArrayLike) -> np.ndarray:
            return tl.minimum(x, y)

        @staticmethod
        def fast_expf(x: ArrayLike) -> np.ndarray:
            return tl.exp(x)  # 快速近似指数

        @staticmethod
        def llrint(x: ArrayLike) -> np.ndarray:
            return np.rint(x).astype(np.int64)

    class extra:
        class cuda:
            class libdevice:
                @staticmethod
                def pow(x: ArrayLike, y: ArrayLike) -> np.ndarray:
                    return np.power(x, y)

                @staticmethod
                def round(x: ArrayLike) -> np.ndarray:
                    return np.round(x)

                @staticmethod
                def llrint(x: ArrayLike) -> np.ndarray:
                    return np.rint(x).astype(np.int64)

    class experimental:
        class descriptor:
            @staticmethod
            def load(desc: Any, indices: Tuple[int, ...]) -> Any:
                if isinstance(desc, dict):
                    shape = desc.get("shape", ())
                    dtype = desc.get("dtype", np.float32)
                    return np.zeros(shape, dtype=dtype)
                return desc

            @staticmethod
            def store(desc: Any, indices: Tuple[int, ...], value: Any) -> None:
                """将数据存储到描述符"""
                if isinstance(desc, dict):
                    desc["value"] = value
                    desc["indices"] = indices
            
if __name__ == "__main__":
    x = tl.zeros((3, 3))
    y = tl.full((3, 3), 2.0)
    z = tl.maximum(x, y)
    
    pid = tl.program_id(0)
    
    exp_result = tl.exp(x)
    
    print("Triton mock library loaded successfully!")
    print("Test result shape:", z.shape)
    print("Program ID:", pid)
