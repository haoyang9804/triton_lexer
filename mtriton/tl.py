import numpy as np
from typing import TypeVar, Callable, Any, Union, List, Tuple, Optional, Sequence
import functools
import random
import math
import threading

T = TypeVar('T')
ArrayLike = Union[np.ndarray, List, Tuple]

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

    # Memory/pointer operations
    @staticmethod
    def load(ptr: ArrayLike, mask: Optional[ArrayLike] = None, other: Optional[Any] = None,
             cache: Optional[str] = None, evict: Optional[str] = None, volatile: bool = False) -> np.ndarray:
        """Load data tensor from memory location defined by pointer"""
        if isinstance(ptr, np.ndarray):
            result = ptr.copy()
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if other is not None:
                    result = np.where(mask, result, other)
                else:
                    result = np.where(mask, result, 0)
            return result
        return np.asarray(ptr)

    @staticmethod
    def store(ptr: ArrayLike, value: Any, mask: Optional[ArrayLike] = None,
              cache: Optional[str] = None, evict: Optional[str] = None) -> None:
        """Store data tensor to memory location defined by pointer"""
        if isinstance(ptr, np.ndarray):
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                ptr[mask] = value
            else:
                ptr[:] = value

    @staticmethod 
    def make_block_ptr(base: np.ndarray, shape: Tuple[int, ...], strides: Tuple[int, ...],
                      offsets: Tuple[int, ...], block_shape: Tuple[int, ...], order: Tuple[int, ...]) -> np.ndarray:
        """Returns a pointer to a block in the parent tensor"""
        return base  # Simplified implementation

    @staticmethod
    def advance(ptr: ArrayLike, offsets: Sequence[int]) -> np.ndarray:
        """Advances tensor pointer by offsets"""
        return np.asarray(ptr)  # Simplified implementation

    @staticmethod
    def make_tensor_descriptor(base: Any, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> dict:
        """Creates a tensor descriptor object"""
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
