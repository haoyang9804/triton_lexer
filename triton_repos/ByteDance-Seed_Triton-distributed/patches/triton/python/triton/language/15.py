


from __future__ import annotations

from warnings import warn
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional, Tuple
import builtins
from ..runtime.jit import jit
import inspect
import os

from .._C.libtriton import ir
from . import semantic
from ._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape

T = TypeVar('T')

TRITON_BUILTIN = "__triton_builtin__"

PropagateNan = ir.PROPAGATE_NAN


def builtin(fn: T) -> T:
    
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


def _tensor_member_fn(fn: T) -> T:
    
    assert callable(fn)
    orig_sig = inspect.signature(fn)
    
    has_args = len(orig_sig.parameters.keys() - {"_builder", "_generator"}) > 1

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    
    
    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name='self')
    new_sig = orig_sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"
    
    if is_builtin(fn):
        setattr(wrapper, TRITON_BUILTIN, True)

    setattr(tensor, fn.__name__, wrapper)
    return fn


def _unwrap_iterable(x):
    
    if len(x) == 1:
        
        
        
        
        
        
        
        
        
        
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass

    return x


def is_builtin(fn) -> bool:
    
    return getattr(fn, TRITON_BUILTIN, False)


@builtin
def to_tensor(x, _builder=None):
    return semantic.to_tensor(x, _builder)







class const:
    
    pass


class constexpr:
    

    def __init__(self, value):
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value
        self.type = constexpr

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __index__(self):
        return self.value

    
    
    
    
    def __add__(self, other):
        return constexpr(self.value + _constexpr_to_value(other))

    def __radd__(self, other):
        return constexpr(_constexpr_to_value(other) + self.value)

    def __sub__(self, other):
        return constexpr(self.value - _constexpr_to_value(other))

    def __rsub__(self, other):
        return constexpr(_constexpr_to_value(other) - self.value)

    def __mul__(self, other):
        return constexpr(self.value * _constexpr_to_value(other))

    def __mod__(self, other):
        return constexpr(self.value % _constexpr_to_value(other))

    def __rmul__(self, other):
        return constexpr(_constexpr_to_value(other) * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / _constexpr_to_value(other))

    def __rtruediv__(self, other):
        return constexpr(_constexpr_to_value(other) / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // _constexpr_to_value(other))

    def __rfloordiv__(self, other):
        return constexpr(_constexpr_to_value(other) // self.value)

    def __gt__(self, other):
        return constexpr(self.value > _constexpr_to_value(other))

    def __rgt__(self, other):
        return constexpr(_constexpr_to_value(other) > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= _constexpr_to_value(other))

    def __rge__(self, other):
        return constexpr(_constexpr_to_value(other) >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < _constexpr_to_value(other))

    def __rlt__(self, other):
        return constexpr(_constexpr_to_value(other) < self.value)

    def __le__(self, other):
        return constexpr(self.value <= _constexpr_to_value(other))

    def __rle__(self, other):
        return constexpr(_constexpr_to_value(other) <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == _constexpr_to_value(other))

    def __ne__(self, other):
        return constexpr(self.value != _constexpr_to_value(other))

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & _constexpr_to_value(other))

    def logical_and(self, other):
        return constexpr(self.value and _constexpr_to_value(other))

    def __or__(self, other):
        return constexpr(self.value | _constexpr_to_value(other))

    def __xor__(self, other):
        return constexpr(self.value ^ _constexpr_to_value(other))

    def logical_or(self, other):
        return constexpr(self.value or _constexpr_to_value(other))

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value**_constexpr_to_value(other))

    def __rpow__(self, other):
        return constexpr(_constexpr_to_value(other)**self.value)

    def __rshift__(self, other):
        return constexpr(self.value >> _constexpr_to_value(other))

    def __lshift__(self, other):
        return constexpr(self.value << _constexpr_to_value(other))

    def __not__(self):
        return constexpr(not self.value)

    def __iter__(self):
        return iter(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)


CONSTEXPR_0 = constexpr(0)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, constexpr) else o


def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.type.scalar.primitive_bitwidth
        if shift_value.value >= bitwidth:
            warn(
                f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) for type '{value.dtype}'. This may result in undefined behavior."
            )


class base_value:
    
    type: base_type

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        
        raise NotImplementedError


class base_type:

    def __eq__(self, other):
        raise NotImplementedError("Types must implement __eq__")

    def __ne__(self, other):
        return not (self == other)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        
        raise NotImplementedError

    def mangle(self) -> str:
        raise NotImplementedError(f"NYI: Type mangling for type {self.__class__}")

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        raise NotImplementedError







class dtype(base_type):
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    class KIND(Enum):
        BOOLEAN = 0
        INTEGRAL = 1
        FLOATING = 2

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.primitive_bitwidth = 8
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.primitive_bitwidth = 8
                self.exponent_bias = 16
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.primitive_bitwidth = 16
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.primitive_bitwidth = 16
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.primitive_bitwidth = 32
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 52
                self.primitive_bitwidth = 64
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')
        elif name == 'void':
            self.primitive_bitwidth = 0

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def kind(self):
        
        if self.is_bool():
            return dtype.KIND.BOOLEAN
        elif self.is_int():
            return dtype.KIND.INTEGRAL
        else:
            assert self.is_floating()
            return dtype.KIND.FLOATING

    def get_int_max_value(self):
        if self.is_int_signed():
            return 2**(self.int_bitwidth - 1) - 1
        if self.is_int_unsigned():
            return 2**self.int_bitwidth - 1
        assert False

    def get_int_min_value(self):
        if self.is_int_signed():
            return -2**(self.int_bitwidth - 1)
        if self.is_int_unsigned():
            return 0
        assert False

    @staticmethod
    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, ))

    @property
    def scalar(self):
        return self

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name.startswith("fp8"):
            if self.name not in builder.options.supported_fp8_dtypes:
                raise ValueError(f'type {self} not supported in this architecture. '
                                 f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')
            if self.name in builder.options.deprecated_fp8_dtypes:
                warn(f"{self.name} is deprecated in this architecture and will be removed in a future triton release")

        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_ty()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_ty()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_ty()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):
        return self.name

    def codegen_name(self):
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        
        return self.name

    def __repr__(self):
        
        return f'triton.language.{self.codegen_name()}'

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        if self.is_int():
            SIGNED = dtype.SIGNEDNESS.SIGNED
            prefix = 'i' if self.int_signedness == SIGNED else 'u'
            return prefix + str(self.int_bitwidth)
        if self.is_floating():
            return str(self)
        if self.is_void():
            return 'V'
        return super().mangle()





_DtypeClass = dtype


class pointer_type(dtype):

    def __init__(self, element_ty: dtype, address_space: int = 1, const: bool = False):
        element_ty = _unwrap_if_constexpr(element_ty)
        if not isinstance(element_ty, dtype):
            raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
        self.element_ty = element_ty
        self.address_space = address_space
        self.const = const
        self.name = f'pointer<{element_ty}>' if not const else f'const_pointer<{element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def is_const(self):
        return self.const

    def __eq__(self, other: pointer_type) -> bool:
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space and self.const == other.const

    def __ne__(self, other: pointer_type) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.element_ty, "tt_ptr"))

    @property
    def scalar(self):
        return self

    def mangle(self) -> str:
        return f"P{self.element_ty.mangle()}"


class nv_tma_desc_type(pointer_type):

    def __init__(self, const=True, address_space=0):
        super().__init__(uint8, const=const, address_space=address_space)
        self.name = 'nv_tma_desc_type'


class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty

        
        
        assert (isinstance(shape, (list, tuple)))

        
        self.shape = tuple(_unwrap_shape(shape))
        if not self.shape:
            raise TypeError('0d block_type is forbidden')

        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> Tuple[int]:
        return self.shape

    def __eq__(self, other) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    @property
    def scalar(self):
        return self.element_ty

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        return f'{elt}S{shape}S'


class tuple_type(base_type):

    def __init__(self, types, fields=None):
        self.types = types
        self.fields = fields or [''] * len(types)
        self.name = '[' + ','.join([f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + ']'

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]):
        for ty in self.types:
            if not isinstance(ty, constexpr):
                ty._flatten_ir_types(builder, out)

    def __getitem__(self, index: int) -> dtype:
        return self.types[index]

    def __eq__(self, other):
        return type(self) is type(other) and self.types == other.types and self.fields == other.fields

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tuple, int]:
        values = []
        for ty in self.types:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values.append(value)
        return tuple(values, self), cursor

    def mangle(self):
        return 'T' + '_'.join(ty.mangle for ty in self.types) + 'T'


class slice_type(dtype):

    def __init__(self):
        self.name = 'slice_type'



void = dtype('void')
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8e5 = dtype('fp8e5')
float8e5b16 = dtype('fp8e5b16')
float8e4nv = dtype('fp8e4nv')
float8e4b8 = dtype('fp8e4b8')
float8e4b15 = dtype('fp8e4b15')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')

pi32_t = pointer_type(int32)


def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    if bitwidth == 1:
        return int1
    elif bitwidth == 8 and signed:
        return int8
    elif bitwidth == 8 and not signed:
        return uint8
    elif bitwidth == 16 and signed:
        return int16
    elif bitwidth == 16 and not signed:
        return uint16
    elif bitwidth == 32 and signed:
        return int32
    elif bitwidth == 32 and not signed:
        return uint32
    elif bitwidth == 64 and signed:
        return int64
    elif bitwidth == 64 and not signed:
        return uint64
    else:
        raise ValueError(f'Unsupported bitwidth {bitwidth} and signedness {signed}')







class tensor(base_value):
    

    def __init__(self, handle, type: dtype):
        
        super().__init__()
        
        self.handle = handle
        
        self.shape = type.shape if type.is_block() else ()
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        self.type = type  
        
        self.dtype = type.scalar
        self.shape = tuple([constexpr(s) for s in self.shape])

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _builder=None):
        return add(self, other, sanitize_overflow=True, _builder=_builder)

    @builtin
    def __radd__(self, other, _builder=None):
        return add(other, self, sanitize_overflow=True, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder=None):
        return sub(self, other, sanitize_overflow=True, _builder=_builder)

    @builtin
    def __rsub__(self, other, _builder=None):
        return sub(other, self, sanitize_overflow=True, _builder=_builder)

    @builtin
    def __mul__(self, other, _builder=None):
        return mul(self, other, sanitize_overflow=True, _builder=_builder)

    @builtin
    def __rmul__(self, other, _builder=None):
        return mul(other, self, sanitize_overflow=True, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.truediv(self, other, _builder)

    @builtin
    def __rtruediv__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.floordiv(self, other, _builder)

    @builtin
    def __rfloordiv__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.floordiv(other, self, _builder)

    @builtin
    def __mod__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.mod(self, other, _builder)

    @builtin
    def __rmod__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.mod(other, self, _builder)

    
    @builtin
    def __neg__(self, _builder=None):
        return semantic.minus(self, _builder)

    @builtin
    def __invert__(self, _builder=None):
        return semantic.invert(self, _builder)

    

    @builtin
    def __and__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.and_(self, other, _builder)

    @builtin
    def __rand__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.and_(other, self, _builder)

    @builtin
    def __or__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.or_(self, other, _builder)

    @builtin
    def __ror__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.or_(other, self, _builder)

    @builtin
    def __xor__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.xor_(self, other, _builder)

    @builtin
    def __rxor__(self, other, _builder=None):
        other = _unwrap_if_constexpr(other)
        return semantic.xor_(other, self, _builder)

    @builtin
    def __lshift__(self, other, _builder=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        return semantic.shl(self, other, _builder)

    @builtin
    def __rlshift__(self, other, _builder=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        return semantic.shl(other, self, _builder)

    @builtin
    def __rshift__(self, other, _builder=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return semantic.ashr(self, other, _builder)
        else:
            return semantic.lshr(self, other, _builder)

    @builtin
    def __rrshift__(self, other, _builder=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return semantic.ashr(other, self, _builder)
        else:
            return semantic.lshr(other, self, _builder)

    
    @builtin
    def __gt__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.greater_than(other, self, _builder)

    
    @builtin
    def __ge__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.greater_equal(self, other, _builder)

    @builtin
    def __rge__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.greater_equal(other, self, _builder)

    
    @builtin
    def __lt__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.less_than(other, self, _builder)

    
    @builtin
    def __le__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.less_equal(other, self, _builder)

    
    @builtin
    def __eq__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.equal(self, other, _builder)

    @builtin
    def __req__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.equal(other, self, _builder)

    @builtin
    def __ne__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.not_equal(self, other, _builder)

    @builtin
    def __rne__(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.not_equal(other, self, _builder)

    @builtin
    def logical_and(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.logical_and(self, other, _builder)

    @builtin
    def logical_or(self, other, _builder=None):
        other = semantic.to_tensor(other, _builder)
        return semantic.logical_or(self, other, _builder)

    
    
    @builtin
    def __not__(self, _builder=None):
        return semantic.not_(self, _builder)

    @builtin
    def __getitem__(self, slices, _builder=None):
        import builtins
        if isinstance(slices, (builtins.slice, slice, constexpr)) or slices is None:
            slices = [slices]
        if isinstance(slices, tuple):
            slices = slices.values
        ret = self
        for dim, sl in enumerate(slices):
            if sl is None or isinstance(sl, constexpr) and sl.value is None:
                ret = semantic.expand_dims(ret, dim, _builder)
            elif isinstance(sl, (builtins.slice, slice)) and sl.start is None and sl.stop is None and sl.step is None:
                pass
            else:
                raise ValueError(f"unsupported tensor index: {sl}")
        return ret

    @property
    def T(self):
        
        assert False, "Transposition must be created by the AST Visitor"

    @builtin
    def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _builder=None):
        
        return cast(self, dtype, fp_downcast_rounding, bitcast, _builder=_builder)

    
    
    
    
    
    def broadcast_to(self, *shape) -> tensor:
        ...

    def trans(self, *dims) -> tensor:
        ...

    def permute(self, *dims) -> tensor:
        ...

    def split(self) -> tuple[tensor, tensor]:
        ...

    def view(self, *shape) -> tensor:
        ...

    def reshape(self, *shape) -> tensor:
        ...

    def expand_dims(self, axis) -> tensor:
        ...

    def cast(self, dtype, fp_downcast_rounding=None, bitcast=False) -> tensor:
        ...

    def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="") -> tensor:
        ...

    def advance(self, offsets) -> tensor:
        ...

    def atomic_cas(self, cmp, val, sem=None, scope=None) -> tensor:
        ...

    def atomic_xchg(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_add(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_max(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_min(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_and(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_or(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_xor(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def exp(self) -> tensor:
        ...

    def log(self) -> tensor:
        ...

    def cos(self) -> tensor:
        ...

    def sin(self) -> tensor:
        ...

    def sqrt(self) -> tensor:
        ...

    def rsqrt(self) -> tensor:
        ...

    def abs(self) -> tensor:
        ...

    def reduce(self, axis, combine_fn, keep_dims=False) -> tensor:
        ...

    def associative_scan(self, axis, combine_fn, reverse=False) -> tensor:
        ...

    def gather(self, indices, axis) -> tensor:
        ...

    def histogram(self, num_bins) -> tensor:
        ...

    def cdiv(self, div) -> tensor:
        ...

    def sigmoid(self) -> tensor:
        ...

    def softmax(self, ieee_rounding=False) -> tensor:
        ...

    def ravel(self) -> tensor:
        ...

    def max(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmax(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def min(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmin(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def sum(self, axis=None, keep_dims=False, dtype=None) -> tensor:
        ...

    def xor_sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def cumsum(self, axis=0, reverse=False) -> tensor:
        ...

    def cumprod(self, axis=0, reverse=False) -> tensor:
        ...

    def sort(self, dim: constexpr = None, descending: constexpr = CONSTEXPR_0) -> tensor:
        ...

    def flip(self, dim=None) -> tensor:
        ...


class tuple(base_value):

    def __init__(self, args: list, type: tuple_type = None):
        self.values = [i for i in args]

        def get_type(x):
            if isinstance(x, dtype):
                return dtype
            if isinstance(x, int):
                return constexpr
            return x.type

        self.type = type or tuple_type([get_type(x) for x in self.values])

    def __getitem__(self, idx: constexpr):
        if isinstance(idx, int):
            idx = constexpr(idx)
        if isinstance(idx, constexpr):
            return self.values[idx]
        else:
            import builtins
            assert isinstance(idx, (slice, builtins.slice))
            return tuple(self.values[idx.start:idx.stop:idx.step])

    def __getattr__(self, name):
        return self.values[self.type.fields.index(name)]

    
    def __setitem__(self, idx: constexpr, value):
        if isinstance(idx, int):
            idx = constexpr(idx)
        assert isinstance(idx, constexpr)
        self.values[idx] = value

    def __add__(self, other):
        if isinstance(other, list):
            other = tuple(other)
        return tuple(self.values + other.values)
        

    def __mul__(self, other):
        assert isinstance(other, constexpr)
        return tuple(self.values * other.value)

    def __eq__(self, other):
        import builtins
        if isinstance(other, (list, builtins.tuple)):
            other = tuple(other)
        return constexpr(self.values == other.values)

    def __hash__(self):
        import builtins
        return hash(builtins.tuple(self.values))

    def __str__(self):
        return str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def _flatten_ir(self, handles: List[ir.value]):
        for v in self.values:
            v._flatten_ir(handles)


class slice:

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
        self.type = slice_type()


class tensor_descriptor_base_type(base_type):

    def __init__(self, block_type: block_type):
        self.block_type = block_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        value = tensor_descriptor_base(handles[cursor], self.block_type)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(builder.create_tensor_descriptor_type(self.block_type.to_ir(builder)))

    def __str__(self) -> str:
        
        return f"tensor_descriptor<{self.block_type}>"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False
        return self.block_type == other.block_type

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}"


class tensor_descriptor_base(base_value):
    

    def __init__(self, handle, block_type: block_type):
        
        super().__init__()

        self.handle = handle  
        self.type = tensor_descriptor_base_type(block_type)  

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, offsets: Sequence[constexpr | tensor], _builder=None) -> tensor:
        
        return semantic.descriptor_load(self, offsets, "", "", _builder)

    @builtin
    def store(self, offsets: Sequence[constexpr | tensor], value: tensor, _builder=None) -> tensor:
        
        return semantic.descriptor_store(self, value, offsets, _builder)

    @builtin
    def gather(self, *args, _builder=None) -> tensor:
        
        assert len(args) == 2, f"descriptor gather only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return semantic.descriptor_gather(self, x_offsets, y_offset, "", "", _builder)

    @builtin
    def scatter(self, value, *args, _builder=None) -> tensor:
        
        assert len(args) == 2, f"descriptor scatter only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return semantic.descriptor_scatter(self, value, x_offsets, y_offset, _builder)


class tensor_descriptor_type(tensor_descriptor_base_type):

    def __init__(self, block_type: block_type, shape_type: tuple_type, strides_type: tuple_type):
        self.block_type = block_type
        self.shape_type = shape_type
        self.strides_type = strides_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        shape = shape.values
        strides = strides.values
        value = tensor_descriptor(handle, shape, strides, self.block_type)
        return value, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        super()._flatten_ir_types(builder, out)
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return super().__eq__(other) and (self.shape_type == other.shape_type) and (self.strides_type
                                                                                    == other.strides_type)


class tensor_descriptor(tensor_descriptor_base):
    

    def __init__(self, handle, shape: List[tensor], strides: List[tensor], block_type: block_type):
        
        
        super().__init__(handle, block_type)
        self.type = tensor_descriptor_type(
            block_type,
            shape_type=tuple_type([s.type for s in shape]),
            strides_type=tuple_type([s.type for s in strides]),
        )
        
        self.shape = shape
        self.strides = strides

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        handles.extend(s.handle for s in self.shape)
        handles.extend(s.handle for s in self.strides)


def get_bool_env_var(var_name):
    v = os.getenv(var_name, "0")
    return v == "1" or v == "true" or v == "on"





def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


@builtin
def program_id(axis, _builder=None):
    
    
    
    
    
    
    
    
    axis = _constexpr_to_value(axis)
    return semantic.program_id(axis, _builder)


@builtin
def num_programs(axis, _builder=None):
    
    axis = _constexpr_to_value(axis)
    return semantic.num_programs(axis, _builder)







@builtin
def arange(start, end, _builder=None):
    start = _constexpr_to_value(start)
    end = _constexpr_to_value(end)
    return semantic.arange(start, end, _builder)


arange.__doc__ = f


def _unwrap_shape(shape):
    shape = _constexpr_to_value(shape)
    return [_constexpr_to_value(s) for s in shape]


def _shape_check_impl(shape):
    shape = _unwrap_shape(shape)
    validate_block_shape(shape)
    return shape


@builtin
def full(shape, value, dtype, _builder=None):
    
    shape = _shape_check_impl(shape)
    value = _constexpr_to_value(value)
    dtype = _constexpr_to_value(dtype)
    return semantic.full(shape, value, dtype, _builder)







@builtin
def broadcast(input, other, _builder=None):
    
    return semantic.broadcast_impl_value(input, other, _builder)


@_tensor_member_fn
@builtin
def broadcast_to(input, *shape, _builder=None):
    
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return semantic.broadcast_impl_shape(input, shape, _builder)


@_tensor_member_fn
@builtin
def trans(input: tensor, *dims, _builder=None):
    
    dims = _unwrap_iterable(dims)
    if not dims:
        dims = (1, 0)
    return semantic.permute(input, dims, _builder)


@_tensor_member_fn
@builtin
def permute(input, *dims, _builder=None):
    
    dims = _unwrap_iterable(dims)
    return semantic.permute(input, dims, _builder)


@builtin
def cat(input, other, can_reorder=False, _builder=None):
    
    return semantic.cat(input, other, can_reorder, _builder)


@builtin
def join(a, b, _builder=None):
    
    return semantic.join(a, b, _builder)


@jit
def _take_first(a, b):
    return a


@_tensor_member_fn
@builtin
def split(a, _builder=None, _generator=None) -> tuple[tensor, tensor]:
    
    
    
    
    was_rank_1 = len(a.shape) == 1
    if was_rank_1:
        a = semantic.expand_dims(a, 0, _builder)

    out_lhs, out_rhs = semantic.split(a, _builder)

    if was_rank_1:
        
        out_lhs = typing.cast(tensor, reduce(out_lhs, None, _take_first, _builder=_builder, _generator=_generator))
        out_rhs = typing.cast(tensor, reduce(out_rhs, None, _take_first, _builder=_builder, _generator=_generator))

    return out_lhs, out_rhs


@_tensor_member_fn
@builtin
def view(input, *shape, _builder=None):
    
    warn("view is deprecated, please use reshape with can_reorder being true.")
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return semantic.reshape(input, shape, can_reorder=True, builder=_builder)


@_tensor_member_fn
@builtin
def reshape(input, *shape, can_reorder=False, _builder=None):
    
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return semantic.reshape(input, shape, can_reorder, _builder)


def _wrap_axis(axis, ndim):
    if not (-ndim <= axis < ndim):
        raise ValueError(f"invalid axis {axis}. Expected {-ndim} <= axis < {ndim}")

    return axis if axis >= 0 else axis + ndim


@_tensor_member_fn
@builtin
def expand_dims(input, axis, _builder=None):
    
    input = semantic.to_tensor(input, _builder)
    axis = _constexpr_to_value(axis)
    axes = list(axis) if isinstance(axis, (Sequence, tuple)) else [axis]
    new_ndim = len(input.shape) + len(axes)
    axes = [_wrap_axis(_constexpr_to_value(d), new_ndim) for d in axes]

    if len(set(axes)) != len(axes):
        raise ValueError(f"expand_dims received duplicate axes, normalized axes = {axes}")

    ret = input
    for a in sorted(axes):
        ret = semantic.expand_dims(ret, a, _builder)
    return ret


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _builder=None):
    
    input = semantic.to_tensor(input, _builder)
    dtype = _constexpr_to_value(dtype)
    fp_downcast_rounding = _constexpr_to_value(fp_downcast_rounding)
    bitcast = _constexpr_to_value(bitcast)
    if bitcast:
        return semantic.bitcast(input, dtype, _builder)
    return semantic.cast(input, dtype, _builder, fp_downcast_rounding)







@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=float32,
        _builder=None):
    
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = _builder and "tf32" in _builder.options.allowed_dot_input_precisions
        default_precision = "tf32" if (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
        input_precision = os.getenv("TRITON_F32_DEFAULT", default_precision)

    input_precision = _constexpr_to_value(input_precision)
    out_dtype = _constexpr_to_value(out_dtype)
    max_num_imprecise_acc = _constexpr_to_value(max_num_imprecise_acc)
    return semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)


@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, out_dtype=float32,
               _builder=None):
    
    out_dtype = _constexpr_to_value(out_dtype)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    return semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, out_dtype,
                               _builder)







@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _builder=None):
    
    
    mask = _constexpr_to_value(mask)
    other = _constexpr_to_value(other)
    if mask is not None:
        mask = semantic.to_tensor(mask, _builder)
    if other is not None:
        other = semantic.to_tensor(other, _builder)
    padding_option = _constexpr_to_value(padding_option)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)
    return semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                         volatile, _builder)


@builtin
def _experimental_reinterpret_tensor_descriptor(desc_ptr, block_shape, dtype, _builder=None) -> tensor_descriptor_base:
    
    block_ty = block_type(_constexpr_to_value(dtype), block_shape)
    return semantic.reinterpret_tensor_descriptor(desc_ptr, block_ty, _builder)


@builtin
def _experimental_descriptor_load(desc_pointer, offsets, shape, dtype, _builder=None):
    
    desc = _experimental_reinterpret_tensor_descriptor(desc_pointer, shape, dtype, _builder=_builder)
    return desc.load(offsets, _builder=_builder)


@builtin
def _experimental_descriptor_store(desc_pointer, value, offsets, _builder=None):
    
    desc = _experimental_reinterpret_tensor_descriptor(desc_pointer, value.shape, value.dtype, _builder=_builder)
    return desc.store(offsets, value, _builder=_builder)


@builtin
def load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor],
                           _builder=None) -> tensor:
    
    return desc.load(offsets, _builder=_builder)


@builtin
def store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor], value: tensor,
                            _builder=None) -> tensor:
    
    return desc.store(offsets, value, _builder=_builder)


@_tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _builder=None):
    
    
    value = semantic.to_tensor(value, _builder)
    mask = _constexpr_to_value(mask)
    if mask is not None:
        mask = semantic.to_tensor(mask, _builder)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    return semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, _builder)


@builtin
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _builder=None):
    
    return semantic.make_block_ptr(base, shape, strides, offsets, block_shape, order, _builder)


@_tensor_member_fn
@builtin
def advance(base, offsets, _builder=None):
    
    return semantic.advance(base, offsets, _builder)


@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _builder=None,
) -> tensor_descriptor:
    
    return semantic.make_tensor_descriptor(base, shape, strides, block_shape, _builder)







def _add_atomic_docstr(name: str, has_cmp: bool = False) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f
        if has_cmp:
            docstr += 
        docstr += 
        func.__doc__ = docstr
        return func

    return _decorator


@_tensor_member_fn
@builtin
@_add_atomic_docstr("compare-and-swap", has_cmp=True)
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _builder=None):
    cmp = semantic.to_tensor(cmp, _builder)
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    return semantic.atomic_cas(pointer, cmp, val, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_xchg(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_add(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_max(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_min(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_and(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_or(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = semantic.to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_xor(pointer, val, mask, sem, scope, _builder)







@builtin
def where(condition, x, y, _builder=None):
    
    condition = semantic.to_tensor(condition, _builder)
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return semantic.where(condition, x, y, _builder)







@builtin
def add(x, y, sanitize_overflow: constexpr = True, _builder=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return semantic.add(x, y, sanitize_overflow, _builder)


@builtin
def sub(x, y, sanitize_overflow: constexpr = True, _builder=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return semantic.sub(x, y, sanitize_overflow, _builder)


@builtin
def mul(x, y, sanitize_overflow: constexpr = True, _builder=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return semantic.mul(x, y, sanitize_overflow, _builder)


@builtin
def minimum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _builder=None):
    
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x = _promote_bfloat16_to_float32(x, _builder=_builder)
    y = _promote_bfloat16_to_float32(y, _builder=_builder)
    propagate_nan = _constexpr_to_value(propagate_nan)
    return semantic.minimum(x, y, propagate_nan, _builder)


@builtin
def maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _builder=None):
    
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x = _promote_bfloat16_to_float32(x, _builder=_builder)
    y = _promote_bfloat16_to_float32(y, _builder=_builder)
    propagate_nan = _constexpr_to_value(propagate_nan)
    return semantic.maximum(x, y, propagate_nan, _builder)


@builtin
def clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _builder=None):
    
    x = semantic.to_tensor(x, _builder)
    min = semantic.to_tensor(min, _builder)
    max = semantic.to_tensor(max, _builder)
    x = _promote_bfloat16_to_float32(x, _builder=_builder)
    min = _promote_bfloat16_to_float32(min, _builder=_builder)
    max = _promote_bfloat16_to_float32(max, _builder=_builder)

    propagate_nan = _constexpr_to_value(propagate_nan)

    return semantic.clamp(x, min, max, propagate_nan, _builder)







def _add_reduction_docstr(name: str, return_indices_arg: str = None, tie_break_arg: str = None,
                          dtype_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = 
        if return_indices_arg is not None:
            docstr += f
        if tie_break_arg is not None:
            docstr += f
        if dtype_arg is not None:
            docstr += f

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@contextmanager
def _insertion_guard(builder):
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)


@_tensor_member_fn
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _builder=None, _generator=None):
    
    if isinstance(input, tensor):
        return reduce((input, ), axis, combine_fn, keep_dims=keep_dims, _builder=_builder, _generator=_generator)[0]

    def make_combine_region(reduce_op):
        param_types = [t.type.scalar for t in input] * 2
        region = reduce_op.get_region(0)
        with _insertion_guard(_builder):
            to_ir = lambda T: T.to_ir(_builder)
            block = _builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _builder=_builder)
        return t

    axis = _constexpr_to_value(axis)
    keep_dims = _constexpr_to_value(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = semantic.reduction(input, axis, make_combine_region, _builder)
    if keep_dims:
        if axis is not None:
            ret = tuple(expand_dims(t, axis, _builder=_builder) for t in ret)
        else:
            ret = tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def _promote_bfloat16_to_float32(t, _builder=None):
    scalar_ty = t.type.scalar

    
    if scalar_ty is bfloat16:
        return t.to(float32, _builder=_builder)
    return t


@builtin
def _reduce_with_indices(input, axis, combine_fn, keep_dims=False, _builder=None, _generator=None):
    axis = _constexpr_to_value(axis)
    n = input.shape[axis]
    index = arange(0, n, _builder=_builder)

    if len(input.shape) > 1:
        
        axes_to_expand = [constexpr(d) for d in builtins.range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _builder=_builder)
        index = broadcast_to(index, input.shape, _builder=_builder)

    rvalue, rindices = reduce((input, index), axis, combine_fn, keep_dims=keep_dims, _builder=_builder,
                              _generator=_generator)
    return rvalue, rindices







def _add_scan_docstr(name: str) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = 
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@_tensor_member_fn
@builtin
def associative_scan(input, axis, combine_fn, reverse=False, _builder=None, _generator=None):
    
    if isinstance(input, tensor):
        return associative_scan((input, ), axis, combine_fn, reverse, _builder=_builder, _generator=_generator)[0]

    def make_combine_region(scan_op):
        param_types = [t.type.scalar for t in input] * 2
        region = scan_op.get_region(0)
        with _insertion_guard(_builder):
            to_ir = lambda T: T.to_ir(_builder)
            block = _builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_scan_ret(*handles)

    axis = _constexpr_to_value(axis)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    return semantic.associative_scan(input, axis, make_combine_region, reverse, _builder)


@_tensor_member_fn
@builtin
def histogram(input, num_bins, _builder=None, _generator=None):
    
    num_bins = _constexpr_to_value(num_bins)
    return semantic.histogram(input, num_bins, _builder)


@_tensor_member_fn
@builtin
def gather(src, index, axis, _builder=None):
    
    axis = _constexpr_to_value(axis)
    return semantic.gather(src, index, axis, _builder)







@builtin
def debug_barrier(_builder=None):
    
    return semantic.debug_barrier(_builder)


@builtin
def multiple_of(input, values, _builder=None):
    
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.multiple_of(input, values)


@builtin
def max_contiguous(input, values, _builder=None):
    
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.max_contiguous(input, values)


@builtin
def max_constancy(input, values, _builder=None):
    
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.max_constancy(input, values)


@builtin
def assume(cond, _builder=None):
    
    return semantic.assume(semantic.to_tensor(cond, _builder), _builder)







@builtin
def static_print(*values, sep: str = " ", end: str = "\n", file=None, flush=False, _builder=None):
    
    pass


@builtin
def static_assert(cond, msg="", _builder=None):
    
    pass


@builtin
def device_print(prefix, *args, hex=False, _builder=None):
    
    import string
    prefix = _constexpr_to_value(prefix)
    assert isinstance(prefix, str), f"{prefix} is not string"
    b_ascii = True
    for ch in prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{prefix} is not an ascii string"
    new_args = []
    for arg in args:
        new_args.append(semantic.to_tensor(arg, _builder))
    return semantic.device_print(prefix, new_args, hex, _builder)


@builtin
def device_assert(cond, msg="", _builder=None):
    
    msg = _constexpr_to_value(msg)
    return semantic.device_assert(semantic.to_tensor(cond, _builder), msg, _builder)


@builtin
def inline_asm_elementwise(asm: str, constraints: str, args: Sequence, dtype: Union[dtype, Sequence[dtype]],
                           is_pure: bool, pack: int, _builder=None):
    
    asm = _constexpr_to_value(asm)
    constraints = _constexpr_to_value(constraints)
    pack = _constexpr_to_value(pack)
    is_pure = _constexpr_to_value(is_pure)

    
    try:
        iter(dtype)  
        has_multiple_outputs = True
    except TypeError:
        has_multiple_outputs = False
        dtype = (dtype, )  

    dtype = typing.cast(Sequence[_DtypeClass], dtype)

    res_tys = dtype
    if dispatch_args := [semantic.to_tensor(arg, _builder) for arg in args]:
        bin_op_type_checking = partial(
            semantic.binary_op_type_checking_impl,
            builder=_builder,
            arithmetic_check=False,
            allow_lhs_ptr=True,
            allow_rhs_ptr=True,
        )
        broadcast_arg = dispatch_args[0]
        
        for item in dispatch_args:
            _, broadcast_arg = bin_op_type_checking(item, broadcast_arg)
        if broadcast_arg.shape:
            
            for i, item in enumerate(dispatch_args):
                dispatch_args[i], _ = bin_op_type_checking(item, broadcast_arg)
            res_tys = [block_type(dt, broadcast_arg.shape) for dt in dtype]
    handles = [t.handle for t in dispatch_args]
    call = _builder.create_inline_asm(asm, constraints, handles, [ty.to_ir(_builder) for ty in res_tys], is_pure, pack)

    if not has_multiple_outputs:
        return tensor(call.get_result(0), res_tys[0])
    return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(res_tys))







class static_range:
    

    def __init__(self, arg1, arg2=None, step=None):
        assert isinstance(arg1, constexpr), f"{arg1} used as tl.static_range start value is not a constexpr"
        if step is None:
            self.step = constexpr(1)
        else:
            assert isinstance(step, constexpr), f"{step} used as tl.static_range step value is not a constexpr"
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            assert isinstance(arg2, constexpr), f"{arg2} used as tl.static_range end value is not a constexpr"
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")


class range:
    

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False):
        if step is None:
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor
        self.disallow_acc_multi_buffer = disallow_acc_multi_buffer
        self.flatten = flatten
        self.warp_specialize = warp_specialize

    def __iter__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")







def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple,
             is_pure: bool, _builder=None):
    
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        if ret_shape:
            ret_type = block_type(ret_type, ret_shape)
        return tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder), is_pure), ret_type)


@builtin
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool,
                       _builder=None, check_args=True):
    
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = semantic.to_tensor(dispatch_args[i], _builder)
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if len(arg_types) > 0:
        arg_types = tuple(arg_types)
        arithmetic_check = True
        
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        if check_args:
            
            for item in dispatch_args:
                _, broadcast_arg = semantic.binary_op_type_checking_impl(item, broadcast_arg, _builder,
                                                                         allow_lhs_ptr=True, allow_rhs_ptr=True,
                                                                         arithmetic_check=arithmetic_check)
            
            for i in builtins.range(len(dispatch_args)):
                dispatch_args[i], _ = semantic.binary_op_type_checking_impl(dispatch_args[i], broadcast_arg, _builder,
                                                                            allow_lhs_ptr=True, allow_rhs_ptr=True,
                                                                            arithmetic_check=arithmetic_check)
            if not all_scalar:
                ret_shape = broadcast_arg.shape
    func = _builder.create_extern_elementwise
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, is_pure, _builder)


def binary_op_type_legalization(lhs, rhs, builder):
    
    return semantic.binary_op_type_checking_impl(lhs, rhs, builder)


def extern(fn):
    
    return builtin(fn)
