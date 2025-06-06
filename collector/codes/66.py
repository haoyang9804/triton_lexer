import triton

from packaging.version import Version
from .compile import (
    make_ast_source,
    kernel_name_suffix,
    materialize_c_params,
    dump_c_code,
)

_triton_ver = Version(triton.__version__)
if _triton_ver.major < 3:
    raise RuntimeError("AOT compilation requires triton>=3.0.0")

__all__ = [
    "make_ast_source",
    "kernel_name_suffix",
    "materialize_c_params",
    "dump_c_code",
]
