import sys

import tritonserver
from tritonfrontend._c.tritonfrontend_bindings import (
    AlreadyExistsError,
    InternalError,
    InvalidArgumentError,
    NotFoundError,
    TritonError,
    UnavailableError,
    UnknownError,
    UnsupportedError,
)


ERROR_MAPPING = {
    TritonError: tritonserver.TritonError,
    NotFoundError: tritonserver.NotFoundError,
    UnknownError: tritonserver.UnknownError,
    InternalError: tritonserver.InternalError,
    InvalidArgumentError: tritonserver.InvalidArgumentError,
    UnavailableError: tritonserver.UnavailableError,
    AlreadyExistsError: tritonserver.AlreadyExistsError,
    UnsupportedError: tritonserver.UnsupportedError,
}


def handle_triton_error(func):
    def error_handling_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TritonError:
            exc_type, exc_value, _ = sys.exc_info()

            raise ERROR_MAPPING[exc_type](exc_value) from None

    return error_handling_wrapper
