from typing import Union

import tritonserver
from pydantic import Field
from pydantic.dataclasses import dataclass
from tritonfrontend._api._error_mapping import handle_triton_error
from tritonfrontend._c.tritonfrontend_bindings import (
    InvalidArgumentError,
    TritonFrontendMetrics,
)


class Metrics:
    @dataclass
    class Options:
        address: str = "0.0.0.0"
        port: int = Field(8002, ge=0, le=65535)
        thread_count: int = Field(1, gt=0)

    @handle_triton_error
    def __init__(self, server: tritonserver, options: "Metrics.Options" = None):
        server_ptr = server._ptr()

        if options is None:
            options = Metrics.Options()

        if not isinstance(options, Metrics.Options):
            raise InvalidArgumentError(
                "Incorrect type for options. options argument must be of type Metrics.Options"
            )

        options_dict: dict[str, Union[int, bool, str]] = options.__dict__

        self.triton_frontend = TritonFrontendMetrics(server_ptr, options_dict)

    def __enter__(self):
        self.triton_frontend.start()
        return self

    @handle_triton_error
    def __exit__(self, exc_type, exc_value, traceback):
        self.triton_frontend.stop()
        if exc_type:
            raise exc_type(exc_value)

    @handle_triton_error
    def start(self):
        self.triton_frontend.start()

    @handle_triton_error
    def stop(self):
        self.triton_frontend.stop()
