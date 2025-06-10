from typing import Union

import tritonserver
from pydantic import Field
from pydantic.dataclasses import dataclass
from tritonfrontend._api._error_mapping import handle_triton_error
from tritonfrontend._c.tritonfrontend_bindings import (
    InvalidArgumentError,
    TritonFrontendHttp,
)


class KServeHttp:
    @dataclass
    class Options:
        address: str = "0.0.0.0"
        port: int = Field(8000, ge=0, le=65535)
        reuse_port: bool = False
        thread_count: int = Field(8, gt=0)
        header_forward_pattern: str = ""

    @handle_triton_error
    def __init__(self, server: tritonserver, options: "KServeHttp.Options" = None):
        server_ptr = server._ptr()

        if options is None:
            options = KServeHttp.Options()

        if not isinstance(options, KServeHttp.Options):
            raise InvalidArgumentError(
                "Incorrect type for options. options argument must be of type KServeHttp.Options"
            )

        options_dict: dict[str, Union[int, bool, str]] = options.__dict__

        self.triton_frontend = TritonFrontendHttp(server_ptr, options_dict)

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
