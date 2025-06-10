from enum import IntEnum
from typing import Union

import tritonserver
from pydantic import Field
from pydantic.dataclasses import dataclass
from tritonfrontend._api._error_mapping import handle_triton_error
from tritonfrontend._c.tritonfrontend_bindings import (
    InvalidArgumentError,
    TritonFrontendGrpc,
)


class Grpc_compression_level(IntEnum):
    NONE = 0
    LOW = 1
    MED = 2
    HIGH = 3
    COUNT = 4


class KServeGrpc:
    Grpc_compression_level = Grpc_compression_level

    @dataclass
    class Options:

        address: str = "0.0.0.0"
        port: int = Field(8001, ge=0, le=65535)
        reuse_port: bool = False

        use_ssl: bool = False
        server_cert: str = ""
        server_key: str = ""
        root_cert: str = ""
        use_mutual_auth: bool = False

        keepalive_time_ms: int = Field(7_200_000, ge=0)
        keepalive_timeout_ms: int = Field(20_000, ge=0)
        keepalive_permit_without_calls: bool = False
        http2_max_pings_without_data: int = Field(2, ge=0)
        http2_min_recv_ping_interval_without_data_ms: int = Field(300_000, ge=0)
        http2_max_ping_strikes: int = Field(2, ge=0)
        max_connection_age_ms: int = Field(0, ge=0)
        max_connection_age_grace_ms: int = Field(0, ge=0)

        infer_compression_level: Union[int, Grpc_compression_level] = (
            Grpc_compression_level.NONE
        )
        infer_thread_count: int = Field(2, ge=0)
        infer_allocation_pool_size: int = Field(8, ge=0)
        max_response_pool_size: int = Field(2_147_483_647, ge=0)
        forward_header_pattern: str = ""

        def __post_init__(self):
            if isinstance(self.infer_compression_level, Grpc_compression_level):
                self.infer_compression_level = self.infer_compression_level.value

    @handle_triton_error
    def __init__(self, server: tritonserver, options: "KServeGrpc.Options" = None):
        server_ptr = server._ptr()

        if options is None:
            options = KServeGrpc.Options()

        if not isinstance(options, KServeGrpc.Options):
            raise InvalidArgumentError(
                "Incorrect type for options. options argument must be of type KServeGrpc.Options"
            )

        options_dict: dict[str, Union[int, bool, str]] = options.__dict__

        self.triton_frontend = TritonFrontendGrpc(server_ptr, options_dict)

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
