import json
import logging
import pathlib
from typing import Dict

import numpy as np
from google.protobuf import json_format, text_format

from pytriton.exceptions import PyTritonModelConfigError

from .common import QueuePolicy, TimeoutAction
from .triton_model_config import (
    DeviceKind,
    DynamicBatcher,
    ResponseCache,
    TensorSpec,
    TritonModelConfig,
)

try:
    import tritonclient.grpc as grpc_client
    from tritonclient import utils as client_utils
except ImportError:
    try:
        import tritonclientutils as client_utils
        import tritongrpcclient as grpc_client
    except ImportError:
        client_utils = None
        grpc_client = None

LOGGER = logging.getLogger(__name__)


class ModelConfigParser:

    @classmethod
    def from_dict(cls, model_config_dict: Dict) -> TritonModelConfig:

        LOGGER.debug(
            "Parsing Triton config model from dict: \n%s",
            json.dumps(model_config_dict, indent=4),
        )

        if model_config_dict.get("max_batch_size", 0) > 0:
            batching = True
        else:
            batching = False

        dynamic_batcher_config = model_config_dict.get("dynamic_batching")
        if dynamic_batcher_config is not None:
            batcher = cls._parse_dynamic_batching(dynamic_batcher_config)
        else:
            batcher = None

        instance_group = {
            DeviceKind(entry["kind"]): entry.get("count")
            for entry in model_config_dict.get("instance_group", [])
        }

        decoupled = model_config_dict.get("model_transaction_policy", {}).get(
            "decoupled", False
        )

        backend_parameters_config = model_config_dict.get("parameters", [])
        if isinstance(backend_parameters_config, list):

            LOGGER.debug(
                "backend_parameters_config is a list of strings: %s",
                backend_parameters_config,
            )
            backend_parameters = {name: "" for name in backend_parameters_config}
        elif isinstance(backend_parameters_config, dict):

            LOGGER.debug(
                "backend_parameters_config is a dictionary: %s",
                backend_parameters_config,
            )
            backend_parameters = {
                name: backend_parameters_config[name]["string_value"]
                for name in backend_parameters_config
            }
        else:

            LOGGER.error(
                "Invalid type %s for backend_parameters_config: %s",
                type(backend_parameters_config),
                backend_parameters_config,
            )
            raise TypeError(
                f"Invalid type for backend_parameters_config: {type(backend_parameters_config)}"
            )

        inputs = [
            cls.rewrite_io_spec(item, "input", idx)
            for idx, item in enumerate(model_config_dict.get("input", []))
        ] or None
        outputs = [
            cls.rewrite_io_spec(item, "output", idx)
            for idx, item in enumerate(model_config_dict.get("output", []))
        ] or None

        response_cache_config = model_config_dict.get("response_cache")
        if response_cache_config:
            response_cache = cls._parse_response_cache(response_cache_config)
        else:
            response_cache = None

        return TritonModelConfig(
            model_name=model_config_dict["name"],
            batching=batching,
            max_batch_size=model_config_dict.get("max_batch_size", 0),
            batcher=batcher,
            inputs=inputs,
            outputs=outputs,
            instance_group=instance_group,
            decoupled=decoupled,
            backend_parameters=backend_parameters,
            response_cache=response_cache,
        )

    @classmethod
    def from_file(cls, *, config_path: pathlib.Path) -> TritonModelConfig:

        from tritonclient.grpc import model_config_pb2

        LOGGER.debug("Parsing Triton config model config_path=%s", config_path)

        with config_path.open("r") as config_file:
            payload = config_file.read()
            model_config_proto = text_format.Parse(
                payload, model_config_pb2.ModelConfig()
            )

        model_config_dict = json_format.MessageToDict(
            model_config_proto, preserving_proto_field_name=True
        )
        return ModelConfigParser.from_dict(model_config_dict=model_config_dict)

    @classmethod
    def rewrite_io_spec(cls, item: Dict, io_type: str, idx: int) -> TensorSpec:

        name = item.get("name")
        if not name:
            raise PyTritonModelConfigError(
                f"Name for {io_type} at index {idx} not provided."
            )

        data_type = item.get("data_type")
        if not data_type:
            raise PyTritonModelConfigError(
                f"Data type for {io_type} with name `{name}` not defined."
            )

        data_type_val = data_type.split("_")
        if len(data_type_val) != 2:
            raise PyTritonModelConfigError(
                f"Invalid data type `{data_type}` for {io_type} with name `{name}` not defined. "
                "The expected name is TYPE_{type}."
            )

        data_type = data_type_val[1]
        if data_type == "STRING":
            dtype = np.bytes_
        else:
            dtype = client_utils.triton_to_np_dtype(data_type)
            if dtype is None:
                raise PyTritonModelConfigError(
                    f"Unsupported data type `{data_type}` for {io_type} with name `{name}`"
                )

            dtype = np.dtype("bool") if dtype is bool else dtype

        dims = item.get("dims", [])
        if not dims:
            raise PyTritonModelConfigError(
                f"Dimension for {io_type} with name `{name}` not defined."
            )

        shape = tuple(int(s) for s in dims)

        optional = item.get("optional", False)
        return TensorSpec(
            name=item["name"], shape=shape, dtype=dtype, optional=optional
        )

    @classmethod
    def _parse_dynamic_batching(cls, dynamic_batching_config: Dict) -> DynamicBatcher:

        default_queue_policy = None
        default_queue_policy_config = dynamic_batching_config.get(
            "default_queue_policy"
        )
        if default_queue_policy_config:
            default_queue_policy = QueuePolicy(
                timeout_action=TimeoutAction(
                    default_queue_policy_config.get(
                        "timeout_action", TimeoutAction.REJECT.value
                    )
                ),
                default_timeout_microseconds=int(
                    default_queue_policy_config.get("default_timeout_microseconds", 0)
                ),
                allow_timeout_override=bool(
                    default_queue_policy_config.get("allow_timeout_override", False)
                ),
                max_queue_size=int(
                    default_queue_policy_config.get("max_queue_size", 0)
                ),
            )

        priority_queue_policy = None
        priority_queue_policy_config = dynamic_batching_config.get(
            "priority_queue_policy"
        )
        if priority_queue_policy_config:
            priority_queue_policy = {}
            for priority, queue_policy_config in priority_queue_policy_config.items():
                queue_policy = QueuePolicy(
                    timeout_action=TimeoutAction(
                        queue_policy_config.get(
                            "timeout_action", TimeoutAction.REJECT.value
                        )
                    ),
                    default_timeout_microseconds=int(
                        queue_policy_config.get("default_timeout_microseconds", 0)
                    ),
                    allow_timeout_override=bool(
                        queue_policy_config.get("allow_timeout_override", False)
                    ),
                    max_queue_size=int(queue_policy_config.get("max_queue_size", 0)),
                )
                priority_queue_policy[int(priority)] = queue_policy

        batcher = DynamicBatcher(
            preferred_batch_size=dynamic_batching_config.get("preferred_batch_size"),
            max_queue_delay_microseconds=int(
                dynamic_batching_config.get("max_queue_delay_microseconds", 0)
            ),
            preserve_ordering=bool(
                dynamic_batching_config.get("preserve_ordering", False)
            ),
            priority_levels=int(dynamic_batching_config.get("priority_levels", 0)),
            default_priority_level=int(
                dynamic_batching_config.get("default_priority_level", 0)
            ),
            default_queue_policy=default_queue_policy,
            priority_queue_policy=priority_queue_policy,
        )
        return batcher

    @classmethod
    def _parse_response_cache(cls, response_cache_config: Dict) -> ResponseCache:

        response_cache = ResponseCache(
            enable=bool(response_cache_config["enable"]),
        )
        return response_cache
