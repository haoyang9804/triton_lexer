import os
import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tao_triton.python.model.triton_model import TritonModel

CHANNEL_MODES = ["rgb", "bgr", "l"]


class VisualChangeNetModel(TritonModel):

    def __init__(
        self,
        max_batch_size,
        input_names,
        output_names,
        channels,
        height,
        width,
        data_format,
        triton_dtype,
        channel_mode="RGB",
    ):

        super().__init__(
            max_batch_size,
            input_names,
            output_names,
            channels,
            height,
            width,
            data_format,
            triton_dtype,
        )
        self.scale = 1.0

    @staticmethod
    def parse_model(model_metadata, model_config):

        print(len(model_metadata.inputs))
        if len(model_metadata.inputs) != 2:
            raise Exception(
                "expecting 2 input, got {}".format(len(model_metadata.inputs))
            )

        if len(model_metadata.outputs) != 5:
            raise Exception(
                "expecting 5 output, got {}".format(len(model_metadata.outputs))
            )

        if len(model_config.input) != 2:
            raise Exception(
                "expecting 2 input in model configuration, got {}".format(
                    len(model_config.input)
                )
            )
        if len(model_config.output) != 5:
            raise Exception(
                "expecting 5 input in model configuration, got {}".format(
                    len(model_config.output)
                )
            )

        input_metadata = model_metadata.inputs
        input_config = model_config.input
        output_metadata = model_metadata.outputs

        for _, data in enumerate(output_metadata):
            if data.datatype != "FP32":
                raise Exception(
                    "expecting output datatype to be FP32, model '"
                    + data.name
                    + "' output type is "
                    + data.datatype
                )

        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if type(input_config) == tuple:
            for i in range(len(input_config)):
                if len(input_metadata[i].shape) != expected_input_dims:
                    raise Exception(
                        "expecting input to have {} dimensions, model '{}' input has {}".format(
                            expected_input_dims,
                            model_metadata.name,
                            len(input_metadata[i].shape),
                        )
                    )

        if type(input_config) == tuple:
            for i in range(len(input_config)):
                if type(input_config[i].format) == str:
                    FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
                    input_config[i].format = FORMAT_ENUM_TO_INT[input_config[i].format]

        if type(input_config) == tuple:
            for i in range(len(input_config)):
                if (input_config[i].format != mc.ModelInput.FORMAT_NCHW) and (
                    input_config[i].format != mc.ModelInput.FORMAT_NHWC
                ):
                    raise Exception(
                        "unexpected input format "
                        + mc.ModelInput.Format.Name(input_config[i].format)
                        + ", expecting "
                        + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
                        + " or "
                        + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
                    )

        if type(input_config) == tuple:
            for i in range(len(input_config)):
                if input_config[i].format == mc.ModelInput.FORMAT_NHWC:
                    h = input_metadata[i].shape[1 if input_batch_dim else 0]
                    w = input_metadata[i].shape[2 if input_batch_dim else 1]
                    c = input_metadata[i].shape[3 if input_batch_dim else 2]
                else:
                    c = input_metadata[i].shape[1 if input_batch_dim else 0]
                    h = input_metadata[i].shape[2 if input_batch_dim else 1]
                    w = input_metadata[i].shape[3 if input_batch_dim else 2]

        print(
            model_config.max_batch_size,
            [input_meta.name for input_meta in input_metadata],
            [data.name for data in output_metadata],
            c,
            h,
            w,
            input_config[0].format,
            input_metadata[0].datatype,
        )
        return (
            model_config.max_batch_size,
            [input_meta.name for input_meta in input_metadata],
            [data.name for data in output_metadata],
            c,
            h,
            w,
            input_config[0].format,
            input_metadata[0].datatype,
        )
