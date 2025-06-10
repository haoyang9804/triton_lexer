import os
import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tao_triton.python.model.triton_model import TritonModel

CHANNEL_MODES = ["rgb", "bgr", "l"]


class ClassificationModel(TritonModel):

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
        if channels == 1:
            self.mean = [117.3786]
        elif channels == 3:
            self.mean = [103.939, 116.779, 123.68]

        if channel_mode.lower() == "rgb":

            self.mean.reverse()
        self.mean = np.asarray(self.mean).astype(np.float32)
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            self.mean = self.mean[:, np.newaxis, np.newaxis]

    @staticmethod
    def parse_model(model_metadata, model_config):

        if len(model_metadata.inputs) != 1:
            raise Exception(
                "expecting 1 input, got {}".format(len(model_metadata.inputs))
            )
        if len(model_metadata.outputs) != 1:
            raise Exception(
                "expecting 1 output, got {}".format(len(model_metadata.outputs))
            )

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config.input)
                )
            )

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception(
                "expecting output datatype to be FP32, model '"
                + model_metadata.name
                + "' output type is "
                + output_metadata.datatype
            )

        output_batch_dim = model_config.max_batch_size > 0
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, model_metadata.name, len(input_metadata.shape)
                )
            )

        if type(input_config.format) == str:
            FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
            input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

        if (input_config.format != mc.ModelInput.FORMAT_NCHW) and (
            input_config.format != mc.ModelInput.FORMAT_NHWC
        ):
            raise Exception(
                "unexpected input format "
                + mc.ModelInput.Format.Name(input_config.format)
                + ", expecting "
                + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
                + " or "
                + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
            )

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        print(
            model_config.max_batch_size,
            input_metadata.name,
            output_metadata.name,
            c,
            h,
            w,
            input_config.format,
            input_metadata.datatype,
        )

        return (
            model_config.max_batch_size,
            input_metadata.name,
            [output_metadata.name],
            c,
            h,
            w,
            input_config.format,
            input_metadata.datatype,
        )
