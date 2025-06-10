import os
import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tao_triton.python.model.triton_model import TritonModel


class PoseClassificationModel(TritonModel):

    def __init__(
        self,
        max_batch_size,
        input_names,
        output_names,
        channels,
        seq_length,
        num_joint,
        num_person,
        triton_dtype,
    ):

        self.max_batch_size = max_batch_size
        self.input_names = input_names
        self.output_names = output_names
        self.channels = channels
        self.seq_length = seq_length
        self.num_joint = num_joint
        self.num_person = num_person
        self.triton_dtype = triton_dtype

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

        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 4 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, model_metadata.name, len(input_metadata.shape)
                )
            )

        c = input_metadata.shape[1 if input_batch_dim else 0]
        t = input_metadata.shape[2 if input_batch_dim else 1]
        v = input_metadata.shape[3 if input_batch_dim else 2]
        m = input_metadata.shape[4 if input_batch_dim else 3]

        print(
            model_config.max_batch_size,
            input_metadata.name,
            output_metadata.name,
            c,
            t,
            v,
            m,
            input_metadata.datatype,
        )

        return (
            model_config.max_batch_size,
            input_metadata.name,
            [output_metadata.name],
            c,
            t,
            v,
            m,
            input_metadata.datatype,
        )
