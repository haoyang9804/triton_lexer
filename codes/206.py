from abc import abstractmethod
import os

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

import numpy as np


class TritonModel(object):

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
    ):

        self.max_batch_size = max_batch_size
        self.input_names = input_names
        self.output_names = output_names
        self.c = channels
        assert channels in [
            1,
            3,
        ], "TAO Toolkit models only support 1 or 3 channel inputs."
        self.h = height
        self.w = width
        self.data_format = data_format
        self.triton_dtype = triton_dtype
        self.scale = 1
        if channels == 3:
            self.mean = [0.0, 0.0, 0.0]
        else:
            self.mean = [0]
        self.mean = np.asarray(self.mean).astype(np.float32)
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            self.mean = self.mean[:, np.newaxis, np.newaxis]

    @staticmethod
    def parse_model(model_metadata, model_config):

        raise NotImplementedError("Base class doesn't implement this method.")

    @classmethod
    def from_metadata(cls, model_metadata, model_config):

        parsed_outputs = cls.parse_model(model_metadata, model_config)
        (
            max_batch_size,
            input_names,
            output_names,
            channels,
            height,
            width,
            data_format,
            triton_dtype,
        ) = parsed_outputs
        return cls(
            max_batch_size,
            input_names,
            output_names,
            channels,
            height,
            width,
            data_format,
            triton_dtype,
        )

    def get_config(self):

        config_dict = {
            "data_format": self.data_format,
            "max_batch_size": self.max_batch_size,
            "channels": self.c,
            "width": self.w,
            "height": self.h,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "triton_dtype": self.triton_dtype,
        }
        return config_dict

    def preprocess(self, image):

        image = (image - self.mean) * self.scale
        return image
