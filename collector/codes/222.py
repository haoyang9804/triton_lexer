import json
import logging
import pathlib

import cloudpickle
import torch
import triton_python_backend_utils as pb_utils

from merlin.systems.triton.utils import triton_error_handling, triton_multi_request
from nvtabular.inference.triton import _convert_string2pytorch_dtype, _convert_tensor

LOG = logging.getLogger("nvtabular")

sparse_value_marker = "__values"
sparse_offsets_marker = "__offsets"


class TritonPythonModel:

    def initialize(self, args):

        repository_path = pathlib.Path(args["model_repository"])
        model_version = str(args["model_version"])

        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent

        model_path = repository_path / model_version / "model.pkl"

        with open(str(model_path), "rb") as model_file:
            self.model = cloudpickle.load(model_file)

        model_path = repository_path / model_version / "model.pth"
        self.model.load_state_dict(torch.load(str(model_path)))
        self.model.eval()

        self.model_config = json.loads(args["model_config"])

        model_info_path = repository_path / model_version / "model_info.json"
        self.model_info = None
        model_info_file = pathlib.Path(model_info_path)
        if model_info_file.exists():
            with open(str(model_info_path), encoding="utf-8") as json_file:
                self.model_info = json.load(json_file)

        self.inputs = {}
        self.sparse_inputs = {}
        self.outputs = {}
        len_svm = len(sparse_value_marker)
        len_snm = len(sparse_offsets_marker)

        for val in self.model_config["input"]:
            name = val["name"]

            if len(name) > len_svm:
                if name[-len_svm:] == sparse_value_marker:
                    self.sparse_inputs[name[0 : (len(name) - len_svm)]] = (
                        _convert_string2pytorch_dtype(val["data_type"])
                    )
                elif name[-len_snm:] != sparse_offsets_marker:
                    self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])
            else:
                if len(name) > len_snm:
                    if name[-len_snm:] != sparse_offsets_marker:
                        self.inputs[name] = _convert_string2pytorch_dtype(
                            val["data_type"]
                        )
                else:
                    self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])

        for val in self.model_config["output"]:
            self.outputs[val["name"]] = _convert_string2pytorch_dtype(val["data_type"])

    @triton_multi_request
    @triton_error_handling
    def execute(self, request):

        with torch.no_grad():

            input_dict = {}
            for name, dtype in self.inputs.items():

                if self.model_info["use_fix_dtypes"]:
                    dtype = _convert_dtype(dtype)
                input_dict[name] = torch.tensor(
                    _convert_tensor(pb_utils.get_input_tensor_by_name(request, name)),
                    dtype=dtype,
                ).cuda()

            for name, dtype in self.sparse_inputs.items():

                input_val = _convert_tensor(
                    pb_utils.get_input_tensor_by_name(
                        request, name + sparse_value_marker
                    )
                )
                input_lengths = _convert_tensor(
                    pb_utils.get_input_tensor_by_name(
                        request, name + sparse_offsets_marker
                    )
                )
                input_lengths = torch.tensor(input_lengths, dtype=torch.int64)
                input_values = torch.tensor(input_val, dtype=dtype)

                sparse_to_dense = False
                seq_limit = 0
                if self.model_info is not None:
                    if self.model_info["sparse_max"].get(name) is not None:
                        sparse_to_dense = True
                        seq_limit = self.model_info["sparse_max"][name]

                if seq_limit == 0:
                    seq_limit = int(input_lengths.max())

                input_dict[name] = _build_sparse_tensor(
                    input_values, input_lengths, seq_limit, sparse_to_dense
                )

            out = self.model(input_dict)
            if not isinstance(out, dict):
                raise ValueError("output of the forward function should be a dict")

            pred = out.get("predictions")
            if pred is None:
                raise KeyError(
                    "output of the forward function should have a bucket named as predictions"
                )

            pred_numpy = pred.cpu().detach().numpy()

            output_info = self.model_config["output"][0]
            return pb_utils.Tensor(output_info["name"], pred_numpy)


def _get_indices(lengths, device="cuda"):

    offsets = torch.cat((torch.tensor([1]), lengths), 0)
    offsets = offsets.cumsum(0)
    row_ids = torch.arange(len(offsets) - 1)
    row_ids_repeated = torch.repeat_interleave(row_ids, lengths)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], lengths)
    col_ids = torch.arange(len(row_offset_repeated)) - row_offset_repeated + 1
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices.T


def _get_sparse_tensor(
    values, indices, num_rows, seq_limit, sparse_as_dense, device="cuda"
):

    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, torch.Size([num_rows, seq_limit]), device=device
    )
    if sparse_as_dense:
        sparse_tensor = sparse_tensor.to_dense()
    return sparse_tensor


def _build_sparse_tensor(values, lengths, seq_limit, sparse_as_dense, device="cuda"):

    indices = _get_indices(lengths, device)
    num_rows = len(lengths)
    return _get_sparse_tensor(
        values, indices, num_rows, seq_limit, sparse_as_dense, device
    )


def _convert_dtype(dtype):

    if dtype in [torch.float64, torch.float32, torch.float16]:
        return torch.float32
    if dtype in [
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]:
        return torch.long

    raise ValueError(f"Can't convert dtype {dtype})")
