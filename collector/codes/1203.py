import subprocess
import logging
import json
from typing import Union
from pathlib import Path
import tritonclient.grpc.model_config_pb2 as mc
from google.protobuf import json_format, text_format

from .server_config import TritonServerConfig

from triton_cli.common import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TritonServerUtils:

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._trtllm_utils = TRTLLMUtils(self._model_repo_path)

    def get_launch_command(
        self,
        server_config: TritonServerConfig,
        cmd_as_list: bool,
        env_cmds=[],
    ) -> Union[str, list]:

        if (
            self._trtllm_utils.has_trtllm_model()
            and self._trtllm_utils.get_world_size() > 1
        ):
            logger.info(
                f"Launching server with world size: {self._trtllm_utils.get_world_size()}"
            )
            cmd = self._trtllm_utils.mpi_run(server_config)
        else:
            cmd = (
                env_cmds + [server_config.server_path()] + server_config.to_args_list()
            )

        if cmd_as_list:
            return cmd
        else:
            return " ".join(cmd)


class TRTLLMUtils:

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._trtllm_model_config_path = self._find_trtllm_model_config_path()
        self._is_trtllm_model = self._trtllm_model_config_path is not None
        self._supported_args = ["model-repository"]

        self._world_size = -1
        if self._is_trtllm_model:
            self._world_size = self._parse_world_size()

    def has_trtllm_model(self) -> bool:

        return self._is_trtllm_model

    def get_world_size(self) -> int:

        return self._world_size

    def get_engine_path(self) -> str:

        return str(self._get_engine_path(self._trtllm_model_config_path))

    def mpi_run(self, server_config: TritonServerConfig) -> str:

        unsupported_args = server_config.get_unsupported_args(self._supported_args)
        if unsupported_args:
            raise Exception(
                f"The following args are not currently supported by this model: {unsupported_args}"
            )

        cmd = ["mpirun", "--allow-run-as-root"]
        for i in range(self._world_size):
            cmd += ["-n", "1", "/opt/tritonserver/bin/tritonserver"]
            cmd += [
                f"--model-repository={self._model_repo_path}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
                ":",
            ]
        return cmd

    def _find_trtllm_model_config_path(self) -> Path:

        try:
            match = subprocess.check_output(
                [
                    "grep",
                    "-r",
                    "--include",
                    "*.pbtxt",
                    'backend: "tensorrtllm"',
                    self._model_repo_path,
                ]
            )

            return Path(match.decode().split(":")[0])
        except subprocess.CalledProcessError:

            return None

    def _get_engine_path(self, config_path: Path) -> Path:

        try:
            with open(config_path) as config_file:
                config = text_format.Parse(config_file.read(), mc.ModelConfig())
                json_config = json.loads(
                    json_format.MessageToJson(config, preserving_proto_field_name=True)
                )
                return Path(json_config["parameters"]["gpt_model_path"]["string_value"])
        except KeyError as e:
            raise Exception(
                f"Unable to extract engine path from config file {config_path}. Key error: {str(e)}"
            )
        except OSError:
            raise Exception(
                f"Failed to open config file for tensorrt_llm. Searched: {config_path}"
            )

    def _parse_world_size(self) -> int:

        assert (
            self._is_trtllm_model
        ), "World size cannot be parsed from a model repository that does not contain a TRT LLM model."
        try:
            engine_path = self._get_engine_path(self._trtllm_model_config_path)
            engine_config_path = engine_path / "config.json"
            with open(engine_config_path) as json_data:
                data = json.load(json_data)

                config = (
                    data.get("builder_config")
                    if data.get("builder_config") is not None
                    else data.get("build_config")
                )
                if not config:
                    raise Exception(f"Unable to parse config from {engine_config_path}")
                tp = int(config.get("tensor_parallel", 1))
                pp = int(config.get("pipeline_parallel", 1))
                return tp * pp
        except OSError:
            raise Exception(f"Unable to open {engine_config_path}")


class VLLMUtils:

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._vllm_model_config_path = self._find_vllm_model_config_path()
        self._is_vllm_model = self._vllm_model_config_path is not None

    def has_vllm_model(self) -> bool:

        return self._is_vllm_model

    def get_vllm_model_huggingface_id_or_path(self) -> str:

        return self._find_vllm_model_huggingface_id_or_path()

    def _find_vllm_model_config_path(self) -> Path:

        try:
            match = subprocess.check_output(
                [
                    "grep",
                    "-r",
                    "--include",
                    "*.pbtxt",
                    'backend: "vllm"',
                    self._model_repo_path,
                ]
            )

            return Path(match.decode().split(":")[0])
        except subprocess.CalledProcessError:

            return None

    def _find_vllm_model_huggingface_id_or_path(self) -> str:

        assert (
            self._is_vllm_model
        ), "model Huggingface Id or path cannot be parsed from a model repository that does not contain a vLLM model."
        try:

            model_version_path = self._vllm_model_config_path.parent / "1"
            model_config_json_file = model_version_path / "model.json"
            with open(model_config_json_file) as json_data:
                data = json.load(json_data)
                model_id = data.get("model")
                if not model_id:
                    raise Exception(
                        f"Unable to parse config from {model_config_json_file}"
                    )
                return model_id
        except OSError:
            raise Exception(f"Unable to open {model_config_json_file}")
