import asyncio
import base64
import json
import logging
import multiprocessing
import os
import pathlib
import threading
import traceback
import typing
from concurrent.futures import Future as ConcurrentFuture

import triton_python_backend_utils as pb_utils

from . import communication, data
from .communication import (
    HandleResponsesCoro,
    HandshakeServer,
    PyTritonResponseFlags,
    RequestsServer,
    get_config_from_handshake_server,
)
from .data import (
    PROTOCOL_VERSION,
    Base64SerializerDeserializer,
    TensorStoreSerializerDeserializer,
)
from .telemetry import TracableModel
from .types import Request, Response, ResponsesOrError

LOGGER = logging.getLogger(__name__)


def _update_loggers():
    def get_triton_backend_logger():
        try:

            import triton_python_backend_utils as pb_utils

            logger = pb_utils.Logger
            logger.error = logger.log_error
            logger.warning = logger.log_warn
            logger.info = logger.log_info
            logger.debug = logger.log_verbose

        except (ImportError, AttributeError):
            logger = logging.getLogger("backend")
            root_logger = logging.getLogger()
            if root_logger.level <= logging.INFO:
                multiprocessing.util.log_to_stderr(logging.INFO)
        return logger

    logger = get_triton_backend_logger()
    global LOGGER
    LOGGER = logger
    data.LOGGER = logger
    communication.LOGGER = logger
    communication.SERVER_LOGGER = logger


class TritonRequestsServer:

    def __init__(
        self,
        url: str,
        responses_handle_fn: HandleResponsesCoro,
        serializer_deserializer,
        model_config: typing.Dict[str, typing.Any],
    ):

        self._model_config = model_config
        self._model_inputs_names = [
            model_input["name"] for model_input in model_config["input"]
        ]
        self._server = RequestsServer(url, responses_handle_fn)
        self._serializer_deserializer = serializer_deserializer

    def run(self):

        self._server.run()

    def shutdown(self):

        self._server.shutdown()

    def push(self, requests_id: bytes, triton_requests, spans=None):

        self._server.wait_till_running()
        kwargs = {"requests_id": requests_id, "triton_requests": triton_requests}
        if spans is not None:
            kwargs["spans"] = spans
        return asyncio.run_coroutine_threadsafe(
            self._send_requests(**kwargs), self._server.server_loop
        )

    def _wrap_request(self, triton_request, inputs, span=None) -> Request:
        request = {}
        for input_name in inputs:
            input_tensor = pb_utils.get_input_tensor_by_name(triton_request, input_name)
            if input_tensor is not None:
                request[input_name] = input_tensor.as_numpy()
        kwargs = {}
        if span is not None:
            kwargs["span"] = span
        return Request(
            data=request,
            parameters=json.loads(triton_request.parameters()),
            requested_output_names=list(triton_request.requested_output_names()),
            **kwargs,
        )

    async def _send_requests(
        self, requests_id: bytes, triton_requests, spans=None
    ) -> ConcurrentFuture:
        requests = triton_requests
        if spans is None:
            spans = [None] * len(triton_requests)
        requests_with_spans = zip(triton_requests, spans)

        requests = [
            self._wrap_request(triton_request, self._model_inputs_names, span)
            for triton_request, span in requests_with_spans
        ]
        requests_payload = self._serializer_deserializer.serialize_requests(requests)

        responses_future = ConcurrentFuture()
        await self._server.send_requests(
            requests_id, requests_payload, responses_future
        )
        return responses_future


def _wrap_response(response: Response, requested_outputs_names, model_outputs_dict):
    if response.data is not None:
        only_requested = {
            key: value
            for key, value in response.data.items()
            if key in requested_outputs_names
        }
        casted = {
            key: value.astype(
                pb_utils.triton_string_to_numpy(model_outputs_dict[key]["data_type"])
            )
            for key, value in only_requested.items()
        }
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(name, value) for name, value in casted.items()
            ]
        )
    else:
        return None


class BatchResponsesHandler:

    def __init__(self, requests_map, serializer_deserializer, model_outputs_dict):

        self._requests_map = requests_map
        self._serializer_deserializer = serializer_deserializer
        self._model_outputs_dict = model_outputs_dict

    async def handle_responses(
        self,
        scope: typing.Dict[str, typing.Any],
        responses_queue: asyncio.Queue,
        responses_future: ConcurrentFuture,
    ):

        requests_id: bytes = scope["requests_id"]
        triton_requests = self._requests_map[requests_id]

        eos = False
        triton_responses_or_error = None
        while not eos:
            try:
                flags, responses_payload = await responses_queue.get()
                eos = flags & PyTritonResponseFlags.EOS
                error = flags & PyTritonResponseFlags.ERROR

                if error:
                    assert eos
                    triton_responses_or_error = pb_utils.TritonModelException(
                        responses_payload.decode("utf-8")
                    )
                elif responses_payload:

                    assert triton_responses_or_error is None
                    responses = self._serializer_deserializer.deserialize_responses(
                        responses_payload
                    )
                    triton_responses_or_error = [
                        _wrap_response(
                            response,
                            request.requested_output_names(),
                            self._model_outputs_dict,
                        )
                        for request, response in zip(triton_requests, responses)
                    ]
            except asyncio.CancelledError:
                LOGGER.warning(
                    f"Cancelled responses handler for requests={requests_id.hex()}"
                )
                triton_responses_or_error = pb_utils.TritonModelException(
                    "Cancelled responses handler"
                )
                eos = True
            finally:
                if not error:
                    self._serializer_deserializer.free_responses_resources(
                        responses_payload
                    )
                responses_queue.task_done()

        self._requests_map.pop(requests_id)
        responses_future.set_result(triton_responses_or_error)
        return triton_responses_or_error


class DecoupledResponsesHandler:

    def __init__(self, requests_map, serializer_deserializer, model_outputs_dict):

        self._requests_map = requests_map
        self._serializer_deserializer = serializer_deserializer
        self._model_outputs_dict = model_outputs_dict

    async def handle_responses(
        self,
        scope: typing.Dict[str, typing.Any],
        responses_queue: asyncio.Queue,
        responses_future: ConcurrentFuture,
    ) -> typing.Optional[ResponsesOrError]:

        requests_id: bytes = scope["requests_id"]
        loop = asyncio.get_running_loop()
        triton_requests = self._requests_map[requests_id]
        triton_senders = [request.get_response_sender() for request in triton_requests]

        eos = False
        while not eos:
            try:
                flags, responses_payload = await responses_queue.get()

                eos = flags & PyTritonResponseFlags.EOS
                error = flags & PyTritonResponseFlags.ERROR

                triton_responses = None
                if error:
                    triton_responses = [
                        pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(
                                responses_payload.decode("utf-8")
                            )
                        )
                        for _ in triton_senders
                    ]
                else:
                    responses = self._serializer_deserializer.deserialize_responses(
                        responses_payload
                    )
                    triton_responses = [
                        _wrap_response(
                            response,
                            request.requested_output_names(),
                            self._model_outputs_dict,
                        )
                        for request, response in zip(triton_requests, responses)
                    ]

                triton_flags = 0
                if eos:
                    triton_flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    triton_responses = triton_responses or [None] * len(triton_senders)

                assert len(triton_responses) == len(triton_senders)
                send_responses_futures = [
                    loop.run_in_executor(None, sender.send, response, triton_flags)
                    for sender, response in zip(triton_senders, triton_responses)
                ]
                await asyncio.gather(*send_responses_futures)
            except asyncio.CancelledError:
                LOGGER.warning(
                    f"Cancelled responses handler for requests={requests_id.hex()}"
                )
                triton_flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                triton_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error="Cancelled responses handler")
                )
                send_responses_futures = [
                    loop.run_in_executor(
                        None, sender.send, triton_response, triton_flags
                    )
                    for sender in triton_senders
                ]
                await asyncio.gather(*send_responses_futures)
            finally:
                if not error:
                    self._serializer_deserializer.free_responses_resources(
                        responses_payload
                    )
                responses_queue.task_done()

        self._requests_map.pop(requests_id)
        responses_future.set_result(None)


class TritonInferenceHandlerConfigGenerator:

    def __init__(self, data_socket: typing.Union[str, pathlib.Path]):

        self._data_socket = pathlib.Path(data_socket)

    def get_config(self) -> typing.Dict[str, typing.Any]:

        return {
            "protocol_version": PROTOCOL_VERSION,
            "data_socket": self._data_socket.as_posix(),
            "authkey": base64.encodebytes(
                multiprocessing.current_process().authkey
            ).decode("ascii"),
        }


class TritonPythonModel:

    def __init__(self):

        self._model_config = None
        self._model_inputs = None
        self._model_outputs = None
        self._model_instance_name = None
        self._decoupled_model = None
        self._serializer_deserializer = None
        self._requests_server = None
        self._requests_server_thread = None
        self._handshake_server = None
        self._loop = None
        self._frontend = None
        self._requests = None
        self._id_counter = 0
        self._tracable_model = None

    def initialize(self, args):

        _update_loggers()

        if bool(os.environ.get("PYTRITON_VIZTRACER")):
            from viztracer import VizTracer

            self._tracer = VizTracer(
                log_async=True, log_gc=True, tracer_entries=10000000, pid_suffix=True
            )
            self._tracer.register_exit()
            self._tracer.start()

        try:
            model_name = args["model_name"]

            self._model_config = model_config = json.loads(args["model_config"])
            self._model_inputs = {
                model_input["name"]: model_input
                for model_input in model_config["input"]
            }
            self._model_outputs = {
                model_output["name"]: model_output
                for model_output in model_config["output"]
            }
            self._model_instance_name = args.get("model_instance_name")
            self._decoupled_model = model_config.get(
                "model_transaction_policy", {}
            ).get("decoupled", False)

            workspace_path = pathlib.Path(
                model_config["parameters"]["workspace-path"]["string_value"]
            )

            self._tracable_model = TracableModel()
            if "trace-config" in model_config["parameters"]:
                self._tracable_model.configure_tracing(
                    model_config["parameters"]["trace-config"]["string_value"]
                )

            LOGGER.debug(f"Model instance name: {self._model_instance_name}")
            LOGGER.debug(f"Decoupled model: {self._decoupled_model}")
            LOGGER.debug(f"Workspace path: {workspace_path}")
            LOGGER.debug(f"Model inputs: {self._model_inputs}")
            LOGGER.debug(f"Model outputs: {self._model_outputs}")

            data_socket = workspace_path / f"{model_name}-data.sock"
            if os.environ.get("PYTRITON_NO_TENSORSTORE"):
                self._serializer_deserializer = Base64SerializerDeserializer()
            else:
                self._serializer_deserializer = TensorStoreSerializerDeserializer()

            handshake_socket = workspace_path / f"{model_name}-config.sock"
            model_first_instance_name = "_".join(
                self._model_instance_name.split("_")[:-1] + ["0"]
            )
            if self._model_instance_name == model_first_instance_name:
                inference_handler_config = TritonInferenceHandlerConfigGenerator(
                    data_socket
                ).get_config()
                self._serializer_deserializer.start(data_socket)

                self._handshake_server = HandshakeServer(
                    handshake_socket, inference_handler_config
                )
                self._handshake_server.start()

            else:
                inference_handler_config = get_config_from_handshake_server(
                    handshake_socket
                )
                LOGGER.debug(f"Loaded configuration from {handshake_socket}")

                authkey = base64.decodebytes(
                    inference_handler_config["authkey"].encode("ascii")
                )
                self._serializer_deserializer.connect(data_socket, authkey=authkey)

            self._id_counter = 0
            self._requests = {}

            server_socket_path = (
                workspace_path / f"{self._model_instance_name}-server.sock"
            )
            handler_class = (
                DecoupledResponsesHandler
                if self._decoupled_model
                else BatchResponsesHandler
            )
            LOGGER.debug(f"Using {handler_class.__name__} for handling responses")
            self._requests_server = TritonRequestsServer(
                url=f"ipc://{server_socket_path.as_posix()}",
                responses_handle_fn=handler_class(
                    self._requests, self._serializer_deserializer, self._model_outputs
                ).handle_responses,
                serializer_deserializer=self._serializer_deserializer,
                model_config=self._model_config,
            )

            def _run_server():
                _update_loggers()
                self._requests_server.run()

            self._requests_server_thread = threading.Thread(
                target=_run_server, name="requests-server", daemon=True
            )
            self._requests_server_thread.start()
        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(
                f"Model initialize error: {msg}"
            ) from None

    def execute(self, triton_requests):

        try:
            spans = self._tracable_model.start_requests_spans(triton_requests)

            def _generate_id():
                self._id_counter = (self._id_counter + 1) % 2**32
                return self._id_counter.to_bytes(4, "big")

            requests_id = _generate_id()
            while requests_id in self._requests:
                requests_id = _generate_id()
            self._requests[requests_id] = triton_requests

            handle_responses_task_async_future = self._requests_server.push(
                requests_id, triton_requests, spans
            )

            if not self._decoupled_model:
                handle_responses_concurrent_future = (
                    handle_responses_task_async_future.result()
                )
                triton_responses_or_error = handle_responses_concurrent_future.result()

                self._tracable_model.end_requests_spans(
                    spans, triton_responses_or_error
                )

                if triton_responses_or_error is not None and isinstance(
                    triton_responses_or_error, Exception
                ):
                    raise triton_responses_or_error
            else:
                triton_responses_or_error = None

                self._tracable_model.end_requests_spans(
                    spans, triton_responses_or_error
                )

            return triton_responses_or_error
        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model execute error: {msg}") from None

    def finalize(self) -> None:

        LOGGER.debug(f"[{self._model_instance_name}] Finalizing backend instance")
        LOGGER.debug(f"[{self._model_instance_name}] Closing requests server")
        self._requests_server.shutdown()
        self._requests_server_thread.join()

        LOGGER.debug(
            f"[{self._model_instance_name}] Closing requests/responses serializer/deserializer"
        )
        self._serializer_deserializer.close()
        self._serializer_deserializer = None

        LOGGER.debug(f"[{self._model_instance_name}] Closing handshake server")
        if self._handshake_server:
            self._handshake_server.close()
            self._handshake_server = None

        LOGGER.debug(f"[{self._model_instance_name}] Finalized.")
        self._model_instance_name = None
