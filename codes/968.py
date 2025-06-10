import asyncio
import contextlib
import itertools
import logging
import socket
import time
import warnings
from concurrent.futures import Future
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Tuple, Union

import gevent
import numpy as np
import tritonclient.grpc
import tritonclient.grpc.aio
import tritonclient.http
import tritonclient.http.aio
import tritonclient.utils

from pytriton.client.asyncio_utils import (
    asyncio_get_model_config,
    asyncio_wait_for_model_ready,
)
from pytriton.client.exceptions import (
    PyTritonClientClosedError,
    PyTritonClientInferenceServerError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientQueueFullError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from pytriton.client.utils import (
    _DEFAULT_NETWORK_TIMEOUT_S,
    _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
    TritonUrl,
    get_model_config,
    wait_for_model_ready,
    wait_for_server_ready,
)
from pytriton.client.warnings import NotSupportedTimeoutWarning
from pytriton.model_config.triton_model_config import TritonModelConfig

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SYNC_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
_DEFAULT_FUTURES_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
DEFAULT_INFERENCE_TIMEOUT_S = 60.0


_IOType = Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]


def _verify_inputs_args(inputs, named_inputs):
    if not inputs and not named_inputs:
        raise PyTritonClientValueError("Provide input data")
    if not bool(inputs) ^ bool(named_inputs):
        raise PyTritonClientValueError(
            "Use either positional either keyword method arguments convention"
        )


def _verify_parameters(
    parameters_or_headers: Optional[Dict[str, Union[str, int, bool]]] = None,
):
    if parameters_or_headers is None:
        return
    if not isinstance(parameters_or_headers, dict):
        raise PyTritonClientValueError("Parameters and headers must be a dictionary")
    for key, value in parameters_or_headers.items():
        if not isinstance(key, str):
            raise PyTritonClientValueError("Parameter/header key must be a string")
        if not isinstance(value, (str, int, bool)):
            raise PyTritonClientValueError(
                "Parameter/header value must be a string, integer or boolean"
            )


class BaseModelClient:

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):

        self._init_timeout_s = (
            _DEFAULT_SYNC_INIT_TIMEOUT_S if init_timeout_s is None else init_timeout_s
        )
        self._inference_timeout_s = (
            DEFAULT_INFERENCE_TIMEOUT_S
            if inference_timeout_s is None
            else inference_timeout_s
        )
        self._network_timeout_s = min(_DEFAULT_NETWORK_TIMEOUT_S, self._init_timeout_s)

        self._general_client = self.create_client_from_url(
            url, network_timeout_s=self._network_timeout_s
        )
        self._infer_client = self.create_client_from_url(
            url, network_timeout_s=self._inference_timeout_s
        )

        self._model_name = model_name
        self._model_version = model_version

        self._request_id_generator = itertools.count(0)

        self._monkey_patch_client()

        if model_config is not None:
            self._model_config = model_config
            self._model_ready = None if ensure_model_is_ready else True

        else:
            self._model_config = None
            self._model_ready = None
        self._lazy_init: bool = lazy_init

        self._handle_lazy_init()

    @classmethod
    def from_existing_client(cls, existing_client: "BaseModelClient"):

        kwargs = {}

        if hasattr(existing_client, "_model_config"):
            kwargs["model_config"] = existing_client._model_config
            kwargs["ensure_model_is_ready"] = False

        new_client = cls(
            url=existing_client._url,
            model_name=existing_client._model_name,
            model_version=existing_client._model_version,
            init_timeout_s=existing_client._init_timeout_s,
            inference_timeout_s=existing_client._inference_timeout_s,
            **kwargs,
        )

        return new_client

    def create_client_from_url(
        self, url: str, network_timeout_s: Optional[float] = None
    ):

        self._triton_url = TritonUrl.from_url(url)
        self._url = self._triton_url.without_scheme
        self._triton_client_lib = self.get_lib()
        self._monkey_patch_client()

        if self._triton_url.scheme == "grpc":

            network_timeout_s = (
                _DEFAULT_NETWORK_TIMEOUT_S
                if network_timeout_s is None
                else network_timeout_s
            )
            warnings.warn(
                f"tritonclient.grpc doesn't support timeout for other commands than infer. Ignoring network_timeout: {network_timeout_s}.",
                NotSupportedTimeoutWarning,
                stacklevel=1,
            )

        triton_client_init_kwargs = self._get_init_extra_args()

        _LOGGER.debug(
            "Creating InferenceServerClient for %s with %s",
            self._triton_url.with_scheme,
            triton_client_init_kwargs,
        )
        return self._triton_client_lib.InferenceServerClient(
            self._url, **triton_client_init_kwargs
        )

    def get_lib(self):

        raise NotImplementedError

    @property
    def _next_request_id(self) -> str:

        if getattr(self, "_request_id_generator", None) is None:
            self._request_id_generator = itertools.count(0)
        return str(next(self._request_id_generator))

    def _get_init_extra_args(self):
        timeout = self._inference_timeout_s

        if self._triton_url.scheme != "http":
            return {}

        kwargs = {
            "network_timeout": timeout,
            "connection_timeout": timeout,
        }
        return kwargs

    def _monkey_patch_client(self):
        pass

    def _get_model_config_extra_args(self):

        return {}

    def _handle_lazy_init(self):
        raise NotImplementedError


def _run_once_per_lib(f):
    def wrapper(_self):
        if _self._triton_client_lib not in wrapper.patched:
            wrapper.patched.add(_self._triton_client_lib)
            return f(_self)

    wrapper.patched = set()
    return wrapper


class ModelClient(BaseModelClient):

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):

        super().__init__(
            url=url,
            model_name=model_name,
            model_version=model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
            model_config=model_config,
            ensure_model_is_ready=ensure_model_is_ready,
        )

    def get_lib(self):

        return {"grpc": tritonclient.grpc, "http": tritonclient.http}[
            self._triton_url.scheme.lower()
        ]

    def __enter__(self):

        return self

    def __exit__(self, *_):

        self.close()

    def load_model(self, config: Optional[str] = None, files: Optional[dict] = None):

        self._general_client.load_model(self._model_name, config=config, files=files)

    def unload_model(self):

        self._general_client.unload_model(self._model_name)

    def close(self):

        _LOGGER.debug("Closing ModelClient")
        try:
            if self._general_client is not None:
                self._general_client.close()
            if self._infer_client is not None:
                self._infer_client.close()
            self._general_client = None
            self._infer_client = None
        except Exception as e:
            _LOGGER.error("Error while closing ModelClient resources: %s", e)
            raise e

    def wait_for_model(self, timeout_s: float):

        if self._general_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")
        wait_for_model_ready(
            self._general_client,
            self._model_name,
            self._model_version,
            timeout_s=timeout_s,
        )

    @property
    def is_batching_supported(self):

        return self.model_config.max_batch_size > 0

    def wait_for_server(self, timeout_s: float):

        wait_for_server_ready(self._general_client, timeout_s=timeout_s)

    @property
    def model_config(self) -> TritonModelConfig:

        if not self._model_config:
            if self._general_client is None:
                raise PyTritonClientClosedError("ModelClient is closed")

            self._model_config = get_model_config(
                self._general_client,
                self._model_name,
                self._model_version,
                timeout_s=self._init_timeout_s,
            )
        return self._model_config

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:

        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if self.is_batching_supported:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {
                    name: data[np.newaxis, ...] for name, data in named_inputs.items()
                }

        result = self._infer(inputs or named_inputs, parameters, headers)

        return self._debatch_result(result)

    def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:

        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if not self.is_batching_supported:
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {self.model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        return self._infer(inputs or named_inputs, parameters, headers)

    def _wait_and_init_model_config(self, init_timeout_s: float):
        if self._general_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")

        should_finish_before_s = time.time() + init_timeout_s
        self.wait_for_model(init_timeout_s)
        self._model_ready = True
        timeout_s = max(0.0, should_finish_before_s - time.time())
        self._model_config = get_model_config(
            self._general_client,
            self._model_name,
            self._model_version,
            timeout_s=timeout_s,
        )

    def _create_request(self, inputs: _IOType):
        if self._infer_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")

        if not self._model_ready:
            self._wait_and_init_model_config(self._init_timeout_s)

        if isinstance(inputs, Tuple):
            inputs = {
                input_spec.name: input_data
                for input_spec, input_data in zip(self.model_config.inputs, inputs)
            }

        inputs_wrapped = []

        inputs: Dict[str, np.ndarray]

        for input_name, input_data in inputs.items():
            if input_data.dtype == object and not isinstance(
                input_data.reshape(-1)[0], bytes
            ):
                raise RuntimeError(
                    f"Numpy array for {input_name!r} input with dtype=object should contain encoded strings \
                    \\(e.g. into utf-8\\). Element type: {type(input_data.reshape(-1)[0])}"
                )
            if input_data.dtype.type == np.str_:
                raise RuntimeError(
                    "Unicode inputs are not supported. "
                    f"Encode numpy array for {input_name!r} input (ex. with np.char.encode(array, 'utf-8'))."
                )
            triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
            infer_input = self._triton_client_lib.InferInput(
                input_name, input_data.shape, triton_dtype
            )
            infer_input.set_data_from_numpy(input_data)
            inputs_wrapped.append(infer_input)

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name)
            for output_spec in self.model_config.outputs
        ]
        return inputs_wrapped, outputs_wrapped

    def _infer(self, inputs: _IOType, parameters, headers) -> Dict[str, np.ndarray]:
        if self.model_config.decoupled:
            raise PyTritonClientInferenceServerError(
                "Model config is decoupled. Use DecoupledModelClient instead."
            )

        inputs_wrapped, outputs_wrapped = self._create_request(inputs)

        try:
            _LOGGER.debug("Sending inference request to Triton Inference Server")
            response = self._infer_client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                headers=headers,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                parameters=parameters,
                **self._get_infer_extra_args(),
            )
        except tritonclient.utils.InferenceServerException as e:

            if "Deadline Exceeded" in e.message():
                raise PyTritonClientTimeoutError(
                    f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s. Message: {e.message()}"
                ) from e

            raise PyTritonClientInferenceServerError(
                f"Error occurred during inference request. Message: {e.message()}"
            ) from e
        except socket.timeout as e:
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except OSError as e:
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e

        if isinstance(response, tritonclient.http.InferResult):
            outputs = {
                output["name"]: response.as_numpy(output["name"])
                for output in response.get_response()["outputs"]
            }
        else:
            outputs = {
                output.name: response.as_numpy(output.name)
                for output in response.get_response().outputs
            }

        return outputs

    def _get_numpy_result(self, result):
        if isinstance(result, tritonclient.grpc.InferResult):
            result = {
                output.name: result.as_numpy(output.name)
                for output in result.get_response().outputs
            }
        else:
            result = {
                output["name"]: result.as_numpy(output["name"])
                for output in result.get_response()["outputs"]
            }
        return result

    def _debatch_result(self, result):
        if self.is_batching_supported:
            result = {name: data[0] for name, data in result.items()}
        return result

    def _handle_lazy_init(self):
        if not self._lazy_init:
            self._wait_and_init_model_config(self._init_timeout_s)

    def _get_infer_extra_args(self):
        if self._triton_url.scheme == "http":
            return {}

        kwargs = {"client_timeout": self._inference_timeout_s}
        return kwargs

    @_run_once_per_lib
    def _monkey_patch_client(self):

        _LOGGER.info("Patch ModelClient %s", self._triton_url.scheme)
        if not hasattr(self._triton_client_lib.InferenceServerClient, "__del__"):
            return

        old_del = self._triton_client_lib.InferenceServerClient.__del__

        def _monkey_patched_del(self):

            try:
                old_del(self)
            except gevent.exceptions.InvalidThreadUseError:
                _LOGGER.info(
                    "gevent.exceptions.InvalidThreadUseError in __del__ of InferenceServerClient"
                )
            except Exception as e:
                _LOGGER.error("Exception in __del__ of InferenceServerClient: %s", e)

        self._triton_client_lib.InferenceServerClient.__del__ = _monkey_patched_del


class DecoupledModelClient(ModelClient):

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):

        super().__init__(
            url,
            model_name,
            model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
            model_config=model_config,
            ensure_model_is_ready=ensure_model_is_ready,
        )
        if self._triton_url.scheme == "http":
            raise PyTritonClientValueError(
                "DecoupledModelClient is only supported for grpc protocol"
            )
        self._queue = Queue()
        self._lock = Lock()

    def close(self):

        _LOGGER.debug("Closing DecoupledModelClient")
        if self._lock.acquire(blocking=False):
            try:
                super().close()
            finally:
                self._lock.release()
        else:
            _LOGGER.warning("DecoupledModelClient is stil streaming answers")
            self._infer_client.stop_stream(False)
            super().close()

    def _infer(self, inputs: _IOType, parameters, headers):
        if not self._lock.acquire(blocking=False):
            raise PyTritonClientInferenceServerError("Inference is already in progress")
        if not self.model_config.decoupled:
            raise PyTritonClientInferenceServerError(
                "Model config is coupled. Use ModelClient instead."
            )

        inputs_wrapped, outputs_wrapped = self._create_request(inputs)
        if parameters is not None:
            raise PyTritonClientValueError(
                "DecoupledModelClient does not support parameters"
            )
        if headers is not None:
            raise PyTritonClientValueError(
                "DecoupledModelClient does not support headers"
            )
        try:
            _LOGGER.debug("Sending inference request to Triton Inference Server")
            if self._infer_client._stream is None:
                self._infer_client.start_stream(
                    callback=lambda result, error: self._response_callback(
                        result, error
                    )
                )

            self._infer_client.async_stream_infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                enable_empty_final_response=True,
                **self._get_infer_extra_args(),
            )
        except tritonclient.utils.InferenceServerException as e:

            if "Deadline Exceeded" in e.message():
                raise PyTritonClientTimeoutError(
                    f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s. Message: {e.message()}"
                ) from e

            raise PyTritonClientInferenceServerError(
                f"Error occurred during inference request. Message: {e.message()}"
            ) from e
        except socket.timeout as e:
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except OSError as e:
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        _LOGGER.debug("Returning response iterator")
        return self._create_response_iterator()

    def _response_callback(self, response, error):
        _LOGGER.debug("Received response from Triton Inference Server: %s", response)
        if error:
            _LOGGER.error("Error occurred during inference request. Message: %s", error)
            self._queue.put(error)
        else:
            actual_response = response.get_response()

            triton_final_response = actual_response.parameters.get(
                "triton_final_response"
            )
            if triton_final_response and triton_final_response.bool_param:
                self._queue.put(None)
            else:
                result = self._get_numpy_result(response)
                self._queue.put(result)

    def _create_response_iterator(self):
        try:
            while True:
                try:
                    item = self._queue.get(self._inference_timeout_s)
                except Empty as e:
                    message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s"
                    _LOGGER.error(message)
                    raise PyTritonClientTimeoutError(message) from e
                if isinstance(item, Exception):
                    message = f"Error occurred during inference request. Message: {item.message()}"
                    _LOGGER.error(message)
                    raise PyTritonClientInferenceServerError(message) from item

                if item is None:
                    break
                yield item
        finally:
            self._lock.release()

    def _debatch_result(self, result):
        if self.is_batching_supported:
            result = (
                {name: data[0] for name, data in result_.items()} for result_ in result
            )
        return result

    def _get_infer_extra_args(self):

        kwargs = {}

        return kwargs


class AsyncioModelClient(BaseModelClient):

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
        model_config: Optional[TritonModelConfig] = None,
        ensure_model_is_ready: bool = True,
    ):

        super().__init__(
            url=url,
            model_name=model_name,
            model_version=model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
            model_config=model_config,
            ensure_model_is_ready=ensure_model_is_ready,
        )

    def get_lib(self):

        return {"grpc": tritonclient.grpc.aio, "http": tritonclient.http.aio}[
            self._triton_url.scheme.lower()
        ]

    async def __aenter__(self):

        _LOGGER.debug("Entering AsyncioModelClient context")
        try:
            if not self._lazy_init:
                _LOGGER.debug(
                    "Waiting in AsyncioModelClient context for model to be ready"
                )
                await self._wait_and_init_model_config(self._init_timeout_s)
                _LOGGER.debug("Model is ready in AsyncioModelClient context")
            return self
        except Exception as e:
            _LOGGER.error(
                "Error occurred during AsyncioModelClient context initialization"
            )
            await self.close()
            raise e

    async def __aexit__(self, *_):

        await self.close()
        _LOGGER.debug("Exiting AsyncioModelClient context")

    async def close(self):

        _LOGGER.debug("Closing InferenceServerClient")
        await self._general_client.close()
        await self._infer_client.close()
        _LOGGER.debug("InferenceServerClient closed")

    async def wait_for_model(self, timeout_s: float):

        _LOGGER.debug("Waiting for model %s to be ready", self._model_name)
        try:
            await asyncio.wait_for(
                asyncio_wait_for_model_ready(
                    self._general_client,
                    self._model_name,
                    self._model_version,
                    timeout_s=timeout_s,
                ),
                self._init_timeout_s,
            )
        except asyncio.TimeoutError as e:
            message = f"Timeout while waiting for model {self._model_name} to be ready for {self._init_timeout_s}s"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e

    @property
    async def model_config(self):

        try:
            if not self._model_config:
                kwargs = self._get_model_config_extra_args()
                _LOGGER.debug("Obtaining model config for %s", self._model_name)

                self._model_config = await asyncio.wait_for(
                    asyncio_get_model_config(
                        self._general_client,
                        self._model_name,
                        self._model_version,
                        timeout_s=self._init_timeout_s,
                        **kwargs,
                    ),
                    self._init_timeout_s,
                )
                _LOGGER.debug("Obtained model config for %s", self._model_name)
            return self._model_config
        except asyncio.TimeoutError as e:
            message = f"Timeout while waiting for model {self._model_name} to be ready for {self._init_timeout_s}s"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e

    async def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ):

        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        _LOGGER.debug("Running inference for %s", self._model_name)
        model_config = await self.model_config
        _LOGGER.debug("Model config for %s obtained", self._model_name)

        model_supports_batching = model_config.max_batch_size > 0
        if model_supports_batching:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {
                    name: data[np.newaxis, ...] for name, data in named_inputs.items()
                }

        _LOGGER.debug("Running _infer for %s", self._model_name)
        result = await self._infer(inputs or named_inputs, parameters, headers)
        _LOGGER.debug("_infer for %s finished", self._model_name)
        if model_supports_batching:
            result = {name: data[0] for name, data in result.items()}

        return result

    async def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ):

        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        _LOGGER.debug("Running inference for %s", self._model_name)
        model_config = await self.model_config
        _LOGGER.debug("Model config for %s obtained", self._model_name)

        model_supports_batching = model_config.max_batch_size > 0
        if not model_supports_batching:
            _LOGGER.error("Model %s doesn't support batching", model_config.model_name)
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        _LOGGER.debug("Running _infer for %s", self._model_name)
        result = await self._infer(inputs or named_inputs, parameters, headers)
        _LOGGER.debug("_infer for %s finished", self._model_name)
        return result

    async def _wait_and_init_model_config(self, init_timeout_s: float):

        try:
            should_finish_before_s = time.time() + init_timeout_s
            _LOGGER.debug("Waiting for model %s to be ready", self._model_name)

            await asyncio.wait_for(self.wait_for_model(init_timeout_s), init_timeout_s)
            _LOGGER.debug("Model %s is ready", self._model_name)
            self._model_ready = True

            timeout_s = max(0.0, should_finish_before_s - time.time())
            _LOGGER.debug("Obtaining model config for %s", self._model_name)
            self._model_config = await asyncio.wait_for(
                asyncio_get_model_config(
                    self._general_client,
                    self._model_name,
                    self._model_version,
                    timeout_s=timeout_s,
                ),
                timeout_s,
            )
            _LOGGER.debug("Model config for %s obtained", self._model_name)
        except asyncio.TimeoutError as e:
            _LOGGER.error(
                "Timeout exceeded while waiting for model %s to be ready",
                self._model_name,
            )
            raise PyTritonClientTimeoutError(
                f"Timeout exceeded while waiting for model {self._model_name} to be ready"
            ) from e

    def _validate_input(self, input_name, input_data):
        if input_data.dtype == object and not isinstance(
            input_data.reshape(-1)[0], bytes
        ):
            raise RuntimeError(
                f"Numpy array for {input_name!r} input with dtype=object should contain encoded strings \
                \\(e.g. into utf-8\\). Element type: {type(input_data.reshape(-1)[0])}"
            )
        if input_data.dtype.type == np.str_:
            raise RuntimeError(
                "Unicode inputs are not supported. "
                f"Encode numpy array for {input_name!r} input (ex. with np.char.encode(array, 'utf-8'))."
            )

    async def _execute_infer(
        self, model_config, inputs_wrapped, outputs_wrapped, parameters, headers
    ) -> Any:
        try:
            _LOGGER.debug("Sending InferRequest for %s", self._model_name)
            kwargs = self._get_infer_extra_args()
            response = await self._infer_client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                headers=headers,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                parameters=parameters,
                **kwargs,
            )
        except asyncio.exceptions.TimeoutError as e:

            message = f"Timeout exceeded while running inference for {self._model_name}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except tritonclient.utils.InferenceServerException as e:
            message = f"Error occurred on Triton Inference Server side:\n {e.message()}"
            _LOGGER.error(message)
            if "Deadline Exceeded" in e.message():

                raise PyTritonClientTimeoutError(message) from e
            else:
                raise PyTritonClientInferenceServerError(message) from e
        _LOGGER.debug("Received InferResponse for %s", self._model_name)
        outputs = {
            output_spec.name: response.as_numpy(output_spec.name)
            for output_spec in model_config.outputs
        }
        return outputs

    async def _infer(self, inputs: _IOType, parameters, headers):
        if self._model_ready:
            _LOGGER.debug("Waiting for model %s config", self._model_name)
            await self._wait_and_init_model_config(self._init_timeout_s)
            _LOGGER.debug("Model wait finished for %s", self._model_name)

        _LOGGER.debug("Obtaining config for %s", self._model_name)
        model_config = await self.model_config
        _LOGGER.debug("Model config for %s obtained", self._model_name)
        if model_config.decoupled:
            raise PyTritonClientInferenceServerError(
                "Model config is decoupled. Use DecouploedAsyncioModelClient instead."
            )

        if isinstance(inputs, Tuple):
            inputs = {
                input_spec.name: input_data
                for input_spec, input_data in zip(model_config.inputs, inputs)
            }

        inputs_wrapped = []
        for input_name, input_data in inputs.items():
            if isinstance(input_data, np.ndarray):
                self._validate_input(input_name, input_data)
                triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
                infer_input = self._triton_client_lib.InferInput(
                    input_name, input_data.shape, triton_dtype
                )
                infer_input.set_data_from_numpy(input_data)
                input_wrapped = infer_input
                inputs_wrapped.append(input_wrapped)
            else:
                raise PyTritonClientValueError(
                    f"Input {input_name} is not a numpy array. Got {type(input_data)} instead."
                )

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name)
            for output_spec in model_config.outputs
        ]
        return await self._execute_infer(
            model_config, inputs_wrapped, outputs_wrapped, parameters, headers
        )

    def _handle_lazy_init(self):

        pass

    def _get_init_extra_args(self):

        if self._triton_url.scheme != "http":
            return {}

        kwargs = {
            "conn_timeout": self._inference_timeout_s,
        }
        return kwargs

    def _get_infer_extra_args(self):
        if self._triton_url.scheme == "http":
            return {}

        kwargs = {"client_timeout": self._inference_timeout_s}
        return kwargs


class AsyncioDecoupledModelClient(AsyncioModelClient):

    async def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ):

        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        _LOGGER.debug("Running inference for %s", self._model_name)
        model_config = await self.model_config
        _LOGGER.debug("Model config for %s obtained", self._model_name)

        model_supports_batching = model_config.max_batch_size > 0
        if model_supports_batching:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {
                    name: data[np.newaxis, ...] for name, data in named_inputs.items()
                }

        _LOGGER.debug("Running _infer for %s", self._model_name)
        result = self._infer(inputs or named_inputs, parameters, headers)
        _LOGGER.debug("_infer for %s finished", self._model_name)

        async for item in result:
            if model_supports_batching:
                debatched_item = {name: data[0] for name, data in item.items()}
                yield debatched_item
            else:
                yield item

    async def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ):

        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        _LOGGER.debug("Running inference for %s", self._model_name)
        model_config = await self.model_config
        _LOGGER.debug("Model config for %s obtained", self._model_name)

        model_supports_batching = model_config.max_batch_size > 0
        if not model_supports_batching:
            _LOGGER.error("Model %s doesn't support batching", model_config.model_name)
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        _LOGGER.debug("Running _infer for %s", self._model_name)
        result = self._infer(inputs or named_inputs, parameters, headers)
        _LOGGER.debug("_infer for %s finished", self._model_name)
        async for item in result:
            yield item

    async def _execute_infer(
        self, model_config, inputs_wrapped, outputs_wrapped, parameters, headers
    ) -> Any:

        error_raised_inside_async_request_iterator = set()
        try:
            _LOGGER.debug("Sending InferRequest for %s", self._model_name)
            kwargs = self._get_infer_extra_args()

            async def async_request_iterator(errors):
                _LOGGER.debug(
                    "Begin creating InferRequestHeader for %s", self._model_name
                )
                try:
                    yield {
                        "model_name": self._model_name,
                        "inputs": inputs_wrapped,
                        "outputs": outputs_wrapped,
                        "request_id": self._next_request_id,
                        "sequence_id": 0,
                        "sequence_start": True,
                        "sequence_end": True,
                    }
                except Exception as e:
                    _LOGGER.error(
                        "Error occurred while creating InferRequestHeader for %s",
                        self._model_name,
                    )
                    errors.add(e)
                    raise e
                _LOGGER.debug(
                    "End creating InferRequestHeader for %s", self._model_name
                )

            response_iterator = self._infer_client.stream_infer(
                inputs_iterator=async_request_iterator(
                    error_raised_inside_async_request_iterator
                ),
                headers=headers,
                **kwargs,
            )
            _LOGGER.debug("End preparing InferRequest for %s", self._model_name)
            while True:
                try:
                    try:
                        response = await asyncio.wait_for(
                            response_iterator.__anext__(),
                            self._inference_timeout_s,
                        )
                    except asyncio.TimeoutError as e:
                        message = f"Timeout while waiting for model {self._model_name} to return next response {self._inference_timeout_s}s"
                        _LOGGER.error(message)
                        raise PyTritonClientTimeoutError(message) from e
                    result, error = response
                    _LOGGER.debug("Received InferResponse for %s", self._model_name)
                    if error is not None:
                        raise error
                    else:
                        partial_output = {
                            output_spec.name: result.as_numpy(output_spec.name)
                            for output_spec in model_config.outputs
                        }
                    yield partial_output
                except StopAsyncIteration:
                    break
            _LOGGER.debug("End receiving InferResponse for %s", self._model_name)

        except asyncio.exceptions.TimeoutError as e:

            message = f"Timeout exceeded while running inference for {self._model_name}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except tritonclient.utils.InferenceServerException as e:
            message = f"Error occurred on Triton Inference Server side:\n {e.message()}"
            _LOGGER.error(message)
            if "Deadline Exceeded" in e.message():

                raise PyTritonClientTimeoutError(message) from e
            else:
                raise PyTritonClientInferenceServerError(message) from e
        except asyncio.exceptions.CancelledError as e:
            _LOGGER.error(
                "CancelledError occurred while streaming inference for %s",
                self._model_name,
            )

            if len(error_raised_inside_async_request_iterator) > 0:
                _LOGGER.error(
                    "Re-raising error raised inside async_request_iterator for %s ",
                    self._model_name,
                )
                raise error_raised_inside_async_request_iterator.pop() from None
            else:
                raise e

    async def _infer(self, inputs: _IOType, parameters, headers):
        if self._model_ready:
            _LOGGER.debug("Waiting for model %s config", self._model_name)
            await self._wait_and_init_model_config(self._init_timeout_s)
            _LOGGER.debug("Model wait finished for %s", self._model_name)

        _LOGGER.debug("Obtaining config for %s", self._model_name)
        model_config = await self.model_config
        _LOGGER.debug("Model config for %s obtained", self._model_name)
        if not model_config.decoupled:
            raise PyTritonClientInferenceServerError(
                "Model config is coupled. Use AsyncioModelClient instead."
            )

        if isinstance(inputs, Tuple):
            inputs = {
                input_spec.name: input_data
                for input_spec, input_data in zip(model_config.inputs, inputs)
            }

        inputs_wrapped = []
        for input_name, input_data in inputs.items():
            if isinstance(input_data, np.ndarray):
                self._validate_input(input_name, input_data)
                triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
                infer_input = self._triton_client_lib.InferInput(
                    input_name, input_data.shape, triton_dtype
                )
                infer_input.set_data_from_numpy(input_data)
                input_wrapped = infer_input
                inputs_wrapped.append(input_wrapped)
            else:
                raise PyTritonClientValueError(
                    f"Input {input_name} is not a numpy array. Got {type(input_data)} instead."
                )

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name)
            for output_spec in model_config.outputs
        ]
        result = self._execute_infer(
            model_config, inputs_wrapped, outputs_wrapped, parameters, headers
        )
        async for item in result:
            yield item

    def _get_infer_extra_args(self):
        if self._triton_url.scheme == "http":
            raise PyTritonClientValueError(
                "AsyncioDecoupledModelClient is only supported for grpc protocol"
            )
        warnings.warn(
            f"tritonclient.aio.grpc doesn't support client_timeout parameter {self._inference_timeout_s} for infer_stream",
            NotSupportedTimeoutWarning,
            stacklevel=1,
        )
        return {}


@contextlib.contextmanager
def _hub_context():
    hub = gevent.get_hub()
    try:
        yield hub
    finally:
        hub.destroy()


_INIT = "init"
_WAIT_FOR_MODEL = "wait_for_model"
_MODEL_CONFIG = "model_config"
_INFER_BATCH = "infer_batch"
_INFER_SAMPLE = "infer_sample"
_CLOSE = "close"


class FuturesModelClient:

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        max_workers: int = 128,
        max_queue_size: int = 128,
        non_blocking: bool = False,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
    ):

        self._url = url
        self._model_name = model_name
        self._model_version = model_version
        self._threads = []
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size
        self._non_blocking = non_blocking

        if self._max_workers is not None and self._max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        if self._max_queue_size is not None and self._max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than 0")

        kwargs = {}
        if self._max_queue_size is not None:
            kwargs["maxsize"] = self._max_queue_size
        self._queue = Queue(**kwargs)
        self._queue.put((_INIT, None, None))
        self._init_timeout_s = (
            _DEFAULT_FUTURES_INIT_TIMEOUT_S
            if init_timeout_s is None
            else init_timeout_s
        )
        self._inference_timeout_s = inference_timeout_s
        self._closed = False
        self._lock = Lock()
        self._existing_client = None

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.close()

    def close(self, wait=True):

        if self._closed:
            return
        _LOGGER.debug("Closing FuturesModelClient.")

        self._closed = True
        for _ in range(len(self._threads)):
            self._queue.put((_CLOSE, None, None))

        if wait:
            _LOGGER.debug("Waiting for futures to finish.")
            for thread in self._threads:
                thread.join()

    def wait_for_model(self, timeout_s: float) -> Future:

        return self._execute(
            name=_WAIT_FOR_MODEL,
            request=timeout_s,
        )

    def model_config(self) -> Future:

        return self._execute(name=_MODEL_CONFIG)

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Future:

        return self._execute(
            name=_INFER_SAMPLE,
            request=(inputs, parameters, headers, named_inputs),
        )

    def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Future:

        return self._execute(
            name=_INFER_BATCH, request=(inputs, parameters, headers, named_inputs)
        )

    def _execute(self, name, request=None):
        if self._closed:
            raise PyTritonClientClosedError("FutureModelClient is already closed")
        self._extend_thread_pool()
        future = Future()
        if self._non_blocking:
            try:
                self._queue.put_nowait((future, request, name))
            except Full as e:
                raise PyTritonClientQueueFullError("Queue is full") from e
        else:
            kwargs = {}
            if self._inference_timeout_s is not None:
                kwargs["timeout"] = self._inference_timeout_s
            try:
                self._queue.put((future, request, name), **kwargs)
            except Full as e:
                raise PyTritonClientQueueFullError("Queue is full") from e
        return future

    def _extend_thread_pool(self):
        if self._closed:
            return

        with self._lock:
            if not self._queue.empty() and (
                self._max_workers is None or len(self._threads) < self._max_workers
            ):
                _LOGGER.debug("Create new thread")
                thread = Thread(target=self._worker)
                self._threads.append(thread)
                thread.start()
            else:
                _LOGGER.debug("No need to create new thread")

    def _client_request_executor(self, client, request, name):
        _LOGGER.debug("Running %s for %s", name, self._model_name)
        if name == _INFER_SAMPLE:
            inputs, parameters, headers, named_inputs = request
            result = client.infer_sample(
                *inputs,
                parameters=parameters,
                headers=headers,
                **named_inputs,
            )
        elif name == _INFER_BATCH:
            inputs, parameters, headers, named_inputs = request
            result = client.infer_batch(
                *inputs,
                parameters=parameters,
                headers=headers,
                **named_inputs,
            )
        elif name == _MODEL_CONFIG:
            result = client.model_config
        elif name == _WAIT_FOR_MODEL:
            timeout_s = request
            result = client.wait_for_model(timeout_s)
        else:
            raise PyTritonClientValueError(f"Unknown request name {name}")
        self._set_existing_client(client)
        return result

    def _create_client(self, lazy_init):
        _LOGGER.debug("Creating ModelClient lazy_init=%s", lazy_init)
        return ModelClient(
            self._url,
            self._model_name,
            self._model_version,
            lazy_init=lazy_init,
            init_timeout_s=self._init_timeout_s,
            inference_timeout_s=self._inference_timeout_s,
        )

    def _set_existing_client(self, client):
        if client._model_config is not None:
            with self._lock:
                if self._existing_client is None:
                    _LOGGER.debug("Setting existing client")
                    self._existing_client = client

    def _remove_existing_client(self, client):
        if client is not None:
            with self._lock:
                if self._existing_client is not None:
                    if self._existing_client is client:
                        _LOGGER.debug("Resetting existing client")
                        self._existing_client = None

    def _worker(self):
        _LOGGER.debug("Starting worker thread")
        client = None

        with _hub_context():
            while True:
                future, request, name = self._queue.get()
                if future == _CLOSE:
                    _LOGGER.debug("Closing thread")
                    self._queue.task_done()
                    break
                if future == _INIT:
                    with self._lock:
                        if self._existing_client is None:
                            try:
                                _LOGGER.debug("Initial client creation")
                                client = self._create_client(False)
                                _LOGGER.debug("Setting existing client")
                                self._existing_client = client
                            except Exception as e:
                                _LOGGER.warning(
                                    "Error %s occurred during init for %s",
                                    e,
                                    self._model_name,
                                )
                    continue
                try:
                    if client is None:
                        with self._lock:
                            if self._existing_client is not None:
                                _LOGGER.debug(
                                    "Creating new client from existing client"
                                )
                                client = ModelClient.from_existing_client(
                                    self._existing_client
                                )
                    if client is None:
                        _LOGGER.debug("Creating new client")
                        client = self._create_client(name == _WAIT_FOR_MODEL)
                    with client:
                        self._set_existing_client(client)
                        while True:
                            try:
                                result = self._client_request_executor(
                                    client, request, name
                                )
                                _LOGGER.debug(
                                    "Finished %s for %s", name, self._model_name
                                )
                                future.set_result(result)
                                self._queue.task_done()
                            except Exception as e:
                                _LOGGER.error(
                                    "Error %s occurred during %s for %s",
                                    e,
                                    name,
                                    self._model_name,
                                )
                                future.set_exception(e)
                                self._queue.task_done()
                                break
                            future, request, name = self._queue.get()
                            if future == _CLOSE:
                                _LOGGER.debug("Closing thread")
                                self._queue.task_done()
                                return
                except Exception as e:
                    _LOGGER.error(
                        "Error %s occurred during %s for %s", e, name, self._model_name
                    )
                    future.set_exception(e)
                    self._queue.task_done()
                finally:
                    self._remove_existing_client(client)
                    client = None
        _LOGGER.debug("Finishing worker thread")
