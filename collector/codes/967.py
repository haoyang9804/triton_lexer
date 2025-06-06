import asyncio
import logging
import time
from typing import Optional, Union

import aiohttp
import grpc
import tritonclient.grpc
import tritonclient.http

from pytriton.client.exceptions import (
    PyTritonClientModelUnavailableError,
    PyTritonClientTimeoutError,
)
from pytriton.client.utils import (
    LATEST_MODEL_VERSION,
    ModelState,
    parse_grpc_response,
    parse_http_response,
)
from pytriton.model_config.parser import ModelConfigParser

aio_clients = Union[
    tritonclient.grpc.aio.InferenceServerClient,
    tritonclient.http.aio.InferenceServerClient,
]

_LOGGER = logging.getLogger(__name__)

_DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S = 60.0
_DEFAULT_ASYNC_SLEEP_FACTOR_S = 0.1


async def asyncio_get_model_state(
    client: aio_clients,
    model_name: str,
    model_version: Optional[str] = None,
) -> ModelState:

    _LOGGER.debug("Obtaining model %s state", model_name)
    repository_index = await client.get_model_repository_index()
    _LOGGER.debug("Model repository index obtained")
    if isinstance(repository_index, list):
        models_states = parse_http_response(models=repository_index)
    else:
        models_states = parse_grpc_response(models=repository_index.models)

    if model_version is None:
        requested_model_states = {
            version: state
            for (name, version), state in models_states.items()
            if name == model_name
        }
        if not requested_model_states:
            return ModelState.UNAVAILABLE
        else:
            requested_model_states = sorted(
                requested_model_states.items(), key=lambda item: int(item[0])
            )
            latest_version, latest_version_state = requested_model_states[-1]
            _LOGGER.debug(
                "Model %s latest version: %s state: %s",
                model_name,
                latest_version,
                latest_version_state,
            )
            return latest_version_state
    else:
        key = (model_name, model_version)
        if key not in models_states:
            return ModelState.UNAVAILABLE
        else:
            model_state = models_states[key]
            _LOGGER.debug(
                "Model %s version %s state: %s", model_name, model_version, model_state
            )
            return model_state


async def asyncio_get_model_config(
    client: aio_clients,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
):

    should_finish_before = time.time() + timeout_s
    _LOGGER.debug("Obtaining model %s config (timeout=%0.2f)", model_name, timeout_s)
    try:
        _LOGGER.debug("Waiting for model %s to be ready", model_name)
        await asyncio.wait_for(
            asyncio_wait_for_model_ready(
                client,
                model_name=model_name,
                model_version=model_version,
                timeout_s=timeout_s,
            ),
            timeout_s,
        )

        model_version = model_version or ""

        timeout_s = max(0, should_finish_before - time.time())
        if isinstance(client, tritonclient.grpc.aio.InferenceServerClient):
            _LOGGER.debug("Obtaining model %s config as_json=True", model_name)
            response = await asyncio.wait_for(
                client.get_model_config(model_name, model_version, as_json=True),
                timeout_s,
            )
            model_config = response["config"]
        else:
            _LOGGER.debug("Obtaining model %s config", model_name)
            model_config = await asyncio.wait_for(
                client.get_model_config(model_name, model_version), timeout_s
            )
        _LOGGER.debug("Model config obtained")
        model_config = ModelConfigParser.from_dict(model_config)
        _LOGGER.debug("Model config: %s", model_config)
        return model_config
    except asyncio.TimeoutError as e:
        message = f"Timeout while waiting for model {model_name} config (timeout={timeout_s:0.2f})"
        _LOGGER.error(message)
        raise PyTritonClientTimeoutError(message) from e


async def asyncio_wait_for_server_ready(
    asyncio_client: aio_clients,
    sleep_time_s: float,
):

    _LOGGER.debug("Waiting for server to be ready")
    try:
        while True:
            try:
                _LOGGER.debug("Waiting for server to be ready")
                server_ready = await asyncio_client.is_server_ready()
                _LOGGER.debug("Waiting for server to be live")
                server_live = await asyncio_client.is_server_live()
            except tritonclient.utils.InferenceServerException:

                server_live = False
                server_ready = False
            except aiohttp.client_exceptions.ClientConnectorError:

                server_live = False
                server_ready = False
            except RuntimeError:

                server_live = False
                server_ready = False
            except grpc._cython.cygrpc.UsageError:

                server_live = False
                server_ready = False
            if server_ready and server_live:
                break
            _LOGGER.debug("Sleeping for %0.2f seconds", sleep_time_s)
            await asyncio.sleep(sleep_time_s)
    except asyncio.TimeoutError as e:

        message = "Timeout while waiting for model"
        _LOGGER.error(message)
        raise PyTritonClientTimeoutError(message) from e
    _LOGGER.debug("Server is ready")


async def asyncio_wait_for_model_status_loaded(
    asyncio_client: aio_clients,
    model_name: str,
    sleep_time_s: float,
    model_version: Optional[str] = None,
):

    model_version = model_version or ""
    model_version_msg = model_version or LATEST_MODEL_VERSION
    _LOGGER.debug("Waiting for model %s, %s to be ready", model_name, model_version_msg)
    try:
        while True:
            _LOGGER.debug("Checking if model %s is ready", model_name)
            is_model_ready = await asyncio_client.is_model_ready(
                model_name, model_version
            )
            if is_model_ready:
                break
            _LOGGER.debug("Sleeping for %s seconds", sleep_time_s)
            await asyncio.sleep(sleep_time_s)
    except asyncio.TimeoutError as e:
        message = f"Timeout while waiting for model {model_name} state (timeout={sleep_time_s:0.2f})"
        _LOGGER.error(message)
        raise PyTritonClientTimeoutError(message) from e
    _LOGGER.debug("Model %s, %s is ready", model_name, model_version_msg)


async def asyncio_wait_for_model_ready(
    asyncio_client: aio_clients,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
):

    _LOGGER.debug(
        "Waiting for model %s to be ready (timeout=%0.2f)", model_name, timeout_s
    )
    sleep_time_s = timeout_s * _DEFAULT_ASYNC_SLEEP_FACTOR_S
    try:
        should_finish_before = time.time() + timeout_s
        await asyncio.wait_for(
            asyncio_wait_for_server_ready(asyncio_client, sleep_time_s), timeout_s
        )
        _LOGGER.debug("Waiting for model %s to be ready", model_name)
        timeout_s = max(0, should_finish_before - time.time())
        await asyncio.wait_for(
            asyncio_wait_for_model_status_loaded(
                asyncio_client,
                model_name=model_name,
                model_version=model_version,
                sleep_time_s=sleep_time_s,
            ),
            timeout_s,
        )
    except PyTritonClientModelUnavailableError as e:
        _LOGGER.error("Failed to obtain model %s config error %s", model_name, e)
        raise e
    except asyncio.TimeoutError as e:
        _LOGGER.error("Failed to obtain model %s config error %s", model_name, e)
        raise PyTritonClientTimeoutError(
            f"Timeout while waiting for model {model_name} to be ready (timeout={timeout_s:0.2f})"
        ) from e
    _LOGGER.debug("Model %s is ready", model_name)
