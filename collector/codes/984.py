import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import openai
import requests
import tritonserver

sys.path.append(os.path.join(Path(__file__).resolve().parent, "..", "openai_frontend"))
from engine.triton_engine import TritonLLMEngine
from frontend.fastapi_frontend import FastApiFrontend


def setup_server(model_repository: str):
    server: tritonserver.Server = tritonserver.Server(
        model_repository=model_repository,
        log_verbose=0,
        log_info=True,
        log_warn=True,
        log_error=True,
    ).start(wait_until_ready=True)
    return server


def setup_fastapi_app(tokenizer: str, server: tritonserver.Server, backend: str):
    engine: TritonLLMEngine = TritonLLMEngine(
        server=server, tokenizer=tokenizer, backend=backend
    )
    frontend: FastApiFrontend = FastApiFrontend(engine=engine)
    return frontend.app


class OpenAIServer:
    API_KEY = "EMPTY"
    START_TIMEOUT = 240

    def __init__(
        self,
        cli_args: List[str],
        *,
        env_dict: Optional[Dict[str, str]] = None,
    ) -> None:

        self.host = "localhost"
        self.port = 9000

        env = os.environ.copy()
        if env_dict is not None:
            env.update(env_dict)

        this_dir = Path(__file__).resolve().parent
        script_path = this_dir / ".." / "openai_frontend" / "main.py"
        self.proc = subprocess.Popen(
            ["python3", script_path] + cli_args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        self._wait_for_server(
            url=self.url_for("health", "ready"), timeout=self.START_TIMEOUT
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            wait_secs = 30
            self.proc.wait(wait_secs)
        except subprocess.TimeoutExpired:

            self.proc.kill()

    def _wait_for_server(self, *, url: str, timeout: float):
        start = time.time()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    break
            except Exception as err:
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from err

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self):
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.API_KEY,
        )

    def get_async_client(self):
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.API_KEY,
        )
