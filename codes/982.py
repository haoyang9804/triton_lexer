import argparse
import signal
from functools import partial

import tritonserver
from engine.triton_engine import TritonLLMEngine
from frontend.fastapi_frontend import FastApiFrontend


def signal_handler(
    server, openai_frontend, kserve_http_frontend, kserve_grpc_frontend, signal, frame
):
    print(f"Received {signal=}, {frame=}")

    shutdown(server, openai_frontend, kserve_http_frontend, kserve_grpc_frontend)


def shutdown(server, openai_frontend, kserve_http, kserve_grpc):
    print("Shutting down Triton OpenAI-Compatible Frontend...")
    openai_frontend.stop()

    if kserve_http:
        print("Shutting down Triton KServe HTTP Frontend...")
        kserve_http.stop()

    if kserve_grpc:
        print("Shutting down Triton KServe GRPC Frontend...")
        kserve_grpc.stop()

    print("Shutting down Triton Inference Server...")
    server.stop()


def start_kserve_frontends(server, args):
    http_service, grpc_service = None, None
    try:
        from tritonfrontend import KServeGrpc, KServeHttp

        http_options = KServeHttp.Options(address=args.host, port=args.kserve_http_port)
        http_service = KServeHttp(server, http_options)
        http_service.start()

        grpc_options = KServeGrpc.Options(address=args.host, port=args.kserve_grpc_port)
        grpc_service = KServeGrpc(server, grpc_options)
        grpc_service.start()

    except ModuleNotFoundError:

        print(
            "[WARNING] The 'tritonfrontend' package was not found. "
            "KServe frontends won't be available through this application without it. "
            "Check /opt/tritonserver/python for tritonfrontend*.whl and pip install it if present."
        )
    return http_service, grpc_service


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton Inference Server with OpenAI-Compatible RESTful API server."
    )

    triton_group = parser.add_argument_group("Triton Inference Server")
    triton_group.add_argument(
        "--model-repository",
        type=str,
        required=True,
        help="Path to the Triton model repository holding the models to be served",
    )
    triton_group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace ID or local folder path of the Tokenizer to use for chat templates",
    )
    triton_group.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["vllm", "tensorrtllm"],
        help="Manual override of Triton backend request format (inputs/output names) to use for inference",
    )
    triton_group.add_argument(
        "--lora-separator",
        type=str,
        default=None,
        help="LoRA name selection may be appended to the model name following this separator if the separator is provided",
    )
    triton_group.add_argument(
        "--tritonserver-log-verbose-level",
        type=int,
        default=0,
        help="The tritonserver log verbosity level",
    )
    triton_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Address/host of frontends (default: '0.0.0.0')",
    )
    triton_group.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        help="Specify the parser for handling tool calling related response text. Options include: 'llama3' and 'mistral'.",
    )

    triton_group.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The path to the custom Jinja chat template file. This is useful if you'd like to use a different chat template than the one provided by the model.",
    )

    openai_group = parser.add_argument_group("Triton OpenAI-Compatible Frontend")
    openai_group.add_argument(
        "--openai-port", type=int, default=9000, help="OpenAI HTTP port (default: 9000)"
    )
    openai_group.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )

    kserve_group = parser.add_argument_group("Triton KServe Frontend")
    kserve_group.add_argument(
        "--enable-kserve-frontends",
        action="store_true",
        help="Enable KServe Predict v2 HTTP/GRPC frontends (disabled by default)",
    )
    kserve_group.add_argument(
        "--kserve-http-port",
        type=int,
        default=8000,
        help="KServe Predict v2 HTTP port (default: 8000)",
    )
    kserve_group.add_argument(
        "--kserve-grpc-port",
        type=int,
        default=8001,
        help="KServe Predict v2 GRPC port (default: 8001)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    server: tritonserver.Server = tritonserver.Server(
        model_repository=args.model_repository,
        log_verbose=args.tritonserver_log_verbose_level,
        log_info=True,
        log_warn=True,
        log_error=True,
    ).start(wait_until_ready=True)

    engine: TritonLLMEngine = TritonLLMEngine(
        server=server,
        tokenizer=args.tokenizer,
        backend=args.backend,
        lora_separator=args.lora_separator,
        tool_call_parser=args.tool_call_parser,
        chat_template=args.chat_template,
    )

    openai_frontend: FastApiFrontend = FastApiFrontend(
        engine=engine,
        host=args.host,
        port=args.openai_port,
        log_level=args.uvicorn_log_level,
    )

    kserve_http, kserve_grpc = None, None
    if args.enable_kserve_frontends:
        kserve_http, kserve_grpc = start_kserve_frontends(server, args)

    signal.signal(
        signal.SIGINT,
        partial(signal_handler, server, openai_frontend, kserve_http, kserve_grpc),
    )
    signal.signal(
        signal.SIGTERM,
        partial(signal_handler, server, openai_frontend, kserve_http, kserve_grpc),
    )

    openai_frontend.start()


if __name__ == "__main__":
    main()
