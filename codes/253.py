import os
import importlib

import pytest

import triton
import triton.language as tl


@pytest.fixture(scope="class", params=["0", "1"])
def triton_interpret(request):

    os.environ["TRITON_INTERPRET"] = request.param
    importlib.reload(triton)
    importlib.reload(tl)
    yield request.param
    os.environ.pop("TRITON_INTERPRET", None)
