import os
from typing import Any, Callable

environment_variables: dict[str, Callable[[], Any]] = {
    # Whether to print debug info.
    "NANOVLLM_ENABLE_DEBUG": lambda: bool(int(os.getenv("NANOVLLM_ENABLE_DEBUG", "0"))),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
