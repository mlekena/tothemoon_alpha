from typing import Any


class Status:
    def __init__(self) -> None:
        self.message = ""

    # https://github.com/python/mypy/issues/2783#issuecomment-276596902
    def __eq__(self, other: Any) -> bool:
        return self.message == other.message  # type: ignore


class Success(Status):
    def __init__(self) -> None:
        self.message = "SUCCESS"


class Failure(Status):
    def __init__(self) -> None:
        self.message = "FAILURE"
