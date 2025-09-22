"""Logger logging jax_dna optimization results to the console."""

from pathlib import Path

import typing_extensions

from jax_dna.ui.loggers.logger import Logger, Status


class ConsoleLogger(Logger):
    """Console logger."""

    def __init__(self, log_dir: str | Path | None = None) -> "ConsoleLogger":
        """Initialize the console logger."""
        super().__init__(log_dir)

    @typing_extensions.override
    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
    ) -> None:
        super().log_metric(name, value, step)

        print(f"Step: {step}, {name}: {value}")  # noqa: T201 -- we intend to print to the console

    def _update_status(self, name: str, status: Status) -> None:
        return print(name, status)  # noqa: T201 -- we intend to print to the console