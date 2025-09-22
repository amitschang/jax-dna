from aim import Run

from .logger import Logger, Status


class AimLogger(Logger):
    """Logger that emits metrics and status to Aim."""

    def __init__(self, aim_run: Run = None) -> "AimLogger":
        """Initialize the AimLogger.

        Args:
            aim_run: Aim Run object (optional).
        """
        super().__init__()
        self.aim_run = aim_run if aim_run is not None else Run()

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric value to Aim."""
        value = float(value)  # Give aim python object (iso jax/numpy array obj)
        self.aim_run.track(value, name=name, step=step)

    def _update_status(self, name: str, status: Status) -> None:
        """Log status changes to Aim."""
        self.aim_run.track(str(status), name=f"status/{name}")
