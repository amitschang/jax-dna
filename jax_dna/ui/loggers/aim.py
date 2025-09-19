from pathlib import Path
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
        self.aim_run.track(value, name=name, step=step)

    def __update_status(self, name: str, status: Status) -> None:
        """Log status changes to Aim."""
        self.aim_run.track(str(status), name=f"status/{name}")

    def update_simulator_status(self, name: str, status: Status) -> None:
        """Update simulator status."""
        self.__update_status(name, status)

    def set_simulator_started(self, name: str) -> None:
        """Set simulator status to STARTED."""
        self.__update_status(name, Status.STARTED)

    def set_simulator_running(self, name: str) -> None:
        """Set simulator status to RUNNING."""
        self.__update_status(name, Status.RUNNING)

    def set_simulator_complete(self, name: str) -> None:
        """Set simulator status to COMPLETE."""
        self.__update_status(name, Status.COMPLETE)

    def set_simulator_error(self, name: str) -> None:
        """Set simulator status to ERROR."""
        self.__update_status(name, Status.ERROR)

    def update_objective_status(self, name: str, status: Status) -> None:
        """Update objective status."""
        self.__update_status(name, status)

    def set_objective_started(self, name: str) -> None:
        """Set objective status to STARTED."""
        self.__update_status(name, Status.STARTED)

    def set_objective_running(self, name: str) -> None:
        """Set objective status to RUNNING."""
        self.__update_status(name, Status.RUNNING)

    def set_objective_complete(self, name: str) -> None:
        """Set objective status to COMPLETE."""
        self.__update_status(name, Status.COMPLETE)

    def set_objective_error(self, name: str) -> None:
        """Set objective status to ERROR."""
        self.__update_status(name, Status.ERROR)

    def update_observable_status(self, name: str, status: Status) -> None:
        """Update observable status."""
        self.__update_status(name, status)

    def set_observable_started(self, name: str) -> None:
        """Set observable status to STARTED."""
        self.__update_status(name, Status.STARTED)

    def set_observable_running(self, name: str) -> None:
        """Set observable status to RUNNING."""
        self.__update_status(name, Status.RUNNING)

    def set_observable_complete(self, name: str) -> None:
        """Set observable status to COMPLETE."""
        self.__update_status(name, Status.COMPLETE)

    def set_observable_error(self, name: str) -> None:
        """Set observable status to ERROR."""
        self.__update_status(name, Status.ERROR)


class MultiLogger(Logger):
    """Logger that routes logs to multiple other loggers."""

    def __init__(self, loggers: list[Logger], log_dir: str | Path | None = None) -> "MultiLogger":
        """Initialize MultiLogger.

        Args:
            loggers: List of Logger instances to route logs to.
            log_dir: Optional log directory for disk logs (not used by MultiLogger itself).
        """
        super().__init__(log_dir)
        self.loggers = loggers

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric value to all configured loggers."""
        for logger in self.loggers:
            logger.log_metric(name, value, step)

    def update_simulator_status(self, name: str, status: Status) -> None:
        """Update simulator status in all loggers."""
        for logger in self.loggers:
            logger.update_simulator_status(name, status)

    def set_simulator_started(self, name: str) -> None:
        """Set simulator status to STARTED in all loggers."""
        for logger in self.loggers:
            logger.set_simulator_started(name)

    def set_simulator_running(self, name: str) -> None:
        """Set simulator status to RUNNING in all loggers."""
        for logger in self.loggers:
            logger.set_simulator_running(name)

    def set_simulator_complete(self, name: str) -> None:
        """Set simulator status to COMPLETE in all loggers."""
        for logger in self.loggers:
            logger.set_simulator_complete(name)

    def set_simulator_error(self, name: str) -> None:
        """Set simulator status to ERROR in all loggers."""
        for logger in self.loggers:
            logger.set_simulator_error(name)

    def update_objective_status(self, name: str, status: Status) -> None:
        """Update objective status in all loggers."""
        for logger in self.loggers:
            logger.update_objective_status(name, status)

    def set_objective_started(self, name: str) -> None:
        """Set objective status to STARTED in all loggers."""
        for logger in self.loggers:
            logger.set_objective_started(name)

    def set_objective_running(self, name: str) -> None:
        """Set objective status to RUNNING in all loggers."""
        for logger in self.loggers:
            logger.set_objective_running(name)

    def set_objective_complete(self, name: str) -> None:
        """Set objective status to COMPLETE in all loggers."""
        for logger in self.loggers:
            logger.set_objective_complete(name)

    def set_objective_error(self, name: str) -> None:
        """Set objective status to ERROR in all loggers."""
        for logger in self.loggers:
            logger.set_objective_error(name)

    def update_observable_status(self, name: str, status: Status) -> None:
        """Update observable status in all loggers."""
        for logger in self.loggers:
            logger.update_observable_status(name, status)

    def set_observable_started(self, name: str) -> None:
        """Set observable status to STARTED in all loggers."""
        for logger in self.loggers:
            logger.set_observable_started(name)

    def set_observable_running(self, name: str) -> None:
        """Set observable status to RUNNING in all loggers."""
        for logger in self.loggers:
            logger.set_observable_running(name)

    def set_observable_complete(self, name: str) -> None:
        """Set observable status to COMPLETE in all loggers."""
        for logger in self.loggers:
            logger.set_observable_complete(name)

    def set_observable_error(self, name: str) -> None:
        """Set observable status to ERROR in all loggers."""
        for logger in self.loggers:
            logger.set_observable_error(name)
