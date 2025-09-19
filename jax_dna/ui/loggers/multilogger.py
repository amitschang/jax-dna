"""MultiLogger: Logger that routes logs to multiple other loggers."""

from pathlib import Path

from .logger import Logger, Status


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
