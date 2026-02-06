"""Rich-based live training monitor for JAX training loops."""

import time
from typing import Any, Dict, Optional


class TrainingMonitor:
    """Live terminal display with progress bar and metrics table.

    Usage::

        with TrainingMonitor(total_updates=1000, config_dict=cfg) as mon:
            # inside jax.debug.callback:
            mon.update(step=i, metrics={"loss": 0.5})
            mon.log("Checkpoint saved")

    Falls back to no-op if ``rich`` is not installed.
    """

    def __init__(
        self,
        total_updates: int,
        config_dict: Optional[Dict[str, Any]] = None,
        tracked_seed: Optional[int] = None,
        title: str = "Training",
    ):
        self.total_updates = total_updates
        self.config = config_dict or {}
        self.tracked_seed = tracked_seed
        self.title = title

        self._first_seed_seen: Optional[int] = None
        self._current_metrics: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._live = None
        self._progress = None
        self._task_id = None

        try:
            from rich.console import Console  # noqa: F401
            self._rich_available = True
        except ImportError:
            self._rich_available = False

    # ── context manager ──────────────────────────────────────

    def __enter__(self):
        if not self._rich_available:
            return self

        from rich.live import Live
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._task_id = self._progress.add_task(
            f"[green]{self.title}", total=self.total_updates
        )

        self._start_time = time.time()
        self._live = Live(
            self._generate_layout(), refresh_per_second=4, console=self._console
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live is not None:
            self._live.stop()
        return False

    # ── public API ───────────────────────────────────────────

    def update(self, step: int, metrics: Dict[str, Any], seed: Optional[int] = None):
        """Update progress bar and metrics table."""
        if not self._rich_available or self._live is None:
            return
        if not self._should_track(seed):
            return

        # Compute SPS from env_step if available
        if self._start_time is not None and "env_step" in metrics:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                sps = int(metrics["env_step"]) / elapsed
                metrics = {**metrics, "SPS": sps}

        self._current_metrics = metrics
        self._progress.update(self._task_id, completed=step)
        self._live.update(self._generate_layout())

    def log(self, message: str):
        """Print a timestamped message above the live display."""
        if not self._rich_available or self._live is None:
            return
        self._console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {message}")

    # ── internals ────────────────────────────────────────────

    @property
    def _console(self):
        from rich.console import Console
        if not hasattr(self, "__console"):
            self.__console = Console()
        return self.__console

    def _should_track(self, seed_value: Optional[int]) -> bool:
        """With vmap, only track the first seed seen (or a configured one)."""
        if seed_value is None:
            return True
        if self._first_seed_seen is None:
            self._first_seed_seen = seed_value
        if self.tracked_seed is not None:
            return seed_value == self.tracked_seed
        return seed_value == self._first_seed_seen

    def _generate_layout(self):
        from rich.table import Table
        from rich.panel import Panel
        from rich.console import Group

        # Metrics table
        table = Table(
            title="Current Metrics",
            expand=True,
            border_style="blue",
            title_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        if self._current_metrics:
            for key, value in self._current_metrics.items():
                if key == "SPS":
                    table.add_row(key, f"{value:,.0f}")
                elif isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
        else:
            table.add_row("Status", "Initializing...")

        return Group(
            Panel(self._progress, style="white on blue", border_style="blue"),
            Panel(table, border_style="green"),
        )
