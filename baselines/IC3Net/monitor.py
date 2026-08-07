"""Training monitor with rich terminal UI.

Ported from the CoGrid reference implementation. Provides a live progress bar
and metrics table for IC3Net-family training.
"""
import sys
import time
from typing import Any, Dict, Optional

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console, Group
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class TrainingMonitor:
    """Rich-based live training monitor with progress bar and metrics table."""

    def __init__(
        self,
        total_updates: int,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        if not _HAS_RICH:
            raise ImportError(
                "TrainingMonitor requires the 'rich' package. "
                "Install with: pip install rich"
            )

        self.total_updates = total_updates
        self.console = Console()
        self.live = None
        self.config = config_dict or {}
        self.last_metrics: Dict[str, Any] = {}
        self.current_step = 0

        # Initialize Progress Bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.task_id = self.progress.add_task(
            "[green]Training...", total=total_updates
        )

    def __enter__(self):
        self.live = Live(
            self._generate_layout(),
            refresh_per_second=4,
            console=self.console,
        )
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.stop()

    def _generate_layout(self):
        # Metrics Table
        table = Table(
            title="Current Metrics",
            expand=True,
            border_style="blue",
            title_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        if self.last_metrics:
            for key, value in self.last_metrics.items():
                if isinstance(value, float):
                    val_str = f"{value:.4f}"
                else:
                    val_str = str(value)
                table.add_row(key, val_str)
        else:
            table.add_row("Status", "Initializing...")

        return Group(
            Panel(self.progress, style="white on blue", border_style="blue"),
            Panel(table, border_style="green"),
        )

    def update(self, step: int, metrics: Dict[str, Any]):
        """Update the live display with new metrics."""
        self.current_step = step
        self.last_metrics = metrics

        self.progress.update(self.task_id, completed=step)

        if self.live:
            self.live.update(self._generate_layout())

    def log(self, message: str):
        """Print a message above the live display."""
        if self.live:
            self.console.print(
                f"[dim]{time.strftime('%H:%M:%S')}[/dim] {message}"
            )
        else:
            print(message)


class TrainingMonitorInterface:
    """Wrapper that gracefully falls back to plain-text logging when rich
    is not available or we're not running in a terminal."""

    def __init__(
        self,
        total_updates: int,
        config_dict: Optional[Dict[str, Any]] = None,
        use_rich: bool = True,
    ):
        self.total_updates = total_updates
        self.config_dict = config_dict or {}
        self._monitor: Optional[TrainingMonitor] = None
        self._use_rich = use_rich and self._is_tty() and _HAS_RICH

    @staticmethod
    def _is_tty() -> bool:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())

    def __enter__(self) -> "TrainingMonitorInterface":
        if self._use_rich:
            try:
                self._monitor = TrainingMonitor(
                    total_updates=self.total_updates,
                    config_dict=self.config_dict,
                )
                self._monitor.__enter__()
            except Exception:
                self._use_rich = False
                self._monitor = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._monitor is not None:
            self._monitor.__exit__(exc_type, exc_val, exc_tb)

    def update(self, step: int, metrics: Dict[str, Any]) -> None:
        if self._monitor is not None:
            self._monitor.update(step, metrics)
        else:
            parts = [f"step={step}/{self.total_updates}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                elif v is not None:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts), flush=True)

    def log(self, message: str) -> None:
        if self._monitor is not None:
            self._monitor.log(message)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
