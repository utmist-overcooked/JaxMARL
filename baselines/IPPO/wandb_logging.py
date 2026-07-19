"""W&B logging helpers for IPPO v3 trainers.

The JAX training loop emits device arrays through ``jax.debug.callback``.
W&B history works best when every logged metric is a finite scalar and the
actual x-axis is declared explicitly. This module keeps that conversion in one
place so the CNN and RNN trainers produce the same chart-friendly payloads.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
import wandb


_ACTIVE_MONITOR = None
_CAPTURE_HISTORY_TABLE = False
_HISTORY_ROWS = []

_PREFERRED_HISTORY_COLUMNS = (
    "env_step",
    "update_step",
    "returned_episode",
    "returned_episode_returns",
    "returned_episode_lengths",
    "original_reward",
    "shaped_reward",
    "reward_sum",
    "base_reward",
    "base_reward_per_step",
    "combined_reward",
    "combined_reward_per_step",
    "mean_reward",
    "max_reward",
    "delivery",
    "delivery_count.agent_0",
    "delivery_count.agent_1",
    "event/delivery",
    "event/dish_pickup",
    "event/dish_to_goal_progress",
    "event/drop",
    "event/order_added",
    "event/order_expired",
    "event/pickup",
    "event/pot_burn",
    "event/pot_placement",
    "event/pot_start_cooking",
    "event_rate/delivery",
    "event_rate/dish_pickup",
    "event_rate/dish_to_goal_progress",
    "event_rate/drop",
    "event_rate/order_added",
    "event_rate/order_expired",
    "event_rate/pickup",
    "event_rate/pot_burn",
    "event_rate/pot_placement",
    "event_rate/pot_start_cooking",
    "order/active_count",
    "order/front_type",
    "loss/total",
    "loss/value",
    "loss/policy",
    "loss/entropy",
    "anneal_factor",
)


def reset_wandb_logging(capture_history_table: bool = False) -> None:
    """Reset per-run logging state."""
    global _CAPTURE_HISTORY_TABLE, _HISTORY_ROWS
    _CAPTURE_HISTORY_TABLE = bool(capture_history_table)
    _HISTORY_ROWS = []


def set_active_monitor(monitor) -> None:
    """Attach or clear the optional terminal training monitor."""
    global _ACTIVE_MONITOR
    _ACTIVE_MONITOR = monitor


def define_wandb_metrics() -> None:
    """Declare metric x-axes without wildcarding non-scalar media/table keys."""
    if wandb.run is None:
        return

    wandb.define_metric("env_step")
    for metric_name in _PREFERRED_HISTORY_COLUMNS:
        if metric_name != "env_step":
            wandb.define_metric(metric_name, step_metric="env_step")


def log_training_metrics(metric: Dict[str, Any]) -> None:
    """Log one sanitized PPO-update payload to W&B and the terminal monitor."""
    payload = _sanitize_payload(_flatten_metric_dict(metric))
    env_step = _first_int(payload.get("env_step", payload.get("update_step", 0)))
    update_step = _first_int(payload.get("update_step", 0))

    payload["env_step"] = env_step
    payload["update_step"] = update_step
    payload.pop("step", None)

    if _CAPTURE_HISTORY_TABLE:
        _HISTORY_ROWS.append(dict(payload))

    if wandb.run is not None:
        wandb.log(payload, step=env_step)

    if _ACTIVE_MONITOR is not None:
        _ACTIVE_MONITOR.update(update_step, _monitor_payload(payload))


def log_history_table_to_wandb() -> None:
    """Upload an optional scalar history table for custom chart construction."""
    if not _CAPTURE_HISTORY_TABLE or wandb.run is None or not _HISTORY_ROWS:
        return

    all_columns = sorted({column for row in _HISTORY_ROWS for column in row})
    columns = [column for column in _PREFERRED_HISTORY_COLUMNS if column in all_columns]
    seen = set(columns)
    columns.extend(column for column in all_columns if column not in seen)

    table = wandb.Table(
        columns=columns,
        data=[[row.get(column) for column in columns] for row in _HISTORY_ROWS],
    )
    wandb.run.summary["training_history/rows"] = len(_HISTORY_ROWS)
    wandb.run.summary["training_history/columns"] = len(columns)
    wandb.log({"training_history/table": table}, step=int(_HISTORY_ROWS[-1]["env_step"]))


def _flatten_metric_dict(metric: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in metric.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metric_dict(value, name))
        else:
            flat[name] = value
    return flat


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in payload.items():
        scalar = _finite_scalar(value)
        if scalar is not None:
            sanitized[key] = scalar
    return sanitized


def _finite_scalar(value: Any) -> Optional[float]:
    arr = np.asarray(value)
    if arr.shape != () and arr.size != 1:
        return None
    try:
        scalar = arr.reshape(()).item()
    except ValueError:
        return None
    if isinstance(scalar, (bool, np.bool_)):
        return int(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, (float, np.floating)):
        scalar = float(scalar)
        return scalar if math.isfinite(scalar) else None
    return None


def _first_int(value: Any, default: int = 0) -> int:
    scalar = _finite_scalar(value)
    if scalar is not None:
        return int(scalar)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for item in value:
            return _first_int(item, default)
    return default


def _monitor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        ("env_step", "env_step"),
        ("base_reward_per_step", "base_rew/step"),
        ("combined_reward_per_step", "combined_rew/step"),
        ("combined_reward", "combined_rew"),
        ("delivery", "delivery"),
        ("event/pickup", "pickup"),
        ("event/drop", "drop"),
        ("event/pot_placement", "pot_place"),
        ("event/pot_start_cooking", "pot_start"),
        ("event/dish_pickup", "dish_pickup"),
        ("event/dish_to_goal_progress", "dish_to_goal"),
        ("event/pot_burn", "pot_burn"),
        ("event/order_expired", "order_expired"),
        ("event/order_added", "order_added"),
        ("order/front_type", "order_front"),
        ("order/active_count", "orders_active"),
        ("loss/total", "loss"),
        ("loss/value", "value_loss"),
        ("loss/entropy", "entropy"),
        ("anneal_factor", "anneal"),
    )
    return {label: payload[key] for key, label in keys if key in payload}
