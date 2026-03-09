import csv
import json
import os
from datetime import datetime
from pathlib import Path

import wandb


ENTITY = os.getenv("WANDB_ENTITY", "dannyb3334-university-of-toronto")
PROJECT = os.getenv("WANDB_PROJECT", "overcookedv3_ippo_20260306_032919")
TARGET_NAMES = ["expert-sweep-1"]  # Target all runs


def _json_safe(value):
	if isinstance(value, datetime):
		return value.isoformat()
	return value


def _to_plain(value, seen=None):
	if seen is None:
		seen = set()

	if isinstance(value, (str, int, float, bool)) or value is None:
		return value

	if isinstance(value, datetime):
		return value.isoformat()

	obj_id = id(value)
	if obj_id in seen:
		return "<circular_ref>"

	if isinstance(value, dict):
		seen.add(obj_id)
		return {str(k): _to_plain(v, seen) for k, v in value.items()}

	if isinstance(value, (list, tuple, set)):
		seen.add(obj_id)
		return [_to_plain(v, seen) for v in value]

	return str(value)


def collect_runs():
	api = wandb.Api()
	path = f"{ENTITY}/{PROJECT}"
	runs = list(api.runs(path))

	records = []
	for run in runs:
		user = getattr(run, "user", None)
		sweep = getattr(run, "sweep", None)
		records.append(
			{
				"id": run.id,
				"name": run.name,
				"display_name": run.display_name,
				"path": run.path,
				"url": run.url,
				"project": run.project,
				"entity": run.entity,
				"state": run.state,
				"group": getattr(run, "group", None),
				"job_type": getattr(run, "job_type", None),
				"notes": getattr(run, "notes", None),
				"tags": list(run.tags) if run.tags is not None else [],
				"sweep_id": sweep.id if sweep is not None else None,
				"created_at": _json_safe(getattr(run, "created_at", None)),
				"heartbeat_at": _json_safe(getattr(run, "heartbeat_at", None)),
				"updated_at": _json_safe(getattr(run, "updated_at", None)),
				"host": getattr(run, "host", None),
				"user": getattr(user, "username", None),
				"config": _to_plain(dict(run.config)),
				"summary": _to_plain(dict(run.summary)),
				"system_metrics": _to_plain(dict(getattr(run, "system_metrics", {}))),
				"history_keys": list(run.history_keys) if run.history_keys else [],
				"commit": getattr(run, "commit", None),
				"description": getattr(run, "description", None),
			}
		)
	return records


def main():
	records = collect_runs()

	full_file = Path("wandb_runs_full.json")
	full_file.write_text(json.dumps(records, indent=2, default=_json_safe), encoding="utf-8")

	rows_file = Path("wandb_runs_table.csv")
	with rows_file.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"id",
				"name",
				"state",
				"group",
				"job_type",
				"created_at",
				"updated_at",
				"url",
			],
		)
		writer.writeheader()
		for record in records:
			writer.writerow({k: record.get(k) for k in writer.fieldnames})

	print(f"Fetched {len(records)} runs from {ENTITY}/{PROJECT}")
	for record in records:
		print(f"  {record['name']:40s}  state={record['state']}")
	print(f"Wrote full run export: {full_file.resolve()}")
	print(f"Wrote run table export: {rows_file.resolve()}")


if __name__ == "__main__":
	main()