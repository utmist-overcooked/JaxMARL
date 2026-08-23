"""Protein hyperparameter sweep driver for the Overcooked V3 macro-action baselines.

All four macro trainers -- ``mappo_macro_boundary``, ``mappo_macro_every_step``,
``mappo_macro_replan`` and ``mappo_macro_every_step_comm`` -- expose the *same*
entry point::

    run_experiment(config, make_train, experiment_name)

so a single sweep driver can target any of them. This script wraps PufferLib's
Protein optimizer (vendored in ``protein.py``) around that entry point:

    1. ``protein.suggest()`` proposes a hyperparameter dict.
    2. We merge it into the trainer's base config and call ``run_experiment``.
    3. We read the best evaluation return the run achieved, measure its
       wall-clock cost, and feed both back with ``protein.observe()``.

Protein keeps two Gaussian Processes (score and log-cost) and searches the
Pareto frontier of past trials, trading predicted score against predicted cost
(bounded by ``max_suggestion_cost`` wall-clock seconds).

Example
-------
    cd baselines/MAPPO
    python protein_sweep.py --target every_step --max-runs 30 \
        --override TOTAL_TIMESTEPS=2000000 WANDB_MODE=offline

Notes
-----
* Structurally invalid suggestions (e.g. ``BATCH_SIZE`` not divisible by
  ``NUM_MINIBATCHES``) raise inside ``initialize_config``; the driver catches
  that, reports the trial to Protein as ``is_failure=True`` and moves on rather
  than aborting the whole sweep.
* Each trial runs a single seed (``NUM_SEEDS=1``) under its own ``SAVE_PATH``
  subdirectory so per-trial ``best_eval.json`` files never collide.
* Protein's cost-aware ``early_stop`` is *not* wired into the scanned JAX
  training loop (that would need host callbacks mid-``lax.scan``); the sweep
  observes only the completed-run score and wall-clock cost.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Ensure the sibling trainer modules (and their ``from mappo_macro_common import``)
# resolve regardless of the caller's working directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from protein import Protein, Random, ParetoGenetic  # noqa: E402

_CONFIG_DIR = _THIS_DIR / "config"
_SWEEP_CONFIG_DIR = _CONFIG_DIR / "sweep"

# target -> (trainer module, base training config name, required ENV_NAME,
#            default sweep-space config name)
TARGETS = {
    "boundary": (
        "mappo_macro_boundary",
        "mappo_macro_boundary",
        "overcooked_v3_macro",
        "protein",
    ),
    "every_step": (
        "mappo_macro_every_step",
        "mappo_macro_every_step",
        "overcooked_v3_macro_interruptible",
        "protein",
    ),
    "replan": (
        "mappo_macro_replan",
        "mappo_macro_replan",
        "overcooked_v3_macro_interruptible",
        "protein",
    ),
    "every_step_comm": (
        "mappo_macro_every_step_comm",
        "mappo_macro_every_step",
        "overcooked_v3_macro_interruptible",
        "protein_comm",
    ),
}

_OPTIMIZERS = {"protein": Protein, "random": Random, "pareto_genetic": ParetoGenetic}


def _load_container(path: Path) -> dict:
    """Load a YAML config file into a plain resolved dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _coerce_scalar(text: str):
    """Parse a CLI ``KEY=VALUE`` value into bool/int/float/None/str."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _apply_overrides(config: dict, overrides) -> dict:
    """Apply ``KEY=VALUE`` CLI overrides onto the base config (in place)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        config[key.strip()] = _coerce_scalar(raw.strip())
    return config


def _typed_merge(config: dict, hypers: dict) -> dict:
    """Write suggested hyperparameters into the config, preserving int/float type.

    Protein returns rounded ints for integer/pow2 spaces already, but as numpy
    scalars; we coerce to native Python types and keep any key that the base
    config declares as ``int`` an ``int`` (e.g. ``UPDATE_EPOCHS``, ``HIDDEN_SIZE``).
    """
    for key, value in hypers.items():
        original = config.get(key)
        if isinstance(original, bool):
            config[key] = bool(round(float(value)))
        elif isinstance(original, int):
            config[key] = int(round(float(value)))
        else:
            config[key] = float(value)
    return config


def _read_best_eval(output_dir: Path, result) -> float:
    """Recover a trial's score: prefer the saved best eval, else the metric array."""
    best_path = output_dir / "best_eval.json"
    if best_path.is_file():
        try:
            return float(json.loads(best_path.read_text())["eval_return"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    # Fallback: max over the per-update eval_return array the trainer returns.
    try:
        metrics = result["metrics"] if isinstance(result, dict) else result[0]["metrics"]
        eval_returns = np.asarray(metrics["eval_return"], dtype=float)
        finite = eval_returns[np.isfinite(eval_returns)]
        if finite.size:
            return float(finite.max())
    except (KeyError, TypeError, IndexError):
        pass
    return float("nan")


def run_sweep(args) -> None:
    module_name, base_config_name, required_env, default_sweep = TARGETS[args.target]

    base_config = _load_container(_CONFIG_DIR / f"{args.base_config or base_config_name}.yaml")
    _apply_overrides(base_config, args.override)
    if base_config.get("ENV_NAME") != required_env:
        raise ValueError(
            f"Target {args.target!r} requires ENV_NAME={required_env!r}, "
            f"but base config has {base_config.get('ENV_NAME')!r}"
        )

    sweep_config = _load_container(
        Path(args.sweep_config) if args.sweep_config
        else _SWEEP_CONFIG_DIR / f"{default_sweep}.yaml"
    )
    method = sweep_config.get("method", "protein")
    optimizer_cls = _OPTIMIZERS[method]

    # Import the trainer lazily so `--help` works without jax/wandb installed.
    trainer = __import__(module_name)
    make_train = trainer.make_train
    run_experiment = trainer.run_experiment

    save_root = Path(args.save_path) / args.target
    save_root.mkdir(parents=True, exist_ok=True)
    results_path = save_root / "sweep_results.jsonl"

    optimizer_kwargs = {}
    if method == "protein":
        optimizer_kwargs = dict(
            max_suggestion_cost=args.max_suggestion_cost,
            use_gpu=not args.no_gpu,
            cost_param=None,  # wall-clock cost; not part of the search space
        )
    optimizer = optimizer_cls(sweep_config, **optimizer_kwargs)

    max_runs = args.max_runs or sweep_config.get("max_runs", 20)
    print(
        f"[protein_sweep] target={args.target} method={method} "
        f"max_runs={max_runs} cost_budget={args.max_suggestion_cost}s "
        f"search_dims={optimizer.hyperparameters.num}"
    )

    best = {"score": -np.inf, "trial": None, "hypers": None}
    for trial_idx in range(max_runs):
        hypers, info = optimizer.suggest(fill=None)

        experiment_name = f"protein_{args.target}_trial{trial_idx:03d}"
        trial_config = copy.deepcopy(base_config)
        _typed_merge(trial_config, hypers)
        trial_config["NUM_SEEDS"] = 1
        trial_config["SAVE_PATH"] = str(save_root / "runs")

        output_dir = save_root / "runs" / experiment_name / "seed_0"
        print(
            f"\n[trial {trial_idx + 1}/{max_runs}] {experiment_name}\n"
            f"  suggested: {json.dumps(_json_safe(hypers))}"
        )

        start = time.time()
        try:
            result = run_experiment(trial_config, make_train, experiment_name)
            cost = time.time() - start
            score = _read_best_eval(output_dir, result)
            is_failure = not np.isfinite(score)
        except Exception as exc:  # structural / numerical failure -> report, continue
            cost = time.time() - start
            score = float("nan")
            is_failure = True
            print(f"  trial FAILED after {cost:.1f}s: {type(exc).__name__}: {exc}")

        optimizer.observe(hypers, score, cost, is_failure=is_failure)

        record = {
            "trial": trial_idx,
            "experiment": experiment_name,
            "hypers": _json_safe(hypers),
            "score": None if is_failure else score,
            "cost_seconds": cost,
            "is_failure": bool(is_failure),
            "predicted": _json_safe(info) if info else None,
        }
        with results_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

        if not is_failure and score > best["score"]:
            best = {"score": score, "trial": trial_idx, "hypers": _json_safe(hypers)}
        status = "FAILED" if is_failure else f"score={score:.4f}"
        print(f"  {status}  cost={cost:.1f}s  best_so_far={best['score']:.4f}")

    print("\n[protein_sweep] done.")
    if best["trial"] is not None:
        print(f"  best trial: {best['trial']}  score={best['score']:.4f}")
        print(f"  best hypers: {json.dumps(best['hypers'], indent=2)}")
        (save_root / "best.json").write_text(json.dumps(best, indent=2))
    print(f"  full log: {results_path}")


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protein hyperparameter sweep over the Overcooked V3 macro MAPPO baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", required=True, choices=sorted(TARGETS),
        help="Which macro trainer to sweep.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="Number of trials to run (defaults to sweep config's max_runs, else 20).",
    )
    parser.add_argument(
        "--sweep-config", default=None,
        help="Path to a Protein sweep-space YAML (defaults per target under config/sweep/).",
    )
    parser.add_argument(
        "--base-config", default=None,
        help="Base training config name under config/ (defaults to the target's own).",
    )
    parser.add_argument(
        "--max-suggestion-cost", type=float, default=3600.0,
        help="Wall-clock seconds ceiling Protein uses to reject over-costly candidates.",
    )
    parser.add_argument(
        "--save-path", default="models/protein_sweep",
        help="Root directory for per-trial checkpoints and the sweep log.",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force Protein's GP models onto CPU even if a GPU is available.",
    )
    parser.add_argument(
        "--override", nargs="*", default=None, metavar="KEY=VALUE",
        help="Override base training config entries (e.g. TOTAL_TIMESTEPS=2000000 WANDB_MODE=offline).",
    )
    return parser


def main() -> None:
    run_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()
