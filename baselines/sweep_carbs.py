"""
CARBS hyperparameter sweep for JaxMARL baselines.

Drives any baseline algorithm (IPPO, MAPPO, QLearning) through CARBS
(Cost Aware Pareto-Region Bayesian Search) for hyperparameter optimization.

Prerequisites:
    pip install -e ./carbs

Usage:
    # IPPO on MPE
    python baselines/sweep_carbs.py \
        --script baselines/IPPO/ippo_ff_mpe.py \
        --config baselines/IPPO/config/ippo_ff_mpe.yaml \
        --sweep-config baselines/sweep_configs/ippo_ff_mpe.yaml \
        --trials 20

    # QLearning (IQL) on MPE - needs --alg-config for the algorithm sub-config
    python baselines/sweep_carbs.py \
        --script baselines/QLearning/iql_rnn.py \
        --config baselines/QLearning/config/config.yaml \
        --alg-config baselines/QLearning/config/alg/ql_rnn_mpe.yaml \
        --sweep-config baselines/sweep_configs/iql_rnn_mpe.yaml \
        --trials 20

    # With wandb logging
    python baselines/sweep_carbs.py \
        --script baselines/IPPO/ippo_ff_mpe.py \
        --config baselines/IPPO/config/ippo_ff_mpe.yaml \
        --sweep-config baselines/sweep_configs/ippo_ff_mpe.yaml \
        --trials 20 \
        --wandb-project my-sweep-project
"""

import argparse
import copy
import importlib.util
import inspect
import os
import sys
import time
import traceback

# Ensure the carbs sub-package is importable when running from the repo root.
# The cloned carbs/ directory at the repo root can shadow the installed package,
# so we insert carbs/ itself onto sys.path so that `carbs.carbs` resolves correctly.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_carbs_dir = os.path.join(_repo_root, "carbs")
if os.path.isdir(_carbs_dir) and _carbs_dir not in sys.path:
    sys.path.insert(0, _carbs_dir)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from carbs import CARBS, Param, LogSpace, LinearSpace, LogitSpace, ObservationInParam
from carbs.utils import CARBSParams, WandbLoggingParams


# ---------------------------------------------------------------------------
# Sweep config loading
# ---------------------------------------------------------------------------

def load_sweep_config(sweep_config_path):
    """Load a sweep YAML and convert to CARBS Param list.

    Returns:
        params: List[Param] for CARBS
        sweep_cfg: dict with top-level sweep settings
    """
    sweep_cfg = OmegaConf.to_container(OmegaConf.load(sweep_config_path))
    params = []
    for name, spec in sweep_cfg["params"].items():
        space_type = spec.pop("space")
        center = spec.pop("center")

        if space_type == "log":
            space = LogSpace(**spec)
        elif space_type == "linear":
            space = LinearSpace(**spec)
        elif space_type == "logit":
            space = LogitSpace(**spec)
        else:
            raise ValueError(f"Unknown space type '{space_type}' for param '{name}'")

        params.append(Param(name=name, space=space, search_center=center))
    return params, sweep_cfg


# ---------------------------------------------------------------------------
# Dynamic module import
# ---------------------------------------------------------------------------

def load_baseline_module(script_path):
    """Dynamically import a baseline training script and return the module."""
    script_path = os.path.abspath(script_path)
    spec = importlib.util.spec_from_file_location("baseline_module", script_path)
    module = importlib.util.module_from_spec(spec)

    # Ensure jaxmarl and baselines are importable
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec.loader.exec_module(module)
    return module


def detect_make_train_signature(module):
    """Detect whether make_train takes (config) or (config, env).

    Returns:
        "standard" if make_train(config)
        "with_env" if make_train(config, env)
        "with_rng" if make_train(config, rng_init) (mabrax pattern)
    """
    sig = inspect.signature(module.make_train)
    param_names = list(sig.parameters.keys())
    if len(param_names) == 1:
        return "standard"
    elif len(param_names) == 2:
        second = param_names[1]
        if second == "env":
            return "with_env"
        elif second == "rng_init":
            return "with_rng"
        else:
            return "with_env"  # default assumption for 2-arg
    return "standard"


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def load_base_config(config_path, alg_config_path=None):
    """Load Hydra-style config from YAML files.

    For QLearning scripts, --alg-config provides the algorithm sub-config
    which gets merged as config["alg"].
    """
    config = OmegaConf.to_container(OmegaConf.load(config_path))
    if alg_config_path:
        alg_config = OmegaConf.to_container(OmegaConf.load(alg_config_path))
        config["alg"] = alg_config
    return config


def apply_suggestion(base_config, suggestion, has_alg_key=False):
    """Override config values with CARBS suggestion.

    If the config uses the QLearning pattern with an "alg" sub-dict,
    suggestion keys are applied to the merged config.
    """
    config = copy.deepcopy(base_config)

    # For QLearning, merge alg into top-level config
    if has_alg_key and "alg" in config:
        config = {**config, **config["alg"]}

    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        if key in config:
            if isinstance(config[key], int):
                config[key] = int(round(value))
            elif isinstance(config[key], float):
                config[key] = float(value)
            else:
                config[key] = value

    # Recompute derived values that depend on swept params
    if "NUM_STEPS" in config and "NUM_ENVS" in config and "TOTAL_TIMESTEPS" in config:
        config["NUM_UPDATES"] = int(
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )

    return config


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

class MetricCollector:
    """Collects metrics from wandb.log or io_callback during training.

    Some baseline scripts return metrics in the output dict, others only
    log them via wandb callbacks. This collector patches wandb.log to
    capture returned_episode_returns regardless.
    """

    def __init__(self):
        self.episode_returns = []
        self._original_wandb_log = None

    def patch_wandb(self):
        """Monkey-patch wandb.log to capture metrics."""
        import wandb

        self._original_wandb_log = wandb.log

        def patched_log(data, *args, **kwargs):
            if isinstance(data, dict) and "returned_episode_returns" in data:
                val = data["returned_episode_returns"]
                if hasattr(val, "item"):
                    val = val.item()
                self.episode_returns.append(float(val))
            # Only forward to original wandb.log if a run is active
            if wandb.run is not None and self._original_wandb_log is not None:
                self._original_wandb_log(data, *args, **kwargs)

        wandb.log = patched_log

    def unpatch_wandb(self):
        """Restore original wandb.log."""
        if self._original_wandb_log is not None:
            import wandb
            wandb.log = self._original_wandb_log
            self._original_wandb_log = None

    def get_objective(self):
        """Return mean of last 100 collected episode returns."""
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns[-100:]))


def extract_objective_from_output(out):
    """Try to extract objective from make_train output dict.

    Returns float objective or None if metrics not available in output.
    """
    if "metrics" not in out:
        return None
    metrics = out["metrics"]
    if "returned_episode_returns" not in metrics:
        return None

    returns = metrics["returned_episode_returns"]
    # returns may be (NUM_UPDATES,) or (NUM_SEEDS, NUM_UPDATES)
    returns = np.array(returns)
    if returns.ndim == 2:
        # Average over seeds, take last 100 updates
        returns = returns.mean(axis=0)
    if len(returns) == 0:
        return 0.0
    last_n = min(100, len(returns))
    return float(returns[-last_n:].mean())


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(module, config, rng, make_train_type):
    """Run a single training trial.

    Returns:
        dict with "objective", "cost", "success"
    """
    collector = MetricCollector()
    collector.patch_wandb()

    start_time = time.time()
    try:
        if make_train_type == "with_env":
            env, _ = module.env_from_config(copy.deepcopy(config))
            train_fn = jax.jit(module.make_train(config, env))
        elif make_train_type == "with_rng":
            rng, rng_init = jax.random.split(rng)
            train_fn = jax.jit(module.make_train(config, rng_init))
        else:
            train_fn = jax.jit(module.make_train(config))

        out = train_fn(rng)

        # Block until computation is done
        jax.block_until_ready(out)
        cost = time.time() - start_time

        # Try to get objective from output dict first
        objective = extract_objective_from_output(out)
        if objective is None:
            # Fall back to collected wandb metrics
            objective = collector.get_objective()

        collector.unpatch_wandb()
        return {"objective": objective, "cost": cost, "success": True}

    except Exception as e:
        cost = time.time() - start_time
        collector.unpatch_wandb()
        print(f"  Trial failed: {e}")
        traceback.print_exc()
        return {"objective": 0.0, "cost": cost, "success": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CARBS hyperparameter sweep for JaxMARL baselines"
    )
    parser.add_argument(
        "--script", type=str, required=True,
        help="Path to baseline .py file (e.g. baselines/IPPO/ippo_ff_mpe.py)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to base Hydra YAML config"
    )
    parser.add_argument(
        "--alg-config", type=str, default=None,
        help="Path to algorithm sub-config YAML (QLearning scripts)"
    )
    parser.add_argument(
        "--sweep-config", type=str, required=True,
        help="Path to CARBS sweep config YAML"
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of CARBS trials (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="W&B project name for CARBS logging (disabled if not set)"
    )
    parser.add_argument(
        "--wandb-group", type=str, default=None,
        help="W&B group name for CARBS logging"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/sweep",
        help="Directory to save CARBS checkpoints (default: checkpoints/sweep)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Disable wandb for training runs (CARBS handles its own wandb logging)
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")

    # Load baseline module
    print(f"Loading baseline script: {args.script}")
    module = load_baseline_module(args.script)
    make_train_type = detect_make_train_signature(module)
    print(f"  make_train signature: {make_train_type}")

    # Load configs
    base_config = load_base_config(args.config, args.alg_config)
    has_alg_key = args.alg_config is not None

    # Force single seed for sweep trials (faster)
    base_config["NUM_SEEDS"] = 1

    # Load CARBS sweep config
    params, sweep_cfg = load_sweep_config(args.sweep_config)
    print(f"  Sweep params: {[p.name for p in params]}")

    # Configure CARBS
    enable_wandb = args.wandb_project is not None
    wandb_params = WandbLoggingParams(
        project_name=args.wandb_project or "",
        group_name=args.wandb_group or os.path.basename(args.script).replace(".py", ""),
        run_name=f"carbs_sweep",
    )

    carbs_config = CARBSParams(
        better_direction_sign=sweep_cfg.get("better_direction_sign", 1),
        is_wandb_logging_enabled=enable_wandb,
        wandb_params=wandb_params,
        num_random_samples=sweep_cfg.get("num_random_samples", min(4, args.trials)),
        is_saved_on_every_observation=True,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    carbs = CARBS(carbs_config, params)

    # Sweep loop
    print(f"\n{'='*60}")
    print(f"CARBS Sweep")
    print(f"  Script:     {args.script}")
    print(f"  Config:     {args.config}")
    if args.alg_config:
        print(f"  Alg config: {args.alg_config}")
    print(f"  Sweep cfg:  {args.sweep_config}")
    print(f"  Trials:     {args.trials}")
    print(f"  Seed:       {args.seed}")
    print(f"  W&B:        {'enabled (' + args.wandb_project + ')' if enable_wandb else 'disabled'}")
    print(f"{'='*60}\n")

    best_objective = float("-inf")
    best_suggestion = None

    for trial_idx in range(args.trials):
        print(f"\n{'#'*60}")
        print(f"# Trial {trial_idx + 1}/{args.trials}")
        print(f"{'#'*60}")

        # Get suggestion from CARBS
        suggest_output = carbs.suggest()
        suggestion = suggest_output.suggestion

        # Print suggested hyperparameters
        print("  Suggested hyperparameters:")
        for key, value in suggestion.items():
            if key != "suggestion_uuid":
                print(f"    {key}: {value}")

        # Build trial config
        trial_config = apply_suggestion(base_config, suggestion, has_alg_key)

        # Override wandb to disabled for training
        trial_config["WANDB_MODE"] = "disabled"

        # Use a different seed per trial for variety
        rng = jax.random.PRNGKey(args.seed + trial_idx)

        # Run training
        result = run_trial(module, trial_config, rng, make_train_type)

        # Report to CARBS
        observation = ObservationInParam(
            input=suggestion,
            output=result["objective"],
            cost=result["cost"],
            is_failure=not result["success"],
        )
        carbs.observe(observation)

        # Track best
        if result["success"] and result["objective"] > best_objective:
            best_objective = result["objective"]
            best_suggestion = {
                k: v for k, v in suggestion.items() if k != "suggestion_uuid"
            }

        status = "OK" if result["success"] else "FAILED"
        print(f"\n  Result: [{status}] objective={result['objective']:.4f}, "
              f"cost={result['cost']:.1f}s")
        if best_suggestion:
            print(f"  Best so far: {best_objective:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Sweep complete! {args.trials} trials finished.")
    if best_suggestion:
        print(f"Best objective: {best_objective:.4f}")
        print(f"Best hyperparameters:")
        for k, v in best_suggestion.items():
            print(f"  {k}: {v}")
    else:
        print("No successful trials.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
