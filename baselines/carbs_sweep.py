"""Modular CARBS (Cost Aware pareto-Region Bayesian Search) sweep utility.

Parses hyperparameter search spaces from a CARBS_SWEEP section in a Hydra
YAML config and provides a simple suggest/observe API for any training script.

How to integrate into a new training script
============================================

1. Add sweep config to your YAML (e.g. config/my_training.yaml):

    "TUNE": False
    "CARBS_NUM_TRIALS": 50

    "CARBS_SWEEP":
      "LR":
        space: "log"          # "log" or "linear"
        scale: 0.5            # controls search width in transformed space
        min: 1.0e-5
        max: 1.0e-2
      "NUM_ENVS":
        space: "log"
        scale: 0.5
        min: 32
        max: 2048
        is_integer: true      # round to nearest integer (default: false)
        round_pow2: true       # round to nearest power of 2 (default: false)
      "GAE_LAMBDA":
        space: "linear"
        scale: 0.05
        min: 0.8
        max: 1.0

   Search centers are read automatically from the main config values
   (e.g. LR: 0.00025 in the YAML), so don't duplicate them in CARBS_SWEEP.

   Use "log" for params spanning orders of magnitude (LR, ENT_COEF, batch sizes).
   Use "linear" for params in a narrow range (GAE_LAMBDA, CLIP_EPS).
   Set round_pow2: true for batch-related integers that must be powers of 2.

2. Add a tune() function to your training script:

    def tune(config):
        from carbs_sweep import CARBSSweep

        sweep = CARBSSweep(config)
        for trial in range(sweep.num_trials):
            suggestion = sweep.suggest()
            trial_config = sweep.apply_suggestion(suggestion)
            trial_config["WANDB_MODE"] = "disabled"

            start_time = time.time()
            try:
                # ... run training with trial_config, get final_return ...
                sweep.observe(suggestion, output=final_return, cost=time.time() - start_time)
            except Exception as e:
                sweep.observe_failure(suggestion, cost=time.time() - start_time)

        sweep.print_summary()

3. Gate your main() on the TUNE flag:

    @hydra.main(...)
    def main(config):
        config = OmegaConf.to_container(config, resolve=True)
        if config.get("TUNE", False):
            tune(config)
        else:
            single_run(config)

4. Run:

    # Normal single run
    python my_training.py

    # CARBS sweep
    python my_training.py TUNE=True CARBS_NUM_TRIALS=30

Results are saved to $SCRATCH/jaxmarl/carbs_sweep/<layout_name>/<timestamp>/.
Wandb logs are saved offline; sync after the job with the command printed at the end.
"""

import copy
import datetime
import math
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import wandb
from carbs import CARBS, CARBSParams, Param, LogSpace, LinearSpace, ObservationInParam


class CARBSSweep:
    def __init__(self, config: dict, save_dir: Optional[str] = None):
        self.config = config
        self.num_trials = int(config.get("CARBS_NUM_TRIALS", 50))

        sweep_def = config.get("CARBS_SWEEP")
        if sweep_def is None:
            raise ValueError("Config must contain a CARBS_SWEEP section")
        self.sweep_def = sweep_def

        # Build save directory under $SCRATCH: carbs_sweep/<layout>/<timestamp>/
        if save_dir is None:
            scratch = os.environ.get("SCRATCH", ".")
            env_kwargs = config.get("ENV_KWARGS", {})
            layout = env_kwargs.get("layout", "default") if env_kwargs else "default"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(scratch, "jaxmarl", "carbs_sweep", layout, timestamp)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Parse param definitions from YAML
        self._pow2_params = set()
        param_spaces = self._build_params()

        # Build CARBS optimizer
        carbs_cfg = config.get("CARBS_PARAMS", {}) or {}
        carbs_params = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=False,
            resample_frequency=0,
            num_random_samples=carbs_cfg.get("num_random_samples", 4),
            initial_search_radius=carbs_cfg.get("initial_search_radius", 0.3),
            checkpoint_dir=checkpoint_dir,
            is_saved_on_every_observation=True,
        )
        self.carbs = CARBS(carbs_params, param_spaces)

        # Tracking
        self.all_results: List[Dict[str, Any]] = []
        self.best_return = float("-inf")
        self.best_config: Optional[dict] = None

        # Wandb offline logging for the sweep itself
        wandb_dir = os.path.join(self.save_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        env_kwargs = config.get("ENV_KWARGS", {})
        layout = env_kwargs.get("layout", "default") if env_kwargs else "default"
        self.wandb_run = wandb.init(
            dir=wandb_dir,
            project=config.get("PROJECT", "") or "carbs-sweep",
            tags=["CARBS", "sweep", layout],
            config=config,
            mode="offline",
            name=f"carbs_sweep_{layout}",
        )

    def _build_params(self) -> List[Param]:
        params = []
        for name, spec in self.sweep_def.items():
            if name not in self.config:
                raise ValueError(
                    f"CARBS_SWEEP param '{name}' not found as a top-level config key. "
                    f"Search center must come from an existing config value."
                )

            search_center = self.config[name]
            space_type = spec.get("space", "log")
            scale = spec.get("scale", 0.5)
            is_integer = spec.get("is_integer", False)

            if spec.get("round_pow2", False):
                self._pow2_params.add(name)

            if space_type == "log":
                space = LogSpace(
                    scale=scale,
                    min=spec.get("min", 0.0),
                    max=spec.get("max", float("inf")),
                    is_integer=is_integer,
                )
            elif space_type == "linear":
                space = LinearSpace(
                    scale=scale,
                    min=spec.get("min", float("-inf")),
                    max=spec.get("max", float("inf")),
                    is_integer=is_integer,
                )
            else:
                raise ValueError(f"Unknown space type '{space_type}' for param '{name}'")

            params.append(Param(name=name, space=space, search_center=search_center))
        return params

    def suggest(self) -> dict:
        raw = self.carbs.suggest().suggestion
        return self._sanitize(raw)

    def apply_suggestion(self, suggestion: dict) -> dict:
        trial_config = copy.deepcopy(self.config)
        for k, v in suggestion.items():
            if k in trial_config:
                trial_config[k] = v
        return trial_config

    def observe(self, suggestion: dict, output: float, cost: float) -> None:
        self.carbs.observe(
            ObservationInParam(input=suggestion, output=output, cost=cost)
        )
        result = {"suggestion": dict(suggestion), "return": output, "cost": cost}
        self.all_results.append(result)

        if output > self.best_return:
            self.best_return = output
            self.best_config = {
                k: v for k, v in suggestion.items() if k != "__carbs_suggestion_id"
            }
        self._save_results()

        # Log to wandb
        log_dict = {"trial": len(self.all_results), "return": output, "cost": cost, "best_return": self.best_return}
        for k, v in suggestion.items():
            if not k.startswith("__"):
                log_dict[f"suggestion/{k}"] = v
        wandb.log(log_dict)

    def observe_failure(self, suggestion: dict, cost: float) -> None:
        self.carbs.observe(
            ObservationInParam(
                input=suggestion, output=0.0, cost=cost, is_failure=True
            )
        )
        self.all_results.append(
            {"suggestion": dict(suggestion), "return": None, "cost": cost}
        )
        self._save_results()

        # Log failure to wandb
        wandb.log({"trial": len(self.all_results), "return": 0.0, "cost": cost, "is_failure": 1, "best_return": self.best_return})

    def _sanitize(self, suggestion: dict) -> dict:
        s = dict(suggestion)
        for key in self._pow2_params:
            if key in s:
                s[key] = int(2 ** round(math.log2(s[key])))
        return s

    def _save_results(self) -> None:
        path = os.path.join(self.save_dir, "carbs_results.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "results": self.all_results,
                    "best_config": self.best_config,
                    "best_return": self.best_return,
                },
                f,
            )

    def print_summary(self) -> None:
        n_success = sum(1 for r in self.all_results if r["return"] is not None)
        n_fail = len(self.all_results) - n_success
        print(f"\n{'='*60}")
        print(f"CARBS sweep complete: {n_success} success, {n_fail} failures")
        print(f"Best return: {self.best_return:.4f}")
        if self.best_config:
            print(f"Best config:")
            for k, v in self.best_config.items():
                print(f"  {k}: {v}")
        print(f"Results saved to {self.save_dir}/carbs_results.pkl")

        # Finish wandb run and print sync command
        wandb_dir = os.path.join(self.save_dir, "wandb")
        wandb.finish()
        # Find the offline run directory
        run_dir = None
        if os.path.isdir(wandb_dir):
            for d in sorted(os.listdir(wandb_dir), reverse=True):
                if d.startswith("offline-run-"):
                    run_dir = os.path.join(wandb_dir, d)
                    break
        if run_dir:
            print(f"\nTo sync wandb logs, run:\n  wandb sync {run_dir}")

    @staticmethod
    def format_suggestion(suggestion: dict) -> str:
        parts = []
        for k, v in suggestion.items():
            if k.startswith("__"):
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.6g}")
            else:
                parts.append(f"{k}={v}")
        return "  ".join(parts)
