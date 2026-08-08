# AGENTS.md

This file provides guidance to AI agents working in this repository.

## Project Overview

JaxMARL is a JAX-native multi-agent reinforcement learning library. Work in this
fork focuses primarily on Overcooked V3 and, in particular, its macro-action
interface and macro-action MAPPO baselines.

The macro environment is a layer on top of Overcooked V3, not a separate game.
`overcooked_v3` owns the grid state, observations, primitive movement,
interactions, rewards, orders, pots, conveyors, buttons, barriers, and other
world updates. `overcooked_v3_macro` adds temporally extended actions, navigation,
valid-action masks, and macro bookkeeping, then emits one primitive Overcooked V3
action per agent on every environment step. Changes to macro behavior therefore
often require understanding both packages.

## Tech Stack

Python 3.10+ · JAX/JAXlib · Flax · Optax · Distrax · Chex · Hydra/OmegaConf ·
Weights &amp; Biases · pytest · pygame · Pillow/Matplotlib

The code is designed around JAX transformations (`jit`, `vmap`, and `lax.scan`).
Environment state uses fixed-shape arrays and JAX-compatible dataclasses.

## Development Setup

Assume `uv` is available. Use it to create the repository-local environment and
install environment, algorithm, and development dependencies:

```bash
uv venv venv
source venv/bin/activate
uv pip install -e '.[algs,dev]'
```

If `uv` is not installed, fall back to the standard-library environment and
pip:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[algs,dev]'
```

In that case, replace `uv run python` with `python` in all commands below.

Install the JAX/JAXlib build appropriate for the machine's CPU or accelerator.
Before running any Python command in this repository, always run:

```bash
source venv/bin/activate
```

Run commands from the repository root. Hydra configuration files live beside
their baseline family and command-line overrides use `KEY=value`, for example
`ENV_KWARGS.layout=cramped_room`.

## Key Commands

### Tests

```bash
# Focused environment, macro layer, visualizer, and trainer utility tests
uv run python -m pytest tests/overcooked_v3 tests/overcooked_v3_macro \
  tests/baselines/test_mappo_macro_utils.py

# Entire test suite
uv run python -m pytest tests
```

### Play and visualize

```bash
# Play primitive-action Overcooked V3; use --list to list registered layouts
uv run python scripts/play_overcooked_v3.py --layout cramped_room

# Render a scripted macro-action episode, including the planner flood field
uv run python scripts/scripted_overcooked_v3_macro_cramped_room.py \
  --layout cramped_room --flood-fill \
  --output artifacts/overcooked_v3_macro_scripted.gif

# Launch the visual layout editor
uv run python -m jaxmarl.tools.layout_editor_v3
```

### Train macro-action MAPPO

```bash
# Select a new macro only at macro boundaries; uses SMDP/event-time returns
uv run python baselines/MAPPO/mappo_macro_boundary.py \
  ENV_KWARGS.layout=cramped_room

# Select a desired macro every primitive step; a new choice interrupts the old one
uv run python baselines/MAPPO/mappo_macro_every_step.py \
  ENV_KWARGS.layout=cramped_room

# Learn separate macro-selection and CONTINUE/REPLAN decisions
uv run python baselines/MAPPO/mappo_macro_replan.py \
  ENV_KWARGS.layout=cramped_room
```

The default configs disable W&B and save local outputs below
`models/mappo_macro/<trainer>/seed_0/`. Override values such as
`TOTAL_TIMESTEPS`, `NUM_ENVS`, `WANDB_MODE`, or `SAVE_PATH` through Hydra.

### Roll out a trained policy to GIF

```bash
uv run python scripts/visualize_macro_mappo_rollout.py \
  --variant boundary \
  --run-dir models/mappo_macro/mappo_macro_boundary/seed_0 \
  --checkpoint-label final \
  --output artifacts/mappo_macro_boundary.gif
```

`--variant` must be `boundary`, `every_step`, or `replan` and must match the
trainer that produced the run. The run directory must contain `config.yaml` and
`final_actor.safetensors`. For primitive-action IPPO checkpoints, use
`scripts/generate_ippo_v3_gif.py` instead.

## Architecture

```text
macro MAPPO trainer
  -> MacroWorldStateWrapper / LogWrapper
  -> jaxmarl.make("overcooked_v3_macro[_interruptible]")
  -> macro selection, masks, flood-fill navigation, macro state
  -> one primitive action per agent
  -> OvercookedV3 primitive transition and world systems
  -> observations, rewards, dones, info
```

### Overcooked V3

- `jaxmarl/environments/overcooked_v3/overcooked.py` — public environment class,
constructor validation, and compatibility wrappers.
- `config.py`, `state.py`, and `common.py` — static configuration, the environment
state pytree, and shared enums/data structures.
- `reset.py` and `initialization.py` — reset pipeline and fixed-shape state setup.
- `step.py` and `agent_step.py` — primitive timestep orchestration, movement,
collision handling, and agent action phase.
- `movement.py` and `interactions.py` — movement helpers and item/pot/delivery
interactions.
- `observations.py` — per-agent default and featurized observations.
- `systems/` — pots, order queues, conveyors, barriers, and moving walls.
- `layouts.py` — built-in layouts, ASCII parsing, and JSON layout loading.
- `jaxmarl/viz/overcooked_v3_visualizer.py` — state rendering and GIF animation.

### Macro-action layer

- `jaxmarl/environments/overcooked_v3_macro/overcooked.py` — the 17-action macro
interface, committed and interruptible variants, valid-action masks,
barrier-aware flood-fill planning, macro completion, and macro state fields.
- `jaxmarl/environments/overcooked_v3_macro/STEP_ENV.md` — detailed walkthrough of
a macro step and the boundary between macro planning and base mechanics.
- `jaxmarl/registration.py` — registers `overcooked_v3`,
`overcooked_v3_macro`, and `overcooked_v3_macro_interruptible`.

Keep base mechanics in `overcooked_v3`. Keep macro selection, target choice,
navigation, availability, and termination in `overcooked_v3_macro`. A macro step
must continue to call the base transition so the two environments do not drift.

### Baselines

- `baselines/MAPPO/mappo_macro_boundary.py` — asynchronous boundary-only macro
decisions with discounted intra-macro rewards and SMDP GAE.
- `baselines/MAPPO/mappo_macro_every_step.py` — interruptible macro choice on each
primitive step.
- `baselines/MAPPO/mappo_macro_replan.py` — hierarchical actor with separate
macro and learned CONTINUE/REPLAN heads.
- `baselines/MAPPO/mappo_macro_common.py` — shared actor/critic definitions,
observation augmentation, action masking, PPO updates, evaluation, logging,
checkpointing, and resume support. Keep variant-specific rollout semantics in
the three trainer files rather than hiding them here.
- `baselines/MAPPO/config/mappo_macro_*.yaml` — Hydra configs for those trainers.
- `baselines/IPPO/`, `baselines/QLearning/`, and `baselines/IC3Net/` — other
baseline families, including primitive-action Overcooked V3 experiments.

## Repository Structure

- `jaxmarl/environments/` — registered multi-agent environments; the current
focus is `overcooked_v3/` and `overcooked_v3_macro/`.
- `baselines/` — mostly single-file training implementations grouped by algorithm
family, with adjacent Hydra configs.
- `scripts/` — play tools, layout utilities, experiment launchers, evaluations,
and checkpoint-to-GIF renderers.
- `tests/overcooked_v3/` and `tests/overcooked_v3_macro/` — base and macro
environment correctness, JIT, system, layout, and visualization coverage.
- `tests/baselines/` — focused checks for baseline math and integration.
- `docs/` — environment notes, experiment runbooks, and investigation reports.
- `layout_pictures/` — rendered examples of Overcooked V3 layouts.
- `pyproject.toml` — package metadata, dependencies, pytest settings, and the
  `overcooked-editor` entry point.

## Code Style

- JAX code can be complicated and difficult to follow, so avoid extracting
  one- to three-line functions. Keep short operations near the surrounding code
  that gives them context.
- Every function must have at least a one-line docstring describing what it
  does. Add more detail whenever inputs, outputs, invariants, or JAX behavior are
  not obvious.
- Inline comments are encouraged. Complex matrix and array operations can be
  hard to follow, so add high-level comments that explain the purpose of the
  operations or blocks of code.

## Implementation Notes

- Preserve JIT compatibility: avoid data-dependent Python control flow, dynamic
shapes, mutation, or host conversions inside traced environment/training code.
- Fixed-size limits in `overcooked_v3/settings.py` are part of the compiled state
shape. Update configuration, initialization, observations, and tests together
when changing them.
- Preserve agent-keyed dictionaries and the `done["__all__"]` convention at the
public environment boundary.
- Add focused tests to the matching Overcooked V3, macro, or baseline test
directory. Include JIT coverage for environment transition changes.
- Macro training budgets are measured in primitive environment steps. Respect
each trainer's divisibility and rollout-boundary checks rather than silently
truncating a run.

## Writing PRs

In the PR description, first describe the previous code path and behavior, then explain how the PR changes it. Follow with bullet points describing the concrete behavioral impact: what agents, gameplay, training, evaluation, or artifacts did before and what they do after the change.

If the PR affects gameplay, make a gif with a scripted policy that you code up to show the functionality difference between the old gameplay and the new gameplay, and then put this at the VERY TOP of the PR description.

Agents are also encouraged to create and add more GIFs whenever they are relevant
and help reviewers understand the impact of the PR.
