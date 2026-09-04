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
Weights & Biases · pytest · pygame · Pillow/Matplotlib

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

There are three macro-action variants, and three communication trainers layered
on top of them. Every argument below already has a value in the trainer's Hydra
config, so a bare `uv run python <trainer>.py` runs. The commands here spell out
the arguments that **define the experiment** — the ones to review and edit
deliberately — and each is followed by the constraints that will abort the run
at startup if violated. Everything not listed is in "Optional arguments" at the
end of this section.

`ENV_NAME` is fixed per trainer and validated on entry: passing the wrong one
raises rather than silently training the wrong environment.

#### 1. Boundary (no comm) — decisions only at macro boundaries, SMDP returns

```bash
uv run python baselines/MAPPO/mappo_macro_boundary.py \
  ENV_NAME=overcooked_v3_macro \
  ENV_KWARGS.layout=cramped_room \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.max_macro_steps=150 \
  ENV_KWARGS.agent_view_size=2 \
  NUM_ENVS=16 NUM_STEPS=400 NUM_MINIBATCHES=4 \
  TOTAL_TIMESTEPS=20000000 \
  USE_RNN=true \
  SAVE_PATH=models/mappo_macro
```

- `NUM_STEPS` must be a multiple of `ENV_KWARGS.max_steps` so every pending
  macro is flushed before the update.
- `USE_RNN=true` requires `NUM_ACTORS` (= `NUM_ENVS` × 2) divisible by
  `NUM_MINIBATCHES`.

#### 2. Every-step (no comm) — a new macro may be chosen every primitive step

```bash
uv run python baselines/MAPPO/mappo_macro_every_step.py \
  ENV_NAME=overcooked_v3_macro_interruptible \
  ENV_KWARGS.layout=race_against_the_clock \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.max_macro_steps=150 \
  ENV_KWARGS.agent_view_size=2 \
  NUM_ENVS=16 NUM_STEPS=125 NUM_MINIBATCHES=4 \
  TOTAL_TIMESTEPS=5000000 \
  USE_RNN=true \
  SAVE_PATH=models/mappo_macro
```

- Uses the interruptible environment; `NUM_STEPS` is unconstrained here.

#### 3. Replan — separate macro-selection and CONTINUE/REPLAN heads

```bash
uv run python baselines/MAPPO/mappo_macro_replan.py \
  ENV_NAME=overcooked_v3_macro_interruptible \
  ENV_KWARGS.layout=cramped_room \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.max_macro_steps=150 \
  ENV_KWARGS.agent_view_size=1 \
  NUM_ENVS=16 NUM_STEPS=125 NUM_MINIBATCHES=4 \
  TOTAL_TIMESTEPS=20000000 \
  SAVE_PATH=models/mappo_macro
```

- **No `USE_RNN` support.** This trainer is MLP-only; passing `USE_RNN` has no
  effect. Under partial observability (`agent_view_size` set) a memoryless
  policy cannot distinguish states that differ only in unobserved history.

#### 4. Boundary + comm (two-stage) — comm module on a FROZEN boundary actor

Train `mappo_macro_boundary.py` first, then point this at its checkpoint.

```bash
uv run python baselines/MAPPO/mappo_macro_boundary_comm.py \
  ENV_NAME=overcooked_v3_macro \
  ENV_KWARGS.layout=pressure_gated_circuit \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.max_macro_steps=150 \
  ENV_KWARGS.agent_view_size=2 \
  NUM_ENVS=16 NUM_STEPS=400 NUM_MINIBATCHES=4 \
  TOTAL_TIMESTEPS=20000000 \
  USE_RNN=true COMM_USE_MEMORY=true \
  VOCAB_SIZE=2 MESSAGE_EMBED_DIM=2 COMM_HIDDEN_SIZE=64 \
  FROZEN_ACTOR_PATH=models/mappo_macro/mappo_macro_boundary/seed_0/final_actor.safetensors \
  FROZEN_CRITIC_PATH=models/mappo_macro/mappo_macro_boundary/seed_0/final_critic.safetensors \
  SAVE_PATH=models/mappo_macro
```

- **Every obs-affecting argument must match the frozen run** — `layout`,
  `agent_view_size`, and anything changing the macro action count. The frozen
  actor's first layer is sized for that run's observation width; a mismatch is
  rejected at startup by `validate_frozen_actor_matches_env`, which checks the
  architecture (recurrent vs MLP) before the width. Safest is to copy
  `ENV_KWARGS` verbatim from `<frozen run>/config.yaml`.
- `USE_RNN` must equal the frozen run's value. `COMM_USE_MEMORY=true` requires
  `USE_RNN=true`.
- This run's own `best_actor.safetensors` holds the **comm module**, not a macro
  actor.

#### 5. Every-step + comm (two-stage) — comm on a FROZEN every-step actor

Shares `config/mappo_macro_every_step.yaml` with trainer 2, so the comm keys are
already present (no Hydra `+` prefix needed).

```bash
uv run python baselines/MAPPO/mappo_macro_every_step_comm.py \
  ENV_NAME=overcooked_v3_macro_interruptible \
  ENV_KWARGS.layout=race_against_the_clock \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.max_macro_steps=150 \
  ENV_KWARGS.agent_view_size=2 \
  NUM_ENVS=16 NUM_STEPS=125 NUM_MINIBATCHES=4 \
  TOTAL_TIMESTEPS=5000000 \
  VOCAB_SIZE=2 MESSAGE_EMBED_DIM=2 COMM_HIDDEN_SIZE=64 \
  FROZEN_ACTOR_PATH=models/mappo_macro/mappo_macro_every_step/seed_0/best_actor.safetensors \
  FROZEN_CRITIC_PATH=models/mappo_macro/mappo_macro_every_step/seed_0/best_critic.safetensors \
  SAVE_PATH=models/mappo_macro
```

- **This trainer is MLP-only.** It builds a plain `Actor`, so the frozen
  checkpoint must come from a `mappo_macro_every_step.py` run trained with
  `USE_RNN=false`. The shared config ships `USE_RNN: true`, so a frozen actor
  produced with the defaults will be rejected as an architecture mismatch.
- Obs-affecting arguments must match the frozen run, as in trainer 4.

#### 6. Boundary + joint comm — actor, critic and comm trained TOGETHER

Nothing is frozen, so there is no `FROZEN_*` path. This is the trainer to use
for information-asymmetry experiments.

```bash
uv run python baselines/MAPPO/mappo_macro_boundary_joint_comm.py \
  ENV_NAME=overcooked_v3_macro \
  ENV_KWARGS.layout=follow_the_leader_nerfed \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.max_macro_steps=150 \
  ENV_KWARGS.agent_view_size=2 \
  ENV_KWARGS.resample_recipe_on_delivery=true \
  NUM_ENVS=16 NUM_STEPS=400 NUM_MINIBATCHES=4 \
  TOTAL_TIMESTEPS=20000000 \
  USE_RNN=true COMM_USE_MEMORY=true \
  VOCAB_SIZE=2 MESSAGE_EMBED_DIM=2 COMM_HIDDEN_SIZE=64 \
  COMM_MODE=normal COMM_INJECTION=concat COMM_CHANNEL=dial \
  SAVE_PATH=models/mappo_macro
```

- `COMM_CHANNEL=dial` requires `COMM_INJECTION=concat` (the gradient reaches the
  speaker through the actor's input, which the `bias` head does not provide) and
  `NUM_ENVS` divisible by `NUM_MINIBATCHES` (minibatches are split by
  environment so an environment's agents stay together for message routing).
- Checkpoints hold **both** halves as one `{"actor", "comm"}` tree, unlike the
  two-stage trainers. `COMM_INJECTION` changes the actor's input width, so
  `concat` and `bias` checkpoints are not interchangeable.

##### Control experiments (trainer 6)

`COMM_MODE` holds architecture, parameter count and training budget fixed and
varies only what the channel carries, so a gain can be attributed to the
information rather than to the extra capacity. Run `normal` against `self`;
`normal` − `self` is the causal estimate.

| `COMM_MODE` | Listener receives | Isolates |
| --- | --- | --- |
| `normal` | the partner's message | treatment |
| `self` | its own message back | same capacity, zero transfer |
| `shuffled` | the partner's message from a random environment | same message statistics, no correlation with this episode's recipe |
| `constant` | a fixed symbol | channel fully severed |

`ORACLE_RECIPE_OBS=true` appends the true recipe to every agent's observation,
removing the information asymmetry entirely. It is the **upper bound** for any
protocol: if a policy handed the recipe still will not condition on it, the
blocker is the task setup rather than the channel. Implemented in
`MacroWorldStateWrapper`, so `overcooked_v3` is untouched, and available to every
macro trainer — but it is only a config key in `mappo_macro_boundary.yaml` and
`mappo_macro_boundary_joint_comm.yaml`; elsewhere pass `+ORACLE_RECIPE_OBS=true`.
Run it on trainer 1 (no comm). Running it on a comm trainer makes the message
redundant and the protocol will correctly collapse, which tests nothing.

#### Optional arguments

None of these need to appear in a run command; they all have config defaults.

**Optimization** — `LR`, `ANNEAL_LR`, `UPDATE_EPOCHS`, `GAMMA`, `GAE_LAMBDA`,
`CLIP_EPS`, `ENT_COEF`, `VF_COEF`, `MAX_GRAD_NORM`, `HIDDEN_SIZE`.

**Reward shaping** — `REW_SHAPING_HORIZON` (shaped rewards decay to zero over
this many primitive steps; the burn penalty ramps in over the same horizon),
`ENV_KWARGS.dense_task_shaping`.

**Run management** — `SEED`, `NUM_SEEDS`, `RESUME_FROM`,
`CHECKPOINT_INTERVAL_UPDATES`, `LOG_INTERVAL_UPDATES`, `EVAL_INTERVAL_UPDATES`,
`NUM_EVAL_ENVS`, `EVAL_SEED`.

**W&B** — `WANDB_MODE` (`online`/`offline`/`disabled`), `ENTITY`, `PROJECT`.

**Comm-only** (trainers 4–6) — `MESSAGE_ENT_COEF`, `MESSAGE_LOSS_COEF`.

**Joint-comm only** (trainer 6) — `COMM_GUMBEL_TAU` (straight-through
temperature; lower sharpens the relaxation), `MESSAGE_HEAD_INIT_SCALE` (init
scale of the message-logit head — at exactly `0.0` that kernel is the Jacobian
into the speaker's encoder, so a zero head blocks gradient to `msg_dense1`,
`mem_dense` and the comm GRU entirely; the default `0.01` keeps initial messages
near-uniform while letting the encoder train from step 0).

#### Which evaluation script goes with which trainer

| Trainer | Rollout / evaluation script |
| --- | --- |
| 1, 2, 3 | `scripts/visualize_macro_mappo_rollout.py --variant boundary\|every_step\|replan` |
| 4, 5 | `scripts/visualize_macro_mappo_rollout_comm.py` (detects boundary vs every_step from `ENV_NAME`) |
| 6 | `scripts/visualize_macro_mappo_rollout_joint_comm.py` (add `--intervene --decode` for the positive-listening and positive-signaling diagnostics) |

Each reads architecture flags from the run's own `config.yaml`, so a script and
a checkpoint cannot silently disagree; mismatches are rejected with an
explanation naming the correct script.

The default configs write local outputs below
`models/mappo_macro/<trainer>/seed_0/`. **`SAVE_PATH` plus the trainer name
determines the output directory, so two runs of the same trainer overwrite each
other** — give each experiment its own `SAVE_PATH`.

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
