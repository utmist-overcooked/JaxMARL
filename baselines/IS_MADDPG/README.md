# IS-MADDPG — Intention Sharing Multi-Agent Deep Deterministic Policy Gradient

A JAX implementation of the Intention Sharing MADDPG algorithm for cooperative multi-agent reinforcement learning, trained on the OvercookedV3 environment.

---

## File Overview

### `networks.py`
Defines the neural network architectures. `ISAgentNet` is the decentralised actor for each agent — it takes the agent's observation and received messages as input, produces a discrete environment action, and generates a continuous communication message by attending over an imagined H-step future trajectory. `ISCriticNet` is the centralised critic shared across all agents, conditioned on a one-hot agent identity vector alongside the global state (all agents' observations, actions, and messages).

### `buffer.py`
Implements the joint replay buffer used to store and sample experience across all agents. Uses three functions: `buffer_init` to allocate the buffer, `buffer_add` to write a transition into it, and `buffer_sample` (and `buffer_sample_prioritised`) to sample minibatches for learning. The buffer uses numpy arrays for efficient in-place writes during environment interaction, and converts to JAX arrays at sample time for use in JIT-compiled training steps.

### `loss.py`
Contains the actor and critic loss functions used to train the networks, as well as the `received_messages` helper that builds per-agent communication tensors from the joint message array. The critic loss computes a MSE Bellman error against a bootstrapped target. The actor loss combines a Q-maximisation objective with two auxiliary world model losses - cross-entropy for predicting other agents' actions and MSE for predicting the next observation delta - which train the internal predictors that power the imagined rollout.

### `update.py`
Handles weight updates for both networks. Defines `TrainState` (a NamedTuple holding actor and critic parameters, target network parameters, and optimizer states), `init_train_state` to initialise everything from scratch, `train_step` to apply one full IS-MADDPG update (critic update, actor update, Polyak target update) using `jax.value_and_grad`, and `polyak_update` for exponential moving average target network tracking.

### `policy.py`
Provides the inference-time joint policy used for evaluation and deployment. `PolicyState` holds the trained actor parameters and the per-agent previous messages needed for communication. `get_joint_action` runs the actor for all agents given their current observations, returning a dictionary of discrete actions. `policy_reset` zeros the message state at the start of a new episode, and `policy_load` loads trained parameters from a checkpoint file.

### `train.py`
Contains the generic training loop and shared utilities. `make_train` builds a training function that manages the environment interaction loop, buffer population, and calls to `train_step`. Also provides `save_checkpoint` and `load_checkpoint` for persisting trained parameters, and `DEFAULT_CONFIG` as the base hyperparameter dictionary shared across all environments.

### `run_overcooked_v3.py`
The main entry point for training on OvercookedV3. Handles environment instantiation, automatic observation and action dimension detection via `probe_env`, hyperparameter configuration specific to the Overcooked environment, shaped reward integration from the environment's info dictionary, manual episode reset logic (JaxMARL does not auto-reset on termination), greedy evaluation runs, W&B logging, and checkpoint saving. Run this file directly to start training.

---

## Usage

```bash
# Default: cramped_room layout, 2M steps
python baselines/IS_MADDPG/run_overcooked_v3.py

# Custom layout and timesteps
python baselines/IS_MADDPG/run_overcooked_v3.py --layout asymmetric_advantages --total_timesteps 1000000

# With W&B logging
python baselines/IS_MADDPG/run_overcooked_v3.py --wandb --wandb_entity "your-entity"
```

## Dependencies

- JAX / JAXLIB
- Flax
- Optax
- Chex
- NumPy
- Matplotlib
- WandB (optional, for logging)