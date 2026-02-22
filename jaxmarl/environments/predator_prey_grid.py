"""Grid-based Predator-Prey environment (JAX).

Replicates the discrete grid environment from the IC3Net paper:
  - N predators move on a dim×dim grid trying to reach a fixed prey
  - Observation: one-hot encoded vision window (vocab_size × (2v+1) × (2v+1))
  - Actions: UP(0), RIGHT(1), DOWN(2), LEFT(3), STAY(4)
  - Reward: -0.05 per timestep for all agents; agents on the prey get 0
  - Episode ends when ALL predators have reached the prey, or max_steps

Reference: IC3Net paper, https://arxiv.org/abs/1812.09755
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.multi_agent_env import State as BaseState
from jaxmarl.environments.spaces import Box, Discrete


@struct.dataclass
class PredatorPreyState(BaseState):
    """State for the grid Predator-Prey environment."""
    predator_loc: chex.Array    # (npredator, 2) row, col
    prey_loc: chex.Array        # (1, 2)
    reached_prey: chex.Array    # (npredator,) float: 1.0 if reached


class PredatorPreyGrid(MultiAgentEnv):
    """Grid-based Predator-Prey matching the IC3Net reference implementation.

    Medium config: num_agents=5, dim=10, vision=1, max_steps=40
    Easy config:   num_agents=3, dim=5,  vision=0, max_steps=20
    """

    def __init__(
        self,
        num_agents: int = 3,
        dim: int = 5,
        vision: int = 0,
        max_steps: int = 20,
        mode: str = "mixed",
    ):
        super().__init__(num_agents)

        self.dim = dim
        self.vision = vision
        self.max_steps = max_steps
        self.mode = mode
        self.npredator = num_agents
        self.nprey = 1

        # Grid constants
        self.base = dim * dim
        self.outside_class = self.base + 1
        self.prey_class = self.base + 2
        self.predator_class = self.base + 3
        self.vocab_size = self.base + 4

        # Vision window
        self.vis_size = 2 * vision + 1

        # Flat observation dim: vocab_size * vis_size * vis_size
        self.obs_dim = self.vocab_size * self.vis_size * self.vis_size

        # Agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Reward constants (matching reference)
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0.0       # reward for agent ON prey in mixed mode
        self.POS_PREY_REWARD = 0.05  # reward for agent ON prey in cooperative mode

        # Spaces
        self.observation_spaces = {
            a: Box(0.0, 1.0, (self.vocab_size, self.vis_size, self.vis_size))
            for a in self.agents
        }
        self.action_spaces = {a: Discrete(5) for a in self.agents}

        # Action deltas: UP(0), RIGHT(1), DOWN(2), LEFT(3), STAY(4)
        self._deltas = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], PredatorPreyState]:
        total = self.npredator + self.nprey
        num_cells = self.dim * self.dim
        indices = jax.random.choice(key, num_cells, shape=(total,), replace=False)
        rows = indices // self.dim
        cols = indices % self.dim
        locs = jnp.stack([rows, cols], axis=-1)  # (total, 2)

        state = PredatorPreyState(
            predator_loc=locs[:self.npredator],
            prey_loc=locs[self.npredator : self.npredator + 1],
            reached_prey=jnp.zeros(self.npredator, dtype=jnp.float32),
            step=0,
            done=jnp.bool_(False),
        )
        obs = self._get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: PredatorPreyState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict, PredatorPreyState, Dict, Dict, Dict]:
        action_arr = jnp.stack([actions[a] for a in self.agents])  # (N,)

        # Move predators
        new_locs = self._take_actions(state.predator_loc, state.reached_prey, action_arr)
        state = state.replace(predator_loc=new_locs)

        # Check which predators are on the prey
        on_prey = jnp.all(
            state.predator_loc == state.prey_loc[0], axis=-1
        ).astype(jnp.float32)  # (N,)

        # Rewards (mixed mode: -0.05 per step, 0 when on prey)
        reward = jnp.full(self.npredator, self.TIMESTEP_PENALTY)
        reward = jnp.where(on_prey > 0, self.PREY_REWARD, reward)

        # Update reached
        new_reached = jnp.maximum(state.reached_prey, on_prey)

        # Done: all predators reached prey OR max_steps
        step = state.step + 1
        all_reached = jnp.all(new_reached > 0)
        time_up = step >= self.max_steps
        episode_over = all_reached | time_up

        state = state.replace(
            reached_prey=new_reached,
            step=step,
            done=episode_over,
        )

        obs = self._get_obs(state)
        reward_dict = {a: reward[i] for i, a in enumerate(self.agents)}
        done_dict = {a: episode_over for a in self.agents}
        done_dict["__all__"] = episode_over
        info = {
            "success": jnp.all(new_reached > 0).astype(jnp.float32),
        }
        return obs, state, reward_dict, done_dict, info

    def get_obs(self, state: PredatorPreyState) -> Dict[str, chex.Array]:
        return self._get_obs(state)

    def _take_actions(self, predator_loc, reached_prey, actions):
        """Apply discrete actions. Agents that reached prey don't move."""
        def _move(loc, reached, action):
            delta = self._deltas[action]
            new_loc = jnp.clip(loc + delta, 0, self.dim - 1)
            return jnp.where(reached > 0, loc, new_loc)
        return jax.vmap(_move)(predator_loc, reached_prey, actions)

    def _get_obs(self, state: PredatorPreyState) -> Dict[str, chex.Array]:
        """Build one-hot vision-window observations for all agents."""
        # Build padded grid with cell IDs
        base_grid = jnp.arange(self.base, dtype=jnp.int32).reshape(self.dim, self.dim)
        pad_size = self.vision
        padded_dim = self.dim + 2 * pad_size

        if pad_size > 0:
            padded = jnp.full((padded_dim, padded_dim), self.outside_class, dtype=jnp.int32)
            padded = padded.at[pad_size:pad_size + self.dim, pad_size:pad_size + self.dim].set(base_grid)
        else:
            padded = base_grid

        # One-hot encode grid: (padded_dim, padded_dim, vocab_size)
        grid_oh = jax.nn.one_hot(padded, self.vocab_size)

        # Add predator marks
        def _add_pred(grid, i):
            r = state.predator_loc[i, 0] + pad_size
            c = state.predator_loc[i, 1] + pad_size
            return grid.at[r, c, self.predator_class].add(1.0)

        grid_oh = jax.lax.fori_loop(0, self.npredator, lambda i, g: _add_pred(g, i), grid_oh)

        # Add prey mark
        r_prey = state.prey_loc[0, 0] + pad_size
        c_prey = state.prey_loc[0, 1] + pad_size
        grid_oh = grid_oh.at[r_prey, c_prey, self.prey_class].add(1.0)

        # Extract vision windows for each agent
        def _agent_obs(pred_idx):
            r = state.predator_loc[pred_idx, 0]
            c = state.predator_loc[pred_idx, 1]
            window = jax.lax.dynamic_slice(
                grid_oh, (r, c, 0), (self.vis_size, self.vis_size, self.vocab_size)
            )
            # Transpose to (vocab_size, vis_h, vis_w) to match reference obs_space
            return window.transpose(2, 0, 1)

        all_obs = jax.vmap(_agent_obs)(jnp.arange(self.npredator))  # (N, V, H, W)

        return {a: all_obs[i] for i, a in enumerate(self.agents)}

    # ── Matplotlib rendering (for Visualizer) ──────────────────────────

    # Agent colours (up to 10)
    _COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    def init_render(self, ax, state, **kwargs):
        """Initialise matplotlib render – called once by Visualizer."""
        import numpy as np
        from matplotlib.patches import Circle, RegularPolygon
        import matplotlib.pyplot as plt

        ax.clear()
        ax.set_xlim(-0.5, self.dim - 0.5)
        ax.set_ylim(-0.5, self.dim - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(self.dim))
        ax.set_yticks(range(self.dim))
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_title(f"PredatorPrey {self.dim}×{self.dim}  step {int(state.step)}")

        artists = {}

        # Prey (gold star)
        pr, pc = int(state.prey_loc[0, 0]), int(state.prey_loc[0, 1])
        star = RegularPolygon(
            (pc, pr), numVertices=5, radius=0.35,
            orientation=0, color="gold", ec="darkgoldenrod", lw=1.5, zorder=5,
        )
        ax.add_patch(star)
        artists["prey"] = star

        # Predators (coloured circles with labels)
        pred_patches = []
        pred_labels = []
        locs = np.array(state.predator_loc)
        reached = np.array(state.reached_prey)
        for i in range(self.npredator):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            color = self._COLORS[i % len(self._COLORS)]
            ec = "gold" if reached[i] > 0 else "black"
            lw = 2.5 if reached[i] > 0 else 1.0
            circ = Circle((c, r), 0.3, color=color, ec=ec, lw=lw, zorder=10)
            ax.add_patch(circ)
            lbl = ax.text(c, r, str(i), ha="center", va="center",
                          fontsize=9, fontweight="bold", color="white", zorder=11)
            pred_patches.append(circ)
            pred_labels.append(lbl)

        artists["pred_patches"] = pred_patches
        artists["pred_labels"] = pred_labels
        return artists

    def update_render(self, artists, state, **kwargs):
        """Update an existing matplotlib render – called per frame by Visualizer."""
        import numpy as np

        # Update title
        ax = artists["pred_patches"][0].axes
        ax.set_title(f"PredatorPrey {self.dim}×{self.dim}  step {int(state.step)}")

        # Update prey position
        pr, pc = int(state.prey_loc[0, 0]), int(state.prey_loc[0, 1])
        artists["prey"].xy = (pc, pr)

        # Update predators
        locs = np.array(state.predator_loc)
        reached = np.array(state.reached_prey)
        for i in range(self.npredator):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            artists["pred_patches"][i].center = (c, r)
            artists["pred_labels"][i].set_position((c, r))
            if reached[i] > 0:
                artists["pred_patches"][i].set_edgecolor("gold")
                artists["pred_patches"][i].set_linewidth(2.5)

        return artists
