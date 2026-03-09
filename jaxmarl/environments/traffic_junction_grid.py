"""Grid-based Traffic Junction environment (JAX).

Replicates the discrete traffic junction environment from the IC3Net paper:
  - N car slots; cars spawn stochastically at road endpoints and follow
    pre-defined routes through intersections.
  - Actions: GAS (0) = advance one step along route, BRAKE (1) = stay.
  - Rewards: -0.01 × wait_time per alive car per step; -10 penalty on crash.
  - Episode ends at max_steps.  Success = no crashes occurred.
  - Observation (bool vocab): [last_act, route_id, vision_grid_flat] per car.
    Dead cars get zero observations.

Difficulties:
  - easy:   2 roads (1 intersection), dim→dim+1, 2 routes
  - medium: 4 roads (1 junction), 12 routes
  - hard:   8 roads (4 junctions), 56 routes

Reference: IC3Net paper, https://arxiv.org/abs/1812.09755
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Tuple, Dict
from functools import partial
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.multi_agent_env import State as BaseState
from jaxmarl.environments.spaces import Box, Discrete


# ── Route-finding helpers (numpy, used only at init time) ──────────────

_MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_road_blocks(w, h, difficulty):
    """Return road block slices for the grid."""
    if difficulty == "easy":
        return [np.s_[h // 2, :], np.s_[:, w // 2]]
    elif difficulty == "medium":
        return [
            np.s_[h // 2 - 1 : h // 2 + 1, :],
            np.s_[:, w // 2 - 1 : w // 2 + 1],
        ]
    elif difficulty == "hard":
        return [
            np.s_[h // 3 - 2 : h // 3, :],
            np.s_[2 * h // 3 : 2 * h // 3 + 2, :],
            np.s_[:, w // 3 - 2 : w // 3],
            np.s_[:, 2 * h // 3 : 2 * h // 3 + 2],
        ]
    raise ValueError(f"Unknown difficulty: {difficulty}")


def _get_add_mat(h, w, grid, difficulty):
    """Arrival/finish points, road directions, junction mask."""
    road_dir = grid.copy()
    junction = np.zeros_like(grid)

    if difficulty == "medium":
        arrival = [
            (0, w // 2 - 1),
            (h - 1, w // 2),
            (h // 2, 0),
            (h // 2 - 1, w - 1),
        ]
        finish = [
            (0, w // 2),
            (h - 1, w // 2 - 1),
            (h // 2 - 1, 0),
            (h // 2, w - 1),
        ]
        road_dir[h // 2, :] = 2
        road_dir[h // 2 - 1, :] = 3
        road_dir[:, w // 2] = 4
        junction[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = 1

    elif difficulty == "hard":
        arrival = [
            (0, w // 3 - 2),
            (0, 2 * w // 3),
            (h // 3 - 1, 0),
            (2 * h // 3 + 1, 0),
            (h - 1, w // 3 - 1),
            (h - 1, 2 * w // 3 + 1),
            (h // 3 - 2, w - 1),
            (2 * h // 3, w - 1),
        ]
        finish = [
            (0, w // 3 - 1),
            (0, 2 * w // 3 + 1),
            (h // 3 - 2, 0),
            (2 * h // 3, 0),
            (h - 1, w // 3 - 2),
            (h - 1, 2 * w // 3),
            (h // 3 - 1, w - 1),
            (2 * h // 3 + 1, w - 1),
        ]
        road_dir[h // 3 - 1, :] = 2
        road_dir[2 * h // 3, :] = 3
        road_dir[2 * h // 3 + 1, :] = 4
        road_dir[:, w // 3 - 2] = 5
        road_dir[:, w // 3 - 1] = 6
        road_dir[:, 2 * w // 3] = 7
        road_dir[:, 2 * w // 3 + 1] = 8
        junction[h // 3 - 2 : h // 3, w // 3 - 2 : w // 3] = 1
        junction[2 * h // 3 : 2 * h // 3 + 2, w // 3 - 2 : w // 3] = 1
        junction[h // 3 - 2 : h // 3, 2 * w // 3 : 2 * w // 3 + 2] = 1
        junction[2 * h // 3 : 2 * h // 3 + 2, 2 * w // 3 : 2 * w // 3 + 2] = 1
    else:
        raise ValueError("_get_add_mat only for medium/hard")

    return arrival, finish, road_dir, junction


def _next_move(curr, turn, turn_step, start, grid, road_dir, junction, visited):
    """Find the next cell on the route (port of IC3Net traffic_helper)."""
    h, w = grid.shape
    turn_completed = False
    turn_prog = False
    neigh = []
    for m in _MOVE:
        n = (curr[0] + m[0], curr[1] + m[1])
        if not (0 <= n[0] <= h - 1 and 0 <= n[1] <= w - 1):
            continue
        if not grid[n]:
            continue
        if n in visited:
            continue

        if junction[n] == junction[curr] == 1:
            if (turn == 0 or turn == 2) and (n[0] == start[0] or n[1] == start[1]):
                neigh.append(n)
                if turn == 2:
                    turn_prog = True
            elif turn == 2 and turn_step == 1:
                neigh.append(n)
                turn_prog = True
        elif (
            junction[curr]
            and not junction[n]
            and turn == 2
            and turn_step == 2
            and (abs(start[0] - n[0]) == 2 or abs(start[1] - n[1]) == 2)
        ):
            neigh.append(n)
            turn_completed = True
        elif junction[n] and not junction[curr]:
            neigh.append(n)
        elif turn == 1 and not junction[n] and junction[curr]:
            neigh.append(n)
            turn_completed = True
        elif turn == 0 and junction[curr] and road_dir[n] == road_dir[start]:
            neigh.append(n)
            turn_completed = True
        elif road_dir[n] == road_dir[curr] and not junction[curr]:
            neigh.append(n)

    if neigh:
        return neigh[0], turn_prog, turn_completed
    raise RuntimeError(f"No valid next move from {curr}")


def _goal_reached(place_i, curr, finish_points):
    return curr in finish_points[:place_i] + finish_points[place_i + 1 :]


def _compute_routes_easy(eff_dims):
    """Trivial routes for easy difficulty."""
    h, w = eff_dims
    r0 = np.array([(i, w // 2) for i in range(h)])
    r1 = np.array([(h // 2, i) for i in range(w)])
    return [[r0], [r1]]


def _compute_routes_medium_hard(eff_dims, route_grid, difficulty):
    """Port of IC3Net get_routes for medium/hard."""
    h, w = eff_dims
    arrival, finish, road_dir, junction = _get_add_mat(h, w, route_grid, difficulty)
    n_turn1 = 3
    n_turn2 = 1 if difficulty == "medium" else 3

    routes = []
    for i in range(len(arrival)):
        paths = []
        for t1 in range(n_turn1):
            for t2 in range(n_turn2):
                total_turns = 0
                curr_turn = t1
                path = [arrival[i]]
                visited = {arrival[i]}
                current = arrival[i]
                start = current
                ts = 0
                try:
                    while not _goal_reached(i, current, finish):
                        visited.add(current)
                        current, tp, tc = _next_move(
                            current, curr_turn, ts, start,
                            route_grid, road_dir, junction, visited,
                        )
                        if curr_turn == 2 and tp:
                            ts += 1
                        if tc:
                            total_turns += 1
                            curr_turn = t2
                            ts = 0
                            start = current
                        if total_turns == 2:
                            curr_turn = 0
                        path.append(current)
                except RuntimeError:
                    continue
                paths.append(np.array(path))
                if total_turns == 1:
                    break
        routes.append(paths)
    return routes


# ── JAX State ──────────────────────────────────────────────────────────


@struct.dataclass
class TrafficJunctionState(BaseState):
    """State for the Traffic Junction environment."""
    car_loc: chex.Array           # (ncar, 2) int — row, col
    alive_mask: chex.Array        # (ncar,) float — 1.0 if alive
    car_route_loc: chex.Array     # (ncar,) int — position along route
    chosen_path_idx: chex.Array   # (ncar,) int — index into all_routes
    car_last_act: chex.Array      # (ncar,) int — last action taken
    wait: chex.Array              # (ncar,) float — time alive so far
    cars_in_sys: chex.Array       # () int — count of active cars
    route_id_norm: chex.Array     # (ncar,) float — normalised route id [0,1]
    has_crashed: chex.Array       # () bool — any crash this episode


# ── Environment ────────────────────────────────────────────────────────


class TrafficJunctionGrid(MultiAgentEnv):
    """Grid-based Traffic Junction matching the IC3Net reference.

    Easy:   5 agents, dim=6, vision=0, max_steps=20, add_rate=0.3
    Medium: 10 agents, dim=14, vision=0, max_steps=40, add_rate=0.05
    Hard:   20 agents, dim=18, vision=0, max_steps=80, add_rate=0.05
    """

    def __init__(
        self,
        num_agents: int = 5,
        dim: int = 6,
        vision: int = 0,
        max_steps: int = 20,
        difficulty: str = "easy",
        add_rate: float = 0.3,
    ):
        super().__init__(num_agents)

        self.dim = dim
        self.vision = vision
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.add_rate = add_rate
        self.ncar = num_agents
        self.naction = 2  # GAS=0, BRAKE=1

        # Effective grid dimensions (easy adds 1 for odd size)
        if difficulty == "easy":
            self.eff_dims = (dim + 1, dim + 1)
        else:
            self.eff_dims = (dim, dim)

        # Vocab/Base (following reference exactly, using original dim)
        dim_sum = dim + dim
        base_map = {"easy": dim_sum, "medium": 2 * dim_sum, "hard": 4 * dim_sum}
        self.BASE = base_map[difficulty]
        self.OUTSIDE_CLASS = self.BASE
        self.CAR_CLASS = self.BASE + 2
        self.ROAD_CLASS = 1
        self.vocab_size = self.BASE + 3  # 1 + BASE + 1 + 1

        # Vision window
        self.vis_size = 2 * vision + 1

        # Flat observation dim: act + route_id + vision_grid
        self.obs_dim = 1 + 1 + self.vis_size * self.vis_size * self.vocab_size

        # Reward constants
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10.0

        # Agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Spaces
        self.observation_spaces = {
            a: Box(low=-100.0, high=100.0, shape=(self.obs_dim,))
            for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.naction) for a in self.agents}

        # --- Pre-compute grid and routes (numpy, at init time) ---
        self._setup_grid_and_routes()

    # ── Grid & route setup (numpy) ─────────────────────────────────────

    def _setup_grid_and_routes(self):
        h, w = self.eff_dims

        # Build grid with sequential road indices (bool vocab)
        grid = np.full((h, w), self.OUTSIDE_CLASS, dtype=np.int32)
        blocks = _get_road_blocks(w, h, self.difficulty)
        for blk in blocks:
            grid[blk] = self.ROAD_CLASS

        # route_grid keeps ROAD_CLASS for route-finding
        route_grid = grid.copy()

        # Assign sequential indices to road cells
        start = 0
        for blk in blocks:
            sz = int(np.prod(grid[blk].shape))
            grid[blk] = np.arange(start, start + sz).reshape(grid[blk].shape)
            start += sz

        # Compute routes
        if self.difficulty == "easy":
            route_groups = _compute_routes_easy(self.eff_dims)
        else:
            route_groups = _compute_routes_medium_hard(
                self.eff_dims, route_grid, self.difficulty
            )

        # Flatten groups into a single list of routes
        all_routes = []
        self.n_groups = len(route_groups)
        group_sizes = []
        for grp in route_groups:
            group_sizes.append(len(grp))
            all_routes.extend(grp)

        self.npath = len(all_routes)
        self.max_route_len = max(len(r) for r in all_routes) if all_routes else 1

        # Pad routes to uniform length and convert to JAX
        padded = np.zeros((self.npath, self.max_route_len, 2), dtype=np.int32)
        lengths = np.zeros(self.npath, dtype=np.int32)
        for i, r in enumerate(all_routes):
            padded[i, : len(r)] = r
            lengths[i] = len(r)

        self.jax_routes = jnp.array(padded)         # (npath, max_len, 2)
        self.jax_route_lens = jnp.array(lengths)    # (npath,)

        # Group offsets for spawning
        offsets = np.zeros(self.n_groups + 1, dtype=np.int32)
        for g in range(self.n_groups):
            offsets[g + 1] = offsets[g] + group_sizes[g]
        self.jax_group_offsets = jnp.array(offsets)
        self.jax_group_sizes = jnp.array(group_sizes, dtype=np.int32)

        # Padded grid → JAX (for observations)
        pad_grid = np.pad(
            grid, self.vision, "constant", constant_values=self.OUTSIDE_CLASS
        )
        self.jax_pad_grid = jnp.array(pad_grid, dtype=jnp.int32)

    # ── Spaces ─────────────────────────────────────────────────────────

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    # ── Reset ──────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], TrafficJunctionState]:
        state = TrafficJunctionState(
            car_loc=jnp.zeros((self.ncar, 2), dtype=jnp.int32),
            alive_mask=jnp.zeros(self.ncar, dtype=jnp.float32),
            car_route_loc=jnp.full(self.ncar, -1, dtype=jnp.int32),
            chosen_path_idx=jnp.zeros(self.ncar, dtype=jnp.int32),
            car_last_act=jnp.zeros(self.ncar, dtype=jnp.int32),
            wait=jnp.zeros(self.ncar, dtype=jnp.float32),
            cars_in_sys=jnp.int32(0),
            route_id_norm=jnp.zeros(self.ncar, dtype=jnp.float32),
            has_crashed=jnp.bool_(False),
            step=0,
            done=jnp.bool_(False),
        )
        obs = self._get_obs(state)
        return obs, state

    # ── Step ───────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: TrafficJunctionState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict, TrafficJunctionState, Dict, Dict, Dict]:
        action_arr = jnp.stack([actions[a] for a in self.agents])  # (N,)

        # 1) Apply actions (move / brake / complete)
        state, n_completed = self._apply_actions(state, action_arr)

        # 2) Add new cars stochastically
        state, key = self._add_cars(state, key)

        # 3) Crash detection
        has_crash_vec = self._detect_crashes(state)  # (ncar,) bool
        any_crash = jnp.any(has_crash_vec)
        new_has_crashed = state.has_crashed | any_crash

        # 4) Rewards
        reward = self.TIMESTEP_PENALTY * state.wait
        reward = reward + jnp.where(has_crash_vec, self.CRASH_PENALTY, 0.0)
        reward = state.alive_mask * reward

        # 5) Step and done
        new_step = state.step + 1
        episode_over = new_step >= self.max_steps

        state = state.replace(
            has_crashed=new_has_crashed,
            step=new_step,
            done=episode_over,
        )

        obs = self._get_obs(state)
        reward_dict = {a: reward[i] for i, a in enumerate(self.agents)}
        done_dict = {a: episode_over for a in self.agents}
        done_dict["__all__"] = episode_over
        info = {
            "success": (1.0 - new_has_crashed.astype(jnp.float32)),
        }
        return obs, state, reward_dict, done_dict, info

    def get_obs(self, state: TrafficJunctionState) -> Dict[str, chex.Array]:
        return self._get_obs(state)

    # ── Internal mechanics ─────────────────────────────────────────────

    def _apply_actions(self, state, actions):
        """Move alive & GAS cars along their routes. Complete if at route end."""
        alive = state.alive_mask
        is_alive = alive > 0
        is_gas = actions == 0

        # Wait increments for alive cars
        new_wait = jnp.where(is_alive, state.wait + 1.0, state.wait)

        # Last action (for observation)
        new_last_act = jnp.where(is_alive, actions, state.car_last_act)

        # Advance route position for alive GAS cars
        new_route_loc = jnp.where(
            is_alive & is_gas, state.car_route_loc + 1, state.car_route_loc
        )

        # Check completion (reached end of route)
        route_lens = self.jax_route_lens[state.chosen_path_idx]  # (ncar,)
        reached_end = new_route_loc >= route_lens
        is_completed = is_alive & is_gas & reached_end

        # New car location from route (use clipped index for safe indexing)
        safe_loc = jnp.clip(new_route_loc, 0, self.max_route_len - 1)
        next_loc = self.jax_routes[state.chosen_path_idx, safe_loc]  # (ncar, 2)

        # Update location only for alive GAS non-completed cars
        should_move = is_alive & is_gas & ~reached_end
        new_car_loc = jnp.where(should_move[:, None], next_loc, state.car_loc)

        # Completed cars: mark dead, reset
        new_alive = jnp.where(is_completed, 0.0, alive)
        new_car_loc = jnp.where(
            is_completed[:, None],
            jnp.zeros_like(state.car_loc),
            new_car_loc,
        )
        new_wait = jnp.where(is_completed, 0.0, new_wait)
        new_route_loc = jnp.where(is_completed, -1, new_route_loc)

        n_completed = jnp.sum(is_completed.astype(jnp.int32))
        new_cars_in_sys = state.cars_in_sys - n_completed

        state = state.replace(
            car_loc=new_car_loc,
            alive_mask=new_alive,
            car_route_loc=new_route_loc,
            car_last_act=new_last_act,
            wait=new_wait,
            cars_in_sys=new_cars_in_sys,
        )
        return state, n_completed

    def _add_cars(self, state, key):
        """Stochastically spawn cars at arrival points."""
        n_groups = self.n_groups

        def _try_spawn(carry, group_idx):
            car_loc, alive, rloc, pidx, rid_norm, cis, k = carry
            k, k1, k2 = jax.random.split(k, 3)

            should_try = jax.random.uniform(k1) <= self.add_rate
            has_cap = cis < self.ncar
            has_dead = jnp.any(alive == 0)
            do_spawn = should_try & has_cap & has_dead

            # First dead slot
            dead_idx = jnp.argmin(alive)

            # Pick random route from this group
            g_off = self.jax_group_offsets[group_idx]
            g_sz = self.jax_group_sizes[group_idx]
            route_pick = jax.random.randint(k2, (), 0, g_sz)
            global_idx = g_off + route_pick

            start_loc = self.jax_routes[global_idx, 0]
            norm_rid = jnp.where(
                self.npath > 1,
                global_idx.astype(jnp.float32) / (self.npath - 1),
                0.0,
            )

            # Conditionally update (always writes, but preserves old value
            # when do_spawn is False)
            new_alive_val = jnp.where(do_spawn, 1.0, alive[dead_idx])
            alive = alive.at[dead_idx].set(new_alive_val)

            new_loc_val = jnp.where(do_spawn, start_loc, car_loc[dead_idx])
            car_loc = car_loc.at[dead_idx].set(new_loc_val)

            new_rloc_val = jnp.where(do_spawn, 0, rloc[dead_idx])
            rloc = rloc.at[dead_idx].set(new_rloc_val)

            new_pidx_val = jnp.where(do_spawn, global_idx, pidx[dead_idx])
            pidx = pidx.at[dead_idx].set(new_pidx_val)

            new_rid_val = jnp.where(do_spawn, norm_rid, rid_norm[dead_idx])
            rid_norm = rid_norm.at[dead_idx].set(new_rid_val)

            # Reset last_act and wait for newly spawned car
            # (not strictly necessary since they start at 0/0.0 anyway)

            cis = jnp.where(do_spawn, cis + 1, cis)
            return (car_loc, alive, rloc, pidx, rid_norm, cis, k), None

        init = (
            state.car_loc,
            state.alive_mask,
            state.car_route_loc,
            state.chosen_path_idx,
            state.route_id_norm,
            state.cars_in_sys,
            key,
        )
        (car_loc, alive, rloc, pidx, rid_norm, cis, key), _ = jax.lax.scan(
            _try_spawn, init, jnp.arange(n_groups), length=n_groups,
        )

        state = state.replace(
            car_loc=car_loc,
            alive_mask=alive,
            car_route_loc=rloc,
            chosen_path_idx=pidx,
            route_id_norm=rid_norm,
            cars_in_sys=cis,
        )
        return state, key

    def _detect_crashes(self, state):
        """Return per-car boolean: True if this car is in a crash."""
        locs = state.car_loc  # (ncar, 2)
        alive = state.alive_mask  # (ncar,)

        # Pairwise: same location
        same = jnp.all(locs[:, None] == locs[None, :], axis=-1)  # (ncar, ncar)
        not_self = ~jnp.eye(self.ncar, dtype=bool)
        both_alive = (alive[:, None] > 0) & (alive[None, :] > 0)
        # Exclude origin (0,0) — dead car parking
        not_origin = jnp.any(locs != 0, axis=-1)  # (ncar,)

        crashed = jnp.any(
            same & not_self & both_alive & not_origin[:, None], axis=1
        )
        return crashed

    def _get_obs(self, state: TrafficJunctionState) -> Dict[str, chex.Array]:
        """Build flat observation per car: [act_norm, route_id_norm, vision_flat]."""
        # One-hot encode padded grid
        grid_oh = jax.nn.one_hot(self.jax_pad_grid, self.vocab_size)  # (H, W, V)

        # Add car marks for alive cars
        def _mark_car(grid, i):
            r = state.car_loc[i, 0] + self.vision
            c = state.car_loc[i, 1] + self.vision
            return grid.at[r, c, self.CAR_CLASS].add(state.alive_mask[i])

        grid_oh = jax.lax.fori_loop(0, self.ncar, lambda i, g: _mark_car(g, i), grid_oh)

        # Extract per-car observations
        def _car_obs(i):
            r = state.car_loc[i, 0]
            c = state.car_loc[i, 1]
            window = jax.lax.dynamic_slice(
                grid_oh, (r, c, 0), (self.vis_size, self.vis_size, self.vocab_size)
            )
            flat_vis = window.reshape(-1)

            act_norm = state.car_last_act[i].astype(jnp.float32) / jnp.maximum(
                self.naction - 1, 1
            )
            rid_norm = state.route_id_norm[i]

            obs = jnp.concatenate([jnp.array([act_norm, rid_norm]), flat_vis])
            # Dead cars get zero observations
            return obs * state.alive_mask[i]

        all_obs = jax.vmap(_car_obs)(jnp.arange(self.ncar))  # (N, obs_dim)
        return {a: all_obs[i] for i, a in enumerate(self.agents)}

    # ── Matplotlib rendering (for Visualizer) ──────────────────────────

    _ROAD_COLOR = "#d0d0d0"
    _CAR_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    def init_render(self, ax, state, **kwargs):
        """Initialise matplotlib render — called once by Visualizer."""
        import numpy as _np
        from matplotlib.patches import Rectangle, Circle

        h, w = self.eff_dims
        ax.clear()
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_facecolor("#f5f5f5")
        ax.set_title(f"TrafficJunction {w}×{h}  step {int(state.step)}")

        artists = {}

        # Draw road cells
        road_patches = []
        np_route_grid = _np.array(self.jax_pad_grid[
            self.vision : self.vision + h, self.vision : self.vision + w
        ])
        for r in range(h):
            for c in range(w):
                if np_route_grid[r, c] != self.OUTSIDE_CLASS:
                    rect = Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor=self._ROAD_COLOR, edgecolor="#bbb", lw=0.3,
                    )
                    ax.add_patch(rect)
                    road_patches.append(rect)
        artists["roads"] = road_patches

        # Draw cars
        car_patches = []
        car_labels = []
        locs = _np.array(state.car_loc)
        alive = _np.array(state.alive_mask)
        for i in range(self.ncar):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            visible = alive[i] > 0
            color = self._CAR_COLORS[i % len(self._CAR_COLORS)]
            circ = Circle(
                (c, r), 0.3, color=color, ec="black", lw=1.0,
                zorder=10, visible=visible,
            )
            ax.add_patch(circ)
            lbl = ax.text(
                c, r, str(i), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                zorder=11, visible=visible,
            )
            car_patches.append(circ)
            car_labels.append(lbl)

        artists["car_patches"] = car_patches
        artists["car_labels"] = car_labels
        return artists

    def update_render(self, artists, state, **kwargs):
        """Update matplotlib render — called per frame by Visualizer."""
        import numpy as _np

        ax = artists["car_patches"][0].axes
        h, w = self.eff_dims
        ax.set_title(f"TrafficJunction {w}×{h}  step {int(state.step)}")

        locs = _np.array(state.car_loc)
        alive = _np.array(state.alive_mask)
        for i in range(self.ncar):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            visible = alive[i] > 0
            artists["car_patches"][i].center = (c, r)
            artists["car_patches"][i].set_visible(visible)
            artists["car_labels"][i].set_position((c, r))
            artists["car_labels"][i].set_visible(visible)

        return artists
