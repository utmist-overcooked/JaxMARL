import jax
import jax.numpy as jnp
from flax import struct
import chex
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete, Box
from typing import Dict
from functools import partial

@struct.dataclass
class State:
    p_pos : chex.Array  # [max_agents, 2] (y, x)
    p_dir: chex.Array  # [max_agents] (0: Right, 1: Down, 2: Left, 3: Up)
    path_idx: chex.Array  # [max_agents] (0: Pre-junction, 1: Post-junction) -> used for turn logic
    active: chex.Array  # [max_agents] (1 if active, 0 if finished)
    path_type: chex.Array  # [max_agents] (0: straight, 1: left, 2: right)
    step: int


class TrafficJunction(MultiAgentEnv):
    def __init__(self, max_agents=10, spawn_prob=0.1, max_steps=100, view_size=3, collision_penalty=-10.0, time_penalty=-0.01, **kwargs):
        self.num_agents = max_agents
        self.spawn_prob = spawn_prob
        self.max_steps = max_steps
        self.view_size = view_size
        self.grid_size = 14

        self.collision_penalty = collision_penalty
        self.time_penalty = time_penalty

        self.agents = [f"car_{i}" for i in range(max_agents)]
        self.agent_range = jnp.arange(max_agents)

        # --- OFF-GRID SPAWN LOCATIONS ---
        # Cars spawn 1 tile OUTSIDE the grid so they "drive in" rather than popping in.
        # Top (Down): [-1, 7], Right (Left): [7, 14], Bottom (Up): [14, 6], Left (Right): [6, -1]
        self.spawn_locations = jnp.array([
            [-1, 7],   # Top
            [7, 14],   # Right
            [14, 6],   # Bottom
            [6, -1]    # Left
        ])
        
        self.spawn_directions = jnp.array([1, 2, 3, 0]) 
        
        # move_vectors[dir] -> [dy, dx]
        self.move_vectors = jnp.array([
            [0, 1],   # 0: Right
            [1, 0],   # 1: Down
            [0, -1],  # 2: Left
            [-1, 0]   # 3: Up
        ])

        self.action_spaces = {agent: Discrete(2) for agent in self.agents}
        self.observation_spaces = {agent: Box(low=-1, high=1, shape=(view_size * view_size,)) for agent in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key=None):
        state = State(
            p_pos=jnp.zeros((self.num_agents, 2), dtype=jnp.int32),
            p_dir=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            path_idx=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            active=jnp.zeros((self.num_agents,), dtype=bool),
            path_type=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            step=0
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step(self, key: chex.PRNGKey, state: State, actions: Dict):
        action_arr = jnp.array([actions[agent] for agent in self.agents])
        gas_intent = (action_arr == 1) & state.active

        # --- PIVOT LOGIC (RHD) ---
        is_pivot = jnp.zeros(self.num_agents, dtype=bool)

        # Right-movers (dir 0) | Lane y=7
        is_pivot = jnp.where((state.p_dir==0) & (state.path_type==1) & (state.p_pos[:,1]==6), True, is_pivot) # Left
        is_pivot = jnp.where((state.p_dir==0) & (state.path_type==2) & (state.p_pos[:,1]==7), True, is_pivot) # Right
        is_pivot = jnp.where((state.p_dir==0) & (state.path_type==0) & (state.p_pos[:,1]==7), True, is_pivot) # Straight

        # Down-movers (dir 1) | Lane x=7
        is_pivot = jnp.where((state.p_dir==1) & (state.path_type==1) & (state.p_pos[:,0]==6), True, is_pivot) # Left
        is_pivot = jnp.where((state.p_dir==1) & (state.path_type==2) & (state.p_pos[:,0]==7), True, is_pivot) # Right
        is_pivot = jnp.where((state.p_dir==1) & (state.path_type==0) & (state.p_pos[:,0]==7), True, is_pivot) # Straight

        # Left-movers (dir 2) | Lane y=6
        is_pivot = jnp.where((state.p_dir==2) & (state.path_type==1) & (state.p_pos[:,1]==7), True, is_pivot) # Left
        is_pivot = jnp.where((state.p_dir==2) & (state.path_type==2) & (state.p_pos[:,1]==6), True, is_pivot) # Right
        is_pivot = jnp.where((state.p_dir==2) & (state.path_type==0) & (state.p_pos[:,1]==6), True, is_pivot) # Straight

        # Up-movers (dir 3) | Lane x=6
        is_pivot = jnp.where((state.p_dir==3) & (state.path_type==1) & (state.p_pos[:,0]==7), True, is_pivot) # Left
        is_pivot = jnp.where((state.p_dir==3) & (state.path_type==2) & (state.p_pos[:,0]==6), True, is_pivot) # Right
        is_pivot = jnp.where((state.p_dir==3) & (state.path_type==0) & (state.p_pos[:,0]==6), True, is_pivot) # Straight

        should_update = state.active & is_pivot & (state.path_idx == 0)
        
        turn_mod = jnp.where(state.path_type == 1, -1, 0)
        turn_mod = jnp.where(state.path_type == 2, 1, turn_mod)

        new_dir = jnp.where(should_update, (state.p_dir + turn_mod) % 4, state.p_dir)
        final_path_idx = jnp.where(should_update, 1, state.path_idx)

        # --- MOVEMENT ---
        targets = state.p_pos + self.move_vectors[new_dir] * gas_intent[:, None]

        def move_body(i, carry):
            final_pos, occupied_grid, collision_mask = carry
            ty, tx = targets[i, 0], targets[i, 1]
            
            in_bounds = (ty >= 0) & (ty < self.grid_size) & (tx >= 0) & (tx < self.grid_size)
            
            # Check occupancy. 
            # If target is off-grid, occupied_grid value is meaningless (always 0), so is_free=True.
            # If target is on-grid, we check the grid.
            is_free = (occupied_grid[ty, tx] == 0) | (~in_bounds)
            
            can_move = gas_intent[i] & is_free
            
            actual_pos = jnp.where(can_move, targets[i], state.p_pos[i])
            
            # We only mark the grid if the NEW position is actually on the grid.
            # We divert off-grid updates to 0,0 with value 0 (no-op) to prevent wrapping.
            pos_on_grid = (actual_pos[0] >= 0) & (actual_pos[0] < self.grid_size) & \
                          (actual_pos[1] >= 0) & (actual_pos[1] < self.grid_size)
            
            safe_y = jnp.where(pos_on_grid, actual_pos[0], 0)
            safe_x = jnp.where(pos_on_grid, actual_pos[1], 0)
            val_to_add = jnp.where(pos_on_grid, 1, 0)
            
            new_occupied = occupied_grid.at[safe_y, safe_x].add(val_to_add)
            
            # Collision: You collided if you wanted to move, target was IN BOUNDS, but blocked.
            had_collision = gas_intent[i] & ~can_move & in_bounds
            
            return final_pos.at[i].set(actual_pos), new_occupied, collision_mask.at[i].set(had_collision)

        # Initialize occupancy grid
        # We must mask out off-grid start positions so they don't wrap around
        init_occupied = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        
        start_on_grid = (state.p_pos[:, 0] >= 0) & (state.p_pos[:, 0] < self.grid_size) & \
                        (state.p_pos[:, 1] >= 0) & (state.p_pos[:, 1] < self.grid_size)
        
        safe_start_y = jnp.where(start_on_grid, state.p_pos[:, 0], 0)
        safe_start_x = jnp.where(start_on_grid, state.p_pos[:, 1], 0)
        start_vals = jnp.where(start_on_grid & state.active, 1, 0)
        
        init_occupied = init_occupied.at[safe_start_y, safe_start_x].add(start_vals)
        
        final_pos, _, collision_mask = jax.lax.fori_loop(0, self.num_agents, move_body, 
                                                        (state.p_pos, init_occupied, jnp.zeros(self.num_agents, dtype=bool)))

        # --- EXIT LOGIC ---
        target_oob = (targets[:, 0] < 0) | (targets[:, 0] >= self.grid_size) | \
                     (targets[:, 1] < 0) | (targets[:, 1] >= self.grid_size)
        
        # Car removes itself if it actively drives out of bounds after passing junction
        has_exited = target_oob & (final_path_idx == 1) & gas_intent
        is_active = state.active & ~has_exited

        # --- SPAWN ---
        # Spawn off-grid. Use collision check with exact coords to prevent stacking.
        spawn_locs = jnp.array([[-1, 7], [7, 14], [14, 6], [6, -1]])
        
        key_spawn, key_type = jax.random.split(key)
        spawn_rolls = jax.random.uniform(key_spawn, shape=(4,)) < self.spawn_prob
        path_types = jax.random.randint(key_type, shape=(4,), minval=0, maxval=3)

        current_state = state.replace(p_pos=final_pos, p_dir=new_dir, 
                                      path_idx=final_path_idx, active=is_active)

        def attempt_spawn(i, current_state):
            pos = spawn_locs[i]
            # Check for stacking: is there an active car EXACTLY at this off-grid pos?
            clear = ~jnp.any((current_state.active == 1) & jnp.all(current_state.p_pos == pos, axis=-1))
            slot = jnp.argmin(current_state.active)
            can_fill = spawn_rolls[i] & clear & (current_state.active[slot] == 0)
            return current_state.replace(
                active=current_state.active.at[slot].set(jnp.where(can_fill, 1, current_state.active[slot])),
                p_pos=current_state.p_pos.at[slot].set(jnp.where(can_fill, pos, current_state.p_pos[slot])),
                p_dir=current_state.p_dir.at[slot].set(jnp.where(can_fill, self.spawn_directions[i], current_state.p_dir[slot])),
                path_type=current_state.path_type.at[slot].set(jnp.where(can_fill, path_types[i], current_state.path_type[slot])),
                path_idx=current_state.path_idx.at[slot].set(jnp.where(can_fill, 0, current_state.path_idx[slot]))
            )

        final_state = jax.lax.fori_loop(0, 4, attempt_spawn, current_state)
        
        reward_per_agent = (collision_mask * self.collision_penalty) + (final_state.active * self.time_penalty)
        info = {agent: collision_mask[i].astype(jnp.int32) for i, agent in enumerate(self.agents)}
        final_state = final_state.replace(step=final_state.step + 1)
        done = final_state.step >= self.max_steps
        dones = {agent: ~final_state.active[i] | done for i, agent in enumerate(self.agents)}
        dones['__all__'] = done

        return self.get_obs(final_state), final_state, {a: reward_per_agent[i] for i, a in enumerate(self.agents)}, dones, info

    def get_obs(self, state: State):
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        
        # Safe Obs: Only render cars that are strictly ON the grid
        on_grid = (state.p_pos[:, 0] >= 0) & (state.p_pos[:, 0] < self.grid_size) & \
                  (state.p_pos[:, 1] >= 0) & (state.p_pos[:, 1] < self.grid_size)
        
        safe_y = jnp.where(on_grid, state.p_pos[:, 0], 0)
        safe_x = jnp.where(on_grid, state.p_pos[:, 1], 0)
        vals = jnp.where(on_grid & state.active, 1, 0)
        
        grid = grid.at[safe_y, safe_x].set(vals)
        pad = self.view_size // 2
        padded_grid = jnp.pad(grid, pad_width=pad, mode='constant', constant_values=-1)

        @jax.vmap
        def _observation(i):
            y, x = state.p_pos[i, 0], state.p_pos[i, 1]
            crop = jax.lax.dynamic_slice(padded_grid, (y, x), (self.view_size, self.view_size))
            return jnp.where(state.active[i], crop.flatten(), jnp.zeros(self.view_size * self.view_size, dtype=jnp.int32))
        
        obs_array = _observation(self.agent_range)
        return {agent: obs_array[i] for i, agent in enumerate(self.agents)}