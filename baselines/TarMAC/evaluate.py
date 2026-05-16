import matplotlib
matplotlib.use('Agg')  # Stability for macOS/Headless runs
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colormaps
import argparse
import os

from jaxmarl.environments.traffic_junction.traffic_junction import TrafficJunction
from tarmac import TarMAC, TarMACConfig

# ==========================================
# UNIVERSAL CUSTOM VISUALIZER
# ==========================================
class TrafficVisualizer:
    def __init__(self, env, state_seq, interval=150):
        self.env = env
        self.state_seq = state_seq
        self.grid_size = env.grid_size
        self.interval = interval
        
        center = self.grid_size // 2
        self.is_one_way = getattr(env, 'one_way', False)
        
        if self.is_one_way:
            self.road_lanes = [center]
        else:
            if self.grid_size % 2 == 0:
                self.road_lanes = [center - 1, center]
            else:
                self.road_lanes = [center, center + 1]
                
        self.road_min = min(self.road_lanes)
        self.road_width = len(self.road_lanes)
        
        self.cmap = colormaps.get_cmap('tab20')
        self.agent_colors = [self.cmap(i % 20) for i in range(self.env.num_agents)]
        self.dir_to_marker = {0: '>', 1: '^', 2: '<', 3: 'v'}

        self.fig, self.ax = plt.subplots(figsize=(8, 8), facecolor='black')
        self.car_artists = [] 
        self.collision_total = 0  # Running count for the whole episode
        self.init_render()

    def init_render(self):
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        
        x_coords, y_coords = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.ax.scatter(x_coords, y_coords, color='#222222', s=2, zorder=0, marker='.')
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Draw Roads
        road_alpha = 0.9
        self.ax.add_patch(plt.Rectangle((self.road_min - 0.5, -0.5), self.road_width, self.grid_size, color='#FFFFFF', zorder=1, alpha=road_alpha))
        self.ax.add_patch(plt.Rectangle((-0.5, self.road_min - 0.5), self.grid_size, self.road_width, color='#FFFFFF', zorder=1, alpha=road_alpha))

        if not self.is_one_way:
            lane_style = {'color': '#B8860B', 'linestyle': '--', 'linewidth': 2, 'zorder': 2}
            center_line = (self.road_lanes[0] + self.road_lanes[-1]) / 2.0
            self.ax.plot([center_line, center_line], [-0.5, self.grid_size - 0.5], **lane_style)
            self.ax.plot([-0.5, self.grid_size - 0.5], [center_line, center_line], **lane_style)

        # HUD Text
        self.step_counter = self.ax.text(0.1, self.grid_size - 0.6, "Step: 0", color='white', fontweight='bold', fontsize=12)
        self.active_counter = self.ax.text(self.grid_size // 2, self.grid_size - 0.6, "Cars: 0", color='white', fontweight='bold', fontsize=12, ha="center")
        self.collision_counter = self.ax.text(self.grid_size - 0.1, self.grid_size - 0.6, "Collisions: 0", color='#FF4444', fontweight='bold', fontsize=12, ha="right")

    def update(self, frame):
        for artist in self.car_artists:
            artist.remove()
        self.car_artists = []

        full_state = self.state_seq[frame]
        state = full_state.env_state if hasattr(full_state, 'env_state') else full_state
        
        active_mask = np.array(state.active).astype(bool)
        all_pos = np.array(state.p_pos)
        all_dirs = np.array(state.p_dir)
        
        # Check for collisions in the info or state if available
        # In TrafficJunction, collisions usually happen during movement
        # If your step function returns a collision mask, ensure it's in the state_sequence
        # For now, we manually detect if two active cars share a position
        pos_list = [tuple(all_pos[i]) for i in range(self.env.num_agents) if active_mask[i]]
        collision_positions = set([x for x in pos_list if pos_list.count(x) > 1])
        
        # Update running total (only if it's a new frame and not the first frame)
        if frame > 0:
            self.collision_total += len(collision_positions)

        active_indices = np.where(active_mask)[0]
        for i in active_indices:
            y, x = all_pos[i]
            
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                direction = int(all_dirs[i])
                marker = self.dir_to_marker.get(direction, 's')
                
                # Visual Collision Feedback: Red edge if colliding
                is_colliding = tuple(all_pos[i]) in collision_positions
                edge_color = 'red' if is_colliding else 'black'
                line_width = 3.0 if is_colliding else 1.5
                
                sc = self.ax.scatter(x, y, s=500, marker=marker, 
                                     color=self.agent_colors[i], 
                                     edgecolors=edge_color, 
                                     linewidths=line_width, zorder=10)
                self.car_artists.append(sc)

        self.step_counter.set_text(f"Step: {int(state.step)}")
        self.active_counter.set_text(f"Cars: {int(np.sum(active_mask))}")
        self.collision_counter.set_text(f"Total Col: {self.collision_total}")
        
        return self.car_artists + [self.step_counter, self.active_counter, self.collision_counter]

    def animate(self, save_fname="traffic_evaluation.gif"):
        print(f"Generating GIF: {save_fname}...")
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(self.state_seq),
            interval=self.interval, blit=False
        )
        ani.save(save_fname, writer='pillow', fps=10)
        plt.close(self.fig)
        print(f"GIF saved successfully as {save_fname}")


# ==========================================
# EVALUATION LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="update_1000.ckpt")
    
    # Matching your 14x14 / 10 car training defaults
    parser.add_argument("--grid_size", type=int, default=14)
    parser.add_argument("--max_cars", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--p_arrive", type=float, default=0.30)
    parser.add_argument("--no_comm", action="store_true")
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--comm_rounds", type=int, default=2)

    args = parser.parse_args()

    env = TrafficJunction(
        max_steps=args.max_steps,
        max_cars=args.max_cars,
        grid_size=args.grid_size,
        spawn_prob=args.p_arrive
    )
    
    num_agents = len(env.agents)
    act_dim = env.action_space(env.agents[0]).n
    
    config = TarMACConfig(
        hidden_dim=args.hidden_dim, 
        msg_dim=args.msg_dim, 
        key_dim=args.key_dim, 
        num_rounds=args.comm_rounds
    )
    actor = TarMAC(action_dim=act_dim, config=config)
    
    # Init & Load Checkpoint
    rng = jax.random.PRNGKey(0)
    dummy_carry = actor.initialize_carry(1, num_agents)
    obs_shape = env.observation_space(env.agents[0]).shape 
    obs_dim = 11 if obs_shape[0] == 10 else obs_shape[0]
    
    dummy_obs = jnp.zeros((1, 1, num_agents, obs_dim))
    dummy_dones = jnp.zeros((1, 1, num_agents, 1))
    dummy_params = actor.init(rng, dummy_carry, dummy_obs, dummy_dones)
    
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    with open(ckpt_path, "rb") as f:
        import flax.serialization
        raw_state_dict = flax.serialization.msgpack_restore(f.read())
        actor_params = flax.serialization.from_state_dict(dummy_params, raw_state_dict['actor'])

    @jax.jit
    def eval_step(env_state, obs, rnn, rng_step, no_comm):
        obs_tensor = jnp.stack([obs[a] for a in env.agents], axis=0)[None, None, ...]
        if obs_tensor.shape[-1] == 10 and obs_dim == 11:
            obs_tensor = jnp.pad(obs_tensor, ((0,0), (0,0), (0,0), (0,1)))
            
        h_in, msg_in = rnn
        msg_to_use = jax.lax.select(no_comm, jnp.zeros_like(msg_in), msg_in)
        
        new_rnn, (logits_seq, _, _) = actor.apply(actor_params, (h_in, msg_to_use), obs_tensor, jnp.zeros((1, 1, num_agents, 1)))
        actions = jnp.argmax(logits_seq[0], axis=-1)[0]
        # actions = jnp.ones((num_agents,), dtype=jnp.int32)
        actions_dict = {a: actions[i] for i, a in enumerate(env.agents)}
        
        next_obs, next_env_st, _, _, info = env.step(rng_step, env_state, actions_dict)
        return next_env_st, next_obs, new_rnn

    print(f"Evaluating model for {args.max_steps} steps...")
    obs, env_state = env.reset(rng)
    rnn_carry = actor.initialize_carry(1, num_agents)
    state_sequence = [jax.tree_util.tree_map(lambda x: jax.device_get(x), env_state)]
    
    for _ in range(args.max_steps):
        rng, rng_step = jax.random.split(rng)
        env_state, obs, rnn_carry = eval_step(env_state, obs, rnn_carry, rng_step, args.no_comm)
        state_sequence.append(jax.tree_util.tree_map(lambda x: jax.device_get(x), env_state))
        
    save_name = f"eval_14x14_{args.ckpt.replace('.ckpt','')}.gif"
    viz = TrafficVisualizer(env, state_sequence)
    viz.animate(save_name)

if __name__ == "__main__":
    main()