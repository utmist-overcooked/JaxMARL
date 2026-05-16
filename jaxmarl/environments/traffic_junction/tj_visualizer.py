import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import colormaps

class TrafficVisualizer:
    def __init__(self, env, state_seq, interval=200):
        self.env = env
        self.state_seq = state_seq
        self.grid_size = env.grid_size
        self.interval = interval
        
        # 1. DYNAMIC ROAD LOGIC
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
        
        # 2. COLOR MAP
        self.cmap = colormaps.get_cmap('tab20')
        self.agent_colors = [self.cmap(i % 20) for i in range(self.env.num_agents)]

        self.fig, self.ax = plt.subplots(figsize=(7, 7), facecolor='black')
        
        # Initialize the artist here so it persists across updates
        self.agent_scatter = None
        self.init_render()

    def init_render(self):
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        
        # Generate Dot Grid
        x_coords, y_coords = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.ax.scatter(x_coords, y_coords, color='#333333', s=2, zorder=0, marker='.')
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # 3. DRAW DYNAMIC ROADS (White)
        self.ax.add_patch(plt.Rectangle((self.road_min - 0.5, -0.5), self.road_width, self.grid_size, color='#FFFFFF', zorder=1))
        self.ax.add_patch(plt.Rectangle((-0.5, self.road_min - 0.5), self.grid_size, self.road_width, color='#FFFFFF', zorder=1))

        # 4. DRAW CENTER LINES (Gold)
        if not self.is_one_way:
            lane_style = {'color': '#B8860B', 'linestyle': '--', 'linewidth': 2, 'zorder': 2}
            center_line = (self.road_lanes[0] + self.road_lanes[-1]) / 2.0
            self.ax.plot([center_line, center_line], [-0.5, self.grid_size - 0.5], **lane_style)
            self.ax.plot([-0.5, self.grid_size - 0.5], [center_line, center_line], **lane_style)

        # 5. INITIALIZE AGENT SCATTER (Empty initially)
        self.agent_scatter = self.ax.scatter([], [], s=300, marker='s', edgecolors='black', 
                                             linewidths=1.5, zorder=10)
        
        self.step_counter = self.ax.text(0, self.grid_size - 0.4, "Step: 0", 
                                         color='white', fontweight='bold', va="bottom")
        self.active_counter = self.ax.text(self.grid_size - 0.5, self.grid_size - 0.4, "Cars: 0", 
                                           color='white', fontweight='bold', va="bottom", ha="right")

    def update(self, frame):
        full_state = self.state_seq[frame]
        state = full_state.env_state if hasattr(full_state, 'env_state') else full_state
        
        active_mask = np.array(state.active).astype(bool)
        all_pos = np.array(state.p_pos)
        
        # Explicitly filter for agents on the grid
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) > 0:
            # JaxMARL stores as (row, col) -> we need (x, y) which is (col, row)
            # Slice only the active agents
            active_pos = all_pos[active_indices]
            
            # CRITICAL: Matplotlib scatter expects (x, y), so we swap columns
            # [row, col] -> [col, row]
            display_pos = active_pos[:, [1, 0]]
            
            self.agent_scatter.set_offsets(display_pos) 
            
            # Update colors to match the specific active agent IDs
            colors = [self.agent_colors[i] for i in active_indices]
            self.agent_scatter.set_facecolors(colors)
        else:
            self.agent_scatter.set_offsets(np.empty((0, 2)))

        self.step_counter.set_text(f"Step: {int(state.step)}")
        self.active_counter.set_text(f"Cars: {int(np.sum(active_mask))}")
        
        return self.agent_scatter, self.step_counter, self.active_counter

    def animate(self, save_fname="traffic_evaluation.gif"):
        print(f"Generating GIF: {save_fname}...")
        # We use blit=False to ensure the entire background renders correctly on every frame
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(self.state_seq),
            interval=self.interval, blit=False
        )
        if save_fname:
            # Using 10 FPS for a clear, readable simulation speed
            ani.save(save_fname, writer='pillow', fps=10, savefig_kwargs={'facecolor': 'black'})
        plt.close(self.fig)
        print(f"GIF saved successfully as {save_fname}")