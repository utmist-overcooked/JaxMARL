import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import colormaps

class TrafficVisualizer:
    def __init__(self, env, state_seq, interval=150):
        self.env = env
        self.state_seq = state_seq
        self.grid_size = env.grid_size
        self.interval = interval
        
        # 1. Create a distinct color for every possible agent slot
        # 'hsv' or 'tab10' are great for high-contrast colors
        self.cmap = colormaps.get_cmap('tab20')
        self.agent_colors = [self.cmap(i % 20) for i in range(self.env.num_agents)]

        self.fig, self.ax = plt.subplots(figsize=(7, 7), facecolor='black')
        self.init_render()

    def init_render(self):
        self.ax.set_facecolor('black')
        
        # 1. Axis limits
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        
        # 2. CREATE DOT GRID
        # We generate a meshgrid of all (x, y) coordinates
        x_coords, y_coords = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.ax.scatter(x_coords, y_coords, color='#444444', s=2, zorder=0, marker='.')
        
        # Hide standard spines and ticks for a cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # 3. DRAW THE ROADS
        # Vertical Road
        self.ax.add_patch(plt.Rectangle((5.5, -0.5), 2, self.grid_size, color='#FFFFFF', zorder=1))
        # Horizontal Road
        self.ax.add_patch(plt.Rectangle((-0.5, 5.5), self.grid_size, 2, color='#FFFFFF', zorder=1))

        # 4. DRAW CENTER LINES (Darker Amber/Gold)
        # Using a "Goldenrod" or hex #DAA520 for better visibility against white
        lane_style = {'color': '#B8860B', 'linestyle': '--', 'linewidth': 2, 'zorder': 2}
        
        # Vertical center line
        self.ax.plot([6.5, 6.5], [-0.5, self.grid_size - 0.5], **lane_style)
        # Horizontal center line
        self.ax.plot([-0.5, self.grid_size - 0.5], [6.5, 6.5], **lane_style)

        # 5. Initialize Agent Artists
        self.agent_scatter = self.ax.scatter([], [], s=350, marker='s', edgecolors='black', 
                                             linewidths=2, zorder=10)
        
        # Text annotations
        self.step_counter = self.ax.text(0, self.grid_size - 0.5, "Step: 0", 
                                         color='white', fontweight='bold', va="bottom")
        self.active_counter = self.ax.text(self.grid_size - 0.5, self.grid_size - 0.5, "Cars: 0", 
                                           color='white', fontweight='bold', va="bottom", ha="right")
    def update(self, frame):
        state = self.state_seq[frame]
        
        active_mask = np.array(state.active == 1)
        all_pos = np.array(state.p_pos)
        active_pos = all_pos[active_mask]
        
        if len(active_pos) > 0:
            # Swap (row, col) to (x, y)
            self.agent_scatter.set_offsets(active_pos[:, ::-1]) 
            
            # 3. Assign colors only to the agents that are active
            # We pull the pre-defined colors for the active indices
            active_indices = np.where(active_mask)[0]
            colors = [self.agent_colors[i] for i in active_indices]
            self.agent_scatter.set_facecolors(colors)
        else:
            self.agent_scatter.set_offsets(np.empty((0, 2)))

        self.step_counter.set_text(f"Step: {int(state.step)}")
        self.active_counter.set_text(f"Active: {int(np.sum(active_mask))}")
        
        return self.agent_scatter, self.step_counter

    def animate(self, save_fname="traffic_colored.gif"):
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(self.state_seq),
            interval=self.interval, blit=False
        )
        if save_fname:
            ani.save(save_fname, writer='pillow', fps=10, 
                     savefig_kwargs={'facecolor': 'black'})
        plt.show()