import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.animation as animation
import matplotlib.markers as mmarkers
import numpy as np
from matplotlib import colormaps


class TrafficVisualizer:
    def __init__(self, env, state_seq, interval=150):
        self.env = env
        self.state_seq = state_seq
        self.grid_size = env.grid_size
        self.interval = interval
        
        # Dynamic lane indices to match environment logic
        self.mid_low = self.grid_size // 2 - 1
        self.mid_high = self.grid_size // 2
        
        self.cmap = colormaps.get_cmap('tab20')
        self.agent_colors = [self.cmap(i % 20) for i in range(self.env.num_agents)]

        # Generate the octagon once during initialization
        angles = np.linspace(0, 2*np.pi, 9) + (np.pi / 8)
        self.octagon_path = mpath.Path(np.column_stack([np.cos(angles), np.sin(angles)]))

        # Pre-cache the default triangle path too
        self.triangle_path = mmarkers.MarkerStyle('^').get_path()
        
        # CALCULATE SCALING FACTOR 
        # Baseline is grid_size=14. If grid doubles (28), area becomes 1/4th ((14/28)^2).
        self.scale_factor = (14 / self.grid_size) ** 2

        self.fig, self.ax = plt.subplots(figsize=(7, 7), facecolor='black')
        self.init_render()

    def init_render(self):
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        
        # 1. DRAW INDIVIDUAL GRID LINES FOR EACH SQUARE
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.grid(which='minor', color='#333333', lw=0.8, zorder=0)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # 2. DYNAMIC ROAD RENDERING
        # Road width is 2 tiles; it starts at mid_low - 0.5 to center the lanes
        road_start = self.mid_low - 0.5
        
        # Vertical Road
        self.ax.add_patch(plt.Rectangle((road_start, -0.5), 2, self.grid_size, color='#FFFFFF', zorder=1))
        # Horizontal Road
        self.ax.add_patch(plt.Rectangle((-0.5, road_start), self.grid_size, 2, color='#FFFFFF', zorder=1))

        # 3. DYNAMIC CENTER LINES
        # The center line sits exactly between mid_low and mid_high
        center_line_pos = self.mid_low + 0.5
        lane_style = {'color': '#B8860B', 'linestyle': '--', 'linewidth': 2, 'zorder': 2}
        
        # Vertical center line
        self.ax.plot([center_line_pos, center_line_pos], [-0.5, self.grid_size - 0.5], **lane_style)
        # Horizontal center line
        self.ax.plot([-0.5, self.grid_size - 0.5], [center_line_pos, center_line_pos], **lane_style)

        # 4. Initialize Individual Agent Artists
        self.agent_artists = []
        for i in range(self.env.num_agents):
            # Scaled initial size (350 * scale)
            artist = self.ax.scatter([], [], s=350 * self.scale_factor, marker='^', 
                                     facecolor=self.agent_colors[i],
                                     edgecolors='black', linewidths=1.5, zorder=10)
            self.agent_artists.append(artist)
        
        self.step_counter = self.ax.text(0, self.grid_size - 0.5, "Step: 0", 
                                         color='white', fontweight='bold', va="bottom")
        self.active_counter = self.ax.text(self.grid_size - 0.5, self.grid_size - 0.5, "Cars: 0", 
                                           color='white', fontweight='bold', va="bottom", ha="right")

    def update(self, frame):
        state = self.state_seq[frame]
        all_pos = np.array(state.p_pos)
        on_grid = (all_pos[:, 0] >= 0) & (all_pos[:, 0] < self.grid_size) & \
                  (all_pos[:, 1] >= 0) & (all_pos[:, 1] < self.grid_size)
        render_mask = np.array(state.active == 1) & on_grid
        
        for i, artist in enumerate(self.agent_artists):
            if render_mask[i]:
                curr_pos = all_pos[i][::-1]
                artist.set_offsets(curr_pos)
                
                # --- DYNAMIC SIZING ---
                # Apply scale factor to your preferred baselines (400 and 150)
                tri_size = 400 * self.scale_factor
                stop_size = 150 * self.scale_factor
                
                if frame > 0:
                    prev_pos = np.array(self.state_seq[frame-1].p_pos[i][::-1])
                    move_vec = curr_pos - prev_pos
                    
                    if np.all(move_vec == 0):
                        # Just grab the pre-made shape from __init__
                        artist.set_paths([self.octagon_path])
                        artist.set_sizes([stop_size]) # Uses scaled size
                    else:
                        # --- MOVING TRIANGLE ---
                        angle = np.degrees(np.arctan2(move_vec[1], move_vec[0])) - 90
                        t = mmarkers.MarkerStyle('^') 
                        transform = t.get_transform().rotate_deg(angle)
                        rotated_path = self.triangle_path.transformed(transform)
                        artist.set_paths([rotated_path])
                        artist.set_sizes([tri_size])
                else:
                    artist.set_paths([mmarkers.MarkerStyle('^').get_path()])
                    artist.set_sizes([tri_size])
            else:
                artist.set_offsets(np.empty((0, 2)))

        self.step_counter.set_text(f"Step: {int(state.step)}")
        self.active_counter.set_text(f"Active: {int(np.sum(render_mask))}")
        return self.agent_artists + [self.step_counter, self.active_counter]

    def animate(self, save_fname="traffic_colored.gif"):
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(self.state_seq),
            interval=self.interval, blit=False
        )
        if save_fname:
            ani.save(save_fname, writer='pillow', fps=10, 
                     savefig_kwargs={'facecolor': 'black'})
        plt.show()