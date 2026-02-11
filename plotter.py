import matplotlib.pyplot as plt
import numpy as np

class ComparisonPlotter:
    """
    A class to handle the plotting of 1D vs 2D simulation results.
    """
    def __init__(self, r_range, num_steps, model_name="Model", grid_shape=(2, 3)):
        self.r_range = r_range
        self.num_steps = num_steps
        self.grid_shape = grid_shape
        
        # Adaptive figure size based on grid shape
        fig_width = 6 * grid_shape[1]
        fig_height = 5 * grid_shape[0]
        self.fig, axs = plt.subplots(*grid_shape, figsize=(fig_width, fig_height), sharex='col')
        
        # Normalize axs to always be a 2D array for consistent indexing
        if grid_shape == (1, 1):
            self.axs = np.array([[axs]])
        elif grid_shape[0] == 1:
            self.axs = axs.reshape(1, -1)
        elif grid_shape[1] == 1:
            self.axs = axs.reshape(-1, 1)
        else:
            self.axs = axs
            
        self.fig.suptitle(f"{model_name}", fontsize=24, fontweight='bold')

    def _get_ax(self, ax_coords):
        """Get axis with consistent indexing regardless of grid shape."""
        return self.axs[ax_coords[0], ax_coords[1]]

    def plot_spatial_panel(self, ax_coords, title, ylabel, data_1d, data_2d_hist=None, set_point=None):
        """Plots a standard 1D vs 2D comparison over the radius."""
        ax = self._get_ax(ax_coords)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("R")
        ax.grid(True)

        if set_point is not None:
            if np.isscalar(set_point):
                ax.axhline(set_point, color='C1', linestyle='--', label='Set Point')
            else:
                ax.plot(self.r_range, set_point, color='C1', linestyle='--', label='Set Point')

        for i in range(len(data_1d)):
            alpha = (i + 1) / (len(data_1d) + 1)
            ax.plot(self.r_range, data_1d[i], color='C0', alpha=alpha)

            if data_2d_hist is not None and i < len(data_2d_hist):
                ax.plot(self.r_range, data_2d_hist[i], color='C1', linestyle='--', alpha=alpha)

    def plot_ricci(self, ax_coords, data_1d):
        """Plots the 1D Ricci scalar."""
        ax = self._get_ax(ax_coords)
        ax.set_title("Ricci Curvature (1D)")
        ax.set_ylabel("Ricci Scalar")
        ax.set_xlabel("R")
        ax.grid(True)
        for i in range(len(data_1d)):
            ax.plot(self.r_range, data_1d[i], color='C0', alpha=(i + 1) / (len(data_1d) + 1))

    def plot_dissipation(self, ax_coords, power_data, dt):
        """Plots the total dissipation over time."""
        ax = self._get_ax(ax_coords)
        ax.set_title("Total Dissipation (1D)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Dissipation Rate")
        ax.grid(True)

        num_data_points = len(power_data["entropy"])
        time_steps = np.arange(1, num_data_points + 1) * dt
        ax.plot(time_steps, power_data["entropy"], marker='o', linestyle='-', color='C0')

    def finalize_and_save(self, filename="comparison.png"):
        """Adds a legend and saves the figure."""
        handles = [plt.Line2D([0], [0], color='C0', lw=2), plt.Line2D([0], [0], color='C1', lw=2, linestyle='--')]
        labels = ['Solution', 'Set-Point/Homeostasis']
        self.fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.06), ncol=2, framealpha=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(filename)
        plt.close(self.fig)
        print(f"✓ Plot saved to {filename}")