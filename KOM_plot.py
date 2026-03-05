import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotter

def plot_results(file_name):
    # Configure a slick, professional plotting style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')  # Fallback style

    # Fine-tune the aesthetics - significantly increased sizes
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,              # Base font size
        'axes.titlesize': 20,         # Much bigger titles
        'axes.labelsize': 20,         # Bigger axis labels
        # 'axes.titleweight': 'bold',   # Bold titles
        'xtick.labelsize': 14,        # Bigger x-axis ticks
        'ytick.labelsize': 14,        # Bigger y-axis ticks
        'legend.fontsize': 16,        # Bigger legend text
        'legend.title_fontsize': 16,  # Bigger legend title
        'legend.frameon': True,
        'legend.framealpha': 0.95,    # Opaque background for readability
        'legend.fancybox': True,      # Rounded corners
        'legend.loc': 'best',         # Automatically choose the best location
        'lines.linewidth': 3,         # Thicker lines
        'grid.alpha': 0.4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    data_loaded = json.loads(Path(file_name).read_text())
    R_range = np.array(data_loaded["R_range"])
    dt = data_loaded["dt"]
    number_of_lines = data_loaded["number_of_lines"]
    set_point = data_loaded["set_point"]
    plot_data = data_loaded["plot_data_1d"]
    power_data = data_loaded["power_data"]
    num_steps = data_loaded["num_steps"]

    # --- Use the Plotter Class ---
    plotter_instance = plotter.ComparisonPlotter(R_range, num_steps, model_name="KOM")

    # Plot spatial data
    plotter_instance.plot_spatial_panel((0, 0), "Radial Stress (Cauchy)", "Stress", plot_data["radial_stress"])
    plotter_instance.plot_spatial_panel((1, 0), "Hoop Stress (Cauchy)"  , "Stress", plot_data["hoop_stress"])
    plotter_instance.plot_spatial_panel((0, 1), "Radial Strain"         , "Strain", plot_data["radial_strain"], set_point = set_point)
    plotter_instance.plot_spatial_panel((1, 1), "Hoop Strain"           , "Strain", plot_data["hoop_strain"], set_point = set_point)
    plotter_instance.plot_spatial_panel((0, 2), "Radial Growth"         , "Growth", plot_data["radial_growth"])
    plotter_instance.plot_spatial_panel((1, 2), "Hoop Growth"           , "Growth", plot_data["hoop_growth"])

    # Finalize and save
    plotter_instance.finalize_and_save("ODE_KOM_results.png")

    # --- Create separate stimulus plot ---
    stimulus_plotter = plotter.ComparisonPlotter(R_range, num_steps, model_name="KOM", grid_shape=(2, 1))

    stimulus_plotter.plot_spatial_panel((0, 0), "Radial Stimulus (sr)", "Stimulus", plot_data["sr"])
    stimulus_plotter.plot_spatial_panel((1, 0), "Hoop Stimulus (sf)", "Stimulus", plot_data["sf"])

    stimulus_plotter.finalize_and_save("ODE_KOM_stimulus.png")

    # --- Create sigmoid components plot for HOOP (fiber) growth ---
    fig_hoop, axes_hoop = plt.subplots(2, 2, figsize=(14, 10))
    fig_hoop.suptitle("KOM Model - Hoop (Fiber) Growth Sigmoid Components", fontsize=18)
    
    colors = plt.cm.viridis(np.linspace(0, 1, number_of_lines))
    time_labels = [f"t = {int(i * num_steps / (number_of_lines - 1)) * dt:.0f}" for i in range(number_of_lines)]
    
    # Growth term (limits based on current gt)
    ax = axes_hoop[0, 0]
    for i, (data_line, color, label) in enumerate(zip(plot_data["growth_term_gt"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel("Growth Term")
    ax.set_title(r"Growth Limiter: $\frac{1}{1 + e^{f_{slope}(g_\theta - g_{max})}}$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Strain term (based on stimulus sf)
    ax = axes_hoop[0, 1]
    for i, (data_line, color, label) in enumerate(zip(plot_data["strain_term_gt"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel("Strain Term")
    ax.set_title(r"Strain Sigmoid: $\frac{f_{ff,max}}{1 + e^{-f_f(s_f - s_{f,50})}}$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Growth increment F_inc_t
    ax = axes_hoop[1, 0]
    for i, (data_line, color, label) in enumerate(zip(plot_data["F_inc_t"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel("F_inc_t")
    ax.set_title(r"Growth Increment: $F_{inc,\theta} = \tau \cdot growth \cdot strain + 1$")
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No growth')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Hoop growth gt
    ax = axes_hoop[1, 1]
    for i, (data_line, color, label) in enumerate(zip(plot_data["hoop_growth"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel(r"$g_\theta$")
    ax.set_title("Cumulative Hoop Growth")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("ODE_KOM_hoop_sigmoids.png")
    plt.close(fig_hoop)

    # --- Create sigmoid components plot for RADIAL (cross-fiber) growth ---
    fig_rad, axes_rad = plt.subplots(2, 2, figsize=(14, 10))
    fig_rad.suptitle("KOM Model - Radial (Cross-Fiber) Growth Sigmoid Components", fontsize=18)
    
    # Growth term (limits based on current gr)
    ax = axes_rad[0, 0]
    for i, (data_line, color, label) in enumerate(zip(plot_data["growth_term_gr"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel("Growth Term")
    ax.set_title(r"Growth Limiter: $\frac{1}{1 + e^{c_{slope}(g_r - g_{max})}}$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Strain term (based on stimulus sr)
    ax = axes_rad[0, 1]
    for i, (data_line, color, label) in enumerate(zip(plot_data["strain_term_gr"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel("Strain Term")
    ax.set_title(r"Strain Sigmoid: $\frac{f_{cc,max}}{1 + e^{-c_c(s_r - s_{c,50})}}$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Growth increment F_inc_r
    ax = axes_rad[1, 0]
    for i, (data_line, color, label) in enumerate(zip(plot_data["F_inc_r"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel("F_inc_r")
    ax.set_title(r"Growth Increment: $F_{inc,r} = \sqrt{\tau \cdot growth \cdot strain + 1}$")
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No growth')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Radial growth gr
    ax = axes_rad[1, 1]
    for i, (data_line, color, label) in enumerate(zip(plot_data["radial_growth"], colors, time_labels)):
        ax.plot(R_range, data_line, color=color, label=label)
    ax.set_xlabel("R (Reference)")
    ax.set_ylabel(r"$g_r$")
    ax.set_title("Cumulative Radial Growth")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("ODE_KOM_radial_sigmoids.png")
    plt.close(fig_rad)
    
    print("Saved: ODE_KOM_results.png, ODE_KOM_stimulus.png, ODE_KOM_hoop_sigmoids.png, ODE_KOM_radial_sigmoids.png")

if __name__ == "__main__":
    plot_results("KOM_ODE_data.json")