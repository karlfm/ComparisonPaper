import numpy as np
from scipy.optimize import brentq
import time, sys, os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
class VariationalCylinderSolver:
 
    def __init__(self, R, gs, gf, mu, bc, w=1.0, set_point=0.0, gamma=2.0, gMax=1.5):
        self.R  = np.asarray(R, dtype=float)
        self.gs = np.asarray(gs, dtype=float)
        self.gf = np.asarray(gf, dtype=float)
        self.mu = float(mu)
        self.bc = float(bc)       # current inner-wall pressure
        self.w  = float(w)        # dissipation parameter
        self.N  = len(R)
        self.dR = np.diff(R)
        self.set_point = float(set_point)
        self.gamma = float(gamma)
        self.gMax = float(gMax)
 
    # Kinematics
    def compute_r(self, ri):
        """
        r^2(R) = ri^2 + 2 * ∫_{Ri}^{R} gamma_s(s) gamma_f(s) s ds
        integrated with the trapezoidal rule (explicit accumulation, no cumsum).
        """
        f = self.gs * self.gf * self.R

        I = np.zeros(self.N, dtype=float)  # I[i] = ∫_{Ri}^{R_i} f(s) ds
        for i in range(1, self.N):
            ds = self.R[i] - self.R[i - 1]
            I[i] = I[i - 1] + 0.5 * (f[i - 1] + f[i]) * ds

        return np.sqrt(ri**2 + 2.0 * I)

    def compute_p(self, r):
        """
        Compute pressure by integrating:
        p(R) = p(R_i) + ∫ μ(2sγf²/r² + 2s²γf/r² · ∂γf/∂s
                        - s³γsγf³/r⁴ - γs/(sγf)) ds
        with
        p(R_i) = μ (R_i^2 / r(R_i)^2) γf(R_i)^2 - σ_rr(R_i).
        """
        s = self.R

        # dγf/ds
        dgf_ds = np.gradient(self.gf, s)

        # Integrand values at grid points
        integrand = self.mu * (
            2.0 * s * self.gf**2 / r**2
            + 2.0 * s**2 * self.gf * dgf_ds / r**2
            - s**3 * self.gs * self.gf**3 / r**4
            - self.gs / (s * self.gf)
        )

        # Inner boundary value p(R_i)
        Ri = s[0]
        rRi = r[0]
        gfRi = self.gf[0]
        sigma_rr_Ri = -self.bc
        p_i = self.mu * (Ri**2 / rRi**2) * gfRi**2 - sigma_rr_Ri

        # Trapezoidal integration without cumsum
        p = np.zeros(self.N, dtype=float)
        p[0] = p_i
        for i in range(1, self.N):
            ds = s[i] - s[i - 1]
            p[i] = p[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * ds

        return p

    # PK1 stress
    def PK1_radial(self, r, p):
        """P^{rR} = −μ R γf/(r γs) + p r/(R γs γf)"""
        R, gs, gf, mu = self.R, self.gs, self.gf, self.mu
        return mu * R * gf / (r * gs) - p * r / (R * gs * gf)
 
    # Cauchy stress
    def cauchy_radial(self, r, p):
        """σ^{rr} = p − μ αs²"""
        a_s2 = (self.R * self.gf / r)**2
        return (self.mu * a_s2 - p)
 
    def cauchy_hoop(self, r, p):
        """σ^{θθ} = p − μ αf²"""
        a_f2 = (r / (self.R * self.gf))**2
        return (self.mu * a_f2 - p)
 
    # Root finding: P^{rR}(Ro) = 0
    def solve(self):
        def obj(ri):
            r = self.compute_r(ri)
            p = self.compute_p(r)
            # return self.PK1_radial(r, p)[-1]
            return self.cauchy_radial(r, p)[-1]
        
        lo, hi = 0.3 * self.R[0], 8.0 * self.R[-1]
        pts = np.linspace(lo, hi, 100)
        vals = np.array([obj(x) for x in pts])
 
        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k+1]):
                if vals[k] > 0 and vals[k+1] < 0:
                    ri = brentq(obj, pts[k], pts[k+1], xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p
 
        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k+1]):
                if vals[k] * vals[k+1] < 0:
                    ri = brentq(obj, pts[k], pts[k+1], xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p
 
        raise RuntimeError("No root found")
 
    # Growth rates
    def growth_rate_r(self, r, p):
        stress_term = self.cauchy_hoop(r, p)
        dgt =  (stress_term - self.set_point)

        return dgt
    
    # Time stepping
    def run(self, dt, max_steps=10000, tol=1e-4, print_every=500, return_history=False):
        cur = self
        history = [cur] if return_history else None
        for i in range(1, max_steps + 1):
            _, r, p = cur.solve()
            # ds = np.zeros_like(cur.growth_rate_s(r, p))
            dr = cur.growth_rate_r(r, p)
            mr = np.max(np.abs(dr))
            if mr < tol:
                print(f"  converged at step {i}, max|dγ/dt|={mr:.2e}")
                if return_history:
                    return cur, history
                return cur
            cur = VariationalCylinderSolver(
                self.R, cur.gs, cur.gf * (1 + dt * dr),
                self.mu, self.bc, w=self.w
            )
            if return_history:
                history.append(cur)
            if print_every and i % print_every == 0:
                print(f"  step {i:>5d}  γs=[{cur.gs.min():.4f},{cur.gs.max():.4f}]"
                      f"  γf=[{cur.gf.min():.4f},{cur.gf.max():.4f}]"
                      f"  max|dγ/dt|={mr:.2e}")
        print(f"  did not converge after {max_steps} steps, max|dγ/dt|={mr:.2e}")
        if return_history:
            return cur, history
        return cur
 

#  Run
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    R  = np.linspace(1.0, 2.0, 16)
    N  = len(R)
    mu = 1.0
    p = 0.15   # normotensive
    set_point = mu*0.1
    gamma = 2.0
    gMax = 1.5
 
    # 2. Instant elastic response after pressure change (before growth)
    pressurized = VariationalCylinderSolver(R, np.ones(N), np.ones(N), mu, p, w=1.0, set_point=set_point, gamma=gamma, gMax=gMax)

    # 3. Grow from the pressurized state
    final, history = pressurized.run(dt=0.0001, max_steps=10000, tol=1e-3, return_history=True)
    _, r_f, p_f = final.solve()

    # 8 evenly spaced growth states including first and last
    n_steps = 8
    if len(history) >= 2:
        raw_idx = np.linspace(0, len(history) - 1, num=min(n_steps, len(history)))
        sampled_idx = np.unique(np.round(raw_idx).astype(int))
    else:
        sampled_idx = np.array([0], dtype=int)
    sampled_states = [history[i] for i in sampled_idx]
 
    # 5. Plot (plotter.py style)
    sampled_data = []

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14, 'axes.titlesize': 20, 'axes.labelsize': 20,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 16,
        'lines.linewidth': 3, 'grid.alpha': 0.4,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    n_s = len(sampled_states)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='col')
    fig.suptitle("LT", fontsize=24, fontweight='bold')

    for j, state in enumerate(sampled_states):
        _, r_m, p_m = state.solve()
        alpha = (j + 1) / (n_s + 1)
        axes[0, 0].plot(R, state.cauchy_radial(r_m, p_m), color='C0', alpha=alpha)
        axes[1, 0].plot(R, state.cauchy_hoop(r_m, p_m),   color='C0', alpha=alpha)
        axes[0, 1].plot(R, R * state.gf / r_m,            color='C0', alpha=alpha)
        axes[1, 1].plot(R, r_m / (R * state.gf),          color='C0', alpha=alpha)
        axes[0, 2].plot(R, state.gs,                       color='C0', alpha=alpha)
        axes[1, 2].plot(R, state.gf,                       color='C0', alpha=alpha)

        # Homeostatic γf from eq (27)
        gf_homeo = (r_m / R) * np.sqrt(mu / (set_point + p_m))
        axes[1, 2].plot(R, gf_homeo, color='C1', ls='--', lw=2, alpha=0.5)

        sampled_data.append({
            "sample_number": int(j + 1),
            "history_index": int(sampled_idx[j]),
            "R": R.tolist(), "r": r_m.tolist(),
            "sigma_rr": state.cauchy_radial(r_m, p_m).tolist(),
            "sigma_tt": state.cauchy_hoop(r_m, p_m).tolist(),
            "alpha_s": (R * state.gf / r_m).tolist(),
            "alpha_f": (r_m / (R * state.gf)).tolist(),
            "gamma_s": state.gs.tolist(),
            "gamma_f": state.gf.tolist(),
            "J": (state.gs * state.gf).tolist(),
        })

    axes[1, 0].axhline(set_point, color='C1', linestyle='--', label='Set Point')
    axes[0, 0].set(ylabel="Stress", title="Radial Stress (Cauchy)")
    axes[1, 0].set(ylabel="Stress", title="Hoop Stress (Cauchy)")
    axes[0, 1].set(ylabel="Stretch", title="Radial Stretch")
    axes[1, 1].set(ylabel="Stretch", title="Hoop Stretch")
    axes[0, 2].set(ylabel="Growth", title="Radial Growth")
    axes[1, 2].set(ylabel="Growth", title="Hoop Growth")
    for ax in axes.flat:
        ax.grid(True)
    for ax in axes[1]:
        ax.set_xlabel("R")
    handles = [plt.Line2D([0],[0], color='C0', lw=2), plt.Line2D([0],[0], color='C1', lw=2, linestyle='--')]
    fig.legend(handles, ['Solution', 'Set-Point'], loc='lower center',
               bbox_to_anchor=(0.52, -0.06), ncol=2, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_DIR, "LT.png"))
    plt.close(fig)
    print(f"Saved LT.png")

    # --- Pre-calculate all data for plotting
    print("--- Pre-calculating data for plots ---")
    plot_data_1d = {
        "radial_stress": [], "hoop_stress": [], "radial_stretch": [],
        "hoop_stretch": [], "radial_growth": [], "hoop_growth": [], "displacement": [],
    }

    for state in sampled_states:
        _, r_m, p_m = state.solve()
        plot_data_1d["radial_stress"].append((state.cauchy_radial(r_m, p_m)).tolist())
        plot_data_1d["hoop_stress"].append((state.cauchy_hoop(r_m, p_m)).tolist())
        plot_data_1d["radial_stretch"].append((R * state.gf / r_m).tolist())
        plot_data_1d["hoop_stretch"].append((r_m / (R * state.gf)).tolist())
        plot_data_1d["radial_growth"].append(state.gs.tolist())
        plot_data_1d["hoop_growth"].append(state.gf.tolist())
        plot_data_1d["displacement"].append(r_m.tolist())

    data = {
        "plot_data_1d": plot_data_1d,
        "R_range": R.tolist(),
        "mu": float(mu),
        "p": float(p),
        "number_of_lines": len(sampled_states),
    }

    with open(os.path.join(OUTPUT_DIR, "LT_data.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Original radiuss: ri = {R[0]:.6f} mm, ro = {R[-1]:.6f} mm")
    print(f"  Final radii: ri = {r_f[0]:.6f} mm, ro = {r_f[-1]:.6f} mm")
    print(f" Net growth (integral over domain): J(Ri) = {np.trapezoid(final.gs * final.gf, final.R):.6f}")
    
    export_data = {
        "metadata": {
            "mu": float(mu),
            "p": float(p),
            "dt": 0.0001,
            "max_steps": 20000,
            "tol": 1e-4,
            "n_grid": int(N),
            "n_history_states": int(len(history)),
            "n_sampled_states": int(len(sampled_states)),
        },
        "R": R.tolist(),
        "r_final": r_f.tolist(),
        "sampled_indices": sampled_idx.tolist(),
        "states": sampled_data,
    }
    with open(os.path.join(OUTPUT_DIR, "LT.json"), "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print("Saved plot.")