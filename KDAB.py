import numpy as np
from scipy.optimize import brentq
import os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class VariationalCylinderSolver:
    """
    Shared mechanics solver for a pressurised incompressible neo-Hookean
    cylinder with multiplicative growth  F = A G.

    Growth fields gs(R), gf(R) are stored as arrays on the reference grid R.
    """

    def __init__(self, R, gs, gf, mu, bc):
        self.R  = np.asarray(R, dtype=float)
        self.gs = np.asarray(gs, dtype=float)
        self.gf = np.asarray(gf, dtype=float)
        self.mu = float(mu)
        self.bc = float(bc)
        self.N  = len(R)

    # Kinematics
    def compute_r(self, ri):
        """r²(R) = ri² + 2 ∫_{Ri}^{R} γs γf s ds"""
        f = self.gs * self.gf * self.R
        I = np.zeros(self.N, dtype=float)
        for i in range(1, self.N):
            ds = self.R[i] - self.R[i - 1]
            I[i] = I[i - 1] + 0.5 * (f[i - 1] + f[i]) * ds
        val = ri**2 + 2.0 * I
        val = np.maximum(val, 0.0)
        return np.sqrt(val)

    def compute_p(self, r):
        """Lagrange multiplier from the momentum balance integral (eq A5)."""
        s = self.R
        dgf_ds = np.gradient(self.gf, s)
        integrand = self.mu * (
            2.0 * s * self.gf**2 / r**2
            + 2.0 * s**2 * self.gf * dgf_ds / r**2
            - s**3 * self.gs * self.gf**3 / r**4
            - self.gs / (s * self.gf)
        )
        Ri, rRi, gfRi = s[0], r[0], self.gf[0]
        p_i = self.mu * (Ri**2 / rRi**2) * gfRi**2 + self.bc   # σ_rr(Ri) = -bc

        p = np.zeros(self.N, dtype=float)
        p[0] = p_i
        for i in range(1, self.N):
            ds = s[i] - s[i - 1]
            p[i] = p[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * ds
        return p

    # Cauchy stress
    def cauchy_radial(self, r, p):
        """σ^{rr} = μ αs² − p"""
        return self.mu * (self.R * self.gf / r)**2 - p

    def cauchy_hoop(self, r, p):
        """σ^{θθ} = μ αf² − p"""
        return self.mu * (r / (self.R * self.gf))**2 - p

    # Elastic stretches
    def alpha_s(self, r):
        return self.R * self.gf / r

    def alpha_f(self, r):
        return r / (self.R * self.gf)

    # Root finding: σ_rr(Ro) = 0
    def solve(self):
        def obj(ri):
            r = self.compute_r(ri)
            p = self.compute_p(r)
            return self.cauchy_radial(r, p)[-1]

        lo, hi = 0.3 * self.R[0], 20.0 * self.R[-1]
        pts = np.linspace(lo, hi, 200)
        vals = np.array([obj(x) for x in pts])

        # Physical root: σ_rr(Ro) goes neg → pos
        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k+1]):
                if vals[k] < 0 and vals[k+1] > 0:
                    ri = brentq(obj, pts[k], pts[k+1], xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p

        # Fallback: any sign change
        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k+1]):
                if vals[k] * vals[k+1] < 0:
                    ri = brentq(obj, pts[k], pts[k+1], xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p

        raise RuntimeError("No root found")


# ══════════════════════════════════════════════════════════════════
#  KDAB growth law  (eq 22)
#
#  Isotropic:  γ_s = γ_f = γ
#  Driver:     fiber stretch  α_f = r / (R γ_f)
#  Update:     γ^{i+1} = γ^i · [ β (α_f − 1 − s_hom) + 1 ]^{1/2}
#  Eq. cond:   γ_f = r / ( R (1 + s_hom) )            (eq 23)
# ══════════════════════════════════════════════════════════════════
def run_kdab(R, mu, bc, beta, s_hom, max_steps=2500, tol=1e-3,
             print_every=500, return_history=True):
    """
    Run the KDAB isotropic growth law on the cylinder.
    Returns (final_solver, history_list).
    """
    N = len(R)
    gamma = np.ones(N)          # γ_s = γ_f = γ  (isotropic)

    cur = VariationalCylinderSolver(R, gamma, gamma, mu, bc)
    history = [cur]

    for i in range(1, max_steps + 1):
        _, r, p = cur.solve()

        # Fiber stretch  α_f = r / (R γ_f)
        af = cur.alpha_f(r)

        # Multiplicative growth factor 
        factor = beta * (af - 1.0 - s_hom) + 1.0
        factor = np.maximum(factor, 0.0)       # guard negative sqrt
        mult = np.sqrt(factor)

        new_gamma = cur.gf * mult               # isotropic: gs = gf
        dgamma = np.max(np.abs(new_gamma - cur.gf))

        cur = VariationalCylinderSolver(R, new_gamma, new_gamma, mu, bc)
        history.append(cur)

        if print_every and i % print_every == 0:
            print(f"  step {i:>5d}  γ=[{cur.gf.min():.4f},{cur.gf.max():.4f}]"
                  f"  max|Δγ|={dgamma:.2e}")

    print(f"  did not converge after {max_steps} steps, max|Δγ|={dgamma:.2e}")
    return cur, history


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Parameters
    R   = np.linspace(1.0, 2.0, 65)
    N   = len(R)
    mu  = 1.0
    bc  = 0.15 * mu                       # inner pressure = 0.15 μ
    beta  = 0.01
    s_hom = np.sqrt(11.0 / 10.0) - 1.0

    stretch_set_point = 1.0 + s_hom        # = sqrt(11/10)
    n_steps = 2500

    print(f"KDAB model:  β={beta}, s_hom={s_hom:.5f}, "
          f"stretch set-point={stretch_set_point:.5f}")
    print(f"  bc={bc}, μ={mu}, grid N={N}, steps={n_steps}")

    # Run
    final, history = run_kdab(R, mu, bc, beta, s_hom,
                              max_steps=n_steps, print_every=500)
    _, r_f, p_f = final.solve()

    # Sample 8 evenly-spaced states
    n_lines = 8
    if len(history) >= 2:
        raw_idx = np.linspace(0, len(history) - 1, num=min(n_lines, len(history)))
        sampled_idx = np.unique(np.round(raw_idx).astype(int))
    else:
        sampled_idx = np.array([0], dtype=int)
    sampled = [history[i] for i in sampled_idx]

    # Plot
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

    n_s = len(sampled)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='col')
    fig.suptitle("KDAB", fontsize=24, fontweight='bold')

    for j, state in enumerate(sampled):
        _, r_m, p_m = state.solve()
        alpha = (j + 1) / (n_s + 1)
        axes[0, 0].plot(R, state.cauchy_radial(r_m, p_m), color='C0', alpha=alpha)
        axes[1, 0].plot(R, state.cauchy_hoop(r_m, p_m),   color='C0', alpha=alpha)
        axes[0, 1].plot(R, state.alpha_s(r_m),             color='C0', alpha=alpha)
        axes[1, 1].plot(R, state.alpha_f(r_m),             color='C0', alpha=alpha)
        axes[0, 2].plot(R, state.gs,                       color='C0', alpha=alpha)
        axes[1, 2].plot(R, state.gf,                       color='C0', alpha=alpha)
        gamma_homeo = r_m / (R * stretch_set_point)
        axes[0, 2].plot(R, gamma_homeo, color='C1', ls='--', lw=2, alpha=0.5)
        axes[1, 2].plot(R, gamma_homeo, color='C1', ls='--', lw=2, alpha=0.5)

    axes[1, 1].axhline(stretch_set_point, color='C1', linestyle='--', label='Set Point')
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
    fig.savefig(os.path.join(OUTPUT_DIR, "KDAB_growth.png"))
    plt.close(fig)
    print(f"Saved KDAB_growth.png")

    # Summary
    print(f"\n  Original radii:  Ri={R[0]:.4f},  Ro={R[-1]:.4f}")
    print(f"  Final radii:     ri={r_f[0]:.6f},  ro={r_f[-1]:.6f}")
    print(f"  Final γ range:   [{final.gf.min():.6f}, {final.gf.max():.6f}]")

    # Export JSON
    plot_data_1d = {
        "radial_stress": [], "hoop_stress": [], "radial_stretch": [],
        "hoop_stretch": [], "radial_growth": [], "hoop_growth": [],
        "displacement": [],
    }
    for state in sampled:
        _, r_m, p_m = state.solve()
        plot_data_1d["radial_stress"].append(state.cauchy_radial(r_m, p_m).tolist())
        plot_data_1d["hoop_stress"].append(state.cauchy_hoop(r_m, p_m).tolist())
        plot_data_1d["radial_stretch"].append(state.alpha_s(r_m).tolist())
        plot_data_1d["hoop_stretch"].append(state.alpha_f(r_m).tolist())
        plot_data_1d["radial_growth"].append(state.gs.tolist())
        plot_data_1d["hoop_growth"].append(state.gf.tolist())
        plot_data_1d["displacement"].append(r_m.tolist())

    data = {
        "plot_data_1d": plot_data_1d,
        "R_range": R.tolist(),
        "mu": mu, "bc": bc, "beta": beta, "s_hom": s_hom,
        "number_of_lines": len(sampled),
    }
    with open(os.path.join(OUTPUT_DIR, "KDAB_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("Saved KDAB_data.json")