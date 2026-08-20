import numpy as np
from scipy.optimize import brentq
import os, json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class VariationalCylinderSolver:
    """
    Shared mechanics solver for a pressurised incompressible neo-Hookean
    cylinder with multiplicative growth  F = A G.
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
        f = self.gs * self.gf * self.R
        I = np.zeros(self.N, dtype=float)
        for i in range(1, self.N):
            ds = self.R[i] - self.R[i - 1]
            I[i] = I[i - 1] + 0.5 * (f[i - 1] + f[i]) * ds
        return np.sqrt(np.maximum(ri**2 + 2.0 * I, 0.0))

    def compute_p(self, r):
        s = self.R
        dgf_ds = np.gradient(self.gf, s)
        integrand = self.mu * (
            2.0 * s * self.gf**2 / r**2
            + 2.0 * s**2 * self.gf * dgf_ds / r**2
            - s**3 * self.gs * self.gf**3 / r**4
            - self.gs / (s * self.gf)
        )
        Ri, rRi, gfRi = s[0], r[0], self.gf[0]
        p_i = self.mu * (Ri**2 / rRi**2) * gfRi**2 + self.bc

        p = np.zeros(self.N, dtype=float)
        p[0] = p_i
        for i in range(1, self.N):
            ds = s[i] - s[i - 1]
            p[i] = p[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * ds
        return p

    # Cauchy stress
    def cauchy_radial(self, r, p):
        return self.mu * (self.R * self.gf / r)**2 - p

    def cauchy_hoop(self, r, p):
        return self.mu * (r / (self.R * self.gf))**2 - p

    # Mandel stress trace  tr(M) = σ_rr + σ_θθ = μ(αs² + αf²) − 2p
    def mandel_trace(self, r, p):
        a_s2 = (self.R * self.gf / r)**2
        a_f2 = (r / (self.R * self.gf))**2
        return self.mu * (a_s2 + a_f2) - 2.0 * p

    # Elastic stretches
    def alpha_s(self, r):
        return self.R * self.gf / r

    def alpha_f(self, r):
        return r / (self.R * self.gf)

    # Root finding: σ_rr(Ro) = 0
    def solve(self, ri_guess=None):
        def obj(ri):
            r = self.compute_r(ri)
            p = self.compute_p(r)
            return self.cauchy_radial(r, p)[-1]

        # Warm start
        if ri_guess is not None and ri_guess > 0:
            delta = 0.3 * ri_guess
            lo_g, hi_g = max(0.1, ri_guess - delta), ri_guess + delta
            try:
                v_lo, v_hi = obj(lo_g), obj(hi_g)
                if np.isfinite(v_lo) and np.isfinite(v_hi) and v_lo * v_hi < 0:
                    ri = brentq(obj, lo_g, hi_g, xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p
            except Exception:
                pass

        lo, hi = 0.3 * self.R[0], 20.0 * self.R[-1]
        pts = np.linspace(lo, hi, 100)
        vals = np.array([obj(x) for x in pts])

        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k+1]):
                if vals[k] < 0 and vals[k+1] > 0:
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


# ══════════════════════════════════════════════════════════════════
#  GAPK stress-based growth law  (eq 25)
#
#  γ_s^{i+1} = γ_s^i + (1/τ) · L(γ_s)^ν · (tr(M) − p^crit)
#  γ_f^{i+1} = γ_f^i                        (no hoop growth)
#
#  where  L(γ_s) = (γ*_s − γ_s) / (γ*_s − 1)
#         tr(M)  = μ(αs² + αf²) − 2p
#
#  Equilibrium:  γ_s = γ*_s   or   tr(M) = p^crit
# ══════════════════════════════════════════════════════════════════
def run_gapk_stress(R, mu, bc, inv_tau, nu, p_crit, gamma_star_s,
                    max_steps=3000, tol=1e-6, print_every=500):
    N = len(R)
    gs = np.ones(N)
    gf = np.ones(N)

    cur = VariationalCylinderSolver(R, gs, gf, mu, bc)
    history = [cur]
    ri_prev = None

    for i in range(1, max_steps + 1):
        ri_prev, r, p = cur.solve(ri_guess=ri_prev)

        # Mandel stress trace
        trM = cur.mandel_trace(r, p)

        # Growth limiter  L = ((γ*_s − γ_s) / (γ*_s − 1))^ν
        limiter = ((gamma_star_s - cur.gs) / (gamma_star_s - 1.0)) ** nu

        # Additive update for radial growth (eq 25)
        delta_gs = inv_tau * limiter * (trM - p_crit)
        new_gs = cur.gs + delta_gs

        dgamma = np.max(np.abs(delta_gs))
        cur = VariationalCylinderSolver(R, new_gs, gf, mu, bc)
        history.append(cur)

        if dgamma < tol:
            print(f"  converged at step {i}, max|Δγs|={dgamma:.2e}")
            return cur, history

        if print_every and i % print_every == 0:
            print(f"  step {i:>5d}  γs=[{cur.gs.min():.4f},{cur.gs.max():.4f}]"
                  f"  max|Δγs|={dgamma:.2e}")

    print(f"  did not converge after {max_steps} steps, max|Δγs|={dgamma:.2e}")
    return cur, history


# ══════════════════════════════════════════════════════════════════
#  Main — reproduce paper Figure 5
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Paper parameters
    R   = np.linspace(1.0, 2.0, 65)
    N   = len(R)
    mu  = 1.0
    bc  = 0.15 * mu

    inv_tau      = 0.01
    nu           = 2.0
    p_crit       = mu / 10.0         # Mandel stress set point
    gamma_star_s = 1.5               # radial growth limit
    n_steps      = 3000

    print(f"GAPK stress:  1/τ={inv_tau}, ν={nu}, "
          f"p^crit={p_crit}, γ*_s={gamma_star_s}")
    print(f"  bc={bc}, μ={mu}, grid N={N}, steps={n_steps}")

    # Run
    final, history = run_gapk_stress(
        R, mu, bc, inv_tau, nu, p_crit, gamma_star_s,
        max_steps=n_steps, print_every=500
    )
    _, r_f, p_f = final.solve()

    # Sample 8 evenly-spaced states
    n_lines = 8
    raw_idx = np.linspace(0, len(history) - 1, num=min(n_lines, len(history)))
    sampled_idx = np.unique(np.round(raw_idx).astype(int))
    sampled = [history[i] for i in sampled_idx]

    #  Plot (plotter.py style, custom GridSpec layout)
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

    from matplotlib.gridspec import GridSpec

    n_s = len(sampled)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("GAPK (stress based)", fontsize=24, fontweight='bold', y=0.98)

    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.30,
                  left=0.06, right=0.98, top=0.90, bottom=0.12)

    # Left: Mandel trace centered vertically (middle two rows)
    ax_mandel = fig.add_subplot(gs[1:3, 0])

    # Middle column: stretches (top half / bottom half)
    ax_rs = fig.add_subplot(gs[0:2, 1])
    ax_hs = fig.add_subplot(gs[2:4, 1], sharex=ax_rs)

    # Right column: growth
    ax_rg = fig.add_subplot(gs[0:2, 2])
    ax_hg = fig.add_subplot(gs[2:4, 2], sharex=ax_rg)

    for j, state in enumerate(sampled):
        _, r_m, p_m = state.solve()
        alpha = (j + 1) / (n_s + 1)

        ax_mandel.plot(R, state.mandel_trace(r_m, p_m), color='C0', alpha=alpha)
        ax_rs.plot(R, state.alpha_s(r_m),                color='C0', alpha=alpha)
        ax_hs.plot(R, state.alpha_f(r_m),                color='C0', alpha=alpha)
        ax_rg.plot(R, state.gs,                          color='C0', alpha=alpha)
        ax_hg.plot(R, state.gf,                          color='C0', alpha=alpha)

    # Set-point / limit lines
    ax_mandel.axhline(p_crit, color='C1', linestyle='--', label='Set Point')
    ax_rg.axhline(gamma_star_s, color='C1', linestyle='--', label='Set Point')

    # Labels
    ax_mandel.set(ylabel="Stress", xlabel="$R$", title="Trace of Mandel Stress")
    ax_rs.set(ylabel="Stretch", title="Radial Stretch")
    ax_hs.set(ylabel="Stretch", xlabel="$R$", title="Hoop Stretch")
    ax_rg.set(ylabel="Growth", title="Radial Growth")
    ax_hg.set(ylabel="Growth", xlabel="$R$", title="Hoop Growth")

    for ax in [ax_mandel, ax_rs, ax_hs, ax_rg, ax_hg]:
        ax.grid(True)

    handles = [
        plt.Line2D([0], [0], color='C0', lw=2),
        plt.Line2D([0], [0], color='C1', lw=2, linestyle='--'),
    ]
    fig.legend(handles, ['Solution', 'Set-Point/Growth Limit'], loc='lower center',
               bbox_to_anchor=(0.52, 0.01), ncol=2, framealpha=0.9)

    fig.savefig(os.path.join(OUTPUT_DIR, "GAPK_stress_growth.png"),
                bbox_inches=None, pad_inches=0.1)
    plt.close(fig)
    print(f"Saved GAPK_stress_growth.png")

    # Summary
    _, r_f, p_f = final.solve()
    print(f"\n  Original radii:  Ri={R[0]:.4f},  Ro={R[-1]:.4f}")
    print(f"  Final radii:     ri={r_f[0]:.6f},  ro={r_f[-1]:.6f}")
    print(f"  Final γs range:  [{final.gs.min():.6f}, {final.gs.max():.6f}]")
    print(f"  Final tr(M):     [{final.mandel_trace(r_f, p_f).min():.6f}, "
          f"{final.mandel_trace(r_f, p_f).max():.6f}]")

    # Export JSON
    plot_data_1d = {
        "mandel_trace": [], "radial_stress": [], "hoop_stress": [],
        "radial_stretch": [], "hoop_stretch": [],
        "radial_growth": [], "hoop_growth": [], "displacement": [],
    }
    for state in sampled:
        _, r_m, p_m = state.solve()
        plot_data_1d["mandel_trace"].append(state.mandel_trace(r_m, p_m).tolist())
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
        "mu": mu, "bc": bc,
        "inv_tau": inv_tau, "nu": nu,
        "p_crit": p_crit, "gamma_star_s": gamma_star_s,
        "number_of_lines": len(sampled),
    }
    with open(os.path.join(OUTPUT_DIR, "GAPK_stress_ODE_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("Saved GAPK_stress_ODE_data.json")