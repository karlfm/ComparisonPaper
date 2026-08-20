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

    def cauchy_radial(self, r, p):
        return self.mu * (self.R * self.gf / r)**2 - p

    def cauchy_hoop(self, r, p):
        return self.mu * (r / (self.R * self.gf))**2 - p

    def alpha_s(self, r):
        return self.R * self.gf / r

    def alpha_f(self, r):
        return r / (self.R * self.gf)

    def green_lagrange_ss(self, r):
        """E_ss = ½(α_s² − 1)"""
        return 0.5 * (self.alpha_s(r)**2 - 1.0)

    def green_lagrange_ff(self, r):
        """E_ff = ½(α_f² − 1)"""
        return 0.5 * (self.alpha_f(r)**2 - 1.0)

    def solve(self, ri_guess=None):
        def obj(ri):
            r = self.compute_r(ri)
            p = self.compute_p(r)
            return self.cauchy_radial(r, p)[-1]

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
#  KOM growth law  (eq 9)
#
#  Strain-driven growth in BOTH radial and circumferential
#  directions via sigmoid functions with growth limits.
#
#  Stimuli:
#    s_s = E_ss − E*_ss       (radial Green-Lagrange strain)
#    s_f = E_ff − E*_ff       (fiber  Green-Lagrange strain)
#
#  Radial update (with √):
#    γ_s^{i+1} = γ_s^i · (sigmoid_term + 1)^{1/2}
#
#  Hoop update (no √):
#    γ_f^{i+1} = γ_f^i · (sigmoid_term + 1)
# ══════════════════════════════════════════════════════════════════
def run_kom(R, mu, bc, params, max_steps=3000, tol=1e-6, print_every=500):
    """
    Run the KOM growth law.  `params` is a dict with all KOM parameters.
    Returns (final_solver, history_list).
    """
    # Unpack parameters
    fff_max  = params["fff_max"]
    fcc_max  = params["fcc_max"]
    sf_star  = params["sf_star"]
    ss_star  = params["ss_star"]
    dt_kom   = params["dt"]
    ff       = params["ff"]
    cs       = params["cs"]
    f_slope  = params["f_slope"]
    c_slope  = params["c_slope"]
    gs_lim   = params["gamma_star_s"]
    gf_lim   = params["gamma_star_f"]
    Ess_star = params["Ess_star"]
    Eff_star = params["Eff_star"]

    N  = len(R)
    gs = np.ones(N)
    gf = np.ones(N)

    cur = VariationalCylinderSolver(R, gs, gf, mu, bc)
    history = [cur]
    ri_prev = None

    for i in range(1, max_steps + 1):
        ri_prev, r, p = cur.solve(ri_guess=ri_prev)

        # Green-Lagrange strains
        Ess = cur.green_lagrange_ss(r)
        Eff = cur.green_lagrange_ff(r)

        # Growth stimuli (Kerckhoffs eqs 5–6)
        # Fiber stimulus: s_l = E_ff − E*_ff
        sf = Eff - Eff_star
        # Cross-fiber stimulus: s_t = min(E_cross,max) − E*_ss
        # In plane strain E_zz = 0, so E_cross,max = max(E_ss, 0);
        # for a static loading min_over_cycle = that single value.
        # When E_ss < 0: max(E_ss, 0) = 0
        Ecross = np.minimum(Ess, 0.0)   # min(E_rr, E_zz=0)
        ss = Ecross - Ess_star

        # Radial growth (γ_s)
        k_cc = 1.0 / (1.0 + np.exp(f_slope * (cur.gs - gs_lim)))

        # Sigmoid growth rate — note (ss + ss_star) in negative branch
        rad_pos = k_cc * fcc_max * dt_kom / (1.0 + np.exp(-cs * (ss - ss_star)))
        rad_neg = -fcc_max * dt_kom / (1.0 + np.exp(cs * (ss + ss_star)))
        rad_term = np.where(ss >= 0, rad_pos, rad_neg)

        new_gs = cur.gs * np.sqrt(np.maximum(rad_term + 1.0, 0.0))

        # Hoop growth (γ_f)
        k_ff = 1.0 / (1.0 + np.exp(c_slope * (cur.gf - gf_lim)))

        # note (sf + sf_star) in negative branch
        hoop_pos = k_ff * fff_max * dt_kom / (1.0 + np.exp(-ff * (sf - sf_star)))
        hoop_neg = -fff_max * dt_kom / (1.0 + np.exp(ff * (sf + sf_star)))
        hoop_term = np.where(sf >= 0, hoop_pos, hoop_neg)

        new_gf = cur.gf * (hoop_term + 1.0)

        # Convergence check
        dgamma = max(np.max(np.abs(new_gs - cur.gs)),
                     np.max(np.abs(new_gf - cur.gf)))

        cur = VariationalCylinderSolver(R, new_gs, new_gf, mu, bc)
        history.append(cur)

        if dgamma < tol:
            print(f"  converged at step {i}, max|Δγ|={dgamma:.2e}")
            return cur, history

        if print_every and i % print_every == 0:
            print(f"  step {i:>5d}  γs=[{cur.gs.min():.4f},{cur.gs.max():.4f}]"
                  f"  γf=[{cur.gf.min():.4f},{cur.gf.max():.4f}]"
                  f"  max|Δγ|={dgamma:.2e}")

    print(f"  did not converge after {max_steps} steps, max|Δγ|={dgamma:.2e}")
    return cur, history


#  Main — reproduce paper Figure 7
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Parameters
    R   = np.linspace(1.0, 2.0, 65)
    N   = len(R)
    mu  = 1.0
    bc  = 0.15 * mu

    params = dict(
        fff_max      = 0.3,
        fcc_max      = 0.1,
        sf_star      = 0.06,
        ss_star      = 0.07,
        dt           = 0.01,
        ff           = 150.0,
        cs           = 75.0,
        f_slope      = 40.0,
        c_slope      = 60.0,
        gamma_star_s = 1.5,
        gamma_star_f = 1.5,
        Ess_star     = -0.3,
        Eff_star     = 0.0,
    )
    n_steps = 3000

    print(f"KOM model:  {n_steps} steps, grid N={N}")
    print(f"  bc={bc}, μ={mu}")

    # Run
    final, history = run_kom(R, mu, bc, params,
                             max_steps=n_steps, print_every=500)
    _, r_f, p_f = final.solve()

    # Sample 8 evenly-spaced states
    n_lines = 8
    raw_idx = np.linspace(0, len(history) - 1, num=min(n_lines, len(history)))
    sampled_idx = np.unique(np.round(raw_idx).astype(int))
    sampled = [history[i] for i in sampled_idx]

    #  Plot
    import matplotlib.ticker as mticker

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    plt.rcParams.update({
        'font.family':          'sans-serif',
        'font.sans-serif':      ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':            14,
        'axes.titlesize':       20,
        'axes.labelsize':       20,
        'xtick.labelsize':      14,
        'ytick.labelsize':      14,
        'legend.fontsize':      16,
        'legend.title_fontsize': 16,
        'legend.frameon':       True,
        'legend.framealpha':    0.95,
        'legend.fancybox':      True,
        'lines.linewidth':      3,
        'grid.alpha':           0.4,
        'figure.dpi':           300,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
    })

    # ──Colour: single hue with alpha ramp (matching plotter.py) ──
    n_s = len(sampled)

    # ── Main 2×3 figure ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='col')
    fig.suptitle("KOM", fontsize=24, fontweight='bold')

    # Stretch set-points from Green-Lagrange set-points
    alpha_s_sp = np.sqrt(max(2.0 * params["Ess_star"] + 1.0, 0.0))
    alpha_f_sp = np.sqrt(max(2.0 * params["Eff_star"] + 1.0, 0.0))

    # Panel definitions: (row, col, title, ylabel, data_key, set_point)
    panels = [
        (0, 0, "Radial Stress (Cauchy)", "Stress",  "radial_stress",  None),
        (1, 0, "Hoop Stress (Cauchy)",   "Stress",  "hoop_stress",    None),
        (0, 1, "Radial Stretch",         "Stretch",  "radial_stretch", alpha_s_sp if alpha_s_sp > 0 else None),
        (1, 1, "Hoop Stretch",           "Stretch",  "hoop_stretch",   alpha_f_sp if alpha_f_sp > 0 else None),
        (0, 2, "Radial Growth",          "Growth",  "radial_growth", None),#,  params["gamma_star_s"]),
        (1, 2, "Hoop Growth",            "Growth",  "hoop_growth", None)#,    params["gamma_star_f"]),
    ]

    # Pre-compute all panel data
    panel_data = {
        "radial_stress": [], "hoop_stress": [],
        "radial_stretch": [], "hoop_stretch": [],
        "radial_growth": [], "hoop_growth": [],
    }
    for state in sampled:
        _, r_m, p_m = state.solve()
        panel_data["radial_stress"].append(state.cauchy_radial(r_m, p_m))
        panel_data["hoop_stress"].append(state.cauchy_hoop(r_m, p_m))
        panel_data["radial_stretch"].append(state.alpha_s(r_m))
        panel_data["hoop_stretch"].append(state.alpha_f(r_m))
        panel_data["radial_growth"].append(state.gs)
        panel_data["hoop_growth"].append(state.gf)

    for row, col, title, ylabel, key, sp in panels:
        ax = axes[row, col]
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("R")
        ax.grid(True)

        if sp is not None:
            ax.axhline(sp, color='C1', linestyle='--', label='Set Point')

        for i, d in enumerate(panel_data[key]):
            alpha = (i + 1) / (n_s + 1)
            ax.plot(R, d, color='C0', alpha=alpha)

    # Legend
    handles = [
        plt.Line2D([0], [0], color='C0', lw=2),
        plt.Line2D([0], [0], color='C1', lw=2, linestyle='--'),
    ]
    labels = ['Solution', 'Set-Point']
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.52, -0.06), ncol=2, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_DIR, "KOM_growth.png"))
    fig.savefig(os.path.join(OUTPUT_DIR, "KOM_growth.pdf"))
    plt.close(fig)
    print(f"Saved KOM_growth.png / .pdf")

    #  Residual stress (same style)
    res = VariationalCylinderSolver(R, final.gs, final.gf, mu, 0.0)
    ri_res, r_res, p_res = res.solve()
    srr_res = res.cauchy_radial(r_res, p_res)
    stt_res = res.cauchy_hoop(r_res, p_res)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("KOM — Residual Stress", fontsize=24, fontweight='bold')

    for ax, data, title, ylabel in zip(
            ax2,
            [srr_res, stt_res],
            ["Residual Radial Stress", "Residual Hoop Stress"],
            ["Stress", "Stress"]):
        ax.plot(R, data, color='C0')
        ax.set(title=title, ylabel=ylabel, xlabel="R")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(os.path.join(OUTPUT_DIR, "KOM_residual.png"))
    fig2.savefig(os.path.join(OUTPUT_DIR, "KOM_residual.pdf"))
    plt.close(fig2)
    print(f"Saved KOM_residual.png / .pdf")

    # Summary
    print(f"\n  Original radii:  Ri={R[0]:.4f},  Ro={R[-1]:.4f}")
    print(f"  Final radii:     ri={r_f[0]:.6f},  ro={r_f[-1]:.6f}")
    print(f"  Final γs range:  [{final.gs.min():.6f}, {final.gs.max():.6f}]")
    print(f"  Final γf range:  [{final.gf.min():.6f}, {final.gf.max():.6f}]")
    print(f"  Residual σ^rr:   [{srr_res.min():.6f}, {srr_res.max():.6f}]")
    print(f"  Residual σ^θθ:   [{stt_res.min():.6f}, {stt_res.max():.6f}]")

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
        "mu": mu, "bc": bc, "params": params,
        "number_of_lines": len(sampled),
    }
    with open(os.path.join(OUTPUT_DIR, "KOM_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("Saved KOM_data.json")