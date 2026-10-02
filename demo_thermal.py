"""
RTMsim-Py demo: thermal + cure coupling (1D through-thickness model).

Three runs on the same plate and stack:

  (A) iso-thermal baseline: no set_thermal() — reference
  (B) cold tool / hot resin: set_thermal(...), cure disabled
        → resin cools as it advances, viscosity climbs, fill front slows
  (C) hot iso-T + cure:      set_thermal(...) + enable_cure(),
        T=120°C uniform → cure proceeds during filling; alpha map shows
        more advanced cure where resin sat longest.

Outputs (in output_thermal/):
  fill_iso.png        baseline fill sequence
  fill_cold.png       cold-tool fill sequence
  fill_cure.png       hot+cure fill sequence
  T_field.png         T field at mid-fill for case (B)
  mu_field.png        mu field at mid-fill for case (B)
  alpha_field.png     alpha field at end-fill for case (C)
  history.png         mean gamma + min mu vs time, all three cases
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmsim as rtm


PLY_ANGLES = [0, 90, -45, 45]
P_INLET = 2.0e5
P_INIT = 1.0e5
T_PLY = 0.75e-3
K1 = 3e-10
K2_OVER_K1 = 0.2
PHI = 0.55


AIR_EOS = dict(p_ref=1.01325e5, rho_ref=1.225, gamma=1.4)

# Resin: constant mu for the iso-thermal run; Arrhenius/Castro-Macosko
# viscosity and Kamal-Sourour cure for the thermal runs. The viscosity
# model gives mu(T=333 K, alpha=0) ~ 0.10 Pa s, matching the baseline.
RESIN = (rtm.ResinMaterial('epoxy')
         .set_viscosity(0.10)
         .set_density(960.0)
         .set_thermal_properties(cp=1800.0, conductivity=0.2)
         .set_viscosity_model(mu_inf=2.0e-6, E_mu=3.0e4, mu_max=1.0e3)
         .set_gel_model(alpha_gel=0.6, C1=1.5, C2=1.0)
         .set_cure_kinetics(H_total=350.0e3, A1=2.0e3, A2=2.0e5,
                            E1=5.0e4, E2=6.0e4, m=0.5, n=1.5,
                            alpha_init=0.0))
FABRIC = (rtm.FabricMaterial('glass fabric')
          .set_permeability(K1, K1 * K2_OVER_K1)
          .set_porosity(PHI)
          .set_thermal_properties(density=2540.0, cp=800.0, conductivity=1.0))


def fibre_dir(deg):
    th = np.deg2rad(deg)
    return np.array([np.cos(th), np.sin(th), 0.0])


def build_mesh(side=0.30, n_div=20):
    return rtm.ShellMesh.make_square_plate(side=side, n_div=n_div)


def run_case(mesh, label, thermal=None, cure=False, inlet_radius=0.02):
    """thermal: None (iso-thermal) or dict(T_init, T_inlet, T_tool, h_tool)."""
    stack = rtm.LaminateStack()
    for a in PLY_ANGLES:
        stack.add_ply(FABRIC, T_PLY, fibre_dir(a))
    sim = (rtm.RTMSimulation()
           .set_mesh(mesh)
           .set_process_model(1)
           .set_resin(RESIN)
           .set_laminate(stack)
           .set_pressures(p_inlet=P_INLET, p_init=P_INIT)
           .set_air_eos(**AIR_EOS)
           .set_run_control(tmax=200.0, n_pics=20)
           .add_injection_port((0.0, 0.0, 0.0), radius=inlet_radius))
    if thermal is not None:
        sim.set_thermal(**thermal).enable_cure(cure)
    print(f'  {label} ...')
    t0 = time.time()
    snaps = sim.run()
    print(f'    -> {time.time() - t0:.1f}s, {len(snaps)} snapshots, '
          f't_final={snaps[-1].t:.1f}s')
    return snaps


# ---------- plotting ----------
def _interior(snap):
    return (snap.celltype == rtm.CELL_INTERIOR) | (snap.celltype == rtm.CELL_WALL)


def _setup_axis(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-0.155, 0.155)
    ax.set_ylim(-0.155, 0.155)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_fill_panel(ax, mesh, snap, title=None):
    gamma = snap.gamma.copy()
    gamma[snap.celltype == rtm.CELL_INLET] = 1.0
    tpc = ax.tripcolor(
        mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cellgridid,
        facecolors=gamma, cmap='RdYlBu_r',
        vmin=0.0, vmax=1.0, edgecolors='none',
    )
    _setup_axis(ax)
    if title is not None:
        mean_g = snap.gamma[_interior(snap)].mean()
        ax.set_title(f'{title}\nt = {snap.t:.1f} s   γ = {mean_g:.2f}',
                     fontsize=9)
    return tpc


def plot_sequence(mesh, snaps, suptitle, outpath):
    n_show = min(8, len(snaps))
    indices = np.linspace(0, len(snaps) - 1, n_show, dtype=int)
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows + 0.4))
    axes = np.atleast_2d(axes)
    flat = list(axes.flat)
    tpc_last = None
    for k, idx in enumerate(indices):
        tpc_last = plot_fill_panel(flat[k], mesh, snaps[idx], f'snap {idx}')
    for ax in flat[n_show:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    if tpc_last is not None:
        cbar = fig.colorbar(tpc_last, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.03)
        cbar.set_label('filling fraction γ')
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def pick_mid(snaps, target=0.5):
    best, best_d = None, 1e9
    for s in snaps:
        gi = s.gamma[_interior(s)]
        if gi.size == 0:
            continue
        d = abs(gi.mean() - target)
        if d < best_d:
            best, best_d = s, d
    return best


def plot_field(mesh, snap, scalar, vmin, vmax, cmap, label, title, outpath,
               mask_dry=False):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    plot = scalar.copy()
    if mask_dry:
        # gray-out cells with negligible resin to highlight wetted region
        dry = snap.gamma < 0.05
        plot = np.ma.array(plot, mask=dry & _interior(snap))
    tpc = ax.tripcolor(
        mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cellgridid,
        facecolors=plot, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='none',
    )
    _setup_axis(ax)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(tpc, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_history(snaps_dict, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    palette = {'iso-thermal': 'C7', 'cold tool / hot resin': 'C0',
               'hot iso-T + cure': 'C3'}
    for label, snaps in snaps_dict.items():
        col = palette.get(label, 'k')
        ts = [s.t for s in snaps]
        gs = [s.gamma[_interior(s)].mean() for s in snaps]
        ax1.plot(ts, gs, '-o', ms=3, lw=1.4, color=col, label=label)
        if snaps[0].mu is not None:
            mus = [s.mu[_interior(s) & (s.gamma > 0.05)].mean()
                   if (_interior(s) & (s.gamma > 0.05)).any() else np.nan
                   for s in snaps]
            ax2.plot(ts, mus, '-o', ms=3, lw=1.4, color=col, label=label)
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('mean filling fraction γ')
    ax1.set_title('Fill progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, frameon=False)
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('mean μ over wetted cells [Pa·s]')
    ax2.set_title('Resin viscosity vs time')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    mesh = build_mesh()
    print(f'Mesh: {mesh.N} cells')

    snaps_iso = run_case(mesh, 'iso-thermal baseline')

    snaps_cold = run_case(
        mesh, 'cold-tool / hot resin',
        thermal=dict(T_init=298.15, T_inlet=353.15, T_tool=298.15,
                     h_tool=500.0),
    )

    snaps_cure = run_case(
        mesh, 'hot iso-T + cure',
        thermal=dict(T_init=393.15, T_inlet=393.15, T_tool=393.15,
                     h_tool=200.0),
        cure=True,
    )

    plot_sequence(mesh, snaps_iso,
                  'Iso-thermal baseline (mu = 0.10 Pa s)',
                  os.path.join(outdir, 'fill_iso.png'))
    plot_sequence(mesh, snaps_cold,
                  'Cold tool 25C / hot resin 80C — viscosity rises near front',
                  os.path.join(outdir, 'fill_cold.png'))
    plot_sequence(mesh, snaps_cure,
                  'Hot iso-T 120C + cure — alpha grows during filling',
                  os.path.join(outdir, 'fill_cure.png'))

    mid_cold = pick_mid(snaps_cold, target=0.45)
    if mid_cold is not None and mid_cold.T is not None:
        plot_field(mesh, mid_cold, mid_cold.T - 273.15,
                   vmin=25.0, vmax=80.0, cmap='inferno',
                   label='temperature [°C]',
                   title=f'Cold-tool case: T at t = {mid_cold.t:.1f} s',
                   outpath=os.path.join(outdir, 'T_field.png'))
        plot_field(mesh, mid_cold, mid_cold.mu,
                   vmin=mid_cold.mu[_interior(mid_cold) & (mid_cold.gamma > 0.05)].min(),
                   vmax=mid_cold.mu[_interior(mid_cold) & (mid_cold.gamma > 0.05)].max(),
                   cmap='viridis', label='resin viscosity μ [Pa·s]',
                   title=f'Cold-tool case: μ(T) at t = {mid_cold.t:.1f} s',
                   outpath=os.path.join(outdir, 'mu_field.png'),
                   mask_dry=True)

    last_cure = snaps_cure[-1]
    if last_cure.alpha is not None:
        a_int = last_cure.alpha[_interior(last_cure) & (last_cure.gamma > 0.05)]
        if a_int.size > 0:
            plot_field(mesh, last_cure, last_cure.alpha,
                       vmin=float(a_int.min()), vmax=float(a_int.max()),
                       cmap='magma', label='cure conversion α',
                       title=f'Hot+cure case: α at t = {last_cure.t:.1f} s '
                             f'(end of fill)',
                       outpath=os.path.join(outdir, 'alpha_field.png'),
                       mask_dry=True)

    plot_history(
        {'iso-thermal': snaps_iso,
         'cold tool / hot resin': snaps_cold,
         'hot iso-T + cure': snaps_cure},
        os.path.join(outdir, 'history.png'),
    )

    # ---- summary -----------------------------------------------------------
    # ASCII-only to keep Windows cp1252 console happy.
    print('\n--- summary ---')
    for label, snaps in [('iso',  snaps_iso),
                         ('cold', snaps_cold),
                         ('cure', snaps_cure)]:
        last = snaps[-1]
        fluid = _interior(last)
        msg = (f'  {label}: t={last.t:6.1f}s, '
               f'mean gamma={last.gamma[fluid].mean():.3f}')
        if last.T is not None:
            msg += (f', T={last.T[fluid].min()-273.15:5.1f}..'
                    f'{last.T[fluid].max()-273.15:5.1f} C')
        if last.alpha is not None:
            msg += (f', alpha={last.alpha[fluid].min():.4f}..'
                    f'{last.alpha[fluid].max():.4f}')
        print(msg)
    print(f'\nSaved figures to {outdir}/')


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output_thermal')
    main(outdir)
