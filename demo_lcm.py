"""
RTMsim-Py LCM demo: VARI process with pressure-dependent porosity.

Compares RTM (i_model=1, rigid preform, smooth VOF) to VARI without
flow distribution medium (i_model=3, compactable preform, two-fluid
surrogate EOS, hard VOF cutoff). Same plate, same stack geometry,
same injection pressure.

Per-ply quadratic porosity law:

    phi(p) = porosity + ((porosity_at_p1 - porosity) / p1**2) * p**2

so phi rises from a compressed baseline at p=0 to a relaxed value at
p=p1. The element-effective porosity is the thickness-weighted mean
of the per-ply quadratics (still a single quadratic per cell).

Outputs (in output_lcm/):
  curve.png       phi(p) law for the chosen plies
  fill_rtm.png    8-panel fill sequence for i_model=1
  fill_vari.png   8-panel fill sequence for i_model=3
  porosity.png    porosity field at mid-fill (i_model=3)
  thickness.png   compacted thickness field at mid-fill (i_model=3)
  compare.png     RTM vs VARI fill-front at three stages
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
# Porosity quadratic phi(p) = PHI_0 + ((PHI_1 - PHI_0)/P1**2) * p**2.
# P1 is set to the injection pressure so PHI_1 is literally the porosity
# the preform reaches under the bag-vs-fluid balance at full inlet
# pressure. The quadratic stays monotone and well below 1 throughout the
# operating pressure range.
PHI_0 = 0.50          # compressed by bag at p = 0
PHI_1 = 0.60          # decompacted at p = P1
P1 = P_INLET          # reference pressure for PHI_1 (= injection pressure)
T_PLY = 0.75e-3
K1 = 3e-10
K2_OVER_K1 = 0.2


def make_lcm_ply(deg):
    th = np.deg2rad(deg)
    return rtm.PlyProperties(
        thickness=T_PLY,
        porosity=PHI_0,
        K1=K1, K2=K1 * K2_OVER_K1,
        refdir=np.array([np.cos(th), np.sin(th), 0.0]),
        porosity_at_p1=PHI_1, p1=P1,
    )


def build_mesh_demo(side=0.30, n_div=20, inlet_radius=0.02):
    mesh = rtm.make_square_plate(side=side, n_div=n_div)
    rtm.assign_patch_by_disk(mesh, 0, (0.0, 0.0), inlet_radius)
    return mesh


def run_model(mesh, i_model, label):
    plies = [make_lcm_ply(a) for a in PLY_ANGLES]
    stack = rtm.LaminateStack(plies=plies)
    params = rtm.SimParameters(
        i_model=i_model,
        tmax=200.0, mu_resin=0.10,
        p_inlet=P_INLET, p_init=P_INIT,
        rho_air=1.225, rho_resin=960.0,
        patch_types=[rtm.PATCH_INLET, rtm.PATCH_IGNORE,
                     rtm.PATCH_IGNORE, rtm.PATCH_IGNORE],
        n_pics=20,
        stack=stack,
    )
    print(f'  {label} (i_model={i_model})')
    t0 = time.time()
    snaps = rtm.run_filling(mesh, params)
    print(f'    -> {time.time() - t0:.1f}s, {len(snaps)} snapshots')
    return snaps


# ---------- plotting ----------
def plot_phi_curve(outpath):
    p = np.linspace(0.0, 2.5e5, 200)
    c = (PHI_1 - PHI_0) / P1 ** 2
    phi = PHI_0 + c * p ** 2
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(p / 1e5, phi, '-', lw=2.0, color='crimson',
            label=f'phi(p) = {PHI_0:.2f} + c · p²,  c = {c:.2e} / Pa²')
    ax.axhline(PHI_0, color='gray', ls=':', lw=0.8,
               label=f'phi0 = {PHI_0:.2f} (compressed)')
    ax.axhline(PHI_1, color='gray', ls='--', lw=0.8,
               label=f'phi(p1 = {P1/1e5:.1f} bar) = {PHI_1:.2f}')
    ax.axvline(P_INLET / 1e5, color='steelblue', ls=':', lw=0.8,
               label=f'p_inlet = {P_INLET/1e5:.1f} bar')
    ax.set_xlabel('local fluid pressure  p  [bar]')
    ax.set_ylabel('effective porosity  phi(p)')
    ax.set_title('VARI compactable preform: porosity vs pressure')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _interior_mask(snap):
    return (snap.celltype == 1) | (snap.celltype == -3)


def _set_axis(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-0.155, 0.155)
    ax.set_ylim(-0.155, 0.155)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_fill_panel(ax, mesh, snap, title=None):
    gamma = snap.gamma.copy()
    gamma[snap.celltype == -1] = 1.0
    tpc = ax.tripcolor(
        mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cellgridid,
        facecolors=gamma, cmap='RdYlBu_r',
        vmin=0.0, vmax=1.0, edgecolors='none',
    )
    _set_axis(ax)
    if title:
        mean_g = snap.gamma[_interior_mask(snap)].mean()
        ax.set_title(f'{title}\nt = {snap.t:.1f} s   mean γ = {mean_g:.2f}',
                     fontsize=10)
    return tpc


def plot_sequence(mesh, snaps, suptitle, outpath):
    n_show = min(8, len(snaps))
    indices = np.linspace(0, len(snaps) - 1, n_show, dtype=int)
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(3.25 * cols, 3.4 * rows + 0.4))
    axes = np.atleast_2d(axes)
    flat = list(axes.flat)
    tpc_last = None
    for k, idx in enumerate(indices):
        ax = flat[k]
        tpc_last = plot_fill_panel(ax, mesh, snaps[idx], f'snap {idx}')
    for ax in flat[len(indices):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    if tpc_last is not None:
        cbar = fig.colorbar(tpc_last, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.03)
        cbar.set_label('filling fraction γ')
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def pick_mid_snap(snaps, target=0.5):
    best, best_diff = None, 1e9
    for s in snaps:
        gi = s.gamma[_interior_mask(s)]
        if gi.size == 0:
            continue
        d = abs(gi.mean() - target)
        if d < best_diff:
            best, best_diff = s, d
    return best


def plot_field(mesh, snap, scalar, vmin, vmax, cmap, label, title, outpath):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    tpc = ax.tripcolor(
        mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cellgridid,
        facecolors=scalar, cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolors='none',
    )
    _set_axis(ax)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(tpc, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(mesh, snaps_rtm, snaps_vari, outpath):
    targets = [0.20, 0.45, 0.75]
    rows = [('RTM (i_model=1)', snaps_rtm), ('VARI (i_model=3)', snaps_vari)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    tpc_last = None
    for r, (label, snaps) in enumerate(rows):
        for c, tgt in enumerate(targets):
            best = min(snaps, key=lambda s: abs(
                s.gamma[_interior_mask(s)].mean() - tgt))
            ax = axes[r, c]
            tpc_last = plot_fill_panel(ax, mesh, best,
                                       title=f'target γ ≈ {tgt:.2f}')
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
                ax.yaxis.set_label_coords(-0.05, 0.5)
    fig.suptitle('RTM vs VARI: fill-front at matched fill levels', fontsize=12)
    if tpc_last is not None:
        cbar = fig.colorbar(tpc_last, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.03)
        cbar.set_label('filling fraction γ')
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def report(label, snaps):
    last = snaps[-1]
    interior = _interior_mask(last)
    mean_g = last.gamma[interior].mean()
    print(f'  {label}: t_final = {last.t:.1f} s,  mean gamma = {mean_g:.3f}')
    if last.porosity is not None:
        phi_min = last.porosity[interior].min()
        phi_max = last.porosity[interior].max()
        t_min = last.thickness[interior].min() * 1e3
        t_max = last.thickness[interior].max() * 1e3
        baseline_mm = 4 * T_PLY * 1e3
        print(f'         phi range: {phi_min:.3f} .. {phi_max:.3f}')
        print(f'         thickness range: {t_min:.2f} .. {t_max:.2f} mm  '
              f'(baseline {baseline_mm:.2f} mm)')


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    mesh = build_mesh_demo(side=0.30, n_div=20)
    print(f'Mesh: {mesh.N} cells')
    print(f'Stack: {len(PLY_ANGLES)} plies, baseline phi = {PHI_0}, '
          f'phi at {P1/1e5:.0f} bar = {PHI_1}')

    plot_phi_curve(os.path.join(outdir, 'curve.png'))

    print('\nRunning RTM (i_model=1)...')
    snaps_rtm = run_model(mesh, 1, 'rigid RTM')
    print('Running VARI (i_model=3)...')
    snaps_vari = run_model(mesh, 3, 'compactable VARI')

    plot_sequence(mesh, snaps_rtm,
                  'RTM (i_model=1) — rigid preform',
                  os.path.join(outdir, 'fill_rtm.png'))
    plot_sequence(mesh, snaps_vari,
                  'VARI (i_model=3) — compactable preform',
                  os.path.join(outdir, 'fill_vari.png'))

    mid = pick_mid_snap(snaps_vari, target=0.5)
    if mid is not None and mid.porosity is not None:
        plot_field(mesh, mid, mid.porosity,
                   vmin=PHI_0, vmax=max(PHI_1, mid.porosity.max()),
                   cmap='viridis', label='effective porosity φ(p)',
                   title=f'VARI porosity field at t = {mid.t:.1f} s',
                   outpath=os.path.join(outdir, 'porosity.png'))
        t_mm = mid.thickness * 1e3
        plot_field(mesh, mid, t_mm,
                   vmin=t_mm.min(), vmax=t_mm.max(),
                   cmap='magma', label='compacted thickness  [mm]',
                   title=f'VARI thickness field at t = {mid.t:.1f} s',
                   outpath=os.path.join(outdir, 'thickness.png'))
    else:
        print('  (skipping porosity/thickness field plots — no mid snapshot)')

    plot_comparison(mesh, snaps_rtm, snaps_vari,
                    os.path.join(outdir, 'compare.png'))

    print('\n--- final state ---')
    report('RTM ', snaps_rtm)
    report('VARI', snaps_vari)
    print(f'\nSaved figures to {outdir}/')


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output_lcm')
    main(outdir)
