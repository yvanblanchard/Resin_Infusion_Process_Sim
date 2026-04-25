"""
RTMsim-Py demo: 4-ply stacked laminate (PCOMP-style).

Every element carries the full stack of 4 plies through the thickness;
each ply has its own fibre orientation. The per-element in-plane
permeability is the thickness-weighted sum of rotated ply tensors

    K_eff = (1/t_tot) sum_p R(theta_p) diag(K1, K2) R(theta_p)^T t_p

This is the standard LCM PCOMP shell convention (parallel flow under
the same pressure gradient => flow rates add).

We compare two stacks on the same plate:
  - quasi-isotropic [0/90/-45/+45]:  flow front should be roughly circular
  - unidirectional   [0/0/0/0]:       front extends faster along x

Outputs:
  layout.png      stack overview with ply orientations
  qiso.png        snapshots of [0/90/-45/+45] fill
  ud0.png         snapshots of [0/0/0/0] fill
  comparison.png  side-by-side mid-fill snapshot
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmsim as rtm


def make_ply(deg, t=0.75e-3, K1=3e-10, K2_over_K1=0.2, phi=0.70):
    th = np.deg2rad(deg)
    return rtm.PlyProperties(
        thickness=t, porosity=phi, K1=K1, K2=K1 * K2_over_K1,
        refdir=np.array([np.cos(th), np.sin(th), 0.0]),
    )


def build_mesh_demo(side=0.30, n_div=20, inlet_radius=0.02):
    mesh = rtm.make_square_plate(side=side, n_div=n_div)
    rtm.assign_patch_by_disk(mesh, 0, (0.0, 0.0), inlet_radius)
    return mesh


def run_stack(mesh, ply_angles, label):
    plies = [make_ply(a) for a in ply_angles]
    stack = rtm.LaminateStack(plies=plies)
    params = rtm.SimParameters(
        tmax=200.0, mu_resin=0.10, p_inlet=2.0e5, p_init=1.0e5,
        patch_types=[rtm.PATCH_INLET, rtm.PATCH_IGNORE,
                     rtm.PATCH_IGNORE, rtm.PATCH_IGNORE],
        n_pics=20,
        stack=stack,
    )
    print(f'  {label}: {len(plies)} plies, t_tot={stack.total_thickness*1e3:.2f}mm')
    t0 = time.time()
    snaps = rtm.run_filling(mesh, params)
    print(f'    -> {time.time()-t0:.1f}s, {len(snaps)} snapshots')
    return stack, snaps


# ---------- plotting ----------
def plot_stack_layout(stacks_info, outpath):
    """One panel per stack, each showing ply arrows in a fanned rosette."""
    fig, axes = plt.subplots(1, len(stacks_info), figsize=(4 * len(stacks_info), 4.2))
    if len(stacks_info) == 1:
        axes = [axes]
    for ax, (label, angles) in zip(axes, stacks_info):
        # draw plies as colored bars in a circular rose
        L = 1.0
        colors = plt.cm.tab10.colors
        for i, deg in enumerate(angles):
            th = np.deg2rad(deg)
            x = L * np.cos(th)
            y = L * np.sin(th)
            ax.plot([-x, x], [-y, y], '-', lw=2.5, color=colors[i % 10],
                    label=f'ply {i+1}: {deg:+.0f}°')
        circle = plt.Circle((0, 0), 1.05, color='gray', fill=False, lw=0.8)
        ax.add_patch(circle)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=11)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fontsize=8, frameon=False)
    fig.suptitle('Stacked laminate — fibre orientation per ply\n'
                 '(every element of the mesh carries this same stack through-thickness)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_fill_panel(ax, mesh, snap, title=None):
    nodes = mesh.nodes
    tris = mesh.cellgridid
    gamma = snap.gamma.copy()
    inlet = snap.celltype == -1
    display = np.where(inlet, 1.0, gamma)
    tpc = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], tris,
        facecolors=display, cmap='RdYlBu_r',
        vmin=0.0, vmax=1.0, edgecolors='none',
    )
    ax.set_aspect('equal')
    ax.set_xlim(-0.155, 0.155)
    ax.set_ylim(-0.155, 0.155)
    ax.set_xticks([])
    ax.set_yticks([])
    interior = snap.celltype != -1
    mean_g = gamma[interior].mean()
    if title is None:
        title = f't = {snap.t:.1f} s   mean γ = {mean_g:.2f}'
    else:
        title = f'{title}\nt = {snap.t:.1f} s   mean γ = {mean_g:.2f}'
    ax.set_title(title, fontsize=10)
    return tpc


def plot_sequence(mesh, snaps, suptitle, outpath):
    n_show = min(8, len(snaps))
    indices = np.linspace(0, len(snaps) - 1, n_show, dtype=int)
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.25 * cols, 3.4 * rows + 0.4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    tpc_last = None
    flat = axes.flat
    for k, idx in enumerate(indices):
        ax = next(flat)
        tpc_last = plot_fill_panel(ax, mesh, snaps[idx], f'snap {idx}')
    # turn off any unused axes
    for ax in flat:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    if tpc_last is not None:
        cbar = fig.colorbar(tpc_last, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.03)
        cbar.set_label('filling fraction γ')
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(mesh, snaps_list, labels, outpath):
    """Side-by-side at three stages: early, mid, late."""
    fig, axes = plt.subplots(len(snaps_list), 3, figsize=(11, 3.7 * len(snaps_list)))
    if len(snaps_list) == 1:
        axes = axes.reshape(1, -1)
    tpc_last = None
    for row, (snaps, label) in enumerate(zip(snaps_list, labels)):
        # pick early (~25%), mid (~50%), late (~80%) by mean gamma
        targets = [0.25, 0.50, 0.80]
        chosen = []
        for tgt in targets:
            best = min(snaps,
                       key=lambda s: abs(
                           s.gamma[(s.celltype == 1) | (s.celltype == -3)].mean() - tgt))
            chosen.append(best)
        for col, snap in enumerate(chosen):
            ax = axes[row, col]
            tpc_last = plot_fill_panel(ax, mesh, snap)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
                ax.yaxis.set_label_coords(-0.05, 0.5)
    fig.suptitle('Stacked-laminate fill — quasi-iso vs unidirectional',
                 fontsize=12)
    if tpc_last is not None:
        cbar = fig.colorbar(tpc_last, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.03)
        cbar.set_label('filling fraction γ')
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


def anisotropy_report(mesh, snaps, label):
    cc = mesh.cellcenter
    target = None
    for s in snaps:
        gi = s.gamma[(s.celltype == 1) | (s.celltype == -3)]
        if 0.30 < gi.mean() < 0.55:
            target = s
            break
    if target is None:
        return
    r = np.sqrt(cc[:, 0]**2 + cc[:, 1]**2)
    inner = r > 0.02
    along_x = inner & (np.abs(cc[:, 0]) > 2 * np.abs(cc[:, 1]))
    along_y = inner & (np.abs(cc[:, 1]) > 2 * np.abs(cc[:, 0]))
    diag45 = inner & (np.abs(np.abs(cc[:, 0]) - np.abs(cc[:, 1])) < 0.012)
    print(f'\n{label} at t = {target.t:.1f} s '
          f'(mean γ = {target.gamma[target.celltype != -1].mean():.2f}):')
    print(f'  along x:  γ = {target.gamma[along_x].mean():.3f}')
    print(f'  along y:  γ = {target.gamma[along_y].mean():.3f}')
    print(f'  diagonal: γ = {target.gamma[diag45].mean():.3f}')


def main(outdir):
    os.makedirs(outdir, exist_ok=True)

    mesh = build_mesh_demo(side=0.30, n_div=20)
    print(f'Mesh: {mesh.N} cells')

    layouts = [
        ('[0/90/-45/+45]\n(quasi-isotropic)', [0, 90, -45, 45]),
        ('[0/0/0/0]\n(unidirectional)', [0, 0, 0, 0]),
    ]
    plot_stack_layout(layouts, os.path.join(outdir, 'layout.png'))

    print('\nSolving (first call triggers JIT compile)...')
    stack_qi, snaps_qi = run_stack(mesh, [0, 90, -45, 45], '[0/90/-45/+45]')
    stack_ud, snaps_ud = run_stack(mesh, [0, 0, 0, 0],     '[0/0/0/0]')

    plot_sequence(mesh, snaps_qi,
                  'Quasi-isotropic stack [0/90/-45/+45]',
                  os.path.join(outdir, 'qiso.png'))
    plot_sequence(mesh, snaps_ud,
                  'Unidirectional stack [0/0/0/0]',
                  os.path.join(outdir, 'ud0.png'))
    plot_comparison(mesh, [snaps_qi, snaps_ud],
                    ['quasi-iso', 'unidirectional'],
                    os.path.join(outdir, 'comparison.png'))

    anisotropy_report(mesh, snaps_qi, 'quasi-isotropic')
    anisotropy_report(mesh, snaps_ud, 'unidirectional [0/0/0/0]')

    print(f'\nSaved figures to {outdir}/')


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_4ply')
    main(outdir)
