"""
RTMsim-Py: checks of UDMaterial (unidirectional fabric / slit tape).

  U1  fibre bed: Gebart's K1, K2 against values worked out by hand
      (R = 1, quadratic Vf 0.5, hexagonal Vf 0.6), porosity 1 - Vf, input
      validation
  U2  tape layup: no gap reproduces the fibre bed; rectangular duct
      permeability (slot a^2/12, square 0.0351 a^2); K1 grows and K2 grows
      with the gap, K1 >= K2; a wide gap tends to the gap channel alone
  U3  through the solver (i_model=4, 1-D channel): the front of a UD ply
      along / across the channel follows K1 / K2, and a [0/90] pair of
      equal plies behaves as K = (K1 + K2) / 2

  U4  strongly anisotropic tape (K1/K2 ~ 700) with the fibres across the
      channel, run to the end of fill with flux_scheme="monotone": the fill
      completes, the volume balance closes and the pressure stays between
      p_init and p_inlet in every snapshot (the default "lsq" scheme dips
      below the vent pressure there and leaves the last cells at the vent
      unfilled; its numbers are printed for comparison)

Run:  python verification_ud.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # repo root (rtmsim.py)
import rtmsim as rtm

RESULTS = []


def check(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")


def rel(a, b):
    return abs(a / b - 1.0)


def u1_fibre_bed():
    print("\n[U1] Fibre bed (Gebart)")
    # Worked out by hand for R = 1 m: K / R^2
    cases = [("quad", 0.5, 0.07017544, 0.01292294),
             ("hex", 0.6, 0.02683438, 0.00582479)]
    for packing, Vf, k1_ref, k2_ref in cases:
        m = rtm.UDMaterial.from_fibre_volume_fraction(Vf, 1.0, packing)
        k1, k2 = m.get_permeability()
        e = max(rel(k1, k1_ref), rel(k2, k2_ref))
        check(f"U1 {packing} Vf {Vf}", e < 1e-5 and rel(m.get_porosity(),
              1 - Vf) < 1e-12, f"K1 {k1:.5f}, K2 {k2:.6f} R^2 "
              f"(max rel. diff to hand values {e:.1e})")
    R = 3.5e-6
    a = rtm.UDMaterial.from_fibre_volume_fraction(0.5, 2 * R, "quad")
    b = rtm.UDMaterial.from_fibre_volume_fraction(0.5, R, "quad")
    check("U1 scales with R^2", rel(a.get_permeability()[0],
          4 * b.get_permeability()[0]) < 1e-12, "doubling R gives 4x K")
    ratios = [rtm.UDMaterial.from_fibre_volume_fraction(v, R)
              .get_permeability() for v in (0.3, 0.45, 0.6, 0.7)]
    check("U1 K falls with Vf, K1 > K2",
          all(x[0] > y[0] and x[1] > y[1] for x, y in zip(ratios, ratios[1:]))
          and all(k1 > k2 for k1, k2 in ratios), "Vf 0.3 .. 0.7")
    bad = []
    for kw in (dict(Vf=0.8, fibre_radius=R), dict(Vf=0.0, fibre_radius=R),
               dict(Vf=0.5, fibre_radius=-1.0),
               dict(Vf=0.5, fibre_radius=R, packing="x")):
        try:
            rtm.UDMaterial.from_fibre_volume_fraction(**kw)
            bad.append(kw)
        except ValueError:
            pass
    check("U1 rejects bad input", not bad, "Vf above packing limit, Vf 0, "
          "R < 0, unknown packing")


def u2_tape_layup():
    print("\n[U2] Tape layup")
    w, t, Vf, R = 6.35e-3, 0.15e-3, 0.55, 3.5e-6
    bed = rtm.UDMaterial.from_fibre_volume_fraction(Vf, R)
    m0 = rtm.UDMaterial.from_tape_layup(w, 0.0, t, Vf, R)
    e = max(rel(x, y) for x, y in zip(m0.get_permeability(),
                                      bed.get_permeability()))
    check("U2 no gap = fibre bed", e < 1e-12 and rel(m0.get_porosity(),
          1 - Vf) < 1e-12, f"max rel. diff {e:.1e}")
    slot = rtm._duct_permeability(1e-3, 1e-1)
    sq = rtm._duct_permeability(1e-3, 1e-3)
    check("U2 duct permeability", rel(slot, 1e-6 / 12) < 3e-2
          and rel(sq, 0.035144e-6) < 1e-3,
          f"slot {slot / 1e-6 * 12:.4f} a^2/12 (exact 1 - 0.0187 for b/a = "
          f"100), square {sq / 1e-6:.5f} a^2 (0.03514)")
    gaps = np.array([0.0, 0.05e-3, 0.1e-3, 0.3e-3, 1e-3])
    k = np.array([rtm.UDMaterial.from_tape_layup(w, g, t, Vf, R)
                  .get_permeability() for g in gaps])
    phi = [rtm.UDMaterial.from_tape_layup(w, g, t, Vf, R).get_porosity()
           for g in gaps]
    check("U2 gap raises K1, K2 and porosity",
          np.all(np.diff(k[:, 0]) > 0) and np.all(np.diff(k[:, 1]) > 0)
          and np.all(np.diff(phi) > 0) and np.all(k[:, 0] > k[:, 1]),
          "K1 " + ", ".join(f"{x:.2e}" for x in k[:, 0])
          + f"; K1/K2 {k[0, 0] / k[0, 1]:.1f} -> {k[-1, 0] / k[-1, 1]:.1f}")
    # Wide gap: K1 -> the channel's share of the area
    g = 100e-3
    m = rtm.UDMaterial.from_tape_layup(w, g, t, Vf, R)
    k1 = m.get_permeability()[0]
    k1_ch = g / (w + g) * rtm._duct_permeability(g, t)
    check("U2 wide gap -> channel", rel(k1, k1_ch) < 0.02,
          f"K1 {k1:.3e} vs channel {k1_ch:.3e}")
    check("U2 model inputs kept", m.get_model_inputs()["gap_width"] == g
          and rtm.UDMaterial("x").get_model_inputs() is None, "")
    try:
        rtm.UDMaterial.from_tape_layup(w, -1e-3, t, Vf, R)
        ok = False
    except ValueError:
        ok = True
    check("U2 rejects a negative gap", ok, "")


def grid_mesh(nx, ny, L, W):
    u, v = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1),
                       indexing="ij")
    nodes = np.column_stack([L * u.ravel(), W * v.ravel(),
                             np.zeros(u.size)])
    nid = lambda i, j: i * (ny + 1) + j
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += ([[a, b, c], [a, c, d]] if (i + j) % 2 == 0
                     else [[a, b, d], [b, c, d]])
    return rtm.ShellMesh(nodes, np.array(tris))


L_CH, W_CH, NX_CH = 0.5, 0.02, 100
MU_CH, DP_CH = 0.1, 1e5


def front_error(plies, k_eff, phi):
    """
    Largest relative error of the front of a 1-D channel along x against
    x = sqrt(2 K dp t / (phi mu)), for plies = [(fabric, refdir)]. The
    front is the resin volume beyond the port, as in verification_model4
    V1, and is followed up to 80 % of the exact fill time (the vent column
    is not reached).
    """
    mesh = grid_mesh(NX_CH, 4, L_CH, W_CH)
    cc = mesh.cellcenter
    dx = L_CH / NX_CH
    stack = rtm.LaminateStack()
    for fab, refdir in plies:
        stack.add_ply(fab, 0.2e-3, refdir=refdir)
    t_end = 0.8 * phi * MU_CH * L_CH ** 2 / (2 * k_eff * DP_CH)
    sim = (rtm.RTMSimulation().set_mesh(mesh).set_process_model(4)
           .set_resin(rtm.ResinMaterial("oil").set_viscosity(MU_CH))
           .set_laminate(stack).set_pressures(p_inlet=2e5, p_init=1e5)
           .set_run_control(tmax=t_end, n_pics=100)
           .add_injection_port_cells(np.where(cc[:, 0] < dx)[0])
           .add_vent_cells(np.where(cc[:, 0] > L_CH - dx)[0]))
    sim.run()
    area = mesh.get_cell_areas()
    errs = []
    for sn in sim.get_snapshots()[1:]:
        x_f = (sn.gamma * area)[sn.celltype != rtm.CELL_INLET].sum() / W_CH
        if 0.05 < x_f < L_CH - 2 * dx:
            x_ex = np.sqrt(2 * k_eff * DP_CH * sn.t / (phi * MU_CH))
            errs.append(x_f / x_ex - 1)
    return float(np.max(np.abs(errs))), len(errs)


def u3_solver():
    print("\n[U3] Through the solver (i_model=4, 1-D channel front)")
    ud = rtm.UDMaterial.from_tape_layup(6.35e-3, 0.3e-3, 0.15e-3, 0.55, 3.5e-6)
    k1, k2 = ud.get_permeability()
    phi = ud.get_porosity()
    for name, plies, k in (
            ("fibres along the channel", [(ud, (1, 0, 0))], k1),
            ("fibres across the channel", [(ud, (0, 1, 0))], k2),
            ("[0/90] pair, K = (K1 + K2) / 2",
             [(ud, (1, 0, 0)), (ud, (0, 1, 0))], 0.5 * (k1 + k2))):
        e, n = front_error(plies, k, phi)
        check(f"U3 {name}", e < 0.01,
              f"K {k:.3e}, max |front error| {100 * e:.3f} % over {n} "
              f"snapshots (K1/K2 = {k1 / k2:.0f})")


def u4_monotone():
    print("\n[U4] Tape across the channel, to the end of fill")
    ud = rtm.UDMaterial.from_tape_layup(6.35e-3, 0.3e-3, 0.15e-3, 0.55, 3.5e-6)
    k1, k2 = ud.get_permeability()
    mesh = grid_mesh(NX_CH, 4, L_CH, W_CH)
    cc = mesh.cellcenter
    dx = L_CH / NX_CH
    out = {}
    for scheme in ("lsq", "monotone"):
        sim = (rtm.RTMSimulation().set_mesh(mesh).set_process_model(4)
               .set_resin(rtm.ResinMaterial("oil").set_viscosity(MU_CH))
               .set_laminate(rtm.LaminateStack().add_ply(
                   ud, 0.2e-3, refdir=(0, 1, 0)))
               .set_pressures(p_inlet=2e5, p_init=1e5)
               .set_run_control(tmax=1e7, n_pics=20)
               .set_solver_settings(flux_scheme=scheme)
               .add_injection_port_cells(np.where(cc[:, 0] < dx)[0])
               .add_vent_cells(np.where(cc[:, 0] > L_CH - dx)[0]))
        sim.run()
        pr = sim.get_pressure_results()
        out[scheme] = (sim, pr["p"].min() - 1e5, pr["p"].max() - 2e5)
        n_left = int((sim.get_final_snapshot().gamma < 1).sum())
        print(f"    {scheme}: complete {sim.is_fill_complete()}, cells not "
              f"full {n_left}, pressure {out[scheme][1]:+.1f} Pa below p_init"
              f" / {out[scheme][2]:+.1f} Pa above p_inlet, balance "
              f"{sim.get_volume_balance()['relative_error']:.1e}")
    sim, p_lo, p_hi = out["monotone"]
    bal = sim.get_volume_balance()["relative_error"]
    check("U4 fill completes", sim.is_fill_complete(),
          f"t = {sim.get_total_fill_time():.0f} s, K1/K2 = {k1 / k2:.0f}")
    check("U4 maximum principle", p_lo >= -1e-6 * 1e5 and p_hi <= 1e-6 * 1e5,
          f"pressure range {p_lo:+.2e} / {p_hi:+.2e} Pa beyond [p_init, "
          f"p_inlet]")
    check("U4 volume balance", bal < 1e-9, f"relative error {bal:.1e}")
    t_ex = ud.get_porosity() * MU_CH * (L_CH - dx) ** 2 / (2 * k2 * DP_CH)
    t = sim.get_total_fill_time()
    check("U4 fill time", abs(t / t_ex - 1) < 0.02,
          f"{t:.0f} s vs phi mu (L - dx)^2 / (2 K2 dp) = {t_ex:.0f} s")


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        u1_fibre_bed()
        u2_tape_layup()
        u3_solver()
        u4_monotone()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
