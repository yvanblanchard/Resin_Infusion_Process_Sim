# RTMsim-Py

Python port of [RTMsim](https://github.com/obertscheiderfhwn/RTMsim) (Christof Obertscheider, FHWN) with additional features.
- RTM and VARI manufacturing processes simulation
- Shell mesh with stacked-laminate (Nastran BDF format)
- Every element carries the full ply stack through its thickness
- Only the fibre orientation differs from ply to ply
- Output: filling progress over time, pressure

![alt text](img/VARTM_RTM_image.png)

## Physics summary

- **Governing equations:** compressible Euler + Darcy drag on 2-D shells, VOF filling factor γ ∈ [0, 1].
- **EOS:** adiabatic `p(ρ) = κ ρ^γ`, stabilised by a quadratic lookup-table fit through three reference points.
- **Gradient:** least-squares pressure gradient on cell-center neighbours (fast 2×2 normal-equations variant by default).
- **Flux:** first-order upwind, central ρ on face, upwinded velocity/γ.
- **Momentum update:** explicit pressure + convection, **implicit** Darcy drag with a full 2×2 inverse `−μ K⁻¹ u`.
- **Anisotropy:** per-element 2×2 in-plane permeability tensor `K` derived from the laminate stack (see below). The solver consumes `(Kxx, Kxy, Kyy)` directly — no diagonal-axis assumption.

The above describes process models 1–3 (`set_process_model(1|2|3)`). In models 2/3 the fill state follows from the density through the EOS, so the cells next to a vent, held near the vent pressure, never count as filled.

### Incompressible model (`set_process_model(4)`)

- **Resin:** incompressible, Darcy flow, isothermal, rigid preform; no air EOS needed.
- **Pressure:** each step solves `∇·(h K/μ ∇p) = 0` on the full cells, with
  `p_inlet` on the boundary of the inlet ports, `p_vent` (= `p_init`) on the outer
  (mesh boundary) edges of full vent cells, and `p_init` or the air-pocket
  pressure in the cells not yet full. A vent with no boundary edge is a hole held at `p_vent`.
- **Fill factor:** `φ V df/dt = Q_in` in the cells not yet full; a step ends when the
  next front cells are full (`fill_fraction_per_step`). The fill state never depends on
  pressure: vent cells fill like the rest, and the run completes when every cell is full.
- **Fluxes:** finite volumes on the shell, each half-face in its own cell's plane
  (curved shells, harmonic K across faces). The flux is two-point plus an implicit
  least-squares correction for skewed cells and anisotropic K; front faces are
  two-point (the face is the front), so resin never leaves a front cell.
- **Flux scheme:** `flux_scheme="lsq"` (default) is the two-point flux plus the
  implicit gradient correction above. `flux_scheme="monotone"` builds every flux
  from positive coefficients (nonlinear two-point flux of Le Potier and of
  Lipnikov, Svyatskiy and Vassilevski): the pressure obeys the maximum principle
  for any anisotropy and mesh skew handled by the cone condition (no
  under- or overshoot at vents and ports), and a strongly anisotropic strip
  (K1/K2 = 680, fibres across the flow) fills to the end with a fill time
  within 0.01 % of the exact one, where the default scheme is 1 % off and
  stalls at the vent. It costs an Anderson-accelerated nonlinear iteration in
  every pressure solve (about 15-20 linear solves per step), so runs are about
  5-6 times slower, and it is slightly less accurate on the V3 ellipse
  (axis ratio 3.4 % off at 10 mm against 2.8 %). `get_run_stats()` then also
  reports the iterations, the solves that stopped at `picard_max` and the faces
  without a positive decomposition (`n_cone_fail`, expected 0).
- **Trapped air:** dry regions cut off from every vent are isothermal ideal gas
  (`p V = const`), solved implicitly with the pressure; pockets left in balance with the
  resin are reported by `get_dry_spots()`.
- **Results:** same getters as the other models, plus `get_volume_balance()`,
  `get_dry_spots()` and `get_run_stats()`. Verified against exact solutions in
  `validation/verification_model4.py` (1-D and radial fronts, anisotropic ellipse, curved
  vs flat shell, volume balance, full fill, trapped air).
`

Example:

![alt text](img/fill_sequence_2d.png)

## Install and run

```bash
pip install numpy matplotlib numba scipy trimesh rtree   # scipy: i_model 4
python demo_4ply.py        # RTM filling, 4-ply stack
python demo_lcm.py         # VARI (LCM) process
python demo_thermal.py     # thermal and cure
python mesh_annulusfiller.py
python validation_double_dome.py   # vs. double dome infusion experiments (validation/)
python view_double_dome.py         # 3-D PyVista view of that case (time slider)
python validation_double_dome.py --model 4   # same with the incompressible model
python verification_model4.py      # i_model 4 vs exact solutions (validation/)

pip install pyvista mmgpy  # needed by the STL test only
python test_filling_frame.py   # STL shell, mmgs remesh, cascade ports, PyVista view
```

## API

All inputs are set through setters and results read through getters;
physical inputs have no built-in defaults (`RTMSimulation.validate()`
names anything missing). Numerical constants live in `SolverSettings`.

```python
import rtmsim as rtm

mesh = rtm.ShellMesh.from_stl("part.stl", scale=1e-3, units="mm")  # or from_trimesh / from_arrays
resin = rtm.ResinMaterial("epoxy").set_viscosity(0.1)
fabric = rtm.FabricMaterial("biax").set_permeability(3e-10, 6e-11).set_porosity(0.6)
stack = rtm.LaminateStack().add_ply(fabric, 0.75e-3, refdir=(1, 0, 0))

sim = (rtm.RTMSimulation()
       .set_mesh(mesh).set_process_model(1)
       .set_resin(resin).set_laminate(stack)
       .set_pressures(p_inlet=2e5, p_init=1e5)
       .set_air_eos(p_ref=1.01325e5, rho_ref=1.225, gamma=1.4)
       .set_run_control(tmax=600.0, n_pics=16)
       .add_injection_port((10.0, 0.0, 5.0), radius=5.0)                  # mesh units
       .add_injection_port((200.0, 0.0, 5.0), radius=5.0, t_activate=120)  # cascade port
       .add_vent((390.0, 0.0, 5.0), radius=5.0))
sim.run()
t_fill = sim.get_total_fill_time()
filled = sim.get_filled_mask_at(0.5 * t_fill)     # also: get_fill_time_field, get_fill_curve,
                                                   # get_pressure_at, get_pressure_results, ...
```

Ports are located by coordinates: the point is projected onto the surface
and every cell whose centre lies within `radius` becomes an inlet cell
(`add_injection_port_cells` takes explicit cell ids instead). Thermal and
cure runs add `set_thermal(...)`, `enable_cure()` and the resin/fabric
thermal, viscosity-model and cure-kinetics setters (see `demo_thermal.py`).

### Laminate layout and fibre orientation

`set_laminate(stack)` is the default stack of every element.
`add_laminate_region(stack, cell_ids)` gives other elements a different
stack (other ply count, thicknesses or fabrics, e.g. ply drops or UD
reinforcements); later regions override earlier ones. Each element's
permeability is the thickness-weighted average of its own plies.

A ply's `refdir` is either one global vector or one vector per element
(`(N, 3)`), projected onto each element. `angle_deg` on `add_ply` (scalar
or `(N,)`) and `sim.set_fibre_deviation(angle_deg)` (all plies) rotate it
about the face normal (right-hand rule on the mesh face orientation),
e.g. to apply a draping-simulation deviation. A UD ply is simply a fabric
with `K1 >> K2`.

```python
ud = rtm.FabricMaterial("UD").set_permeability(6e-10, 3e-11).set_porosity(0.5)
hoop = mesh.get_cylindrical_directions(origin=(0, 0, 0), axis=(0, 0, 1), kind="hoop")
base = rtm.LaminateStack().add_ply(fabric, 1e-3, refdir=(1, 0, 0))
pad = (rtm.LaminateStack().add_ply(fabric, 1e-3, refdir=(1, 0, 0))
       .add_ply(ud, 1e-3, refdir=hoop)
       .add_ply(ud, 1e-3, refdir=(1, 0, 0), angle_deg=45.0))
sim.set_laminate(base).add_laminate_region(pad, pad_cells)
sim.set_fibre_deviation(shear_deg)      # optional, (N,) [deg]
stacks, stack_id = sim.get_laminate_map()
```

### Unidirectional fabric and slit tape

`UDMaterial` is a `FabricMaterial` whose K1 (along the fibres), K2 (across)
and porosity are computed from what is known about the material. It goes
into `add_ply` like any fabric; `refdir` is the fibre / tape direction.

```python
# fibre bed: Gebart's model from fibre volume fraction and fibre radius [m]
ud = rtm.UDMaterial.from_fibre_volume_fraction(0.55, fibre_radius=3.5e-6,
                                               packing="quad")   # or "hex"
# slit tape laid side by side with an open gap between the tapes [m]
tape = rtm.UDMaterial.from_tape_layup(tape_width=6.35e-3, gap_width=0.3e-3,
                                      tape_thickness=0.15e-3, Vf=0.55,
                                      fibre_radius=3.5e-6)
tape.get_permeability(), tape.get_porosity()
```

Along the tapes the tape and the open gap (a rectangular duct) flow in
parallel, so the gaps raise K1 strongly; across the tapes the flow crosses
tape and gap in series, the gap taken as free of resistance (an upper
bound). The gap is assumed open over the full thickness; scale `gap_width`
to the open width, or overwrite K1 / K2 / porosity with measured values
through the inherited setters. The flow is still averaged through the stack
thickness: gap channels act through the element tensor, not as a separate
path. `validation/verification_ud.py` checks the constructors and the
tensor through the solver (1-D fronts, `[0/90]` pair).

With strongly anisotropic material (K1/K2 of about 50 or more) and i_model 4,
use `sim.set_solver_settings(flux_scheme="monotone")`, see below: the default
flux can dip below the vent pressure there and leave the last cells next to
a vent unfilled.

A `refdir` nearly normal to some elements gives an ill-defined fibre
angle there and triggers a warning; use per-element directions instead.

The time loop runs compiled (Numba) and the flow kernel is multi-threaded.
The thread count defaults to about one thread per 500 cells, capped at the
machine's thread count. Override it with
`sim.set_solver_settings(n_threads=8)`; results do not depend on it.

The solver needs a reasonably isotropic triangle mesh: raw CAD STL
tessellations (long flat triangles) make it diverge and should be
remeshed first, as `test_filling_frame.py` does with MMG (`mmgs`).

