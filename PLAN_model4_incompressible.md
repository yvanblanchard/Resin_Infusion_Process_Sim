# Implementation plan: `i_model=4`, incompressible resin (Darcy + fill factor)

## Why

In `i_model=2`/`3` the resin and air are one compressible fluid whose density,
and so the fill state, depends only on pressure (power-law EOS). As a result:

- The saturated strip next to the vents sits at 2.5–6.4 kPa. That is below the
  filled threshold p_vent + Δp/16 ≈ 6.6 kPa, so it is never counted as filled.
- Patching the fill rule does not help. Each option was tested on the 800 mm
  flat plate (10 mm cells) against the exact radial-flow solution:

| Variant | Front vs exact solution | Edges fill? |
|---|---|---|
| Density threshold (current) | 4–7 % early | no |
| VOF γ carried with the parcel velocity | 2–20 % early, grows with distance | yes |
| VOF γ carried with mass flux / ρ_resin | 36 % late | no |
| Stiffer EOS (exponent 8, CFL 0.0125) | 15 % early, 13× slower | no |

Fluent (Pierce & Falzon 2017) treats the resin as incompressible, with a
pressure outlet on the boundary face. That is the physical model, and the one
implemented here.

## Physics

- The resin is incompressible and flows by Darcy's law. Inertia is neglected.
- In the wet region the pressure satisfies, at each step,
  `∇·(h K/μ ∇p) = 0`.
- Boundary pressures:
  - `p_inlet` at the inlets
  - `p_vent` at the vents and at the flow front (the air ahead is near vacuum)
- Each cell has a fill factor `f ∈ [0, 1]`, updated from the Darcy fluxes:
  `φ V df/dt = Q_in`. The fill state never depends on pressure.

## Steps

### 1. Discretisation spec

Write a short spec first, then code against it.

- **Unknowns:** pressure in the full cells (`f = 1`).
- **Fixed-pressure cells:**
  - inlets at `p_inlet`
  - vents at `p_vent`
  - front cells (`0 < f < 1`) at `p_vent`, or at the pocket pressure (step 5)
- **Empty cells** are left out of the solve.
- **Fluxes:** two-point flux on each face, using the existing `cc_to_cc_x/y`,
  `face_nx/ny`, `face_area` and the `T11..T22` frame rotations.
- **K across faces:** harmonic average, so regions with different shear meet
  correctly.
- **Anisotropy correction:** the two-point flux is inaccurate when K is strongly
  anisotropic and the mesh does not follow it (K1/K2 reaches 5 at 40° shear).
  Add a deferred correction from the existing least-squares pressure gradient,
  iterated 1–2 times per step.

### 2. Pressure solve

- Assemble a sparse matrix once per step from the current set of wet cells,
  with SciPy. SciPy 1.16 is installed but not listed in `requirements.txt`.
- The matrix is symmetric positive definite:
  - direct solver up to about 50k cells
  - conjugate gradient started from the previous pressure above that

### 3. Time stepping

- Each step ends when the next front cell becomes full:
  `Δt = min over front cells of (1 − f)·φ·V / Q_in`.
- Steps also stop at snapshot times and at cascade-port activation times.
- This is about one pressure solve per cell, roughly 10–40 s for the
  12.8k-cell plate.
- If that proves slow, let several cells fill per step and pass any overflow
  downstream, keeping the resin volume exact.

### 4. Vents and completion

- Vent cells keep `p_vent`. They fill like any other cell, then pass resin out
  of the domain.
- The run is complete when every cell in the fed region is full, vents
  included.
- Keep a running balance (injected = stored + vented) and check that it closes
  to round-off.

### 5. Trapped air

- A dry region cut off from every vent becomes an air pocket, using connected
  components of the not-full cells plus the vents.
- The pocket pressure is isothermal ideal gas, `p = p0 V0 / V_air`. The pocket's
  front cells use this pressure.
- When the pocket pressure reaches the local resin pressure, the pocket stops
  shrinking and is reported as a dry spot.
- At 0.3 kPa initial pressure this barely matters, but it keeps the model valid
  for RTM at 1 atm.

### 6. Integration into `RTMSimulation`

- `set_process_model(4)`; no air EOS required.
- Reused unchanged:
  - laminate stacks and the per-element K tensor
  - porosity
  - ports, vents and cascade ports
  - fed region
  - snapshots (`gamma = f`, `p`)
  - all getters (`get_fill_time_field`, `is_fill_complete`, ...)
- New `SolverSettings` entries: linear solver choice, number of correction
  iterations, fill tolerance.
- Version 1 is isothermal, with constant μ per cell. Coupling temperature and
  cure is a later step.
- Models 1–3 are not touched.

### 7. Verification (`validation/verification_model4.py`)

| Test | Pass criterion |
|---|---|
| V1: 1-D channel, front `x = √(2KΔp·t/(φμ))` | < 1 % |
| V2: flat-plate radial flow vs exact solution | < 2 % at 10 mm cells, converging at 20/10/5 mm |
| V3: anisotropic radial flow, K1/K2 = 5 at 30° | elliptical front: axis ratio √5, angle 30°, timing |
| V4: curved strip (half cylinder) vs the same strip flat | identical fill times (checks cell frames) |
| V5: resin volume balance | error < 1e-10 |
| V6: plate filled to completion | 100 % filled including vents, `complete = True` |
| V7: layout that traps air | dry spot where expected; ideal-gas pocket pressure |

### 8. Rerun the double dome

- Add `--model 4` to `validation_double_dome.py` and `view_double_dome.py`.
- In the viewer legend, the vent entry becomes "fills, held at 0.3 kPa".
- Run three variants: isotropic, shear-dependent, and shear-dependent with K1 on
  the warp (y). The weft axis of the 0°/90° ply is not given in the paper.
- Report front errors against the experiment and Fluent, and confirm the ply
  fills to the edges.

### 9. Documentation

Update the README physics section, the module docstring and `requirements.txt`
(add scipy).

## Out of scope (the paper ignores these too)

- preform compaction
- partial saturation behind the front
- capillary effects

## Order

1. Steps 1–4 with V1, V2, V5 and V6. This is enough to see whether the flow
   front and vent filling come out right.
2. The anisotropy correction, checked with V3.
3. Steps 5–9.
