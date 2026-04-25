# RTMsim-Py — stacked-laminate edition

Python port of [RTMsim](https://github.com/obertscheiderfhwn/RTMsim) (Christof Obertscheider, FHWN), with stacked-laminate support. Single mesh; every element carries the full ply stack through its thickness; only the fibre orientation differs from ply to ply.

## Physics summary

- **Governing equations:** compressible Euler + Darcy drag on 2-D shells, VOF filling factor γ ∈ [0, 1].
- **EOS:** adiabatic `p(ρ) = κ ρ^γ`, stabilised by a quadratic lookup-table fit through three reference points.
- **Gradient:** least-squares pressure gradient on cell-center neighbours (fast 2×2 normal-equations variant by default).
- **Flux:** first-order upwind, central ρ on face, upwinded velocity/γ.
- **Momentum update:** explicit pressure + convection, **implicit** Darcy drag with a full 2×2 inverse `−μ K⁻¹ u`.
- **Anisotropy:** per-element 2×2 in-plane permeability tensor `K` derived from the laminate stack (see below). The solver consumes `(Kxx, Kxy, Kyy)` directly — no diagonal-axis assumption.

### Stacked-laminate assumption

Every element on the shell carries all *N* plies through the thickness. Each ply has its own (thickness *tₖ*, principal permeabilities *K1ₖ*, *K2ₖ*, fibre direction *r̂ₖ*).

The standard LCM shell assumption is that all plies see the same in-plane pressure gradient (parallel flow through the stack). Volumetric flow rates add, which is mathematically equivalent to thickness-weighted averaging of each ply's 2×2 in-plane tensor:

```
        N
       ___
       \                            
K_eff = >    tₖ · R(αₖ) · diag(K1ₖ, K2ₖ) · R(αₖ)ᵀ   /  Σ tₖ
       /
       ‾‾‾
       k=1
```

where `αₖ` is the in-plane angle between ply *k*'s fibre direction and the element's local geometric *e1* axis. This is computed once at setup time in `build_stack_tensor()`.

For the canonical ply orientations on a flat plate with `K1 = 3·10⁻¹⁰`, `K2 = 0.6·10⁻¹⁰` (per-ply ratio 0.2), 4 equal-thickness plies:

| stack             | K_eff principal values        | ratio K2/K1 | major axis |
|-------------------|-------------------------------|:-----------:|:----------:|
| [0/0/0/0]         | (3.0, 0.6) × 10⁻¹⁰            | 0.20        | 0°         |
| [0/0/+45/−45]     | (2.4, 1.2) × 10⁻¹⁰            | 0.50        | 0°         |
| [0/90/+45/−45]    | (1.8, 1.8) × 10⁻¹⁰ (isotropic)| 1.00        | undefined  |

These are exactly what classical laminate theory predicts; the demo prints them at runtime as a self-check.

## What is NOT ported

GUI, Nastran BDF mesh reader, JLD2 binary I/O, multi-model extension stubs (`i_model > 1`).

## Contents

```
rtmsim.py              monolithic solver (~910 lines)
demo_stacks.py         3 stacked-laminate cases on a 30 cm plate
demo_4ply.py           legacy in-plane-patch demo (kept for reference)
README.md              this file
demo_stacks_out/
  comparison.png       3 stacks × (mid, late) snapshots side-by-side
  sequence_*.png       full snapshot sequence per stack
demo_out/              legacy in-plane-patch results
```

## Install and run

```bash
pip install numpy matplotlib numba
python demo_stacks.py
```

Numba JIT compilation happens on first call to `run_filling` (≈3 s). 3200-cell run takes ~3–4 s afterwards.

## Public API for stacked laminates

```python
import rtmsim as rtm
import numpy as np

mesh = rtm.make_square_plate(side=0.3, n_div=40)
rtm.assign_patch_by_disk(mesh, patch_idx=0, center_xy=(0, 0), radius=0.008)

def ply(deg, t=0.75e-3, K1=3e-10, K2=0.6e-10, phi=0.7):
    th = np.deg2rad(deg)
    return rtm.PlyProperties(
        thickness=t, porosity=phi, K1=K1, K2=K2,
        refdir=np.array([np.cos(th), np.sin(th), 0.0]),
    )

stack = rtm.LaminateStack(plies=[ply(0), ply(90), ply(45), ply(-45)])

params = rtm.SimParameters(
    tmax=600.0, mu_resin=0.10, p_inlet=2.0e5, p_init=1.0e5,
    patch_types=[rtm.PATCH_INLET, rtm.PATCH_IGNORE,
                 rtm.PATCH_IGNORE, rtm.PATCH_IGNORE],
    n_pics=10,
    stack=stack,             # <-- enables stack mode
)

snaps = rtm.run_filling(mesh, params)
```

When `params.stack` is provided, `params.main` and `params.patches` are ignored for permeability — every element gets the same stack. The four `patch_*` slots are only used for inlet/outlet flagging.

You can still inspect the per-element tensor directly:

```python
Kxx, Kxy, Kyy = rtm.build_stack_tensor(mesh, stack, None)
# These are the 2x2 components in each element's local frame.
```

To recover the global-frame tensor, transform with the element basis (the demo's `report_effective_K()` shows how).

## Numerical stability notes

- CFL-limited time step `dt = CFL · h_min · μ / (K_eff_max · Δp)`, default `CFL = 0.05`. The largest principal `K_eff` across the mesh is used to size the worst-case Darcy velocity.
- Implicit Darcy drag uses the full 2×2 inverse `K⁻¹` — required for stability when `μ/K` is O(10⁸).
- Pressure normalisation `p → p − p_init + p_eps` keeps the EOS lookup table away from `ρ ≈ 0`.
- Early stop on `mean(γ) > 0.985` rather than `min(γ) > 0.98`; corner cells with small effective face area asymptote below 1.0.
