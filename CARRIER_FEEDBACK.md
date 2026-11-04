# Trying `Carrier` on the BABY-1L tritium chain

Branch `try/carrier-baby-chain` ports SaltCrucible, Bubbler4 and LSCCounter to
carrier ports (pathsim 0.29.0, behavior re-checked on pathsim main c2d44b8).
The scalar blocks stay as they are, the carrier versions subclass them
(`carrier_form.py`, `carrier_channels.py`, `streams.py`), so both can run side by side.

Chemical form, two ways:

- (a) `Form` carrier: `("soluble", "insoluble")`. Crucible writes `Species`, a map block assigns the form.
- (b) `SpeciesForm` carrier: `("H","D","T","He","Z","T_soluble","T_insoluble")`, one carrier everywhere.

## Numerical equivalence with the scalar ports

BABY-1L run 1 inputs (1.22 L, k_top 1.09e-7 m/s, wall ratio 1.49e-2), RKCK54, same tolerances.

| case | quantity | max rel diff, (a) and (b) |
|---|---|---|
| all insoluble, ve = ce = 1 | released top/wall, flux, vials, counter gross/net, cumulative | 0 (bitwise equal) |
| 30% soluble, ve 0.9, ce 0.8, background 0.3 | same | 2.2e-15 |

The test `tests/tritium/test_carrier_chain.py` checks the first case for exact equality and the second at rtol 1e-12.

## What was easy

- Declaring ports: `input_port_labels = {"vials": Vials(0)}`. `LSCCounter` needed only new labels, its code did not change.
- One connection per stream, and the width check catches a stream wired to the wrong port.
- Registers stay flat float64, so solvers, events and `ScheduleList` did not notice. `self.inputs.to_array()[:4]` in an event still works.
- Extending a carrier works (`class SF(Species): keys = Species.keys + (...)`).
- Core `Splitter([0.3, 0.7], carrier=Species())` gives two `Species` streams.
- A core block gets carrier ports with a subclass, no new code:
  ```python
  class SpeciesAmp(Amplifier):
      input_port_labels = {"in": Species(0)}
      output_port_labels = {"out": Species(0)}
  ```

## Awkward, with reproducers

**1. Same width, different carrier connects silently.** Only the width is checked.
```python
class Flipped(Carrier): keys = ("Z", "He", "T", "D", "H")
Connection(src["out"], snk["in"])   # Species(5) -> Flipped(5): builds, H lands on Z
```
Issue #8 says a mismatch fails at build time, that holds for different widths only. Our (b) variant
shows the cost: connect the crucible straight to the bubbler, both `SpeciesForm`, and the vials stay
at zero with no error, because the crucible writes `T` and the bubbler reads `T_soluble`.
Suggestion: compare carrier class (or keys) in `Connection._validate_dimensions`.

**2. Error message has no names.** `Connection(src["out"], snk["in"]["T"])` raises
`Source and target have different number of ports!`. Naming both blocks, ports and widths would save time.

**3. `.T` in the #8 example does not exist.** `pump["out"].T` raises `AttributeError: 'PortReference' object has no attribute 'T'`.
The working form is `pump["out"]["T"]`. Either the comment or the API should change. Also,
`Species.T` fails on the class, the indices exist only on instances (`Species().T`, `len(Species())`),
so layout code has to build throwaway instances.

**4. `Register` read by tuple key fails.** `("out", "T") in block.outputs` is `True`, `block.outputs[("out", "T")]` raises
`IndexError: only integers, slices ...`. The docstring says channels are keys as tuple.

**5. Writing into a carrier view is dropped.**
```python
b.outputs["vials"][2] = 5.0   # no effect, "vials" returns a copy
b.outputs["vials"]            # -> [0. 0. 0. 0.]
```
Reads that look like views but are copies are easy to trip on. A note in the docs, or a read-only array, would help.

**6. Explicit `start` forces hand layout and renumbers scalar ports.** Putting `flux_top` and `flux_wall` (5 wide) at index 2
moved `released_top` from 4 to 12, and int indexing keeps working but means something else:
`Connection(crucible[2], x)` is now channel `H` of `flux_top`. Our scalar tests wired `crucible[i]` for `i in range(6)` and would
wire the wrong channels with no error. Layout needs `Species(2 + len(Species()))`.
Suggestion: let `*_port_labels` take unplaced carriers and place them in dict order after the scalars.

**7. Core blocks reject streams.** `Connection(src["out"], Amplifier())`, `Adder()`, `Integrator()`, `Scope()` all raise the port-count error.
`Scope` takes `sc[0:5]`, but labels do not come from the keys. Splitter and mixer are announced, a carrier-aware gain
(`Amplifier(gain)` on a stream) is also needed for the particles/s to Bq step. We wrote a 20 line `StreamMap` block (matrix on a stream) for it and for assigning the form.

**8. Units and metadata.** A carrier has keys only. Our streams are particles/s of atoms, converted to Bq by a gain,
and nothing on the port says so. A `units` or `meta` class attribute that is shown in the error message and in PathView would be enough to start.

**9. Dead channels.** The crucible only knows tritium, so 4 of the 5 `Species` channels on `flux_top` are always zero.
Fine as a convention, but it makes sums over keys and any conservation report need to know which channels a block owns.

**10. Form as carrier vs channels.** (a) forces a map block between `Species` and `Form`, which is the right place to state the
assumption (here: all crucible release is insoluble) and keeps the crucible isotope only. (b) needs no converter but
`T` = `T_soluble` + `T_insoluble` is a convention nothing enforces, and every block on the path must use the wide carrier.
We lean to (a) or, closer to core, one `Species` port per form.

**11. Event ordering is unchanged.** The counter must be listed before the bubbler to read the vials before the reset at the same time:
```python
Simulation([src, split, counter, bubbler], ...)   # counts 89.1, 89.1
Simulation([src, split, bubbler, counter], ...)   # counts 0.0, 0.0
```
Carriers do not help here, this is the "read before an event at the same time" item from #8.

**12. PathView.** Not tested, we have no PathView build here (pathview#332).

## pathsim bugs

**`Simulation.run(duration)` overshoots with a fixed step.** Still on main (c2d44b8). The first step is never clipped to
the end time, and a fixed-step solver does not clip at all:
```python
sim = Simulation([Constant(1.0), Integrator()], [], Solver=EUF, dt=1.0, log=False)
sim.run(0.25); sim.time   # 1.0, expected 0.25
sim = Simulation([Constant(1.0), Integrator()], [], Solver=EUF, dt=1.0, log=False)
sim.run(2.5);  sim.time   # 3.0, expected 2.5
```
`RKCK54` is clipped after the first step: `run(2.5)` gives 2.5, `run(0.25)` gives 1.0.
Aligned endpoints are fine (#263 is closed).
