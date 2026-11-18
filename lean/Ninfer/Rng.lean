-- SPDX-License-Identifier: Apache-2.0
/-!
# `Ninfer.Rng` — the generator every fixture is drawn from

xorshift64*, written in the open so that a fixture is a function of its seed and not of a
library version. Fixtures are generated and judged by the same Lean program on one machine, so the
transcendental functions behind `normal` need not agree across hosts.
-/

namespace Ninfer

/-- xorshift64* state; never zero. -/
structure Rng where
  s : UInt64

namespace Rng

/-- A generator from any seed: one splitmix64 step, so that seeds 1, 2, 3 are unrelated. -/
def seed (n : UInt64) : Rng :=
  let z := n + 0x9E3779B97F4A7C15
  let z := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  let z := z ^^^ (z >>> 31)
  ⟨if z == 0 then 0x9E3779B97F4A7C15 else z⟩

/-- The next 64 bits. -/
def next (r : Rng) : UInt64 × Rng :=
  let x := r.s
  let x := x ^^^ (x >>> 12)
  let x := x ^^^ (x <<< 25)
  let x := x ^^^ (x >>> 27)
  (x * 0x2545F4914F6CDD1D, ⟨x⟩)

/-- A uniform draw in `[0, 1)` from the top 53 bits. -/
def unit (r : Rng) : Float × Rng :=
  let (v, r) := r.next
  ((v >>> 11).toNat.toFloat / 9007199254740992.0, r)

/-- A standard normal draw (Box–Muller, one value per call). -/
def normal (r : Rng) : Float × Rng :=
  let (u1, r) := r.unit
  let (u2, r) := r.unit
  let u1 := if u1 < 1e-300 then 1e-300 else u1
  (Float.sqrt (-2.0 * Float.log u1) * Float.cos (6.283185307179586 * u2), r)

/-- A uniform integer in `[0, n)` (`n > 0`); the modulo bias is far below anything a fixture sees. -/
def below (r : Rng) (n : Nat) : Nat × Rng :=
  let (v, r) := r.next
  (v.toNat % n, r)

end Rng

end Ninfer
