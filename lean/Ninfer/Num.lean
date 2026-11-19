-- SPDX-License-Identifier: Apache-2.0
/-!
# `Ninfer.Num` — the number formats of the ternary linear

Exact conversions between binary32, binary16 and bfloat16, on bit patterns, and the helpers the
reference model of the ternary linear is written with.

* bfloat16 is the high half of a binary32. Narrowing rounds to nearest even, as the engine's
  `__float2bfloat16_rn` does; a NaN stays a quiet NaN.
* binary16 widens exactly. Narrowing rounds to nearest even, overflows to infinity, and rounds a
  value past the smallest subnormal's midpoint up to it.

Every narrowing is integer arithmetic, so it does not depend on a target's float-to-half
instruction (some round toward zero). The kernels in `Ninfer.Ops.LinearT2.Slang` use the same
algorithm in Slang. `tests/num/run.exs` holds both to the hardware conversions of the host C++
compiler (`_Float16`, `__bf16`), which are independent of this file.

Arithmetic on `Float32` rounds every operation to binary32, and nothing here fuses a multiply and
an add: the model fixes an order and rounds after each step.
-/

namespace Ninfer.Num

/-- Shift left; both operands `UInt32`, so a literal amount cannot default to `Nat`. -/
@[inline] def shl (a n : UInt32) : UInt32 := a <<< n
/-- Shift right; both operands `UInt32`. -/
@[inline] def shr (a n : UInt32) : UInt32 := a >>> n

/-- bfloat16 bits to binary32: the pattern in the high half, exact. -/
def bf16ToF32 (h : UInt16) : Float32 := Float32.ofBits (shl h.toUInt32 16)

/-- binary32 to bfloat16, round to nearest even; a NaN stays a quiet NaN. -/
def f32ToBf16 (x : Float32) : UInt16 :=
  let u : UInt32 := x.toBits
  if (u &&& 0x7FFFFFFF) > 0x7F800000 then (shr u 16 ||| 0x0040).toUInt16
  else
    let r : UInt32 := shr (u + 0x7FFF + (shr u 16 &&& 1)) 16
    r.toUInt16

/-- binary16 bits to binary32, exact. -/
def f16ToF32 (h : UInt16) : Float32 :=
  let w : UInt32 := h.toUInt32
  let sign : UInt32 := shl (w &&& 0x8000) 16
  let e : UInt32 := shr w 10 &&& 0x1F
  let m : UInt32 := w &&& 0x3FF
  if e == 0 then
    if m == 0 then Float32.ofBits sign
    else
      -- A subnormal: m * 2^-24 with the leading bit of m (position p) made the hidden bit.
      let p : Nat := Nat.log2 m.toNat
      let bits : UInt32 := sign ||| shl (p + 103).toUInt32 23 ||| (shl m (23 - p).toUInt32 &&& 0x7FFFFF)
      Float32.ofBits bits
  else if e == 31 then Float32.ofBits (sign ||| 0x7F800000 ||| shl m 13)
  else Float32.ofBits (sign ||| shl (e + 112) 23 ||| shl m 13)

/-- binary32 to binary16, round to nearest even. Overflow gives infinity; a NaN stays a quiet NaN. -/
def f32ToF16 (x : Float32) : UInt16 :=
  let u : UInt32 := x.toBits
  let sign : UInt32 := shr u 16 &&& 0x8000
  let e : Nat := (shr u 23 &&& 0xFF).toNat
  let m : UInt32 := u &&& 0x7FFFFF
  if e == 255 then
    if m == 0 then (sign ||| 0x7C00).toUInt16 else (sign ||| 0x7E00 ||| shr m 13).toUInt16
  else if e ≥ 143 then (sign ||| 0x7C00).toUInt16 -- 2^16 and up: past the largest half
  else if e ≥ 113 then
    -- A normal half: the exponent field is e - 112 (in 1 ..= 30 here).
    let h0 : UInt32 := shl (e - 112).toUInt32 10 ||| shr m 13
    let rem : UInt32 := m &&& 0x1FFF
    let h : UInt32 := if rem > 0x1000 || (rem == 0x1000 && (h0 &&& 1) == 1) then h0 + 1 else h0
    (sign ||| h).toUInt16
  else if e == 0 || e < 102 then sign.toUInt16 -- a binary32 subnormal, or below half the smallest half
  else
    -- A half subnormal k * 2^-24, k = round(significand * 2^-(126 - e)): shift out 126 - e bits.
    let sh : UInt32 := (126 - e).toUInt32
    let sig : UInt32 := m ||| 0x800000
    let q0 : UInt32 := shr sig sh
    let rem : UInt32 := sig &&& (shl 1 sh - 1)
    let half : UInt32 := shl 1 (sh - 1)
    let q : UInt32 := if rem > half || (rem == half && (q0 &&& 1) == 1) then q0 + 1 else q0
    (sign ||| q).toUInt16

/-- Round a binary32 to bfloat16 and widen it again. -/
def bf16Round (x : Float32) : Float32 := bf16ToF32 (f32ToBf16 x)

/-- binary32 round to nearest even to an integer value, ties to even (`rintf`). -/
def roundEven (x : Float32) : Float32 :=
  let f := Float32.floor x
  if x - f == 0.5 then
    if Float32.floor (f * 0.5) * 2.0 == f then f else f + 1.0
  else Float32.round x

end Ninfer.Num
