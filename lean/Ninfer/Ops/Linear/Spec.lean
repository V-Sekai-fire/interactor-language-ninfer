-- SPDX-License-Identifier: Apache-2.0
import Ninfer.Num
import Ninfer.Rng

/-!
# `Ninfer.Ops.Linear.Spec` — the ternary linear, as arithmetic

What the kernels of `Ninfer.Ops.Linear.Kernels` must compute, in Lean. Every rounding is spelled
out, so the result is a function of the inputs: no schedule, device or library appears in it.

* **Weights** (`Weight`): per output row and group of 128 inputs, 32 code bytes and one binary16
  scale. A code decodes to `m - 2 (m & s)`: `00` 0, `01` +1, `11` -1, the illegal `10` 0.
* **A16** (`a16Total`): lane `l` of 32 sums its four products of a group left to right (each
  product of a decoded weight and a bfloat16 activation is exact), scales that partial and adds it to
  its accumulator, group after group; the lanes combine as `l + (l ^ 16)`, then `^ 8, 4, 2, 1`.
  Inputs past the logical K are skipped.
* **A8** (`quantize`, `a8Total`): activations to int8 per token and group of 128 with a binary16
  scale `f16(amax / 127)`; the group's dot is an exact integer, and the total is
  `acc + ((float dot * sw) * sx)`, groups in order.
* **FP64 reference** (`ref64`): the dense weights times the activations in double precision, which is
  what ties the model to the mathematics rather than to itself.

FP32 here is `Float32`, whose every operation rounds to binary32; nothing fuses.
-/

namespace Ninfer.Ops.Linear

open Ninfer Ninfer.Num

/-- Inputs per scale group. -/
def groupK : Nat := 128
/-- Code bytes per row and group. -/
def codeBytes : Nat := 32

/-- A code to -1, 0 or +1: `m - 2 (m & s)` with `m` the low bit and `s` the high bit. -/
def decode (code : Nat) : Int :=
  let m := code &&& 1
  let s := (code >>> 1) &&& 1
  (m : Int) - 2 * ((m &&& s : Nat) : Int)

/-- `decode` as a double: `00` 0, `01` +1, `10` 0, `11` -1. -/
def decodeF (code : Nat) : Float := if code == 1 then 1.0 else if code == 3 then -1.0 else 0.0

/-- Row-split ternary weights: `n` rows of `k` logical inputs, `groups = ceil(k / 128)`. -/
structure Weight where
  n : Nat
  k : Nat
  groups : Nat
  /-- `n * groups * 32` code bytes. -/
  q : ByteArray
  /-- `n * groups` binary16 patterns. -/
  s : Array UInt16

/-- `ceil(k / 128)`. -/
def groupsOf (k : Nat) : Nat := (k + groupK - 1) / groupK

namespace Weight

/-- The 2-bit code of input `i` of group `g` of `row`. -/
def code (w : Weight) (row g i : Nat) : Nat :=
  (w.q.get! ((row * w.groups + g) * codeBytes + i / 4)).toNat >>> (2 * (i % 4)) &&& 3

/-- The decoded row: `groups * 128` entries of -1, 0 or +1 (a double holds each exactly). -/
def decodedRow (w : Weight) (row : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity (w.groups * groupK)
  for g in [0:w.groups] do
    for i in [0:groupK] do
      out := out.push (decodeF (w.code row g i))
  return out

/-- The row's scales as binary32. -/
def scaleRow (w : Weight) (row : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity w.groups
  for g in [0:w.groups] do
    out := out.push (f16ToF32 (w.s[row * w.groups + g]!)).toFloat
  return out

end Weight

/-! ## Fixtures -/

/-- Weights from a seed. Four 64-bit draws fill the 32 code bytes of a row and group. `legal`
rewrites the illegal `10` to `11`; codes past the logical K are zero, as the engine pads. Scales are
uniform in [0.01, 0.2), rounded to binary16. -/
def makeWeight (seed : UInt64) (n k : Nat) (legal : Bool) : Weight := Id.run do
  let groups := groupsOf k
  let mut r := Rng.seed seed
  let mut q := ByteArray.emptyWithCapacity (n * groups * codeBytes)
  let mut s : Array UInt16 := Array.mkEmpty (n * groups)
  for _ in [0:n] do
    for g in [0:groups] do
      for _ in [0:4] do
        let (v, r') := r.next
        r := r'
        for b in [0:8] do
          let byte0 : UInt8 := (v >>> (8 * b).toUInt64).toUInt8
          let byte1 : UInt8 := if legal then byte0 ||| ((byte0 >>> 1) &&& 0x55) else byte0
          -- zero the fields whose input lies past K
          let lane := q.size % codeBytes
          let mut byte := byte1
          for f in [0:4] do
            if g * groupK + lane * 4 + f ≥ k then byte := byte &&& ~~~(3 <<< (2 * f).toUInt8)
          q := q.push byte
      let (u, r') := r.unit
      r := r'
      s := s.push (f32ToF16 (0.01 + 0.19 * u).toFloat32)
  return { n, k, groups, q, s }

/-- bfloat16 activations `[t, k]`: with `spread`, each token drawn at a scale of 0.01, 1 or 30 (the
spread the engine sees), otherwise all at 1; rounded to bfloat16. Returns the patterns and their
binary32 values. -/
def makeActivations (seed : UInt64) (t k : Nat) (spread : Bool := true) : Array UInt16 × FloatArray := Id.run do
  let mut r := Rng.seed seed
  let mut bits : Array UInt16 := Array.mkEmpty (t * k)
  let mut vals := FloatArray.emptyWithCapacity (t * k)
  for _ in [0:t] do
    let (c, r') := r.below 3
    r := r'
    let factor : Float := if spread then #[0.01, 1.0, 30.0][c]! else 1.0
    for _ in [0:k] do
      let (z, r') := r.normal
      r := r'
      let b := f32ToBf16 (z * factor).toFloat32
      bits := bits.push b
      vals := vals.push (bf16ToF32 b).toFloat
  return (bits, vals)

/-! ## A16 -/

/-- One lane's accumulator: its four inputs of every group, in the kernel's order. -/
partial def laneAcc (wr sc x : FloatArray) (groups k l xoff : Nat) : Float32 :=
  let rec go (g : Nat) (acc : Float32) : Float32 :=
    if g < groups then
      let base := g * groupK + l * 4
      let rec part (j : Nat) (p : Float32) : Float32 :=
        if j < 4 then
          if base + j < k then
            part (j + 1) (p + (wr.get! (base + j)).toFloat32 * (x.get! (xoff + base + j)).toFloat32)
          else part (j + 1) p
        else p
      let p := part 0 0
      go (g + 1) (acc + p * (sc.get! g).toFloat32)
    else acc
  go 0 0

/-- The 32 lanes in a fixed tree: `a[l] += a[l + 16]` for `l < 16`, then 8, 4, 2, 1; lane 0 holds
the sum. `sequential` is the control: the lanes added one after the other. -/
def combine (lanes : Array Float32) (sequential : Bool := false) : Float32 :=
  if sequential then (lanes.extract 1 32).foldl (fun a v => a + v) lanes[0]!
  else Id.run do
    let mut a := lanes
    for off in [16, 8, 4, 2, 1] do
      for l in [0:off] do
        a := a.set! l (a[l]! + a[l + off]!)
    return a[0]!

/-- The FP32 total of output (`row`, token at `xoff = token * k`) of the A16 route. -/
def a16Total (wr sc x : FloatArray) (groups k xoff : Nat) (sequential : Bool := false) : Float32 :=
  combine ((List.range 32).toArray.map fun l => laneAcc wr sc x groups k l xoff) sequential

/-- FP32 totals `[t, rows]` of rows `[rowBegin, rowBegin + rows)` against activations `[t, k]`. -/
def a16Totals (w : Weight) (x : FloatArray) (t rowBegin rows : Nat) (sequential : Bool := false) :
    Array Float32 := Id.run do
  let mut out : Array Float32 := Array.replicate (t * rows) 0
  for r in [0:rows] do
    let wr := w.decodedRow (rowBegin + r)
    let sc := w.scaleRow (rowBegin + r)
    for tk in [0:t] do
      out := out.set! (tk * rows + r) (a16Total wr sc x w.groups w.k (tk * w.k) sequential)
  return out

/-! ## A8 -/

/-- Activations to int8 codes (as bytes, two's complement) and binary16 scales, per token and group:
`scale = f16(amax / 127)`, `inv = 1 / scale` (0 for a zero scale), `code = clamp(rne(x * inv), ±127)`.
Codes are padded with zeros to whole groups. -/
def quantize (x : FloatArray) (t k : Nat) : ByteArray × Array UInt16 := Id.run do
  let groups := groupsOf k
  let mut codes := ByteArray.emptyWithCapacity (t * groups * groupK)
  let mut xs : Array UInt16 := Array.mkEmpty (t * groups)
  for tk in [0:t] do
    for g in [0:groups] do
      let mut amax : Float32 := 0
      for i in [0:groupK] do
        let pos := g * groupK + i
        let xv : Float32 := if pos < k then (x.get! (tk * k + pos)).toFloat32 else 0
        amax := if Float32.abs xv > amax then Float32.abs xv else amax
      let sbits := f32ToF16 (amax / 127.0)
      xs := xs.push sbits
      let s := f16ToF32 sbits
      let inv : Float32 := if s > 0 then 1.0 / s else 0
      for i in [0:groupK] do
        let pos := g * groupK + i
        let xv : Float32 := if pos < k then (x.get! (tk * k + pos)).toFloat32 else 0
        let q := roundEven (xv * inv)
        let q := if q > 127.0 then 127.0 else if q < -127.0 then -127.0 else q
        let mag : Nat := (Float32.abs q).toFloat.toUInt64.toNat
        let qi : Int := if q < 0 then -(mag : Int) else (mag : Int)
        codes := codes.push (UInt8.ofNat (qi % 256).toNat)
  return (codes, xs)

/-- A byte as the int8 it holds. -/
def asInt8 (b : UInt8) : Int := if b.toNat ≥ 128 then (b.toNat : Int) - 256 else b.toNat

/-- The FP32 total of one output of the A8 route; `row` of `w`, token `tk` of `codes`/`xs`. -/
def a8Total (w : Weight) (row : Nat) (codes : ByteArray) (xs : Array UInt16) (tk : Nat) : Float32 := Id.run do
  let mut acc : Float32 := 0
  for g in [0:w.groups] do
    let mut dsum : Int := 0
    for i in [0:groupK] do
      dsum := dsum + decode (w.code row g i) * asInt8 (codes.get! (tk * w.groups * groupK + g * groupK + i))
    let sw := f16ToF32 (w.s[row * w.groups + g]!)
    let sx := f16ToF32 (xs[tk * w.groups + g]!)
    let p1 := (Float.ofInt dsum).toFloat32 * sw
    let p2 := p1 * sx
    acc := acc + p2
  return acc

def a8Totals (w : Weight) (codes : ByteArray) (xs : Array UInt16) (t rowBegin rows : Nat) : Array Float32 := Id.run do
  let mut out : Array Float32 := Array.replicate (t * rows) 0
  for r in [0:rows] do
    for tk in [0:t] do
      out := out.set! (tk * rows + r) (a8Total w (rowBegin + r) codes xs tk)
  return out

/-! ## Placing, and the FP64 reference -/

/-- What the kernel leaves in the bfloat16 plane `[t, outLd]`: `init` with columns
`[outOff, outOff + rows)` replaced by the bfloat16 rounding of the total (plus the old value, in FP32,
when `accumulate`). -/
def place (totals : Array Float32) (t rows outLd outOff : Nat) (accumulate : Bool) (init : Array UInt16) :
    Array UInt16 := Id.run do
  let mut out := init
  for tk in [0:t] do
    for r in [0:rows] do
      let at_ := tk * outLd + outOff + r
      let tot := totals[tk * rows + r]!
      let tot := if accumulate then tot + bf16ToF32 init[at_]! else tot
      out := out.set! at_ (f32ToBf16 tot)
  return out

/-- The mathematics: `x · W^T` in double precision, with the weights as stored (decoded, times their
binary16 scale). `[t, rows]`. -/
def ref64 (w : Weight) (x : FloatArray) (t rowBegin rows : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.mk (Array.replicate (t * rows) 0.0)
  for r in [0:rows] do
    let wr := w.decodedRow (rowBegin + r)
    let sc := w.scaleRow (rowBegin + r)
    let mut dense := FloatArray.emptyWithCapacity w.k
    for pos in [0:w.k] do
      dense := dense.push (wr.get! pos * sc.get! (pos / groupK))
    for tk in [0:t] do
      let mut acc : Float := 0
      for pos in [0:w.k] do
        acc := acc + dense.get! pos * x.get! (tk * w.k + pos)
      out := out.set! (tk * rows + r) acc
  return out

/-! ## SwiGLU -/

/-- `silu(gate) * up` over a bfloat16 plane `[t, 2 * inter]` laid out `[gate | up]`, rounded to
bfloat16: `[t, inter]`. `exp` is the host's own, as the device's is its own. -/
def siluMul (plane : Array UInt16) (t inter : Nat) : Array UInt16 := Id.run do
  let mut out : Array UInt16 := Array.mkEmpty (t * inter)
  for tk in [0:t] do
    for r in [0:inter] do
      let g := bf16ToF32 plane[tk * 2 * inter + r]!
      let u := bf16ToF32 plane[tk * 2 * inter + inter + r]!
      let sg := g / (1.0 + Float32.exp (-g))
      out := out.push (f32ToBf16 (sg * u))
  return out

end Ninfer.Ops.Linear
