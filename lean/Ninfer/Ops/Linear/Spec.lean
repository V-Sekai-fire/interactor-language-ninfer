-- SPDX-License-Identifier: Apache-2.0
import Ninfer.Num
import Ninfer.Rng

/-!
# `Ninfer.Ops.Linear.Spec` — the linear op, as arithmetic

What the kernels of `Ninfer.Ops.Linear.Kernels` must compute, in Lean. Every rounding is spelled
out, so the result is a function of the inputs: no schedule, device or library appears in it.

* **Formats** (`Fmt`): the row-split formats of the engine (`row_split_k128_v1`), the ternary `t2`
  with groups of 128 inputs and the signed integers `q4`, `q5`, `q6` (groups of 64) and `q8`
  (groups of 32). K is padded to a multiple of 128. Per output row and group a format has 32 bytes
  of code plane, `hiBytes` of high plane (`q5` keeps a fifth bit of each code there, `q6` two) and
  one binary16 scale. Every format has 32 code bytes, so lane `l` of a kernel owns byte `l` and the
  `group / 32` inputs in it.
* **Weights** (`Weight`): the planes of a format. A ternary code decodes to `m - 2 (m & s)`: `00` 0,
  `01` +1, `11` -1, the illegal `10` 0. A signed-integer code is a two's complement field of
  `bits` bits, as in the engine's `unpack_lowbit_code`. `packGroup` is the engine's
  `pack_lowbit_group`.
* **A16** (`a16Total`): lane `l` of 32 sums its `group / 32` products of a group left to right
  (each product of a decoded weight and a bfloat16 activation is exact), scales that partial and
  adds it to its accumulator, group after group; the lanes combine as `l + (l ^ 16)`, then `^ 8,
  4, 2, 1`. Inputs past the logical K are skipped.
* **A8** (`quantize`, `a8Total`, ternary only): activations to int8 per token and group of 128 with
  a binary16 scale `f16(amax / 127)`; the group's dot is an exact integer, and the total is
  `acc + ((float dot * sw) * sx)`, groups in order.
* **FP64 reference** (`ref64`): the dense weights times the activations in double precision, which
  is what ties the model to the mathematics rather than to itself.

FP32 here is `Float32`, whose every operation rounds to binary32; nothing fuses.
-/

namespace Ninfer.Ops.Linear

open Ninfer Ninfer.Num

/-- Inputs per scale group of the ternary format, and the multiple that K is padded to. -/
def groupK : Nat := 128
/-- Code bytes per row and group, in every format. -/
def codeBytes : Nat := 32

/-- A byte as the int8 it holds. -/
def asInt8 (b : UInt8) : Int := if b.toNat ≥ 128 then (b.toNat : Int) - 256 else b.toNat

/-! ## Formats -/

/-- A row-split format. -/
structure Fmt where
  /-- `t2`, `q4`, `q5`, `q6` or `q8`: the prefix of the names of its kernels and cases. -/
  name : String
  /-- Bits per code. -/
  bits : Nat
  /-- Inputs per scale group. -/
  group : Nat
  /-- Bytes of high plane per row and group: `group * (bits - 4) / 8` above four bits, else none. -/
  hiBytes : Nat
deriving BEq, Repr, Inhabited

namespace Fmt

def t2 : Fmt := { name := "t2", bits := 2, group := 128, hiBytes := 0 }
def q4 : Fmt := { name := "q4", bits := 4, group := 64, hiBytes := 0 }
def q5 : Fmt := { name := "q5", bits := 5, group := 64, hiBytes := 8 }
def q6 : Fmt := { name := "q6", bits := 6, group := 64, hiBytes := 16 }
def q8 : Fmt := { name := "q8", bits := 8, group := 32, hiBytes := 0 }

/-- The signed-integer formats. -/
def ints : List Fmt := [q4, q5, q6, q8]
def all : List Fmt := t2 :: ints
def ofName (s : String) : Option Fmt := all.find? (·.name == s)

/-- The inputs of a group that one of the 32 lanes owns. -/
def perLane (f : Fmt) : Nat := f.group / 32
/-- Scale groups in 128 inputs. -/
def perChunk (f : Fmt) : Nat := groupK / f.group
/-- The smallest and the largest legal code of a signed-integer format (`q8` has no -128). -/
def qmin (f : Fmt) : Int := if f.bits == 8 then -127 else -((1 <<< (f.bits - 1) : Nat) : Int)
def qmax (f : Fmt) : Int := if f.bits == 8 then 127 else ((1 <<< (f.bits - 1) : Nat) : Int) - 1

end Fmt

/-- The code of input `i` of a group of a signed-integer format, as the engine's
`unpack_lowbit_code` reads it: low bits from the nibble of byte `i / 2` (the even input in the low
nibble), the rest from the high plane (`q5` bit `i % 8` of byte `i / 8`, `q6` two bits at `2 i`),
a two's complement field of `bits` bits. `no` and `ho` are where the group starts in the code and
high planes. -/
def unpackCode (f : Fmt) (nib hi : ByteArray) (no ho i : Nat) : Int :=
  if f.bits == 8 then asInt8 (nib.get! (no + i))
  else
    let lowByte := (nib.get! (no + i / 2)).toNat
    let low := if i % 2 == 1 then lowByte >>> 4 else lowByte &&& 15
    let high : Nat :=
      if f.bits == 5 then ((hi.get! (ho + i / 8)).toNat >>> (i % 8)) &&& 1
      else if f.bits == 6 then ((hi.get! (ho + (2 * i) / 8)).toNat >>> ((2 * i) % 8)) &&& 3
      else 0
    let u := low ||| (high <<< 4)
    if u &&& (1 <<< (f.bits - 1)) != 0 then (u : Int) - ((1 <<< f.bits : Nat) : Int) else (u : Int)

/-- The engine's `pack_lowbit_group`: the `group` codes of one group as the 32 bytes of the code
plane and the `hiBytes` of the high plane. -/
def packGroup (f : Fmt) (codes : Array Int) : ByteArray × ByteArray := Id.run do
  let mut nib : Array UInt8 := Array.replicate codeBytes 0
  let mut hi : Array UInt8 := Array.replicate f.hiBytes 0
  for i in [0:f.group] do
    let u : Nat := (codes[i]! % ((1 <<< f.bits : Nat) : Int)).toNat
    if f.bits == 8 then
      nib := nib.set! i (UInt8.ofNat u)
    else
      let low := u &&& 15
      nib := nib.modify (i / 2) fun b => b ||| UInt8.ofNat (if i % 2 == 0 then low else low <<< 4)
      if f.bits == 5 then
        hi := hi.modify (i / 8) fun b => b ||| UInt8.ofNat (((u >>> 4) &&& 1) <<< (i % 8))
      else if f.bits == 6 then
        hi := hi.modify ((2 * i) / 8) fun b => b ||| UInt8.ofNat (((u >>> 4) &&& 3) <<< ((2 * i) % 8))
  return (ByteArray.mk nib, ByteArray.mk hi)

/-! ## Weights -/

/-- A ternary code to -1, 0 or +1: `m - 2 (m & s)` with `m` the low bit and `s` the high bit. -/
def decode (code : Nat) : Int :=
  let m := code &&& 1
  let s := (code >>> 1) &&& 1
  (m : Int) - 2 * ((m &&& s : Nat) : Int)

/-- `decode` as a double: `00` 0, `01` +1, `10` 0, `11` -1. -/
def decodeF (code : Nat) : Float := if code == 1 then 1.0 else if code == 3 then -1.0 else 0.0

/-- Row-split weights: `n` rows of `k` logical inputs, `groups = ceil(k / 128) * (128 / group)`
scale groups per row. -/
structure Weight where
  n : Nat
  k : Nat
  groups : Nat
  /-- `n * groups * 32` code bytes. -/
  q : ByteArray
  /-- `n * groups` binary16 patterns. -/
  s : Array UInt16
  /-- The format. -/
  fmt : Fmt := Fmt.t2
  /-- `n * groups * fmt.hiBytes` bytes of high plane; none for a format without one. -/
  qh : ByteArray := ByteArray.empty

/-- Scale groups of a row of `k` inputs: K padded to a multiple of 128, in groups of `f.group`. -/
def groupsOfFmt (f : Fmt) (k : Nat) : Nat := (k + groupK - 1) / groupK * f.perChunk

/-- `ceil(k / 128)`: the ternary group count. -/
def groupsOf (k : Nat) : Nat := groupsOfFmt Fmt.t2 k

namespace Weight

/-- The 2-bit code of input `i` of group `g` of `row` (the ternary format). -/
def code (w : Weight) (row g i : Nat) : Nat :=
  (w.q.get! ((row * w.groups + g) * codeBytes + i / 4)).toNat >>> (2 * (i % 4)) &&& 3

/-- The signed value of input `i` of group `g` of `row`. -/
def value (w : Weight) (row g i : Nat) : Int :=
  if w.fmt.bits == 2 then decode (w.code row g i)
  else
    let gi := row * w.groups + g
    unpackCode w.fmt w.q w.qh (gi * codeBytes) (gi * w.fmt.hiBytes) i

/-- The decoded row: `groups * group` entries (a double holds each exactly). -/
def decodedRow (w : Weight) (row : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity (w.groups * w.fmt.group)
  for g in [0:w.groups] do
    for i in [0:w.fmt.group] do
      out := out.push (Float.ofInt (w.value row g i))
  return out

/-- The row's scales as binary32. -/
def scaleRow (w : Weight) (row : Nat) : FloatArray := Id.run do
  let mut out := FloatArray.emptyWithCapacity w.groups
  for g in [0:w.groups] do
    out := out.push (f16ToF32 (w.s[row * w.groups + g]!)).toFloat
  return out

end Weight

/-! ## Fixtures -/

/-- Weights from a seed.

* Ternary: four 64-bit draws fill the 32 code bytes of a row and group. `legal` rewrites the
  illegal `10` to `11`; codes past the logical K are zero, as the engine pads.
* Signed integer: one draw per code, uniform over the legal codes, packed by `packGroup`. Codes past
  the logical K are zero, and a group wholly past it has the scale 0, as the engine writes it.

Scales are uniform in [0.01, 0.2), rounded to binary16; with `signedScales`, about half carry the
sign bit, which the engine's fixtures also do (a kernel must apply the stored scale as a signed
multiplier). The signs come from a generator of their own, so they leave every other draw alone. -/
def makeWeight (seed : UInt64) (n k : Nat) (legal : Bool) (f : Fmt := Fmt.t2)
    (signedScales : Bool := false) : Weight := Id.run do
  let groups := groupsOfFmt f k
  let mut r := Rng.seed seed
  let mut rs := Rng.seed (seed ^^^ 0x5BD1E9955BD1E995)
  let mut q := ByteArray.emptyWithCapacity (n * groups * codeBytes)
  let mut qh := ByteArray.emptyWithCapacity (n * groups * f.hiBytes)
  let mut s : Array UInt16 := Array.mkEmpty (n * groups)
  for _ in [0:n] do
    for g in [0:groups] do
      if f.bits == 2 then
        for _ in [0:4] do
          let (v, r') := r.next
          r := r'
          for b in [0:8] do
            let byte0 : UInt8 := (v >>> (8 * b).toUInt64).toUInt8
            let byte1 : UInt8 := if legal then byte0 ||| ((byte0 >>> 1) &&& 0x55) else byte0
            -- zero the fields whose input lies past K
            let lane := q.size % codeBytes
            let mut byte := byte1
            for fi in [0:4] do
              if g * groupK + lane * 4 + fi ≥ k then byte := byte &&& ~~~(3 <<< (2 * fi).toUInt8)
            q := q.push byte
      else
        let span : UInt64 := (f.qmax - f.qmin + 1).toNat.toUInt64
        let mut codes : Array Int := Array.mkEmpty f.group
        for i in [0:f.group] do
          if g * f.group + i < k then
            let (v, r') := r.next
            r := r'
            codes := codes.push (f.qmin + (((v >>> 32) % span).toNat : Int))
          else codes := codes.push 0
        let (nib, hi) := packGroup f codes
        q := q.append nib
        qh := qh.append hi
      let (u, r') := r.unit
      r := r'
      let (sv, rs') := rs.next
      rs := rs'
      let mag : UInt16 := f32ToF16 (0.01 + 0.19 * u).toFloat32
      let mag : UInt16 := if f.bits != 2 && g * f.group ≥ k then 0 else mag
      s := s.push (if signedScales && (sv >>> 63) == 1 then mag ||| 0x8000 else mag)
  return { n, k, groups, q, s, fmt := f, qh }

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

/-- One lane's accumulator: its `group / 32` inputs of every group, in the kernel's order. -/
partial def laneAcc (wr sc x : FloatArray) (groups k l xoff : Nat) (f : Fmt := Fmt.t2) : Float32 :=
  let per := f.perLane
  let rec go (g : Nat) (acc : Float32) : Float32 :=
    if g < groups then
      let base := g * f.group + l * per
      let rec part (j : Nat) (p : Float32) : Float32 :=
        if j < per then
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
def a16Total (wr sc x : FloatArray) (groups k xoff : Nat) (sequential : Bool := false)
    (f : Fmt := Fmt.t2) : Float32 :=
  combine ((List.range 32).toArray.map fun l => laneAcc wr sc x groups k l xoff f) sequential

/-- FP32 totals `[t, rows]` of rows `[rowBegin, rowBegin + rows)` against activations `[t, k]`. -/
def a16Totals (w : Weight) (x : FloatArray) (t rowBegin rows : Nat) (sequential : Bool := false) :
    Array Float32 := Id.run do
  let mut out : Array Float32 := Array.replicate (t * rows) 0
  for r in [0:rows] do
    let wr := w.decodedRow (rowBegin + r)
    let sc := w.scaleRow (rowBegin + r)
    for tk in [0:t] do
      out := out.set! (tk * rows + r) (a16Total wr sc x w.groups w.k (tk * w.k) sequential w.fmt)
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
      dense := dense.push (wr.get! pos * sc.get! (pos / w.fmt.group))
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
