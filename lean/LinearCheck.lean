-- SPDX-License-Identifier: Apache-2.0
import Ninfer.Num
import Ninfer.Rng
import Ninfer.Ops.Linear.Spec
import Ninfer.Ops.Linear.Kernels

/-!
# `ninfer_linear` — fixtures for the linear kernels, and the verdicts on what hosts return

    ninfer_linear plan                             one line per (case, variant): <case> <variant> <kernel> <cpu|gpu>
    ninfer_linear write <dir> <case> [<control>]   inputs, expected outputs and a host script per variant
    ninfer_linear check <dir> <variant>            judge `<variant>.*.out` against the expectation
    ninfer_linear selftest                         the checks that need no host

A *host* (`tests/kernels/cpu_host.cpp`, `tests/kernels/metal_host.swift`) only dispatches: it reads
a script, runs the kernel over the files, and writes the buffers back. Every verdict is made here.

The cases belong to a format (`Fmt`). The ternary format has all four kinds, and its ids are bare
(`e0`, `c1024x5120t1`, `q300t2`, `s192`). The signed-integer formats `q4`, `q5`, `q6` and `q8` have
the A16 route only, as the engine tests them, and their ids start with the format (`q4-e0`,
`q8-c130x5120t2`).

The expectations are the reference model of `Ninfer.Ops.Linear.Spec`:

* `exact` cases: the bfloat16 plane must equal the model's bit for bit, columns outside the written
  range untouched; the `_f32` probes must return the model's FP32 total bit for bit (a bfloat16
  comparison hides a last-bit difference in the FP32 sum, an FP32 one does not).
* `crit` cases: the plane must meet the engine's accuracy criterion against the FP64 reference: the
  constants of `tests/ops/linear/linear_test_common.cpp`, the formula of `reduction_passes` in
  `tests/ops/op_check.h`. Relative L2 error at most one bfloat16 roundoff (A16) or 0.04 (A8), and
  the largest error at most `1/256 + g * max|ref|` with g = 2/256 (A16) or 0.06 (A8).
* `quant` cases: int8 codes and binary16 scales must equal the model's; the fixture holds a zero
  group, an outlier, and a group of exact ties.
* `swiglu`: the projection is exact; SwiGLU may differ by one bfloat16 step on a rare element (the
  device's `exp` is its own).

A control (`flip-row`) hands the kernel weights with one row's signs flipped while the expectation
stays; its verdict must be FAIL.
-/

open Ninfer Ninfer.Num Ninfer.Slang Ninfer.Slang.Dag Ninfer.Ops.Linear

/-! ## Cases -/

inductive Kind | exact | crit | quant | swiglu
deriving BEq, Repr

structure Case where
  id : String
  kind : Kind
  n : Nat
  k : Nat
  t : Nat
  rowBegin : Nat := 0
  rows : Nat
  outLd : Nat
  outOff : Nat := 0
  accumulate : Bool := false
  /-- The weight format. -/
  fmt : Fmt := Fmt.t2

def kindName : Kind → String
  | .exact => "exact" | .crit => "crit" | .quant => "quant" | .swiglu => "swiglu"

def exactCases : List Case :=
  [ { id := "e0", kind := .exact, n := 64, k := 5120, t := 1, rows := 64, outLd := 64 }
  , { id := "e1", kind := .exact, n := 96, k := 200, t := 5, rows := 96, outLd := 96 }
  , { id := "e2", kind := .exact, n := 130, k := 6144, t := 3, rowBegin := 17, rows := 70, outLd := 70 }
  , { id := "e3", kind := .exact, n := 130, k := 5120, t := 4, rowBegin := 30, rows := 40, outLd := 100, outOff := 25 }
  , { id := "e4", kind := .exact, n := 96, k := 384, t := 6, rowBegin := 10, rows := 48, outLd := 48, accumulate := true }
  , { id := "e5", kind := .exact, n := 257, k := 17408, t := 2, rows := 257, outLd := 257 }
  -- one input past a whole group, tokens not a multiple of four, a column range, accumulate
  , { id := "e6", kind := .exact, n := 33, k := 129, t := 7, rowBegin := 1, rows := 31, outLd := 40, outOff := 5, accumulate := true }
  -- K a single lane of a single group
  , { id := "e7", kind := .exact, n := 8, k := 3, t := 5, rows := 8, outLd := 8 } ]

/-- The engine's accuracy criterion, at the shapes of the Mojo suite this replaces (`n`, `k`, and
`t` in 1, 2, 17, 65, 129 up to n = 5120; 1 and 2 above). -/
def critCases : List Case :=
  let mk (n k : Nat) (ts : List Nat) : List Case :=
    ts.map fun t => { id := s!"c{n}x{k}t{t}", kind := .crit, n, k, t, rows := n, outLd := n }
  let ts : List Nat := [1, 2, 17, 65, 129]
  mk 1024 5120 ts ++ mk 130 5120 ts ++ mk 512 200 ts ++ mk 5120 6144 ts ++ mk 34816 5120 [1, 2]

def quantCases : List Case :=
  [ { id := "q5120t3", kind := .quant, n := 1, k := 5120, t := 3, rows := 1, outLd := 1 }
  , { id := "q200t4", kind := .quant, n := 1, k := 200, t := 4, rows := 1, outLd := 1 }
  , { id := "q300t2", kind := .quant, n := 1, k := 300, t := 2, rows := 1, outLd := 1 } ]

def swigluCases : List Case :=
  [ { id := "s192", kind := .swiglu, n := 192, k := 5120, t := 5, rows := 192, outLd := 192 } ]

/-- The A16 cases of a signed-integer format, at the shapes of the ternary ones where they make
sense: ragged K (200, 129, 3, which leave a group partly and wholly past K), a row range, a column
range, `accumulate`, tokens that are not a multiple of four, and a long K. -/
def intCases (f : Fmt) : List Case :=
  let mk (id : String) (n k t rowBegin rows outLd outOff : Nat) (accumulate : Bool) : Case :=
    { id := s!"{f.name}-{id}", kind := .exact, fmt := f, n, k, t, rowBegin, rows, outLd, outOff, accumulate }
  [ mk "e0" 64 5120 1 0 64 64 0 false
  , mk "e1" 96 200 5 0 96 96 0 false
  , mk "e2" 130 6144 3 17 70 70 0 false
  , mk "e3" 130 5120 4 30 40 100 25 false
  , mk "e4" 96 384 6 10 48 48 0 true
  , mk "e5" 40 17408 2 0 40 40 0 false
  , mk "e6" 33 129 7 1 31 40 5 true
  , mk "e7" 8 3 5 0 8 8 0 false ]

/-- The engine's accuracy criterion for a signed-integer format, at the widths of its A16 tests. -/
def intCritCases (f : Fmt) : List Case :=
  let mk (n k : Nat) (ts : List Nat) : List Case :=
    ts.map fun t => { id := s!"{f.name}-c{n}x{k}t{t}", kind := .crit, fmt := f, n, k, t, rows := n, outLd := n }
  mk 1024 5120 [1, 17] ++ mk 130 5120 [1, 2, 17, 65] ++ mk 512 200 [1, 17, 65]

def allCases : List Case :=
  exactCases ++ critCases ++ quantCases ++ swigluCases ++ Fmt.ints.flatMap fun f => intCases f ++ intCritCases f

/-- Whether `pat` occurs in `s`. -/
def hasPart (s pat : String) : Bool := (s.splitOn pat).length > 1

def findCase (id : String) : Option Case := allCases.find? (·.id == id)

/-- The variants of a case: (name, kernel). The `_f32` ones are probes. -/
def variants (c : Case) : List (String × String) :=
  if c.fmt.bits != 2 then
    let k (s : String) := s!"{c.fmt.name}_a16_{s}"
    match c.kind with
    | .exact => [ (k "serial", k "serial"), (k "lanes", k "lanes"), (k "wide", k "wide")
                , (k "serial_f32", k "serial_f32"), (k "lanes_f32", k "lanes_f32"), (k "wide_f32", k "wide_f32") ]
    | .crit => [ (k "serial", k "serial"), (k "lanes", k "lanes") ]
    | _ => []
  else
  match c.kind with
  | .exact =>
    [ ("a16_serial", "t2_a16_serial"), ("a16_lanes", "t2_a16_lanes")
    , ("a16_wide", "t2_a16_wide")
    , ("a16_serial_f32", "t2_a16_serial_f32"), ("a16_lanes_f32", "t2_a16_lanes_f32")
    , ("a16_wide_f32", "t2_a16_wide_f32")
    , ("a8_serial", "t2_a8_serial"), ("a8_lanes", "t2_a8_lanes")
    , ("a8_serial_f32", "t2_a8_serial_f32"), ("a8_lanes_f32", "t2_a8_lanes_f32") ]
  | .crit =>
    [ ("a16_serial", "t2_a16_serial"), ("a16_lanes", "t2_a16_lanes")
    , ("a8_serial", "t2_a8_serial"), ("a8_lanes", "t2_a8_lanes") ]
  | .quant => [("a8_quantize", "t2_a8_quantize")]
  | .swiglu =>
    [ ("a16_serial", "t2_a16_serial"), ("a16_lanes", "t2_a16_lanes")
    , ("silu", "t2_silu_mul") ]

/-! ## Bytes -/

def u16Bytes (a : Array UInt16) : ByteArray :=
  a.foldl (fun (b : ByteArray) (v : UInt16) => (b.push v.toUInt8).push (v >>> (8 : UInt16)).toUInt8)
    (ByteArray.emptyWithCapacity (2 * a.size))

def u32Bytes (a : Array UInt32) : ByteArray :=
  a.foldl (fun (b : ByteArray) (v : UInt32) =>
    (((b.push v.toUInt8).push (v >>> (8 : UInt32)).toUInt8).push (v >>> (16 : UInt32)).toUInt8).push
      (v >>> (24 : UInt32)).toUInt8)
    (ByteArray.emptyWithCapacity (4 * a.size))

def f64Bytes (a : FloatArray) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity (8 * a.size)
  for i in [0:a.size] do
    let v := (a.get! i).toBits
    for s in [0:8] do b := b.push (v >>> (8 * s).toUInt64).toUInt8
  return b

def readU16 (b : ByteArray) : Array UInt16 := Id.run do
  let mut out := Array.mkEmpty (b.size / 2)
  for i in [0:b.size / 2] do
    out := out.push ((b.get! (2 * i)).toUInt16 ||| ((b.get! (2 * i + 1)).toUInt16 <<< 8))
  return out

def readU32 (b : ByteArray) : Array UInt32 := Id.run do
  let mut out := Array.mkEmpty (b.size / 4)
  for i in [0:b.size / 4] do
    let g := fun j => (b.get! (4 * i + j)).toUInt32
    out := out.push (g 0 ||| (g 1 <<< 8) ||| (g 2 <<< 16) ||| (g 3 <<< 24))
  return out

def f32Bits (a : Array Float32) : Array UInt32 := a.map (·.toBits)

/-! ## Fixtures -/

/-- Activations with the special groups of the quantisation cases: a zero group (token 0, group 0),
one outlier (token 1, input 133) and a group of exact ties on the last token, whose amax of 127
gives a scale of exactly 1, so `x * inv` is the value itself and the half-integers are real ties. -/
def specialActivations (abits : Array UInt16) (t k : Nat) : Array UInt16 × FloatArray := Id.run do
  let mut bits := abits
  let tie : List Float := [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, -2.5, -0.5, -1.5, 6.5, 7.5, 8.5]
  for i in [0:128] do
    if i < k then bits := bits.set! i 0
  if k > 133 && t > 1 then bits := bits.set! (k + 133) (f32ToBf16 10000.0)
  let last := (t - 1) * k
  for i in [0:128] do
    if i < k then bits := bits.set! (last + i) 0
  bits := bits.set! last (f32ToBf16 127.0)
  for (v, j) in tie.zipIdx do
    if j + 1 < k then bits := bits.set! (last + 1 + j) (f32ToBf16 v.toFloat32)
  let vals := FloatArray.mk (bits.map fun b => (bf16ToF32 b).toFloat)
  return (bits, vals)

/-- Flip 01 <-> 11 in every field of row `r`'s code bytes (the control). -/
def flipRow (q : ByteArray) (r groups : Nat) : ByteArray := Id.run do
  let mut out := q
  for i in [r * groups * codeBytes : (r + 1) * groups * codeBytes] do
    let b := out.get! i
    out := out.set! i (b ^^^ ((b &&& 0x55) <<< 1))
  return out

/-- XOR 0xFF into every code byte of row `r`: every code of the row changes (the control for the
signed-integer formats, whose codes have no sign to flip bit by bit). -/
def flipRowInt (q : ByteArray) (r groups : Nat) : ByteArray := Id.run do
  let mut out := q
  for i in [r * groups * codeBytes : (r + 1) * groups * codeBytes] do
    out := out.set! i (out.get! i ^^^ 0xFF)
  return out

structure Fixture where
  w : Weight
  abits : Array UInt16
  avals : FloatArray
  init : Array UInt16

def makeFixture (c : Case) : Fixture :=
  let seed : UInt64 := (c.n * 7 + c.k + c.t).toUInt64
  let legal := c.kind != .exact
  -- the ternary fixtures keep their scales positive; every integer format stores half negative
  let w := makeWeight seed c.n c.k legal c.fmt (c.fmt.bits != 2)
  let spread := c.kind == .exact
  let (ab, av) := makeActivations (seed + 1) c.t c.k spread
  let (ab, av) := if c.kind == .quant then specialActivations ab c.t c.k else (ab, av)
  let init : Array UInt16 :=
    if c.accumulate then Id.run do
      let mut r := Rng.seed (seed + 2)
      let mut out : Array UInt16 := Array.mkEmpty (c.t * c.outLd)
      for _ in [0:c.t * c.outLd] do
        let (z, r') := r.normal
        r := r'
        out := out.push (f32ToBf16 (z * 3.0).toFloat32)
      out
    else Array.replicate (c.t * c.outLd) 0xFFFF
  { w, abits := ab, avals := av, init }

/-! ## The host script of one variant -/

def groupsOfCase (c : Case) : Nat := groupsOfFmt c.fmt c.k

def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Script lines: kernel, grid, parameters, then `buf <binding> <bytes per element> <in> <out|->`. -/
def script (c : Case) (variant kernel : String) : String :=
  let g := groupsOfCase c
  let params : List Nat :=
    if kernel == "t2_a8_quantize" then [c.k, g, c.t]
    else if kernel == "t2_silu_mul" then [c.rows / 2, c.t * (c.rows / 2)]
    else [c.k, g, c.t, c.rowBegin, c.rows, c.outLd, c.outOff, if c.accumulate then 1 else 0]
  let grid : Nat × Nat × Nat :=
    if kernel == "t2_a8_quantize" then (ceilDiv (g * c.t) 64, 1, 1)
    else if kernel == "t2_silu_mul" then (ceilDiv (c.t * (c.rows / 2)) 256, 1, 1)
    else if hasPart kernel "_a16_serial" || kernel.startsWith "t2_a8_serial" then (ceilDiv (c.rows * c.t) 64, 1, 1)
    else if hasPart kernel "_a16_wide" then (c.rows, ceilDiv c.t 4, 1)
    else (c.rows, c.t, 1)
  let probe := kernel.endsWith "_f32"
  let dstIn := if probe then "probe0.bin" else "init.bin"
  let dstBytes := if probe then 4 else 2
  let bufs : List String :=
    if kernel == "t2_a8_quantize" then
      [ "buf 1 2 a.bin -", s!"buf 2 1 qinit.bin {variant}.codes.out", s!"buf 3 2 xinit.bin {variant}.xscales.out" ]
    else if kernel == "t2_silu_mul" then
      [ "buf 1 2 plane.bin -", s!"buf 2 2 sinit.bin {variant}.dst.out" ]
    else if hasPart kernel "_a16_" then
      if c.fmt.hiBytes > 0 then
        [ "buf 1 1 q.bin -", "buf 2 1 qh.bin -", "buf 3 2 s.bin -", "buf 4 2 a.bin -"
        , s!"buf 5 {dstBytes} {dstIn} {variant}.dst.out" ]
      else
        [ "buf 1 1 q.bin -", "buf 2 2 s.bin -", "buf 3 2 a.bin -", s!"buf 4 {dstBytes} {dstIn} {variant}.dst.out" ]
    else
      [ "buf 1 1 q.bin -", "buf 2 2 s.bin -", "buf 3 1 codes.bin -", "buf 4 2 xs.bin -"
      , s!"buf 5 {dstBytes} {dstIn} {variant}.dst.out" ]
  String.intercalate "\n"
    ([ s!"kernel {kernel}", s!"grid {grid.1} {grid.2.1} {grid.2.2}"
     , "params " ++ String.intercalate " " (params.map toString) ] ++ bufs) ++ "\n"

/-! ## write -/

def writeCase (dir : String) (c : Case) (control : Option String) : IO UInt32 := do
  IO.FS.createDirAll dir
  let fx := makeFixture c
  let w := fx.w
  let g := groupsOfCase c
  let ternary := c.fmt.bits == 2
  if ternary && g > maxGroups then
    IO.eprintln s!"{c.id}: {g} groups exceed the A8 lanes kernel's {maxGroups}"
    return 2
  let q :=
    if control == some "flip-row" then
      (if ternary then flipRow w.q (c.rowBegin + 7) g else flipRowInt w.q (c.rowBegin + 7) g)
    else w.q
  IO.FS.writeBinFile s!"{dir}/q.bin" q
  if c.fmt.hiBytes > 0 then IO.FS.writeBinFile s!"{dir}/qh.bin" w.qh
  IO.FS.writeBinFile s!"{dir}/s.bin" (u16Bytes w.s)
  -- The activations are followed by NaN. A kernel owns `[t, k]` and nothing after it: the padding of
  -- the last group (where the weight codes are zero) is not its to read, and 0 * NaN is NaN, so a
  -- kernel that reads it fails the verdict instead of passing on a lucky zero.
  let tail : Array UInt16 := Array.replicate (g * c.fmt.group - c.k) 0x7FC0
  IO.FS.writeBinFile s!"{dir}/a.bin" (u16Bytes (fx.abits ++ tail))
  IO.FS.writeBinFile s!"{dir}/init.bin" (u16Bytes fx.init)
  -- outputs start as poison (the engine's 0xff): a position a kernel does not write cannot pass
  IO.FS.writeBinFile s!"{dir}/probe0.bin" (u32Bytes (Array.replicate (c.t * c.rows) 0xFFFFFFFF))
  -- the A8 route and SwiGLU belong to the ternary format
  let (codes, xs) := if ternary then quantize fx.avals c.t c.k else (ByteArray.empty, #[])
  if ternary then
    IO.FS.writeBinFile s!"{dir}/qinit.bin" (ByteArray.mk (Array.replicate (c.t * g * groupK) 0xFF))
    IO.FS.writeBinFile s!"{dir}/xinit.bin" (u16Bytes (Array.replicate (c.t * g) 0xFFFF))
    IO.FS.writeBinFile s!"{dir}/sinit.bin" (u16Bytes (Array.replicate (c.t * (c.rows / 2)) 0xFFFF))
    IO.FS.writeBinFile s!"{dir}/codes.bin" codes
    IO.FS.writeBinFile s!"{dir}/xs.bin" (u16Bytes xs)
  IO.FS.writeFile s!"{dir}/case.txt"
    (s!"id {c.id}\nkind {kindName c.kind}\nn {c.n}\nk {c.k}\nt {c.t}\nrowBegin {c.rowBegin}\nrows {c.rows}\n"
      ++ s!"outLd {c.outLd}\noutOff {c.outOff}\naccumulate {if c.accumulate then 1 else 0}\n")
  match c.kind with
  | .exact =>
    let t16 := a16Totals w fx.avals c.t c.rowBegin c.rows
    IO.FS.writeBinFile s!"{dir}/want_total16.bin" (u32Bytes (f32Bits t16))
    IO.FS.writeBinFile s!"{dir}/want_dst16.bin"
      (u16Bytes (place t16 c.t c.rows c.outLd c.outOff c.accumulate fx.init))
    if ternary then
      let t8 := a8Totals w codes xs c.t c.rowBegin c.rows
      IO.FS.writeBinFile s!"{dir}/want_total8.bin" (u32Bytes (f32Bits t8))
      IO.FS.writeBinFile s!"{dir}/want_dst8.bin"
        (u16Bytes (place t8 c.t c.rows c.outLd c.outOff c.accumulate fx.init))
  | .crit =>
    IO.FS.writeBinFile s!"{dir}/ref64.bin" (f64Bytes (ref64 w fx.avals c.t c.rowBegin c.rows))
  | .quant =>
    IO.FS.writeBinFile s!"{dir}/want_codes.bin" codes
    IO.FS.writeBinFile s!"{dir}/want_xs.bin" (u16Bytes xs)
  | .swiglu =>
    let t16 := a16Totals w fx.avals c.t c.rowBegin c.rows
    let plane := place t16 c.t c.rows c.outLd c.outOff c.accumulate fx.init
    IO.FS.writeBinFile s!"{dir}/want_dst16.bin" (u16Bytes plane)
    IO.FS.writeBinFile s!"{dir}/plane.bin" (u16Bytes plane)
    IO.FS.writeBinFile s!"{dir}/want_silu.bin" (u16Bytes (siluMul plane c.t (c.rows / 2)))
  for (variant, kernel) in variants c do
    IO.FS.writeFile s!"{dir}/script.{variant}.txt" (script c variant kernel)
  return 0

/-! ## check -/

def readCase (dir : String) : IO Case := do
  let txt ← IO.FS.readFile s!"{dir}/case.txt"
  let get (key : String) : Nat :=
    ((txt.splitOn "\n").filterMap fun l =>
      match l.splitOn " " with
      | [k, v] => if k == key then v.toNat? else none
      | _ => none).headD 0
  let id := ((txt.splitOn "\n").filterMap fun l =>
    match l.splitOn " " with | ["id", v] => some v | _ => none).headD ""
  match findCase id with
  | some c => return c
  | none => throw (IO.userError s!"unknown case {id} in {dir}/case.txt (rows {get "rows"})")

/-- Compare two arrays; report the count of differences and the first. -/
def diffBits {α : Type} [BEq α] [ToString α] [Inhabited α] (label : String) (got want : Array α) : String × Bool :=
  if got.size != want.size then (s!"{label}: {got.size} values, expected {want.size}", false)
  else
    let bad := (List.range got.size).filter fun i => got[i]! != want[i]!
    match bad with
    | [] => (s!"{label}: {got.size} values equal", true)
    | i :: _ => (s!"{label}: {bad.length} of {got.size} differ; first at {i}: got {got[i]!}, want {want[i]!}", false)

/-- A bit comparison against a constant, all-zero or mostly-zero reference proves nothing: more than
half the values must be nonzero, with a spread of distinct ones. (A tiny K leaves some totals
legitimately zero, so the bar is half, not nearly all.) -/
def informative (vals : Array UInt32) : Bool :=
  let nonzero := vals.foldl (fun n v => if v != 0 && v != 0x80000000 then n + 1 else n) 0
  let distinct := (vals.toList.eraseDups).length
  nonzero * 2 > vals.size && distinct ≥ min 8 vals.size

def roundoff : Float := 1.0 / 256.0

def checkCrit (c : Case) (isA16 : Bool) (got : Array UInt16) (ref : Array Float) : String × Bool :=
  let (relL2, grossAbs, grossRel) :=
    if isA16 then (roundoff, roundoff, 2 * roundoff) else (0.04, roundoff, 1.5 * 0.04)
  let g := got.map fun b => (bf16ToF32 b).toFloat
  let finite := g.all fun v => v == v && v.abs < 1e30
  let errSq := (List.range g.size).foldl (fun s i => s + (g[i]! - ref[i]!) * (g[i]! - ref[i]!)) 0
  let refSq := ref.foldl (fun s v => s + v * v) 0
  let refMax := ref.foldl (fun m v => if v.abs > m then v.abs else m) 0
  let errMax := (List.range g.size).foldl (fun m i => let e := (g[i]! - ref[i]!).abs; if e > m then e else m) 0
  let rel := Float.sqrt errSq / Float.sqrt refSq
  let limit := grossAbs + grossRel * refMax
  let ok := finite && refMax > 0 && rel ≤ relL2 && errMax ≤ limit
  (s!"{c.id} {if isA16 then "A16" else "A8"}: relative L2 {rel} (limit {relL2}), max error {errMax} (limit {limit}), max |ref| {refMax}", ok)

def check (dir variant : String) : IO UInt32 := do
  let c ← readCase dir
  let rd (name : String) : IO ByteArray := IO.FS.readBinFile s!"{dir}/{name}"
  let probe := variant.endsWith "_f32"
  let isA16 := variant.startsWith "a16" || hasPart variant "_a16_"
  let (msg, ok) ← match c.kind, variant with
    | .quant, _ =>
      let gotC ← rd s!"{variant}.codes.out"
      let gotX ← rd s!"{variant}.xscales.out"
      let wantC ← rd "want_codes.bin"
      let wantX ← rd "want_xs.bin"
      let (m1, o1) := diffBits "int8 codes" gotC.toList.toArray wantC.toList.toArray
      let (m2, o2) := diffBits "binary16 scales" (readU16 gotX) (readU16 wantX)
      let want := wantC.toList.toArray.map asInt8
      let distinct := (want.toList.eraseDups).length
      let nonzero := (want.filter (· != 0)).size
      -- the fixture plants a 127; a reference of few distinct or mostly-zero codes proves little
      let spread := want.contains 127 && distinct ≥ 16 && nonzero * 10 ≥ want.size
      pure (m1 ++ "; " ++ m2 ++ s!"; {want.size} codes, {distinct} distinct, {nonzero} nonzero, 127 present {want.contains 127}",
        o1 && o2 && spread)
    | .swiglu, "silu" =>
      let got := readU16 (← rd "silu.dst.out")
      let want := readU16 (← rd "want_silu.bin")
      let steps := (List.range got.size).map fun i =>
        if got[i]! ≥ want[i]! then (got[i]! - want[i]!).toNat else (want[i]! - got[i]!).toNat
      let maxStep := steps.foldl max 0
      let exact := steps.filter (· == 0) |>.length
      let frac := exact.toFloat / got.size.toFloat
      pure (s!"silu: max {maxStep} bfloat16 steps, {frac} exact of {got.size}", maxStep ≤ 1 && frac > 0.99)
    | .crit, _ =>
      let got := readU16 (← rd s!"{variant}.dst.out")
      let refB ← rd "ref64.bin"
      let ref : Array Float := (List.range (refB.size / 8)).toArray.map fun i =>
        Float.ofBits ((List.range 8).foldl (fun a s => a ||| ((refB.get! (8 * i + s)).toUInt64 <<< (8 * s).toUInt64)) 0)
      pure (checkCrit c isA16 got ref)
    | _, _ =>
      if probe then
        let got := readU32 (← rd s!"{variant}.dst.out")
        let want := readU32 (← rd (if isA16 then "want_total16.bin" else "want_total8.bin"))
        let (m, o) := diffBits s!"{c.id} {variant} FP32 totals" got want
        pure (m ++ s!"; reference informative {informative want}", o && informative want)
      else
        let got := readU16 (← rd s!"{variant}.dst.out")
        let want := readU16 (← rd (if isA16 then "want_dst16.bin" else "want_dst8.bin"))
        let region : Array UInt32 := Id.run do
          let mut r := #[]
          for tk in [0:c.t] do
            for j in [0:c.rows] do r := r.push (want[tk * c.outLd + c.outOff + j]!).toUInt32
          return r
        let (m, o) := diffBits s!"{c.id} {variant} bfloat16 plane" got want
        pure (m ++ s!"; reference informative {informative region}", o && informative region)
  IO.println s!"{if ok then "PASS" else "FAIL"}  {variant}: {msg}"
  return (if ok then 0 else 1)

/-! ## selftest: the DAG of processes -/

section DagTests
open LeanSlang Ninfer.Slang.Dsl

def isOk {α : Type} (r : Except String α) : Bool :=
  match r with
  | .ok _ => true
  | .error _ => false

def why {α : Type} (r : Except String α) : String :=
  match r with
  | .ok _ => "accepted"
  | .error e => e

def occurrences (hay pat : String) : Nat := (hay.splitOn pat).length - 1

/-- The static check, the two lowerings, and a control for each thing the check claims. Returns the
number of failures. -/
def dagSelftest : IO Nat := do
  let mut fails := 0
  let report (name : String) (ok : Bool) (detail : String) : IO Unit :=
    IO.println s!"{if ok then "PASS" else "FAIL"}  {name}: {detail}"
  -- the shipped kernels
  let shipped := kernels {}
  let procs := (shipped.map (·.dag.procs.size)).foldl (· + ·) 0
  let shippedOk := shipped.all fun k => isOk k.dag.check
  report "the DAG of every shipped kernel passes the check" shippedOk s!"{shipped.length} kernels, {procs} processes"
  if !shippedOk then fails := fails + 1
  -- barriers: the fused kernel has one per edge level, no process has any
  let barrierWant : List (String × Nat) :=
    [("t2_a16_lanes", 6), ("t2_a16_wide", 6), ("t2_a8_lanes", 1), ("t2_a16_serial", 0)] ++
    Fmt.ints.flatMap fun f => [(s!"{f.name}_a16_lanes", 6), (s!"{f.name}_a16_wide", 6), (s!"{f.name}_a16_serial", 0)]
  for (name, want) in barrierWant do
    match shipped.find? (·.name == name) with
    | none =>
      report s!"{name} is shipped" false "missing"
      fails := fails + 1
    | some k =>
      match k.fused, k.modules with
      | .ok f, .ok ms =>
        let n := occurrences (LeanSlang.emit f) "GroupMemoryBarrierWithGroupSync()"
        let clean := ms.all fun m =>
          let t := LeanSlang.emit m.shader
          occurrences t "GroupMemoryBarrier" == 0 && occurrences t "groupshared" == 0
        report s!"{name}: {want} barrier(s) fused, none in {ms.length} process module(s)" (n == want && clean)
          s!"{n} in the fused kernel, modules clean {clean}"
        if !(n == want && clean) then fails := fails + 1
      | _, _ =>
        report s!"{name} lowers" false "an error"
        fails := fails + 1
  -- planted faults in the real kernels
  let faulty (m : Mutant) : Except String Unit :=
    match (kernels { mutant := m }).find? (·.name == "t2_a16_lanes") with
    | some k => k.dag.check
    | none => .error "no t2_a16_lanes"
  for (m, label) in ([(Mutant.mergeAll, "every process in one"), (Mutant.mergeFold, "two folds in one"),
                      (Mutant.dropEdges, "no edges")] : List (Mutant × String)) do
    let r := faulty m
    report s!"the check refuses {label}" (!isOk r) (why r)
    if isOk r then fails := fails + 1
  let rev := faulty .reverseEdges
  report "the check accepts reversed edges: only the model's verdict can say they are the wrong way round" (isOk rev) (why rev)
  if !isOk rev then fails := fails + 1
  -- the analyser on small DAGs of its own: each claim has a case that must fail and one that must pass
  let sh : SlangGroupSharedDecl := { name := "sh", elemType := fT, dims := [32] }
  let base : Dag :=
    { shared := [sh], procs := #[], threads := 32, threadId := "gtid", groupId := "gid"
      prelude := [let_ uT "lane" (.member (v "gtid") "x")] }
  let wr (idx rhs : E) : St := setIdx (v "sh") idx rhs
  let rd (idx : E) : E := .index (.var "sh") idx
  let own : Dag := { base with procs := #[{ name := "own", body := [wr (v "lane") (fl 1.0)] }] }
  let neighbour : Dag := { base with procs := #[{ name := "neighbour", body := [wr (v "lane") (rd (v "lane" + u 1))] }] }
  let readsOnly : Dag := { base with procs := #[{ name := "a", body := [let_ fT "x" (rd (v "lane"))] },
                                                 { name := "b", body := [let_ fT "y" (rd (u 0))] }] }
  let two : Dag := { base with procs := #[{ name := "a", body := [wr (v "lane") (fl 1.0)] },
                                          { name := "b", body := [wr (v "lane") (fl 2.0)] }] }
  let barriered : Dag := { base with procs := #[{ name := "barrier", body := [barrier] }] }
  let memoryPrelude : Dag := { own with prelude := own.prelude ++ [let_ fT "x" (rd (u 0))] }
  let cycle : Dag := { two with edges := [(0, 1), (1, 0)] }
  let cases : List (String × Dag × Bool) :=
    [ ("a process whose lanes write their own slots", own, true)
    , ("a lane reads the slot its neighbour writes, in one process", neighbour, false)
    , ("two processes that only read, no edge", readsOnly, true)
    , ("two processes that write one array, no edge", two, false)
    , ("the same two processes with an edge", { two with edges := [(0, 1)] }, true)
    , ("a barrier inside a process", barriered, false)
    , ("a prelude that reads memory", memoryPrelude, false)
    , ("a cycle", cycle, false) ]
  for (label, d, want) in cases do
    let r := d.check
    report s!"analyser: {label}" (isOk r == want) (why r)
    if isOk r != want then fails := fails + 1
  -- the order: a chain runs in sequence, and the levels of a diamond put the two branches together
  let chainOrder := (Dag.chain [] [] ((List.range 4).map fun i => ({ name := s!"p{i}", body := [] } : Proc))).order
  report "a chain runs in position order" (chainOrder.toOption == some [0, 1, 2, 3]) s!"{chainOrder.toOption}"
  if chainOrder.toOption != some [0, 1, 2, 3] then fails := fails + 1
  let diamond : Dag := { procs := (List.range 4).toArray.map fun i => { name := s!"p{i}", body := [] }, edges := [(0, 1), (0, 2), (1, 3), (2, 3)] }
  let lv := diamond.levels.toOption
  report "a diamond has the two branches on one level" (lv == some #[0, 1, 1, 2]) s!"{lv}"
  if lv != some #[0, 1, 1, 2] then fails := fails + 1
  return fails

end DagTests

/-! ## selftest: the checks that need no host -/

def selftest : IO UInt32 := do
  let mut fails := 0
  let report (name : String) (ok : Bool) (detail : String) : IO Unit :=
    IO.println s!"{if ok then "PASS" else "FAIL"}  {name}: {detail}"
  -- the decode formula against the table
  let table : List (Nat × Int) := [(0, 0), (1, 1), (2, 0), (3, -1)]
  let decodeOk := table.all fun (code, want) => decode code == want && decodeF code == Float.ofInt want
  report "decode" decodeOk "00 -> 0, 01 -> +1, 10 -> 0 (illegal), 11 -> -1"
  if !decodeOk then fails := fails + 1
  -- Float32 operations round once: the model's `a + b` equals the double sum rounded to binary32,
  -- which is correct rounding (53 >= 2 * 24 + 2). Fused or double-rounded arithmetic would not.
  let mut r := Rng.seed 7
  let mut badAdd := 0
  let mut badMul := 0
  for _ in [0:200000] do
    let (u, r1) := r.next
    let (v, r2) := r1.next
    r := r2
    let a := Float32.ofBits (u >>> 32).toUInt32
    let b := Float32.ofBits (v >>> 32).toUInt32
    if a == a && b == b && (a.toFloat).abs < 1e30 && (b.toFloat).abs < 1e30 then
      if (a + b).toBits != (a.toFloat + b.toFloat).toFloat32.toBits then badAdd := badAdd + 1
      if (a * b).toBits != (a.toFloat * b.toFloat).toFloat32.toBits then badMul := badMul + 1
  report "Float32 rounds each operation" (badAdd == 0 && badMul == 0) s!"200000 pairs, {badAdd} sums and {badMul} products differ"
  if badAdd + badMul != 0 then fails := fails + 1
  -- the summation order is visible in FP32: the tree and the plain sum disagree somewhere
  let c : Case := { id := "o", kind := .exact, n := 64, k := 5120, t := 4, rows := 64, outLd := 64 }
  let fx := makeFixture c
  let tree := a16Totals fx.w fx.avals c.t 0 c.rows
  let plain := a16Totals fx.w fx.avals c.t 0 c.rows true
  let differ := (List.range tree.size).filter (fun i => tree[i]!.toBits != plain[i]!.toBits) |>.length
  report "the order control sees a different order" (differ > tree.size / 4) s!"{differ} of {tree.size} FP32 totals differ"
  if differ ≤ tree.size / 4 then fails := fails + 1
  -- ... and the bfloat16 plane hides most of it, which is why the probes compare FP32
  let pt := place tree c.t c.rows c.outLd 0 false fx.init
  let pp := place plain c.t c.rows c.outLd 0 false fx.init
  let planeDiffer := (List.range pt.size).filter (fun i => pt[i]! != pp[i]!) |>.length
  report "bfloat16 hides the order" (planeDiffer < differ) s!"{planeDiffer} plane values differ against {differ} totals"
  if planeDiffer ≥ differ then fails := fails + 1
  -- a constant or all-zero reference proves nothing
  report "an all-zero reference is not informative" (!informative (Array.replicate 64 0)) "rejected"
  if informative (Array.replicate 64 0) then fails := fails + 1
  report "a constant reference is not informative" (!informative (Array.replicate 64 0x3f800000)) "rejected"
  if informative (Array.replicate 64 0x3f800000) then fails := fails + 1
  let mostlyZero : Array UInt32 := Array.ofFn (n := 100) fun i => if i.val < 60 then 0 else i.val.toUInt32 + 1
  report "a mostly-zero reference is not informative" (!informative mostlyZero) "60 of 100 zero: rejected"
  if informative mostlyZero then fails := fails + 1
  let varied : Array UInt32 := Array.ofFn (n := 100) fun i => i.val.toUInt32 + 1
  report "a varied reference is informative" (informative varied) "100 distinct nonzero: accepted"
  if !informative varied then fails := fails + 1
  -- the quantisation fixture really contains ties (so a tie-away kernel is caught)
  let qc : Case := { id := "q300t2", kind := .quant, n := 1, k := 300, t := 2, rows := 1, outLd := 1 }
  let qfx := makeFixture qc
  let (codes, xs) := quantize qfx.avals qc.t qc.k
  let lastBase := 1 * groupsOf qc.k * groupK
  let tieCodes := (List.range 8).map fun j => asInt8 (codes.get! (lastBase + 1 + j))
  report "ties round to even" (tieCodes == [0, 2, 2, 4, 4, 6, -2, 0] && xs[groupsOf qc.k]! == 0x3C00)
    s!"codes {tieCodes}, scale bits {xs[groupsOf qc.k]!}"
  if !(tieCodes == [0, 2, 2, 4, 4, 6, -2, 0] && xs[groupsOf qc.k]! == 0x3C00) then fails := fails + 1
  -- the signed-integer formats: every code at every input survives the engine's packer and its reader
  for f in Fmt.ints do
    let span := (f.qmax - f.qmin + 1).toNat
    let mut bad := 0
    for i in [0:f.group] do
      for d in [0:span] do
        let code := f.qmin + (d : Int)
        let codes : Array Int := (Array.replicate f.group (0 : Int)).set! i code
        let (nib, hi) := packGroup f codes
        let back := (List.range f.group).map fun j => unpackCode f nib hi 0 0 j
        if back != codes.toList then bad := bad + 1
    report s!"{f.name}: every code at every input survives pack and unpack" (bad == 0)
      s!"{f.group} inputs x {span} codes, {bad} differ"
    if bad != 0 then fails := fails + 1
  -- ... and the words come out as the engine's `pack_lowbit_group` writes them (worked by hand)
  let vec (f : Fmt) (set : List (Nat × Int)) : ByteArray × ByteArray :=
    packGroup f (set.foldl (fun (a : Array Int) (p : Nat × Int) => a.set! p.1 p.2) (Array.replicate f.group (0 : Int)))
  let w4 := vec Fmt.q4 [(0, -8), (1, 7), (2, -1)]
  let w5 := vec Fmt.q5 [(0, -16), (1, 15), (9, -1)]
  let w6 := vec Fmt.q6 [(0, -32), (1, 31), (5, -1)]
  let w8 := vec Fmt.q8 [(0, -127), (1, 127), (2, -1)]
  let vecOk :=
    (w4.1.get! 0 == 0x78 && w4.1.get! 1 == 0x0F)
    && (w5.1.get! 0 == 0xF0 && w5.1.get! 4 == 0xF0 && w5.2.get! 0 == 0x01 && w5.2.get! 1 == 0x02)
    && (w6.1.get! 0 == 0xF0 && w6.1.get! 2 == 0xF0 && w6.2.get! 0 == 0x06 && w6.2.get! 1 == 0x0C)
    && (w8.1.get! 0 == 0x81 && w8.1.get! 1 == 0x7F && w8.1.get! 2 == 0xFF)
  report "the integer packers write the words the engine's pack_lowbit_group writes" vecOk
    "q4 0x78 0x0F; q5 nibbles 0xF0 and high bits 0x01 0x02; q6 high bits 0x06 0x0C; q8 0x81 0x7F 0xFF"
  if !vecOk then fails := fails + 1
  -- the fixtures: codes in range, zero past K, a wholly padded group has scale 0, half the scales negative
  for f in Fmt.ints do
    let w := makeWeight 5 40 200 true f true
    let mut inRange := true
    let mut pastZero := true
    for row in [0:w.n] do
      for g in [0:w.groups] do
        for i in [0:f.group] do
          let v := w.value row g i
          if v < f.qmin || v > f.qmax then inRange := false
          if g * f.group + i ≥ w.k && v != 0 then pastZero := false
    let padScale := (List.range w.groups).all fun g => g * f.group < w.k || w.s[g]! == 0
    let big := makeWeight 11 64 5120 true f true
    let neg := big.s.foldl (fun n b => if b &&& 0x8000 != 0 then n + 1 else n) 0
    let frac := neg.toFloat / big.s.size.toFloat
    let ok := inRange && pastZero && padScale && frac > 0.4 && frac < 0.6
    report s!"{f.name}: codes in range, zero past K, padded groups unscaled, scales signed" ok
      s!"in range {inRange}, past K zero {pastZero}, padded scale 0 {padScale}, {frac} of scales negative"
    if !ok then fails := fails + 1
  -- the summation order is visible in these formats too
  for f in [Fmt.q4, Fmt.q8] do
    let oc : Case := { id := "o", kind := .exact, fmt := f, n := 64, k := 5120, t := 4, rows := 64, outLd := 64 }
    let ofx := makeFixture oc
    let tr := a16Totals ofx.w ofx.avals oc.t 0 oc.rows
    let pl := a16Totals ofx.w ofx.avals oc.t 0 oc.rows true
    let df := (List.range tr.size).filter (fun i => tr[i]!.toBits != pl[i]!.toBits) |>.length
    report s!"{f.name}: the order control sees a different order" (df > tr.size / 4) s!"{df} of {tr.size} FP32 totals differ"
    if df ≤ tr.size / 4 then fails := fails + 1
  fails := fails + (← dagSelftest)
  return (if fails == 0 then 0 else 1)

/-! ## main -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["plan"] =>
    for c in allCases do
      for (variant, kernel) in variants c do
        IO.println s!"{c.id} {kindName c.kind} {variant} {kernel}"
    return 0
  | ["write", dir, id] | ["write", dir, id, "none"] =>
    match findCase id with
    | some c => writeCase dir c none
    | none => IO.eprintln s!"unknown case {id}"; return 2
  | ["write", dir, id, control] =>
    match findCase id with
    | some c => writeCase dir c (some control)
    | none => IO.eprintln s!"unknown case {id}"; return 2
  | ["check", dir, variant] => check dir variant
  | ["selftest"] => selftest
  | _ =>
    IO.eprintln "usage: ninfer_linear plan | write <dir> <case> [flip-row] | check <dir> <variant> | selftest"
    return 2
