-- SPDX-License-Identifier: Apache-2.0
import Ninfer.Num
import Ninfer.Rng

/-!
# `ninfer_num` — `Ninfer.Num` against the host compiler's hardware conversions

    ninfer_num gen <inputs.txt>                  write the binary32 patterns to convert
    ninfer_num check <oracle.txt> [<control>]    compare with what `tests/num/oracle.cpp` printed

`tests/num/run.exs` runs the three steps. The inputs are every bfloat16 pattern, every rounding
boundary of binary16 and of bfloat16 (each value, the exact midpoint to its successor, and one ulp
either side of each, both signs), and a random sample, so ties and the subnormal range are all
covered, not sampled. `check` exits 1 on any difference. A control swaps in a deliberately wrong
conversion, which must be caught: `tie-away` rounds a tie up instead of to even, `bf16-truncate`
drops the low half.
-/

open Ninfer Ninfer.Num

/-- Hex digits to a number; `none` on anything else. -/
def parseHex (s : String) : Option Nat :=
  s.toList.foldlM (fun acc c =>
    if c.isDigit then some (acc * 16 + (c.toNat - '0'.toNat))
    else if 'a' ≤ c && c ≤ 'f' then some (acc * 16 + (c.toNat - 'a'.toNat + 10))
    else none) 0

def hex (w : Nat) (n : Nat) : String :=
  let ds := Nat.toDigits 16 n
  String.ofList (List.replicate (w - ds.length) '0' ++ ds)

/-- The patterns to convert. -/
def inputs : Array UInt32 := Id.run do
  let mut out : Array UInt32 := #[]
  let signs : List UInt32 := [0, 0x80000000]
  -- binary16: each finite value, the midpoint to the next, one ulp either side.
  for h in [0:0x7C00] do
    let b := (f16ToF32 h.toUInt16).toBits
    let next : UInt32 := if h == 0x7BFF then 0x47800000 else (f16ToF32 (h + 1).toUInt16).toBits
    let mid := b + (next - b) / 2
    for sg in signs do
      for d in [b, b + 1, mid - 1, mid, mid + 1] do
        out := out.push (d ||| sg)
      if b > 0 then out := out.push ((b - 1) ||| sg)
  -- bfloat16: each pattern widened, and the midpoint to the next with one ulp either side.
  for h in [0:0x10000] do
    let b : UInt32 := h.toUInt32 <<< 16
    out := out.push b
    if h &&& 0x7F80 != 0x7F80 then
      for d in [b + 0x7FFF, b + 0x8000, b + 0x8001] do out := out.push d
  -- a random sample of everything else
  let mut r := Rng.seed 20261117
  for _ in [0:262144] do
    let (v, r') := r.next
    r := r'
    out := out.push (v >>> 32).toUInt32
  return out

/-- The deliberately wrong conversions the controls swap in. -/
def f32ToF16TieAway (x : Float32) : UInt16 :=
  -- Round to nearest, but a tie rounds up in magnitude instead of to even.
  let good := f32ToF16 x
  let u := x.toBits
  let e := ((u >>> 23) &&& 0xFF).toNat
  let m := u &&& 0x7FFFFF
  if 113 ≤ e && e < 143 && (m &&& 0x1FFF) == 0x1000 && (good &&& 1) == 0 then good + 1 else good

def f32ToBf16Truncate (x : Float32) : UInt16 :=
  if x.toBits &&& 0x7FFFFFFF > 0x7F800000 then f32ToBf16 x else (x.toBits >>> 16).toUInt16

def isNanF16 (h : UInt16) : Bool := (h &&& 0x7C00) == 0x7C00 && (h &&& 0x3FF) != 0
def isNanBf16 (h : UInt16) : Bool := (h &&& 0x7F80) == 0x7F80 && (h &&& 0x7F) != 0
def isNanF32 (u : UInt32) : Bool := (u &&& 0x7F800000) == 0x7F800000 && (u &&& 0x7FFFFF) != 0

def check (path : String) (control : Option String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let toF16 : Float32 → UInt16 := if control == some "tie-away" then f32ToF16TieAway else f32ToF16
  let toBf16 : Float32 → UInt16 := if control == some "bf16-truncate" then f32ToBf16Truncate else f32ToBf16
  let mut narrow := 0
  let mut widen := 0
  let mut bad16 := 0
  let mut badBf := 0
  let mut badW := 0
  let mut first : Array String := #[]
  for line in lines do
    match line.splitOn " " with
    | ["n", a, b, c] =>
      match parseHex a, parseHex b, parseHex c with
      | some u, some h16, some hbf =>
        narrow := narrow + 1
        let x := Float32.ofBits u.toUInt32
        let g16 := toF16 x
        let gbf := toBf16 x
        if !(g16 == h16.toUInt16 || (isNanF16 g16 && isNanF16 h16.toUInt16)) then
          bad16 := bad16 + 1
          if first.size < 5 then first := first.push s!"f16 {hex 8 u}: got {hex 4 g16.toNat} want {hex 4 h16}"
        if !(gbf == hbf.toUInt16 || (isNanBf16 gbf && isNanBf16 hbf.toUInt16)) then
          badBf := badBf + 1
          if first.size < 5 then first := first.push s!"bf16 {hex 8 u}: got {hex 4 gbf.toNat} want {hex 4 hbf}"
      | _, _, _ => IO.eprintln s!"unreadable line: {line}"; return 2
    | ["w", a, b] =>
      match parseHex a, parseHex b with
      | some h, some u =>
        widen := widen + 1
        let g := (f16ToF32 h.toUInt16).toBits
        if !(g == u.toUInt32 || (isNanF32 g && isNanF32 u.toUInt32)) then
          badW := badW + 1
          if first.size < 5 then first := first.push s!"widen {hex 4 h}: got {hex 8 g.toNat} want {hex 8 u}"
      | _, _ => IO.eprintln s!"unreadable line: {line}"; return 2
    | _ => if !line.isEmpty then IO.eprintln s!"unreadable line: {line}"; return 2
  IO.println s!"narrowing: {narrow} inputs, binary16 differs on {bad16}, bfloat16 differs on {badBf}"
  IO.println s!"widening:  {widen} patterns, binary16 to binary32 differs on {badW}"
  for f in first do IO.println s!"  {f}"
  if narrow < 500000 || widen != 65536 then
    IO.eprintln "the oracle covers too little: a silent skip reads as a pass"
    return 2
  return (if bad16 + badBf + badW == 0 then 0 else 1)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["gen", path] =>
    let ins := inputs
    IO.FS.writeFile path (String.intercalate "\n" (ins.toList.map (hex 8 ·.toNat)) ++ "\n")
    IO.println s!"wrote {ins.size} patterns to {path}"
    return 0
  | ["check", path] => check path none
  | ["check", path, control] => check path (some control)
  | _ =>
    IO.eprintln "usage: ninfer_num gen <inputs> | check <oracle> [tie-away|bf16-truncate]"
    return 2
