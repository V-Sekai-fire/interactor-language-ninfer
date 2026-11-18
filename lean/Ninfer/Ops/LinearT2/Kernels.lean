-- SPDX-License-Identifier: Apache-2.0
import LeanSlang
import Ninfer.Slang.Dsl
import Ninfer.Slang.Kernel
import Ninfer.Ops.LinearT2.Spec

/-!
# `Ninfer.Ops.LinearT2.Kernels` — the ternary linear as Slang, emitted from Lean

The T2G128_F16S ternary linear of `src/ops/linear/t2` (and the `linear_add` and `linear_swiglu`
wrappers) as `LeanSlang` trees. `Ninfer.Ops.LinearT2.Spec` is the reference: it fixes every
rounding, and these kernels do the same arithmetic in the same order, so the result is a function
of the inputs and not of the target or the schedule.

## Weights

Row-split planes. For output row `n` and group `g` of 128 inputs: 32 code bytes (four 2-bit codes
per byte, input `i` of the group in bits `2 (i % 4)` of byte `i / 4`) at `(n * groups + g) * 32`,
and one binary16 scale at `n * groups + g`. A code decodes as `m - 2 (m & s)` with `m` its low bit
and `s` its high bit: `00` is 0, `01` is +1, `11` is -1, and the illegal `10` is 0.

## The two routes

* **A16.** bfloat16 activations `[t, k]`, FP32 accumulation, bfloat16 output. Lane `l` of 32 adds
  its four products of a group left to right, scales that partial by the group's scale, and adds it
  to its accumulator, group after group. The 32 lanes combine in the tree `l + (l ^ 16)`, then
  `^ 8`, `^ 4`, `^ 2`, `^ 1`.
* **A8.** Activations quantised once per token and group of 128 to int8 with one binary16 scale
  (`amax / 127`); the product is an exact int32 dot per group, `acc += (dot * sw) * sx` in FP32,
  groups in order.

## Schedules

* `serial`: one thread per output, looping over the 32 lanes itself. It needs no barrier or
  cross-lane operation, so the C++ target runs it and the Lean model holds it bit for bit.
* `lanes`: a group of 32 threads per output (A16: one lane each, the tree in `groupshared`; A8:
  the groups spread over the lanes, summed in order by lane 0). The same bits as `serial`.
* `wide`: `lanes` over four tokens at once, so each decoded weight is shared by four tokens.

`probe` kernels store the FP32 total (`asuint`) into a `uint` plane instead of rounding to
bfloat16: a bfloat16 comparison hides a last-bit difference in the FP32 sum, an FP32 one does not.
`Mutant` plants one fault in the emitted text; each must be caught by the gate.
-/

namespace Ninfer.Ops.LinearT2

open LeanSlang Ninfer.Slang Ninfer.Slang.Dsl Ninfer.Slang.Dag

/-- The A8 lanes kernel keeps one FP32 per group in `groupshared`: `groups` may not exceed this. -/
def maxGroups : Nat := 256
/-- Tokens one group of the wide A16 kernel covers. -/
def wideTokens : Nat := 4

/-- A fault planted in the emitted text. -/
inductive Mutant
  | clean
  /-- The 32 lanes are added one after the other instead of in the tree. -/
  | sequentialFold
  /-- The illegal code `10` decodes to -2 (two's complement) instead of 0. -/
  | decodeIllegal
  /-- The check that keeps codes past the logical K out of the sum is dropped. -/
  | noGuard
  /-- Quantisation rounds a tie away from zero instead of to even. -/
  | tieAway
  /-- Every process of a barrier kernel in one: no barrier anywhere. -/
  | mergeAll
  /-- Two consecutive folds of the tree in one process: the barrier between them is gone. -/
  | mergeFold
  /-- No edges: processes that must be ordered are not. -/
  | dropEdges
  /-- Every edge reversed: the processes run in the wrong order. -/
  | reverseEdges
deriving BEq, Repr, Inhabited

/-- The DAG under a mutant: the faults of a barrier kernel are faults of its edges. -/
def Mutant.onDag : Mutant → Dag → Dag
  | .mergeAll, d => if d.procs.size > 1 then d.merge 0 (d.procs.size - 1) else d
  | .mergeFold, d => if d.procs.size > 3 then d.merge 1 2 else d
  | .dropEdges, d => { d with edges := [] }
  | .reverseEdges, d => { d with edges := d.edges.map fun (i, j) => (j, i) }
  | _, d => d

/-- How a kernel is emitted. -/
structure Cfg where
  mutant : Mutant := .clean
  /-- Store the FP32 total in a `uint` plane (no bfloat16 rounding, no accumulate). -/
  probe : Bool := false
deriving Inhabited

/-! ## Helper functions -/

/-- `asfloat(h << 16)`: bfloat16 to float. -/
def bf16ToF32Fn : SlangFunctionDecl :=
  fn fT "bf16ToF32" [prm "h" uT] [retE (call "asfloat" [shl (v "h") (u 16)])]

/-- float to bfloat16, round to nearest even; a NaN stays a NaN. -/
def f32ToBf16Fn : SlangFunctionDecl :=
  fn uT "f32ToBf16" [prm "x" fT]
    [ let_ uT "w" (call "asuint" [v "x"])
    , if_ (gt (band (v "w") (u 0x7FFFFFFF)) (u 0x7F800000)) [retE (bor (shr (v "w") (u 16)) (u 0x40))]
    , retE (shr (v "w" + u 0x7FFF + band (shr (v "w") (u 16)) (u 1)) (u 16)) ]

/-- binary16 to float, exact. Subnormals are `m * 2^-24`, a product that is exact. -/
def f16ToF32Fn : SlangFunctionDecl :=
  fn fT "f16ToF32" [prm "h" uT]
    [ let_ uT "sg" (shl (band (v "h") (u 0x8000)) (u 16))
    , let_ uT "e" (band (shr (v "h") (u 10)) (u 0x1F))
    , let_ uT "m" (band (v "h") (u 0x3FF))
    , if_ (eq (v "e") (u 0))
        [ if_ (eq (v "m") (u 0)) [retE (call "asfloat" [v "sg"])]
        , let_ fT "mag" (toF (v "m") * fl 5.9604644775390625e-8)
        , retE (call "asfloat" [bor (call "asuint" [v "mag"]) (v "sg")]) ]
    , if_ (eq (v "e") (u 31))
        [retE (call "asfloat" [bor (bor (v "sg") (u 0x7F800000)) (shl (v "m") (u 13))])]
    , retE (call "asfloat" [bor (bor (v "sg") (shl (v "e" + u 112) (u 23))) (shl (v "m") (u 13))]) ]

/-- float to binary16, round to nearest even; overflow gives infinity. -/
def f32ToF16Fn : SlangFunctionDecl :=
  fn uT "f32ToF16" [prm "x" fT]
    [ let_ uT "w" (call "asuint" [v "x"])
    , let_ uT "sg" (band (shr (v "w") (u 16)) (u 0x8000))
    , let_ uT "e" (band (shr (v "w") (u 23)) (u 0xFF))
    , let_ uT "m" (band (v "w") (u 0x7FFFFF))
    , if_ (eq (v "e") (u 255))
        [ if_ (eq (v "m") (u 0)) [retE (bor (v "sg") (u 0x7C00))]
        , retE (bor (bor (v "sg") (u 0x7E00)) (shr (v "m") (u 13))) ]
    , if_ (ge (v "e") (u 143)) [retE (bor (v "sg") (u 0x7C00))]
    , if_ (ge (v "e") (u 113))
        [ let_ uT "h0" (bor (shl (v "e" - u 112) (u 10)) (shr (v "m") (u 13)))
        , let_ uT "rm" (band (v "m") (u 0x1FFF))
        , let_ uT "h" (v "h0")
        , if_ (or_ (gt (v "rm") (u 0x1000)) (and_ (eq (v "rm") (u 0x1000)) (eq (band (v "h0") (u 1)) (u 1))))
            [setv "h" (v "h0" + u 1)]
        , retE (bor (v "sg") (v "h")) ]
    , if_ (lt (v "e") (u 102)) [retE (v "sg")]
    , let_ uT "shf" (u 126 - v "e")
    , let_ uT "sig" (bor (v "m") (u 0x800000))
    , let_ uT "q0" (shr (v "sig") (v "shf"))
    , let_ uT "rm" (band (v "sig") (shl (u 1) (v "shf") - u 1))
    , let_ uT "mid" (shl (u 1) (v "shf" - u 1))
    , let_ uT "q" (v "q0")
    , if_ (or_ (gt (v "rm") (v "mid")) (and_ (eq (v "rm") (v "mid")) (eq (band (v "q0") (u 1)) (u 1))))
        [setv "q" (v "q0" + u 1)]
    , retE (bor (v "sg") (v "q")) ]

/-- A two-bit code to -1, 0 or +1: `m - 2 (m & s)`. -/
def decodeFn (mt : Mutant) : SlangFunctionDecl :=
  match mt with
  | .decodeIllegal =>
    fn iT "t2Decode" [prm "code" uT]
      [retE (toI (v "code") - i 4 * toI (band (shr (v "code") (u 1)) (u 1)))]
  | _ =>
    fn iT "t2Decode" [prm "code" uT]
      [ let_ uT "mg" (band (v "code") (u 1))
      , let_ uT "sg" (band (shr (v "code") (u 1)) (u 1))
      , retE (toI (v "mg") - i 2 * toI (band (v "mg") (v "sg"))) ]

/-- Round to nearest, ties to even (`rintf`). -/
def roundEvenFn (mt : Mutant) : SlangFunctionDecl :=
  match mt with
  | .tieAway => fn fT "roundEven" [prm "x" fT] [retE (call "round" [v "x"])]
  | _ =>
    fn fT "roundEven" [prm "x" fT]
      [ let_ fT "f" (call "floor" [v "x"])
      , if_ (eq (v "x" - v "f") (fl 0.5))
          [ if_ (eq (call "floor" [v "f" * fl 0.5] * fl 2.0) (v "f")) [retE (v "f")]
          , retE (v "f" + fl 1.0) ]
      , retE (call "round" [v "x"]) ]

/-! ## Kernel descriptions -/

/-- A kernel from its pieces, the mutant (if any) applied to its DAG. -/
def mk (c : Cfg) (name paramsStruct : String) (fields : List String) (bufs : List Buf) (threads : Nat)
    (helpers : List SlangFunctionDecl) (ids : List SlangBinding) (dag : Dag) : Kernel :=
  mkKernel name paramsStruct fields bufs threads helpers ids (c.mutant.onDag dag)

def t2Fields : List String :=
  ["k", "groups", "tokens", "rowBegin", "rows", "outLd", "outOffset", "accumulate"]

/-- The A16 buffers; the output is the bfloat16 plane, or the FP32 total of the probe. -/
def a16Bufs (c : Cfg) : List Buf :=
  [ ⟨"qdata", "uint8_t", false⟩, ⟨"scales", "uint16_t", false⟩, ⟨"act", "uint16_t", false⟩
  , ⟨"dst", if c.probe then "uint" else "uint16_t", true⟩ ]

def a8Bufs (c : Cfg) : List Buf :=
  [ ⟨"qdata", "uint8_t", false⟩, ⟨"scales", "uint16_t", false⟩, ⟨"codes", "int8_t", false⟩
  , ⟨"xscales", "uint16_t", false⟩, ⟨"dst", if c.probe then "uint" else "uint16_t", true⟩ ]

def suffix (c : Cfg) : String := if c.probe then "_f32" else ""

/-- The store of one output: the probe writes the FP32 bits to `[t, rows]`; the plane adds the old
value first when `accumulate` is set, then rounds to bfloat16. -/
def storeStmts (c : Cfg) (total : String) : List St :=
  if c.probe then
    [ setAt "dst" (v "t" * p "rows" + v "loc") (call "asuint" [v total]) ]
  else
    [ let_ uT "oi" (v "t" * p "outLd" + (p "outOffset" + v "loc"))
    , if_ (ne (p "accumulate") (u 0)) [setv total (v total + call "bf16ToF32" [toU (at_ "dst" (v "oi"))])]
    , setAt "dst" (v "oi") (.cast u16T (call "f32ToBf16" [v total])) ]

def helpersA16 (c : Cfg) : List SlangFunctionDecl :=
  [bf16ToF32Fn, f32ToBf16Fn, f16ToF32Fn, decodeFn c.mutant]

/-- The products of one lane for one group: `byte` holds the lane's four codes, `base` is the
input index of its first. `parts` are the scalar partials, one per token. -/
def laneProducts (c : Cfg) (tokens : List Nat) (tokenIdx : Nat → E) : List St :=
  let guarded (body : List St) : List St :=
    if c.mutant == .noGuard then body else [if_ (lt (v "base" + v "j") (p "k")) body]
  (tokens.map fun x => let_ fT s!"part{x}" (fl 0.0)) ++
  [ for_ "j" (u 0) (u 4)
      (guarded
        ([ let_ fT "wf" (toF (call "t2Decode" [band (shr (v "byte") (v "j" * u 2)) (u 3)])) ] ++
         tokens.map fun x =>
           if_ (lt (tokenIdx x) (p "tokens"))
             [ setv s!"part{x}"
                 (v s!"part{x}" + v "wf" * call "bf16ToF32" [toU (at_ "act" (tokenIdx x * p "k" + (v "base" + v "j")))]) ] )) ]

/-- `acc{x} = acc{x} + part{x} * scale`, the product kept apart from the sum. -/
def scaleInto (tokens : List Nat) : List St :=
  tokens.flatMap fun x =>
    [ letP fT s!"pr{x}" (v s!"part{x}" * v "scale"), setv s!"acc{x}" (v s!"acc{x}" + v s!"pr{x}") ]

/-! ### A16, serial: one thread per output, the 32 lanes in a local array -/

def a16Serial (c : Cfg) : Kernel :=
  let tokenIdx : Nat → E := fun _ => v "t"
  let fold : List St :=
    if c.mutant == .sequentialFold then
      [ let_ fT "tot" (at_ "lane" (u 0))
      , for_ "l" (u 1) (u 32) [setv "tot" (v "tot" + at_ "lane" (v "l"))] ]
    else
      ([16, 8, 4, 2, 1].map fun off =>
        for_ "l" (u 0) (u off) [setIdx (v "lane") (v "l") (at_ "lane" (v "l") + at_ "lane" (v "l" + u off))]) ++
      [ let_ fT "tot" (at_ "lane" (u 0)) ]
  let body : List St :=
    [ let_ uT "idx" (.member (v "tid") "x")
    , if_ (ge (v "idx") (p "rows" * p "tokens")) [ret]
    , let_ uT "loc" (v "idx" % p "rows")
    , let_ uT "t" (v "idx" / p "rows")
    , let_ uT "row" (p "rowBegin" + v "loc")
    , arr fT "lane" 32
    , for_ "l" (u 0) (u 32) [setIdx (v "lane") (v "l") (fl 0.0)]
    , for_ "g" (u 0) (p "groups")
        ([ let_ uT "gi" (v "row" * p "groups" + v "g")
         , let_ fT "scale" (call "f16ToF32" [toU (at_ "scales" (v "gi"))])
         , for_ "l" (u 0) (u 32)
             ([ let_ uT "byte" (toU (at_ "qdata" (v "gi" * u 32 + v "l")))
              , let_ uT "base" (v "g" * u 128 + v "l" * u 4) ] ++
              laneProducts c [0] tokenIdx ++
              [ letP fT "pr0" (v "part0" * v "scale")
              , setIdx (v "lane") (v "l") (at_ "lane" (v "l") + v "pr0") ]) ]) ] ++
    fold ++ storeStmts c "tot"
  mk c s!"t2_a16_serial{suffix c}" "T2Params" t2Fields (a16Bufs c) 64 (helpersA16 c) [dtid]
    (Dag.single body)

/-! ### A16, lanes: a 32-thread group per output (and per `U` tokens)

Seven processes in a chain: each lane's partial sum into `sh`, five folds of the tree (`sh[l] +=
sh[l + off]` for `off` 16, 8, 4, 2, 1), lane 0's store. Each edge is a barrier. -/

def a16Lanes (c : Cfg) (tokensPerGroup : Nat) : Kernel :=
  let us := List.range tokensPerGroup
  let tokenIdx : Nat → E := fun x => v "t0" + u x
  let shAt (x : Nat) (e : E) : E := .index (.index (.var "sh") (u x)) e
  let folds : List Proc :=
    if c.mutant == .sequentialFold then []
    else
      [16, 8, 4, 2, 1].map fun off =>
        { name := s!"fold{off}"
          body := [ if_ (lt (v "lane") (u off))
                      (us.map fun x =>
                        setIdx (at_ "sh" (u x)) (v "lane") (shAt x (v "lane") + shAt x (v "lane" + u off))) ] }
  let finish : List St :=
    us.flatMap fun x =>
      let total := s!"tot{x}"
      let sumE : E := if c.mutant == .sequentialFold then v "seq" else shAt x (u 0)
      [ if_ (lt (tokenIdx x) (p "tokens"))
          ((if c.mutant == .sequentialFold then
              [ let_ fT "seq" (shAt x (u 0))
              , for_ "l" (u 1) (u 32) [setv "seq" (v "seq" + shAt x (v "l"))] ]
            else []) ++
           [ let_ fT total sumE
           , let_ uT "t" (tokenIdx x) ] ++ storeStmts c total) ]
  let prelude : List St :=
    [ let_ uT "loc" (.member (v "gid") "x")
    , let_ uT "t0" (.member (v "gid") "y" * u tokensPerGroup)
    , let_ uT "lane" (.member (v "gtid") "x")
    , let_ uT "row" (p "rowBegin" + v "loc") ]
  let partialSums : Proc :=
    { name := "partial"
      body :=
        (us.map fun x => let_ fT s!"acc{x}" (fl 0.0)) ++
        [ for_ "g" (u 0) (p "groups")
            ([ let_ uT "gi" (v "row" * p "groups" + v "g")
             , let_ fT "scale" (call "f16ToF32" [toU (at_ "scales" (v "gi"))])
             , let_ uT "byte" (toU (at_ "qdata" (v "gi" * u 32 + v "lane")))
             , let_ uT "base" (v "g" * u 128 + v "lane" * u 4) ] ++
             laneProducts c us tokenIdx ++ scaleInto us) ] ++
        (us.map fun x => setIdx (at_ "sh" (u x)) (v "lane") (v s!"acc{x}")) }
  let store : Proc := { name := "finish", body := [if_ (eq (v "lane") (u 0)) finish] }
  let name := if tokensPerGroup == 1 then "t2_a16_lanes" else "t2_a16_wide"
  mk c s!"{name}{suffix c}" "T2Params" t2Fields (a16Bufs c) 32 (helpersA16 c) [gid, gtid]
    (Dag.chain prelude [{ name := "sh", elemType := fT, dims := [tokensPerGroup, 32] }]
      ([partialSums] ++ folds ++ [store]))

/-! ### A8 -/

def quantFields : List String := ["k", "groups", "tokens"]

/-- Quantise `act[t, k]` to int8 codes and binary16 scales, one thread per (group, token). -/
def a8Quantize (c : Cfg) : Kernel :=
  let load : List St :=
    [ let_ uT "pos" (v "g" * u 128 + v "ii")
    , let_ fT "xv" (fl 0.0)
    , if_ (lt (v "pos") (p "k")) [setv "xv" (call "bf16ToF32" [toU (at_ "act" (v "t" * p "k" + v "pos"))])] ]
  let body : List St :=
    [ let_ uT "idx" (.member (v "tid") "x")
    , if_ (ge (v "idx") (p "groups" * p "tokens")) [ret]
    , let_ uT "g" (v "idx" % p "groups")
    , let_ uT "t" (v "idx" / p "groups")
    , let_ fT "amax" (fl 0.0)
    , for_ "ii" (u 0) (u 128) (load ++ [setv "amax" (call "max" [v "amax", call "abs" [v "xv"]])])
    , let_ uT "sh" (call "f32ToF16" [v "amax" / fl 127.0])
    , setAt "xscales" (v "t" * p "groups" + v "g") (.cast u16T (v "sh"))
    , let_ fT "s" (call "f16ToF32" [v "sh"])
    , let_ fT "inv" (fl 0.0)
    , if_ (gt (v "s") (fl 0.0)) [setv "inv" (fl 1.0 / v "s")]
    , for_ "ii" (u 0) (u 128)
        (load ++
         [ let_ fT "q" (call "roundEven" [v "xv" * v "inv"])
         , setv "q" (call "max" [fl (-127.0), call "min" [fl 127.0, v "q"]])
         , setAt "codes" (v "t" * p "groups" * u 128 + v "g" * u 128 + v "ii") (.cast i8T (toI (v "q"))) ]) ]
  mk c "t2_a8_quantize" "QuantParams" quantFields
    [⟨"act", "uint16_t", false⟩, ⟨"codes", "int8_t", true⟩, ⟨"xscales", "uint16_t", true⟩] 64
    [bf16ToF32Fn, f16ToF32Fn, f32ToF16Fn, roundEvenFn c.mutant] [dtid] (Dag.single body)

def helpersA8 (c : Cfg) : List SlangFunctionDecl :=
  [bf16ToF32Fn, f32ToBf16Fn, f16ToF32Fn, decodeFn c.mutant]

/-- The exact integer dot of one group, as a statement list leaving it in `dsum`:
`wbase` indexes the row's code bytes, `cbase` the token's int8 codes. -/
def groupDot : List St :=
  [ let_ iT "dsum" (i 0)
  , for_ "b" (u 0) (u 32)
      [ let_ uT "byte" (toU (at_ "qdata" (v "wbase" + v "b")))
      , for_ "j" (u 0) (u 4)
          [ setv "dsum" (v "dsum" + call "t2Decode" [band (shr (v "byte") (v "j" * u 2)) (u 3)]
              * toI (at_ "codes" (v "cbase" + v "b" * u 4 + v "j"))) ] ] ]

/-- `(float(dsum) * sw) * sx` with each product rounded on its own. -/
def groupTerm (out : String) : List St :=
  [ let_ fT "sw" (call "f16ToF32" [toU (at_ "scales" (v "gi"))])
  , let_ fT "sx" (call "f16ToF32" [toU (at_ "xscales" (v "t" * p "groups" + v "g"))])
  , letP fT "p1" (toF (v "dsum") * v "sw")
  , letP fT out (v "p1" * v "sx") ]

def a8Serial (c : Cfg) : Kernel :=
  let body : List St :=
    [ let_ uT "idx" (.member (v "tid") "x")
    , if_ (ge (v "idx") (p "rows" * p "tokens")) [ret]
    , let_ uT "loc" (v "idx" % p "rows")
    , let_ uT "t" (v "idx" / p "rows")
    , let_ uT "row" (p "rowBegin" + v "loc")
    , let_ fT "tot" (fl 0.0)
    , for_ "g" (u 0) (p "groups")
        ([ let_ uT "gi" (v "row" * p "groups" + v "g")
         , let_ uT "wbase" (v "gi" * u 32)
         , let_ uT "cbase" (v "t" * p "groups" * u 128 + v "g" * u 128) ] ++
         groupDot ++ groupTerm "term" ++ [setv "tot" (v "tot" + v "term")]) ] ++
    storeStmts c "tot"
  mk c s!"t2_a8_serial{suffix c}" "T2Params" t2Fields (a8Bufs c) 64 (helpersA8 c) [dtid] (Dag.single body)

/-- A8 over 32 lanes, two processes: lane `l` takes groups `l, l + 32, ...` whole and parks each
`(dot * sw) * sx` in `pg`; then lane 0 adds them in group order, as `serial` does. -/
def a8Lanes (c : Cfg) : Kernel :=
  let rounds := maxGroups / 32
  let prelude : List St :=
    [ let_ uT "loc" (.member (v "gid") "x")
    , let_ uT "t" (.member (v "gid") "y")
    , let_ uT "lane" (.member (v "gtid") "x")
    , let_ uT "row" (p "rowBegin" + v "loc") ]
  let spread : Proc :=
    { name := "spread"
      body :=
        [ for_ "rd" (u 0) (u rounds)
            [ let_ uT "g" (v "rd" * u 32 + v "lane")
            , if_ (lt (v "g") (p "groups"))
                ([ let_ uT "gi" (v "row" * p "groups" + v "g")
                 , let_ uT "wbase" (v "gi" * u 32)
                 , let_ uT "cbase" (v "t" * p "groups" * u 128 + v "g" * u 128) ] ++
                 groupDot ++ groupTerm "term" ++ [setIdx (v "pg") (v "g") (v "term")]) ] ] }
  let total : Proc :=
    { name := "total"
      body :=
        [ if_ (eq (v "lane") (u 0))
            ([ let_ fT "tot" (fl 0.0)
             , for_ "g" (u 0) (p "groups") [setv "tot" (v "tot" + at_ "pg" (v "g"))] ] ++ storeStmts c "tot") ] }
  mk c s!"t2_a8_lanes{suffix c}" "T2Params" t2Fields (a8Bufs c) 32 (helpersA8 c) [gid, gtid]
    (Dag.chain prelude [{ name := "pg", elemType := fT, dims := [maxGroups] }] [spread, total])

/-! ### SwiGLU: `silu(gate) * up` over the `[gate | up]` plane -/

def siluFields : List String := ["inter", "total"]

def siluMulKernel : Kernel :=
  let body : List St :=
    [ let_ uT "idx" (.member (v "tid") "x")
    , if_ (ge (v "idx") (p "total")) [ret]
    , let_ uT "t" (v "idx" / p "inter")
    , let_ uT "r" (v "idx" % p "inter")
    , let_ fT "gt" (call "bf16ToF32" [toU (at_ "plane" (v "t" * (u 2 * p "inter") + v "r"))])
    , let_ fT "up" (call "bf16ToF32" [toU (at_ "plane" (v "t" * (u 2 * p "inter") + p "inter" + v "r"))])
    , let_ fT "sg" (v "gt" / (fl 1.0 + call "exp" [-(v "gt")]))
    , setAt "dst" (v "idx") (.cast u16T (call "f32ToBf16" [v "sg" * v "up"])) ]
  mkKernel "t2_silu_mul" "SiluParams" siluFields [⟨"plane", "uint16_t", false⟩, ⟨"dst", "uint16_t", true⟩] 256
    [bf16ToF32Fn, f32ToBf16Fn] [dtid] (Dag.single body)

/-! ## The kernel set -/

/-- The kernels the library ships, under one configuration. -/
def kernels (c : Cfg) : List Kernel :=
  [ a16Serial c, a16Lanes c 1, a16Lanes c wideTokens, a8Quantize c, a8Serial c, a8Lanes c, siluMulKernel ]

end Ninfer.Ops.LinearT2
