-- SPDX-License-Identifier: Apache-2.0
import LeanSlang
import Ninfer.Slang.Dsl

/-!
# `Ninfer.Slang.Dag` — a kernel with group barriers, as a DAG of barrier-free processes

A `GroupMemoryBarrierWithGroupSync()` cuts a kernel into stretches: every thread of the group runs
one stretch before any thread starts the next. A target that runs the threads of a group one after
another cannot honour the barrier (slangc's C++ target refuses `groupshared` outright, E36107), but
it can honour the cut: run each stretch for every thread, then the next.

Here the cut is the source. A kernel is a `Dag`:

* a node is a `Proc`ess: a barrier-free body that every thread of the group executes, in any order;
* an edge `i → j` says every thread finishes `i` before any thread starts `j`. That is the barrier.

One DAG lowers two ways.

* `fuse`: a single kernel, the processes in topological order with a `GroupMemoryBarrierWithGroupSync`
  between consecutive levels. This is what a GPU runs, and it is the kernel the barrier version
  always was.
* `splitBody`: one kernel per process. The group's `groupshared` arrays become device buffers
  indexed by the group, so they outlive a dispatch, and each process is a dispatch over the whole
  grid. The host runs the processes as a DAG of executes, in topological order, passing the buffers
  along the edges. Nothing in it needs a barrier, so the C++ target runs it.

The two compute the same function because `check` establishes, before either is emitted:

1. no process holds a barrier, and the prelude (the thread-identity lets every process starts with)
   reads no memory;
2. within a process no lane touches a shared slot that another lane writes: the lanes of a process
   can run in any order. Proved by running the process body once per lane on the slots it can
   reach, with the lane index known, and everything else unknown;
3. any two processes that touch the same shared array, one of them writing, are ordered by a path.

Both the evaluator and the walk over-approximate: an index it cannot evaluate is "any slot", a
condition it cannot decide takes both branches. So `check` can reject a safe DAG. For the shared
arrays it does not accept a racy one; it does not look at the device buffers.
-/

namespace Ninfer.Slang.Dag

open LeanSlang Ninfer.Slang.Dsl

/-! ## Rewriting a tree -/

/-- Rewrite every expression top-down: `f` returns the replacement, or `none` to descend. -/
partial def rewriteE (f : E → Except String (Option E)) (e : E) : Except String E := do
  match ← f e with
  | some r => return r
  | none =>
    match e with
    | .index b i => return .index (← rewriteE f b) (← rewriteE f i)
    | .member r n => return .member (← rewriteE f r) n
    | .bin o a b => return .bin o (← rewriteE f a) (← rewriteE f b)
    | .un o a => return .un o (← rewriteE f a)
    | .call fn args => return .call fn (← args.mapM (rewriteE f))
    | .ternary c t g => return .ternary (← rewriteE f c) (← rewriteE f t) (← rewriteE f g)
    | .cast ty a => return .cast ty (← rewriteE f a)
    | other => return other

/-- `rewriteE` over every expression of a statement, and of the statements inside it. -/
partial def rewriteSt (f : E → Except String (Option E)) (s : St) : Except String St := do
  match s with
  | .declare ty n init => return .declare ty n (← init.mapM (rewriteE f))
  | .declarePrecise ty n init => return .declarePrecise ty n (← init.mapM (rewriteE f))
  | .declareArray ty n sz => return .declareArray ty n sz
  | .assign l r => return .assign (← rewriteE f l) (← rewriteE f r)
  | .expr e => return .expr (← rewriteE f e)
  | .ret e => return .ret (← e.mapM (rewriteE f))
  | .ifThen c t e => return .ifThen (← rewriteE f c) (← t.mapM (rewriteSt f)) (← e.mapM (rewriteSt f))
  | .forCount n lo hi body =>
      return .forCount n (← rewriteE f lo) (← rewriteE f hi) (← body.mapM (rewriteSt f))
  | .whileLoop c body => return .whileLoop (← rewriteE f c) (← body.mapM (rewriteSt f))

/-- `a[i0][i1]…`: the root name and the indices, first index first. A bare name has none. -/
partial def chainOf : E → Option (String × List E)
  | .var n => some (n, [])
  | .index b i => (chainOf b).map fun (n, is) => (n, is ++ [i])
  | _ => none

/-- Elements of one group's copy of a shared array. -/
def sharedSize (d : SlangGroupSharedDecl) : Nat := d.dims.foldl (· * ·) 1

/-- Row-major flat index. -/
def flatIndex (dims : List Nat) (idxs : List E) : E :=
  match idxs, dims with
  | i0 :: is, _ :: ds => (List.zip ds is).foldl (fun acc (dm, i) => acc * u dm + i) i0
  | _, _ => u 0

/-- The group's index in the grid: the row of a shared array's copy in its device buffer. -/
def groupLinearName : String := "groupLinear"

/-- An access to a shared array becomes an access to the device buffer of the same name, at the
group's row. Every index has to be there: `sh[x]` for a `[4][32]` array is not a slot. -/
partial def lowerAccess (shared : List SlangGroupSharedDecl) (e : E) : Except String (Option E) :=
  match chainOf e with
  | some (n, idxs@(_ :: _)) =>
      match shared.find? (·.name == n) with
      | none => .ok none
      | some d =>
          if idxs.length != d.dims.length then
            .error s!"{n} has {d.dims.length} dimension(s) and the access has {idxs.length} index(es)"
          else do
            let idxs' ← idxs.mapM (rewriteE (lowerAccess shared))
            return some (.index (.var n) (v groupLinearName * u (sharedSize d) + flatIndex d.dims idxs'))
  | _ => .ok none

def noBarrier : E → Except String (Option E)
  | .call "GroupMemoryBarrierWithGroupSync" _ =>
      .error "a process holds a barrier: the edges of the DAG are the barriers"
  | _ => .ok none

def noMemory : E → Except String (Option E)
  | .index _ _ => .error "the prelude reads memory: it may hold only thread identity and parameters"
  | _ => .ok none

/-! ## What one lane touches -/

/-- Where in a shared array a lane reads or writes: one flat slot, or any (an index the walk cannot
evaluate). -/
inductive Slot
  | one (i : Int)
  | any
deriving BEq, Repr, Inhabited

def Slot.overlaps : Slot → Slot → Bool
  | .any, _ => true
  | _, .any => true
  | .one a, .one b => a == b

def Slot.show : Slot → String
  | .one i => s!"[{i}]"
  | .any => "[?]"

structure Access where
  array : String
  slot : Slot
  write : Bool
deriving BEq, Repr, Inhabited

/-- Known integer values of the variables in scope; `none` is unknown. -/
abbrev Env := List (String × Option Int)

def Env.get (env : Env) (n : String) : Option Int :=
  match env.find? (·.1 == n) with
  | some (_, x) => x
  | none => none

def Env.put (env : Env) (n : String) (x : Option Int) : Env :=
  (n, x) :: env.filter (·.1 != n)

structure Ctx where
  lane : Nat
  /-- The entry parameter that carries `SV_GroupThreadID`. -/
  threadId : String
  shared : List SlangGroupSharedDecl

def binOp (op : String) (x y : Int) : Option Int :=
  let b (c : Bool) : Option Int := some (if c then 1 else 0)
  match op with
  | "+" => some (x + y)
  | "-" => some (x - y)
  | "*" => some (x * y)
  | "/" => if y == 0 then none else some (x / y)
  | "%" => if y == 0 then none else some (x % y)
  | "<" => b (x < y)
  | "<=" => b (x ≤ y)
  | ">" => b (x > y)
  | ">=" => b (x ≥ y)
  | "==" => b (x == y)
  | "!=" => b (x != y)
  | "&&" => b (x != 0 && y != 0)
  | "||" => b (x != 0 || y != 0)
  | "&" => if x ≥ 0 && y ≥ 0 then some (Int.ofNat (x.toNat &&& y.toNat)) else none
  | "|" => if x ≥ 0 && y ≥ 0 then some (Int.ofNat (x.toNat ||| y.toNat)) else none
  | "^" => if x ≥ 0 && y ≥ 0 then some (Int.ofNat (x.toNat ^^^ y.toNat)) else none
  | "<<" => if x ≥ 0 && y ≥ 0 && y < 64 then some (Int.ofNat (x.toNat <<< y.toNat)) else none
  | ">>" => if x ≥ 0 && y ≥ 0 then some (Int.ofNat (x.toNat >>> y.toNat)) else none
  | _ => none

/-- The value of an integer expression in one lane, when it is known. The lane index is
`gtid.x`; parameters, group ids, loaded values and calls are unknown. -/
partial def evalI (cx : Ctx) (env : Env) : E → Option Int
  | .litUint n => some (Int.ofNat n)
  | .litInt x => some x
  | .litBool bv => some (if bv then 1 else 0)
  | .var n => env.get n
  | .member (.var r) f =>
      if r == cx.threadId then some (if f == "x" then Int.ofNat cx.lane else 0) else none
  | .cast _ a => evalI cx env a
  | .un "-" a => (evalI cx env a).map (fun x => -x)
  | .ternary c t f =>
      match evalI cx env c with
      | some x => if x != 0 then evalI cx env t else evalI cx env f
      | none => none
  | .bin op a b =>
      match evalI cx env a, evalI cx env b with
      | some x, some y => binOp op x y
      | _, _ => none
  | _ => none

/-- The flat slot of an access to a declared array, or any slot when an index is unknown or out of range. -/
def slotOf (cx : Ctx) (env : Env) (d : SlangGroupSharedDecl) (idxs : List E) : Slot :=
  match idxs, d.dims with
  | i0 :: is, _ :: ds =>
      if is.length != ds.length then .any
      else
        match (i0 :: is).mapM (evalI cx env) with
        | some (x0 :: xs) =>
            let inRange := (List.zip d.dims (x0 :: xs)).all fun (dm, x) => 0 ≤ x && x < Int.ofNat dm
            let flat := (List.zip ds xs).foldl (fun acc (dm, x) => acc * Int.ofNat dm + x) x0
            if inRange then .one flat else .any
        | _ => .any
  | _, _ => .any

/-- The shared-array reads inside an expression. -/
partial def readsOf (cx : Ctx) (env : Env) (e : E) : List Access :=
  match chainOf e with
  | some (n, idxs@(_ :: _)) =>
      match cx.shared.find? (·.name == n) with
      | some d => ⟨n, slotOf cx env d idxs, false⟩ :: idxs.flatMap (readsOf cx env)
      | none => idxs.flatMap (readsOf cx env)
  | _ =>
      match e with
      | .index b i => readsOf cx env b ++ readsOf cx env i
      | .member r _ => readsOf cx env r
      | .bin _ a b => readsOf cx env a ++ readsOf cx env b
      | .un _ a => readsOf cx env a
      | .call _ args => args.flatMap (readsOf cx env)
      | .ternary c t f => readsOf cx env c ++ readsOf cx env t ++ readsOf cx env f
      | .cast _ a => readsOf cx env a
      | _ => []

/-- Variables a statement list assigns, at any depth. -/
partial def assignedVars : List St → List String
  | [] => []
  | s :: rest =>
    (match s with
     | .assign (.var n) _ => [n]
     | .ifThen _ t e => assignedVars t ++ assignedVars e
     | .forCount _ _ _ b => assignedVars b
     | .whileLoop _ b => assignedVars b
     | _ => []) ++ assignedVars rest

/-- Most iterations a loop with known bounds is unrolled for. -/
def unrollLimit : Int := 4096

/-- Run one statement for one lane: the variables it knows, and every shared slot it may touch. A
condition it cannot decide takes both branches; a loop it cannot unroll is walked once with
everything it assigns unknown. Both over-approximate. -/
partial def walkSt (cx : Ctx) (st : Env × List Access) (s : St) : Env × List Access :=
  let (env, acc) := st
  let block (st : Env × List Access) (b : List St) : Env × List Access := b.foldl (walkSt cx) st
  let unknownLoop (n : String) (st1 : Env × List Access) (body : List St) : Env × List Access :=
    let un := ((assignedVars body).foldl (fun en m => en.put m none) st1.1).put n none
    let (_, a) := block (un, st1.2) body
    (un, a)
  match s with
  | .declare _ n init =>
      match init with
      | some e => (env.put n (evalI cx env e), acc ++ readsOf cx env e)
      | none => (env.put n none, acc)
  | .declarePrecise ty n init => walkSt cx st (.declare ty n init)
  | .declareArray .. => st
  | .assign lhs rhs =>
      let rd := readsOf cx env rhs
      match lhs with
      | .var n => (env.put n (evalI cx env rhs), acc ++ rd)
      | _ =>
        match chainOf lhs with
        | some (n, idxs@(_ :: _)) =>
            let ird := idxs.flatMap (readsOf cx env)
            match cx.shared.find? (·.name == n) with
            | some d => (env, acc ++ rd ++ ird ++ [⟨n, slotOf cx env d idxs, true⟩])
            | none => (env, acc ++ rd ++ ird)
        | _ => (env, acc ++ rd)
  | .expr e => (env, acc ++ readsOf cx env e)
  | .ret e => (env, acc ++ (e.map (readsOf cx env)).getD [])
  | .ifThen c t e =>
      let st1 := (env, acc ++ readsOf cx env c)
      match evalI cx env c with
      | some x => block st1 (if x != 0 then t else e)
      | none =>
          let (_, a1) := block st1 t
          let (_, a2) := block (env, a1) e
          ((assignedVars t ++ assignedVars e).foldl (fun en n => en.put n none) env, a2)
  | .forCount n lo hi body =>
      let st1 := (env, acc ++ readsOf cx env lo ++ readsOf cx env hi)
      match evalI cx env lo, evalI cx env hi with
      | some l, some h =>
          if h - l ≤ unrollLimit then
            let (envE, accE) := (List.range (h - l).toNat).foldl
              (fun st j => block (st.1.put n (some (l + Int.ofNat j)), st.2) body) st1
            (envE.put n none, accE)
          else unknownLoop n st1 body
      | _, _ => unknownLoop n st1 body
  | .whileLoop c body => unknownLoop "" (env, acc ++ readsOf cx env c) body

/-! ## The DAG -/

/-- One process: what every thread of the group runs between two barriers. -/
structure Proc where
  name : String
  body : List St
deriving Inhabited

structure Dag where
  /-- Thread-identity lets (`gid`, `gtid`, parameters). Each process starts with them: they are
  the only locals that cross a barrier. -/
  prelude : List St := []
  shared : List SlangGroupSharedDecl := []
  procs : Array Proc
  /-- `(i, j)`: every thread finishes `procs[i]` before any starts `procs[j]`. -/
  edges : List (Nat × Nat) := []
  /-- Threads in a group, and the names of the `SV_GroupThreadID` and `SV_GroupID` parameters. -/
  threads : Nat := 1
  threadId : String := ""
  groupId : String := ""

/-- One process, no barrier. -/
def Dag.single (body : List St) : Dag := { procs := #[{ name := "all", body }] }

/-- A chain: each process after the one before, with a barrier between each pair. -/
def Dag.chain (prelude : List St) (shared : List SlangGroupSharedDecl) (procs : List Proc) : Dag :=
  { prelude, shared, procs := procs.toArray
    edges := (List.range (procs.length - 1)).map fun i => (i, i + 1) }

/-- Longest-path level of every process. Rejects an edge to a process that is not there, a loop, and a cycle. -/
def Dag.levels (d : Dag) : Except String (Array Nat) := do
  let n := d.procs.size
  for (i, j) in d.edges do
    if i ≥ n || j ≥ n then throw s!"edge ({i}, {j}) names a process that does not exist"
    if i == j then throw s!"edge ({i}, {j}) loops on one process"
  let mut level : Array (Option Nat) := Array.replicate n none
  for round in [0:n] do
    for j in [0:n] do
      if level[j]!.isNone then
        let preds := (d.edges.filter (·.2 == j)).map (·.1)
        if preds.all (fun i => match level[i]! with | some l => l < round | none => false) then
          level := level.set! j (some round)
  let mut out : Array Nat := #[]
  for l in level do
    match l with
    | some x => out := out.push x
    | none => throw "the processes have a cycle"
  return out

/-- Process indices in execution order: by level, then by position. -/
def Dag.order (d : Dag) : Except String (List Nat) := do
  let lv ← d.levels
  return ((List.range d.procs.size).toArray.qsort
    (fun a b => lv[a]! < lv[b]! || (lv[a]! == lv[b]! && a < b))).toList

/-- A barrier between every two consecutive levels. -/
def Dag.fuse (d : Dag) : Except String (List St) := do
  let lv ← d.levels
  let top := lv.foldl max 0
  let mut out := d.prelude
  for l in [0:top + 1] do
    if l > 0 then out := out ++ [barrier]
    for (pr, pl) in d.procs.toList.zip lv.toList do
      if pl == l then out := out ++ pr.body
  return out

/-- Process `i` as a kernel of its own: its shared arrays are device buffers, one row per group. -/
def Dag.splitBody (d : Dag) (i : Nat) : Except String (List St) := do
  let some pr := d.procs[i]? | throw s!"no process {i}"
  let body := d.prelude ++ pr.body
  if d.shared.isEmpty then return body
  if d.groupId.isEmpty then throw "a kernel with shared arrays needs an SV_GroupID parameter to index them by group"
  let gx : E := .member (v d.groupId) "x"
  let gy : E := .member (v d.groupId) "y"
  let gz : E := .member (v d.groupId) "z"
  let head : St := let_ uT groupLinearName (gx + p "gridX" * (gy + p "gridY" * gz))
  return head :: (← body.mapM (rewriteSt (lowerAccess d.shared)))

/-- Merge processes `lo` to `hi` into one: the barriers between them are gone. (A fault to plant,
not something to ship.) -/
def Dag.merge (d : Dag) (lo hi : Nat) : Dag :=
  if lo > hi || hi ≥ d.procs.size then d
  else
    let ps := d.procs.toList
    let merged : Proc := { name := "merged", body := ((ps.drop lo).take (hi - lo + 1)).flatMap (·.body) }
    let remap (k : Nat) : Nat := if k < lo then k else if k ≤ hi then lo else k - (hi - lo)
    let edges := ((d.edges.map fun (i, j) => (remap i, remap j)).filter fun (i, j) => i != j).eraseDups
    { d with procs := (ps.take lo ++ [merged] ++ ps.drop (hi + 1)).toArray, edges }

/-! ## The check -/

def laneAccesses (d : Dag) (lane : Nat) (body : List St) : List Access :=
  let cx : Ctx := { lane, threadId := d.threadId, shared := d.shared }
  ((d.prelude ++ body).foldl (walkSt cx) ([], [])).2.eraseDups

/-- Two lanes of one process that touch a slot, one of them writing it. -/
def intraHazard (d : Dag) (pr : Proc) : Option String :=
  let per := (List.range d.threads).map fun l => (l, laneAccesses d l pr.body)
  per.findSome? fun (a, accA) =>
    accA.findSome? fun w =>
      if !w.write then none
      else
        per.findSome? fun (b, accB) =>
          if a == b then none
          else
            accB.findSome? fun y =>
              if y.array == w.array && w.slot.overlaps y.slot then
                some s!"process {pr.name}: lane {a} writes {w.array}{w.slot.show}, which lane {b} {if y.write then "writes" else "reads"} in the same process; a barrier belongs between them (an edge)"
              else none

/-- The shared arrays a process reads and writes, over all its lanes. -/
def effects (d : Dag) (pr : Proc) : List String × List String :=
  let all := (List.range d.threads).flatMap fun l => laneAccesses d l pr.body
  ((all.filter (!·.write)).map (·.array) |>.eraseDups, (all.filter (·.write)).map (·.array) |>.eraseDups)

def reachMatrix (n : Nat) (edges : List (Nat × Nat)) : Array (Array Bool) := Id.run do
  let mut m := Array.replicate n (Array.replicate n false)
  for (i, j) in edges do
    if i < n && j < n then m := m.set! i (m[i]!.set! j true)
  for k in [0:n] do
    for i in [0:n] do
      for j in [0:n] do
        if m[i]![k]! && m[k]![j]! then m := m.set! i (m[i]!.set! j true)
  return m

/-- Everything `check` promises (see the header). -/
def Dag.check (d : Dag) : Except String Unit := do
  if d.procs.isEmpty then throw "a DAG with no process"
  let _ ← d.levels
  for pr in d.procs do
    for s in pr.body do
      let _ ← rewriteSt noBarrier s
      let _ ← rewriteSt (lowerAccess d.shared) s
  for s in d.prelude do
    let _ ← rewriteSt noMemory s
  if d.shared.isEmpty then return ()
  for pr in d.procs do
    if let some msg := intraHazard d pr then throw msg
  let n := d.procs.size
  let reach := reachMatrix n d.edges
  let eff := d.procs.map (effects d)
  for i in [0:n] do
    for j in [i + 1:n] do
      if !reach[i]![j]! && !reach[j]![i]! then
        let (ri, wi) := eff[i]!
        let (rj, wj) := eff[j]!
        let hit := (wi.find? fun a => rj.contains a || wj.contains a) <|> (wj.find? fun a => ri.contains a)
        if let some a := hit then
          throw s!"processes {d.procs[i]!.name} and {d.procs[j]!.name} both touch {a}, one of them writing, and no edge orders them"
  return ()

end Ninfer.Slang.Dag
