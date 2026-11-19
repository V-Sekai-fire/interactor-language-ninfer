-- SPDX-License-Identifier: Apache-2.0
import LeanSlang
import Ninfer.Slang.Dsl
import Ninfer.Slang.Dag

/-!
# `Ninfer.Slang.Kernel` — a kernel, its buffers, and its two lowerings

A `Kernel` is a `Dag` of barrier-free processes (`Ninfer.Slang.Dag`) with what a host needs to run
it: the constant buffer's fields, the buffers by binding, the helper functions, the entry
parameters. A kernel without a barrier is a DAG of one process.

* `fused` is one Slang module for the targets that run a group in parallel (SPIR-V, Metal): the
  processes in order with a barrier between levels, the shared arrays `groupshared`.
* `modules` is the DAG of executes: one module per process, in execution order, for the targets that
  do not. The shared arrays are buffers (one row per group), and the constant buffer gains the
  grid's `gridX` and `gridY` to find the row. A kernel of one process and no shared array is its
  own module.
-/

namespace Ninfer.Slang

open LeanSlang Ninfer.Slang.Dsl Ninfer.Slang.Dag

/-- A buffer: Slang name, element type name (`uint8_t`, `int8_t`, `uint16_t`, `uint`, `float`), writable. -/
structure Buf where
  name : String
  elem : String
  rw : Bool
deriving Repr

def elemType : String → SlangType
  | "uint" => uT
  | "float" => fT
  | n => .named n

def Buf.global (b : Buf) (binding : Nat) : SlangBinding :=
  glob b.name (if b.rw then .rwBuf (elemType b.elem) else .roBuf (elemType b.elem)) binding

/-- The host's name for a shared array's element type. -/
def typeName : SlangType → Except String String
  | .scalar .float => .ok "float"
  | .scalar .uint => .ok "uint"
  | .scalar .int => .ok "int"
  | t => .error s!"a shared array of {repr t} has no host type"

structure Kernel where
  name : String
  paramsStruct : String
  fields : List String
  /-- Bound at `1, 2, ...` in this order; binding 0 is the constant buffer `params`. -/
  bufs : List Buf
  threads : Nat
  helpers : List SlangFunctionDecl
  /-- The entry parameters: `SV_DispatchThreadID`, or `SV_GroupID` and `SV_GroupThreadID`. -/
  ids : List SlangBinding
  dag : Dag

/-- A kernel from its pieces; the DAG learns the thread count and the names of the group and thread ids. -/
def mkKernel (name paramsStruct : String) (fields : List String) (bufs : List Buf) (threads : Nat)
    (helpers : List SlangFunctionDecl) (ids : List SlangBinding) (dag : Dag) : Kernel :=
  let named (s : Semantic) : String := ((ids.find? (·.semantic == s)).map (·.name)).getD ""
  { name, paramsStruct, fields, bufs, threads, helpers, ids
    dag := { dag with threads, threadId := named .svGroupThreadId, groupId := named .svGroupId } }

def Kernel.entry (k : Kernel) (body : List St) : SlangFunctionDecl :=
  { attrs := [.shaderCompute, .numthreads k.threads 1 1], name := "main", params := k.ids, body }

/-- The kernel as one module with barriers, for GPUs. -/
def Kernel.fused (k : Kernel) : Except String SlangShaderModule := do
  let body ← k.dag.fuse
  return { structs := [{ name := k.paramsStruct, fields := k.fields.map (fun f => fld f uT) }]
           groupShared := k.dag.shared
           globals := paramsCB k.paramsStruct :: (k.bufs.zipIdx.map fun (b, ix) => b.global (ix + 1))
           functions := k.helpers ++ [k.entry body] }

/-- One process as a kernel of its own. -/
structure ProcModule where
  /-- File stem and host name: the kernel's own name for a kernel that is one module. -/
  stem : String
  procName : String
  /-- Position in `Dag.procs`. -/
  index : Nat
  paramsStruct : String
  /-- Fields of the constant buffer, `u32` each. -/
  nFields : Nat
  /-- Buffers by binding, shared arrays last. -/
  bufs : List Buf
  threads : Nat
  shader : SlangShaderModule

/-- The kernel as a DAG of executes, in execution order. -/
def Kernel.modules (k : Kernel) : Except String (List ProcModule) := do
  if k.dag.procs.size == 1 && k.dag.shared.isEmpty then
    let sh ← k.fused
    return [{ stem := k.name, procName := k.dag.procs[0]!.name, index := 0,
              paramsStruct := k.paramsStruct, nFields := k.fields.length, bufs := k.bufs,
              threads := k.threads, shader := sh }]
  let scratch ← k.dag.shared.mapM fun s => do
    pure (⟨s.name, ← typeName s.elemType, true⟩ : Buf)
  let fields := k.fields ++ ["gridX", "gridY"]
  let bufs := k.bufs ++ scratch
  (← k.dag.order).mapM fun i => do
    let pr := k.dag.procs[i]!
    let body ← k.dag.splitBody i
    let shader : SlangShaderModule :=
      { structs := [{ name := k.paramsStruct, fields := fields.map (fun f => fld f uT) }],
        groupShared := [],
        globals := paramsCB k.paramsStruct :: (bufs.zipIdx.map fun (b, ix) => b.global (ix + 1)),
        functions := k.helpers ++ [k.entry body] }
    return { stem := s!"{k.name}_p{i}_{pr.name}", procName := pr.name, index := i,
             paramsStruct := k.paramsStruct, nFields := fields.length, bufs,
             threads := k.threads, shader }

end Ninfer.Slang
