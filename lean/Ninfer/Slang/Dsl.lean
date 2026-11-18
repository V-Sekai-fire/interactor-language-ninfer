-- SPDX-License-Identifier: Apache-2.0
import LeanSlang

/-!
# `Ninfer.Slang.Dsl` — terse constructors for LeanSlang trees

The subset of the helpers in contract-rotation-star's `RotationStar.Slang.Dsl` that integer and
buffer-heavy kernels need, with the bit operators added. Every helper expands to LeanSlang
constructors, so a kernel written with them emits the same text as the same kernel written with
the constructors by hand, and the pins in `Ninfer.Ops.Linear.Kernels` check exactly that.

LeanSlang prints a binary expression fully parenthesised, `(a op b)`, with `op` any string, so the
shift, mask and comparison operators below are plain `.bin` nodes.
-/

namespace Ninfer.Slang.Dsl

open LeanSlang

abbrev E := SlangExpr
abbrev St := SlangStmt

def fT : SlangType := .scalar .float
def uT : SlangType := .scalar .uint
def iT : SlangType := .scalar .int
def u3T : SlangType := .vec .uint 3
/-- Element types of the packed planes: bytes, binary16 and bfloat16 patterns, signed codes. -/
def u8T : SlangType := .named "uint8_t"
def u16T : SlangType := .named "uint16_t"
def i8T : SlangType := .named "int8_t"

scoped instance : HAdd E E E := ⟨fun a b => .bin "+" a b⟩
scoped instance : HSub E E E := ⟨fun a b => .bin "-" a b⟩
scoped instance : HMul E E E := ⟨fun a b => .bin "*" a b⟩
scoped instance : HDiv E E E := ⟨fun a b => .bin "/" a b⟩
scoped instance : Mod E := ⟨fun a b => .bin "%" a b⟩
scoped instance : Neg E := ⟨fun a => .un "-" a⟩

def v (s : String) : E := .var s
/-- An unsigned literal. -/
def u (n : Nat) : E := .litUint n
/-- A signed literal. -/
def i (n : Int) : E := .litInt n
/-- A float literal printed exactly (the shortest round-tripping binary32, with an `f`). -/
def fl (x : Float) : E := .litFloatExact x
def at_ (buf : String) (idx : E) : E := .index (.var buf) idx
/-- A field of the constant buffer `params`. -/
def p (field : String) : E := .member (.var "params") field
def call (f : String) (args : List E) : E := .call f args
def sel (c t f : E) : E := .ternary c t f
def toF (e : E) : E := .cast fT e
def toU (e : E) : E := .cast uT e
def toI (e : E) : E := .cast iT e

def lt (a b : E) : E := .bin "<" a b
def le (a b : E) : E := .bin "<=" a b
def gt (a b : E) : E := .bin ">" a b
def ge (a b : E) : E := .bin ">=" a b
def eq (a b : E) : E := .bin "==" a b
def ne (a b : E) : E := .bin "!=" a b
def and_ (a b : E) : E := .bin "&&" a b
def or_ (a b : E) : E := .bin "||" a b
def band (a b : E) : E := .bin "&" a b
def bor (a b : E) : E := .bin "|" a b
def bxor (a b : E) : E := .bin "^" a b
def shl (a b : E) : E := .bin "<<" a b
def shr (a b : E) : E := .bin ">>" a b

def let_ (ty : SlangType) (n : String) (e : E) : St := .declare ty n (some e)
/-- A `precise` local: the compiler may not fuse or reorder what is assigned to it. -/
def letP (ty : SlangType) (n : String) (e : E) : St := .declarePrecise ty n (some e)
def arr (ty : SlangType) (n : String) (size : Nat) : St := .declareArray ty n size
def setv (n : String) (r : E) : St := .assign (.var n) r
def setAt (buf : String) (idx r : E) : St := .assign (.index (.var buf) idx) r
def setIdx (base idx r : E) : St := .assign (.index base idx) r
def if_ (c : E) (t : List St) (e : List St := []) : St := .ifThen c t e
def for_ (n : String) (lo hi : E) (body : List St) : St := .forCount n lo hi body
def ret : St := .ret none
def retE (e : E) : St := .ret (some e)
/-- `GroupMemoryBarrierWithGroupSync();`: every thread of the group reaches it. -/
def barrier : St := .expr (.call "GroupMemoryBarrierWithGroupSync" [])

def fld (n : String) (ty : SlangType) : SlangBinding :=
  ⟨n, ty, Semantic.none, none, none, .qIn⟩
def glob (n : String) (ty : SlangType) (b : Nat) : SlangBinding :=
  ⟨n, ty, Semantic.none, some b, some 0, .qIn⟩
def paramsCB (structName : String) : SlangBinding := glob "params" (.const structName) 0
def prm (n : String) (ty : SlangType) : SlangBinding := ⟨n, ty, Semantic.none, none, none, .qIn⟩

/-- Thread ids as entry parameters. -/
def dtid : SlangBinding := ⟨"tid", u3T, .svDispatchThreadId, none, none, .qIn⟩
def gid : SlangBinding := ⟨"gid", u3T, .svGroupId, none, none, .qIn⟩
def gtid : SlangBinding := ⟨"gtid", u3T, .svGroupThreadId, none, none, .qIn⟩

/-- A compute entry point `main` over `tx` threads. -/
def entry (tx : Nat) (params : List SlangBinding) (body : List St) : SlangFunctionDecl :=
  { attrs := [.shaderCompute, .numthreads tx 1 1], name := "main", params := params, body := body }

/-- A helper function. -/
def fn (retT : SlangType) (name : String) (params : List SlangBinding) (body : List St) : SlangFunctionDecl :=
  { retType := retT, name := name, params := params, body := body }

end Ninfer.Slang.Dsl
