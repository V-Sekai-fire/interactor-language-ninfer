-- SPDX-License-Identifier: Apache-2.0
import Lake
open Lake DSL

package Ninfer where

-- LeanSlang at the commit that adds litFloatExact, the one contract-rotation-star pins. The
-- repository moved from V-Sekai-fire to V-Sekai-archive and the old URL only redirects, so
-- this names the new one.
require LeanSlang from git
  "https://github.com/V-Sekai-archive/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

-- The library: number formats, the reference models and the Slang emitters.
@[default_target] lean_lib Ninfer

-- Ninfer.Num against the host compiler's hardware conversions (tests/num/run.exs).
lean_exe ninfer_num where
  root := `NumCheck

-- Writes the kernels as Slang for kernels/gen.exs.
lean_exe emit_ninfer where
  root := `EmitKernels
