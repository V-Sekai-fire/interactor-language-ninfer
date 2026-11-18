# Ternary linear in Lean 4 and Slang

This directory holds the T2G128_F16S ternary linear of the engine (`src/ops/linear/t2`). It has a reference model in Lean 4. It also has the Slang kernels that Lean writes. It replaces `mojo/ninfer/ops/linear_t2.mojo`.

Lean fixes every rounding step of the matmuls. On the C++ host and on Metal, the exact test cases return the bytes of the model. The `exp` in `silu_mul` is the exception, because each target has its own `exp`.

## What the model covers

- Weights are row-split planes. Each output row has 32 code bytes and one binary16 scale for each group of 128 inputs. A 2-bit code decodes to `m - 2 (m & s)`. Here `m` is the low bit and `s` is the high bit. The code `10` is illegal and decodes to 0.
- A16 has bfloat16 activations, FP32 accumulation and a bfloat16 output. Lane `l` of 32 adds its four products of a group from left to right. It multiplies that partial sum by the group scale and adds the result to its accumulator. The lanes then combine as `l + (l ^ 16)`, then `^ 8`, `^ 4`, `^ 2` and `^ 1`. The kernel skips inputs past the logical K.
- A8 turns the activations into int8 codes. Each token and group of 128 has one binary16 scale, `f16(amax / 127)`. The dot product of a group is an exact integer. The total is `acc + ((float dot * sw) * sx)`, with the groups in order.
- A kernel can write a row range of the weights into an output plane that has a leading dimension and a column offset. It can also accumulate into the old output.
- SwiGLU is `silu(gate) * up` over a `[gate | up]` plane. The result is a bfloat16 plane.
- The float formats in `Ninfer/Num.lean` round with integer arithmetic. No kernel uses the hardware conversions `f32tof16` or `f16tof32`.

## Layout

```
Ninfer/Num.lean, Rng.lean            float formats, and a seeded generator
Ninfer/Ops/Linear/Spec.lean          the reference model
Ninfer/Ops/Linear/Kernels.lean       the kernels, as LeanSlang trees
Ninfer/Slang/Dsl.lean                short constructors for the trees
Ninfer/Slang/Dag.lean, Kernel.lean   a kernel with barriers, as a DAG of processes
EmitKernels.lean                     emit_ninfer: writes the Slang
LinearCheck.lean                     ninfer_linear: the test inputs, and the verdicts
NumCheck.lean                        ninfer_num: the float formats against the host compiler
kernels/gen.exs, lib.exs, dag.exs    slangc over the Slang; the DAG runner for the C++ host
kernels/slang, spv, metal, cpp       the committed output of gen.exs
tests/num/, tests/kernels/           the runners and the hosts
```

## Barriers and the DAG

A group barrier cuts a kernel into stretches. Every thread finishes one stretch before any thread starts the next. A kernel with barriers is therefore a DAG of processes. A process is a body without a barrier. An edge means that every thread finishes one process before any thread starts the next.

A GPU runs the DAG as one kernel. A barrier separates each pair of levels. Those kernels are in `kernels/spv` and `kernels/metal`. The C++ target of slangc refuses `groupshared` and barriers (error E36107). For that target, each process is a module of its own, in `kernels/cpp`. The host runs the modules one after the other, in topological order. Each shared array becomes a buffer with one row per group. The buffer passes from one execute to the next.

`Dag.check` runs when Lean emits a kernel. It refuses a DAG in three cases:

1. A barrier is inside a process.
2. Two lanes of one process touch the same shared slot, and one of them writes.
3. Two processes touch the same shared array, one of them writes, and no path joins them.

The check over-approximates, so it can refuse a safe DAG. It looks at the shared arrays. It does not look at the device buffers.

## Checks

```sh
lake build                                # the library
lake exe ninfer_linear selftest           # the checks that need no host
elixir tests/num/run.exs                  # float formats against the host compiler
elixir kernels/gen.exs --check            # the committed kernels are what Lean emits
elixir tests/kernels/run.exs [--metal]    # the kernels against the model
```

You need `elan` (it reads `lean-toolchain`), `slangc` 2026.13, a C++17 compiler, Elixir and `spirv-val`. The option `--metal` also needs macOS, with `xcrun metal` and `swiftc`.

`tests/num/run.exs` compares the float formats of Lean with the hardware conversions of the host compiler. It uses 904,446 narrowing inputs and 65,536 widening patterns.

`tests/kernels/run.exs` runs every kernel on the C++ host, as a DAG of executes. With `--metal`, it also runs every kernel on the GPU. For each case, `ninfer_linear` writes the inputs and the expected result. It then judges what the host returns.

- Exact cases (`e0` to `e7`) compare bit for bit. They compare the bfloat16 output. They also compare the FP32 total, through a probe kernel that stores it. The cases cover a row range, an output offset, accumulation, one input past a whole group, and a K of 3. Output buffers start as poison. NaN follows the activations. A kernel that skips a write, or reads past its input, fails.
- Accuracy cases (`c<n>x<k>t<t>`) compare with an FP64 reference. The tolerance is the engine criterion: `tests/ops/linear/linear_test_common.cpp` and `reduction_passes` in `tests/ops/op_check.h`. The shapes come from the Mojo suite that this replaces.
- Quantization cases (`q*`) compare the int8 codes and the binary16 scales bit for bit. The inputs have a zero group, an outlier and a group of exact ties.
- The SwiGLU case (`s192`) compares the projection bit for bit. The SwiGLU output can differ by one bfloat16 step on a rare element. More than 99% of the elements must be exact.
- Each execute of a DAG also runs with the threads of each group in reverse order. For the exact cases, the bytes must not change.

The run also validates each committed SPIR-V module with `spirv-val`. On macOS, it also compiles each Metal module.

The run plants nine kinds of fault. Each one must fail:

- weights with the signs of one row flipped,
- the 32 lanes added in a sequence and not in the tree,
- the illegal code decoded as -2,
- no guard on the codes past the logical K,
- quantization that rounds a tie away from zero,
- all processes merged into one,
- two folds merged into one,
- no edges, and
- reversed edges.

The run fails if one of them passes. `tests/num/run.exs` plants two more faults: ties rounded up, and bfloat16 truncated.

## Limits

- No check runs the CUDA engine. That needs a GPU.
- No check measures speed.
- The model does not fix the `exp` of `silu_mul`. The check allows one bfloat16 step on a rare element.
- Metal runs only on macOS. The CI workflow runs the C++ host only.

## License

Apache-2.0, as the repository.
