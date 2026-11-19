# SPDX-License-Identifier: Apache-2.0
# The ternary linear kernels, from Lean to every target.
#
#   elixir kernels/gen.exs              emit from Lean, then slangc into kernels/{slang,cpp,spv,metal}
#   elixir kernels/gen.exs --no-emit    compile the committed slang/ (no lake)
#   elixir kernels/gen.exs --check      regenerate in a temp dir; exit 1 if any committed file differs
#
#   Lean (`lake exe emit_ninfer`)   -> slang/<kernel>.slang, slang/<stem>.slang, slang/manifest.txt  (committed)
#   slangc -target spirv            -> spv/<kernel>.spv      the kernel with its barriers, for GPUs  (committed)
#   slangc -target metal            -> metal/<kernel>.metal  the same                                 (committed)
#   slangc -target cpp              -> cpp/<stem>_emit.cpp   one per process of the DAG               (committed)
#
# A kernel with group barriers goes to the GPU targets whole. The C++ target refuses it (E36107: no
# groupshared, no barrier), so it gets the same kernel as a DAG of barrier-free processes
# (lean/Ninfer/Slang/Dag.lean): one slang file and one C++ emission per process. A kernel without a
# barrier is one process and its own stem.
#
# slangc is pinned to 2026.13 (SLANGC overrides the binary): another version emits different text and
# --check reports it. `-fp-mode precise` keeps a target from fusing `a * b + c`, so it rounds in the
# order the C++ path does (tests/kernels/run.exs builds the C++ with -ffp-contract=off).
#
# `precise` (the Lean DSL marks every product the model rounds before an add) lands differently per
# target: SPIR-V gets a NoContraction decoration; the C++ glue defines `precise` away and the host
# build runs with -ffp-contract=off; Metal has no such variable qualifier, so it is stripped here
# (kernels/lib.exs) and the Metal build runs with -ffp-contract=off -fno-fast-math. The Metal host
# in tests/kernels then shows the compiler did not fuse after all.
#
# This slangc inlines its 4,000-line C++ prelude into every emit. The prelude is identical for every
# kernel, so it is committed once as cpp/slang-cpp-prelude.h and each emit keeps one `#include` of it.
here = __DIR__
Code.require_file(Path.join(here, "lib.exs"))
Code.require_file(Path.join(here, "dag.exs"))
alias Ninfer.Fs
lean = Path.expand(Path.join(here, ".."))
slangc = Fs.must(Ninfer.Slangc.standalone(System.get_env("SLANGC") || "slangc", Path.join(lean, "build/slangc")))

mode =
  case System.argv() do
    [] -> :emit
    ["--no-emit"] -> :none
    ["--check"] -> :check
    other -> IO.puts("unknown arguments: #{inspect(other)}"); System.halt(2)
  end

die = fn msg -> IO.puts(:stderr, msg); System.halt(1) end

run = fn cmd, args, opts ->
  case System.cmd(cmd, args, [stderr_to_stdout: true] ++ opts) do
    {out, 0} -> {:ok, out}
    {out, status} -> {:error, {cmd, args, status, out}}
  end
end

must = fn result ->
  case result do
    {:ok, out} -> out
    {:error, {cmd, args, status, out}} -> die.("#{cmd} #{Enum.join(args, " ")} exited #{status}:\n#{out}")
  end
end

# Split an emit into the inlined prelude and the module's own text.
marker = "#ifdef SLANG_PRELUDE_NAMESPACE\nusing namespace SLANG_PRELUDE_NAMESPACE;"

split_prelude = fn text ->
  case :binary.match(text, marker) do
    {at, _} ->
      # `at` is a byte offset: split bytes, not graphemes (the prelude holds multi-byte characters).
      header = binary_part(text, 0, at)
      tail = binary_part(text, at, byte_size(text) - at)

      if String.starts_with?(header, "#ifndef SLANG_CPP_PRELUDE_H") do
        {:ok, header, tail}
      else
        {:error, "the emit does not start with the inlined prelude"}
      end

    :nomatch ->
      {:error, "no prelude boundary in the emit (this slangc does not inline it)"}
  end
end

gen = fn out ->
  for d <- ["slang", "cpp", "spv", "metal"], do: Fs.must(Fs.mkdir_p(Path.join(out, d)))

  if mode != :none do
    must.(run.("lake", ["build", "emit_ninfer"], cd: lean))
    must.(run.(Path.join(lean, ".lake/build/bin/emit_ninfer"), [Path.join(out, "slang")], []))
  else
    for f <- Path.wildcard(Path.join(here, "slang/*")), do: Fs.must(Fs.cp(f, Path.join([out, "slang", Path.basename(f)])))
  end

  kernels = Ninfer.Dag.parse_manifest(Fs.must(Fs.read(Path.join(out, "slang/manifest.txt"))))

  # The GPU targets take the kernel whole. Relative paths: slangc writes the input path into a
  # #line directive.
  for k <- Map.keys(kernels) do
    src = "slang/#{k}.slang"

    must.(run.(slangc, ["-target", "spirv", "-profile", "sm_6_5", "-stage", "compute", "-entry", "main",
                        "-fp-mode", "precise", "-o", "spv/#{k}.spv", src], cd: out))
    must.(run.(slangc, ["-target", "metal", "-stage", "compute", "-entry", "main", "-fp-mode", "precise",
                        "-o", "metal/#{k}.metal", src], cd: out))

    metal_path = Path.join(out, "metal/#{k}.metal")

    case Ninfer.Kernels.strip_precise(Fs.must(Fs.read(metal_path))) do
      {:ok, text} -> Fs.must(Fs.write(metal_path, String.trim_trailing(text) <> "\n"))
      {:error, why} -> die.("#{k}.metal: #{why}")
    end
  end

  # The C++ target takes the processes, one emission each.
  headers =
    for stem <- Ninfer.Dag.stems(kernels) do
      raw = Path.join(out, "cpp/#{stem}_emit.cpp")
      must.(run.(slangc, ["-target", "cpp", "-stage", "compute", "-entry", "main", "-o", "cpp/#{stem}_emit.cpp",
                          "slang/#{stem}.slang"], cd: out))

      case split_prelude.(Fs.must(Fs.read(raw))) do
        {:ok, header, tail} ->
          Fs.must(Fs.write(raw, "#include \"slang-cpp-prelude.h\"\n" <> tail))
          header

        {:error, why} ->
          die.("#{stem}: #{why}")
      end
    end
    |> Enum.uniq()

  case headers do
    [header] -> Fs.must(Fs.write(Path.join(out, "cpp/slang-cpp-prelude.h"), String.trim_trailing(header) <> "\n"))
    other -> die.("the modules carry #{length(other)} different preludes; expected one")
  end

  kernels
end

# The files a manifest accounts for, relative to kernels/.
files_of = fn kernels ->
  ["slang/manifest.txt", "cpp/slang-cpp-prelude.h"] ++
    Enum.flat_map(Enum.sort(Map.keys(kernels)), fn k ->
      stems = Enum.map(kernels[k].procs, & &1.stem)

      ["slang/#{k}.slang", "spv/#{k}.spv", "metal/#{k}.metal"] ++
        for(s <- stems, s != k, do: "slang/#{s}.slang") ++ for(s <- stems, do: "cpp/#{s}_emit.cpp")
    end)
end

if mode == :check do
  tmp = Path.join(Fs.must(Fs.tmp_dir()), "ninfer-kernels-#{System.unique_integer([:positive])}")
  Fs.must(Fs.mkdir_p(tmp))
  kernels = gen.(tmp)
  files = files_of.(kernels)

  stale =
    Enum.filter(files, fn f ->
      File.read(Path.join(tmp, f)) != File.read(Path.join(here, f))
    end)

  extra =
    (Path.wildcard(Path.join(here, "{slang,cpp,spv,metal}/*")) |> Enum.map(&Path.relative_to(&1, here))) -- files

  Fs.must(Fs.rm_rf(tmp))
  for f <- stale, do: IO.puts("stale: kernels/#{f}")
  for f <- extra, do: IO.puts("not generated any more: kernels/#{f}")

  if stale == [] and extra == [] do
    IO.puts("kernels current (#{map_size(kernels)} kernels, #{length(files)} files)")
  else
    System.halt(1)
  end
else
  kernels = gen.(here)
  procs = kernels |> Map.values() |> Enum.map(&length(&1.procs)) |> Enum.sum()
  IO.puts("== #{map_size(kernels)} kernels, #{procs} processes, into kernels/{slang,cpp,spv,metal} ==")
end
