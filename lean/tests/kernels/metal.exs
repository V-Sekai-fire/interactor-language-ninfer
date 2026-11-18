# SPDX-License-Identifier: Apache-2.0
# Builds what `tests/kernels/run.exs --metal` runs on the GPU.
#
#   elixir tests/kernels/metal.exs prepare <build-dir>        (<build-dir> = lean/build/kernels)
#
# Into <build-dir>/metal/:
#   lib/<kernel>.metallib   one library per kernel, because every kernel's entry is `main_0`
#   lib/manifest.txt        the committed kernels' manifest and the probes'
#   host                    tests/kernels/metal_host.swift
#
# The committed kernels/metal/*.metal are compiled as they are, so what is tested is what is
# committed; the FP32 probes (`*_f32`, from `emit_ninfer --probe`) are compiled from the probe
# emission that run.exs wrote under <build-dir>/probe/slang. The Metal compiler runs with
# -fno-fast-math -ffp-contract=off: its default permits fusing a multiply and an add, and the model
# rounds after each of them.
here = __DIR__
Code.require_file(Path.join(here, "../../kernels/lib.exs"))
alias Ninfer.Fs
lean = Path.expand(Path.join(here, "../.."))
slangc = System.get_env("SLANGC") || "slangc"

die = fn msg -> IO.puts(:stderr, msg); System.halt(1) end

run = fn cmd, args, opts ->
  case System.cmd(cmd, args, [stderr_to_stdout: true] ++ opts) do
    {out, 0} -> out
    {out, status} -> die.("#{cmd} #{Enum.join(args, " ")} exited #{status}:\n#{out}")
  end
end

kernels_of = fn path ->
  Fs.must(Fs.read(path))
  |> String.split("\n", trim: true)
  |> Enum.flat_map(fn line ->
    case String.split(line, " ") do
      ["kernel", name | _] -> [name]
      _ -> []
    end
  end)
end

case System.argv() do
  ["prepare", build] ->
    out = Path.join(build, "metal")
    Fs.must(Fs.rm_rf(out))
    for d <- ["src", "air", "lib"], do: Fs.must(Fs.mkdir_p(Path.join(out, d)))

    committed = kernels_of.(Path.join(lean, "kernels/slang/manifest.txt"))
    probe_dir = Path.join(build, "probe")
    probes = kernels_of.(Path.join(probe_dir, "slang/manifest.txt")) -- committed

    if probes == [], do: die.("no probe kernels under #{probe_dir}/slang: run.exs writes them first")

    for k <- committed, do: Fs.must(Fs.cp(Path.join(lean, "kernels/metal/#{k}.metal"), Path.join(out, "src/#{k}.metal")))

    for k <- probes do
      metal_src = Path.join(out, "src/#{k}.metal")

      run.(slangc, ["-target", "metal", "-stage", "compute", "-entry", "main", "-fp-mode", "precise",
                    "-o", metal_src, "slang/#{k}.slang"], cd: probe_dir)

      case Ninfer.Kernels.strip_precise(Fs.must(Fs.read(metal_src))) do
        {:ok, text} -> Fs.must(Fs.write(metal_src, text))
        {:error, why} -> die.("#{k}.metal: #{why}")
      end
    end

    for k <- committed ++ probes do
      src = Path.join(out, "src/#{k}.metal")
      air = Path.join(out, "air/#{k}.air")
      run.("xcrun", ["-sdk", "macosx", "metal", "-c", "-ffp-contract=off", "-fno-fast-math", src, "-o", air], [])
      run.("xcrun", ["-sdk", "macosx", "metallib", air, "-o", Path.join(out, "lib/#{k}.metallib")], [])
    end

    manifest =
      [Path.join(lean, "kernels/slang/manifest.txt"), Path.join(probe_dir, "slang/manifest.txt")]
      |> Enum.map(&Fs.must(Fs.read(&1)))
      |> Enum.join("")

    Fs.must(Fs.write(Path.join(out, "lib/manifest.txt"), manifest))
    run.("swiftc", ["-O", Path.join(here, "metal_host.swift"), "-o", Path.join(out, "host")], [])
    IO.puts("metal: #{length(committed)} committed kernels and #{length(probes)} probes in #{out}/lib, host built")

  _ ->
    die.("usage: elixir tests/kernels/metal.exs prepare <build-dir>")
end
