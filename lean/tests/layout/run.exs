# SPDX-License-Identifier: Apache-2.0
#
# The Lean model's weight layout against the engine's own fixtures.
#
#   elixir tests/layout/run.exs           (from lean/)
#   NINFER_ROOT=<tree> elixir tests/layout/run.exs     a tree that holds src/core and tests/ops
#
# `oracle.cpp` includes the engine's `tests/ops/quantized_weight.h`, builds a row-split weight with
# its `make_patterned_weight` and decodes it with its `logical_weight_fp64` and `dot_fp64`. `ninfer_linear
# layout` reads the engine's bytes with the model's decoder and holds the model to the engine:
#
#   * every weight the model decodes equals the engine's, bit for bit;
#   * the columns past K are zero codes, and a group wholly past K has the scale 0;
#   * packing the model's codes again gives the engine's code and high planes, byte for byte;
#   * the model's FP64 reference equals the engine's `dot_fp64`, bit for bit.
#
# Each shape runs with the engine's two code patterns (hashed, coordinate). A control corrupts one
# byte of a plane after the engine has decoded it; its verdict must be FAIL.

Code.require_file(Path.expand("../../kernels/lib.exs", __DIR__))

alias Ninfer.Fs

here = __DIR__
lean = Path.expand("../..", here)
root = System.get_env("NINFER_ROOT") || Path.expand("..", lean)
build = Path.join(lean, "build/layout")
bin = Path.join(lean, ".lake/build/bin/ninfer_linear")

die = fn msg -> IO.puts(:stderr, msg); System.halt(1) end

run = fn cmd, args ->
  case System.cmd(cmd, args, stderr_to_stdout: true) do
    {out, 0} -> {:ok, out}
    {out, status} -> {:error, {status, out}}
  end
end

for d <- ["src/core/weight.h", "tests/ops/quantized_weight.h"] do
  unless File.exists?(Path.join(root, d)), do: die.("layout: #{d} is not under #{root}; set NINFER_ROOT")
end

Fs.must(Fs.rm_rf(build))
Fs.must(Fs.mkdir_p(build))

IO.puts("== build ==")

case run.("lake", ["build", "ninfer_linear"]) do
  {:ok, _} -> :ok
  {:error, {status, out}} -> die.("lake build failed (exit #{status}):\n#{out}")
end

oracle = Path.join(build, "oracle")

case run.("c++", ["-std=c++20", "-O1", "-ffp-contract=off", "-Wall", "-I", Path.join(root, "src"),
                  "-I", Path.join(root, "include"), "-I", Path.join(root, "tests"),
                  Path.join(here, "oracle.cpp"), "-o", oracle]) do
  {:ok, _} -> IO.puts("   the oracle includes the engine's tests/ops/quantized_weight.h from #{root}")
  {:error, {status, out}} -> die.("the oracle did not build (exit #{status}):\n#{out}")
end

# {n, k}: whole groups, a partial group, a lone input, K below one lane's inputs, many rows, a long K
shapes = [{40, 5120}, {9, 200}, {5, 129}, {3, 3}, {17, 1}, {130, 640}, {4, 17408}]
fmts = ["t2", "q4", "q5", "q6", "q8"]

IO.puts("== the model against the engine's weights ==")

verdicts =
  for fmt <- fmts, {n, k} <- shapes, {pattern, seed} <- [{"hashed", 101}, {"coordinate", 7}] do
    dir = Path.join(build, "#{fmt}-#{n}x#{k}-#{pattern}")
    Fs.must(Fs.mkdir_p(dir))

    case run.(oracle, [fmt, "#{n}", "#{k}", "#{seed}", pattern, dir]) do
      {:ok, _} -> :ok
      {:error, {status, out}} -> die.("the oracle failed on #{fmt} #{n}x#{k} (exit #{status}):\n#{out}")
    end

    {ok?, text} =
      case run.(bin, ["layout", dir, fmt]) do
        {:ok, out} -> {true, out}
        {:error, {_, out}} -> {false, out}
      end

    IO.puts("   #{if ok?, do: "ok  ", else: "FAIL"}  #{fmt} #{n}x#{k} #{pattern}: " <> (text |> String.split("\n", trim: true) |> List.last() |> String.slice(0, 150)))
    unless ok?, do: IO.puts("      " <> String.trim(text))
    ok?
  end

IO.puts("== controls: each verdict must be FAIL ==")

# one byte of a plane, after the engine decoded it: a nibble of the code plane, a bit of the high
# plane, the sign of one scale
corrupt = fn fmt, what ->
  src = Path.join(build, "#{fmt}-40x5120-hashed")
  dir = Path.join(build, "control-#{fmt}-#{what}")
  Fs.must(Fs.rm_rf(dir))
  Fs.must(Fs.mkdir_p(dir))
  for f <- ["layout.txt", "logical.bin", "x.bin", "dot.bin"], do: Fs.must(Fs.cp(Path.join(src, f), Path.join(dir, f)))

  payload = Fs.must(Fs.read(Path.join(src, "payload.bin")))
  layout = Fs.must(Fs.read(Path.join(src, "layout.txt")))
  get = fn key ->
    layout
    |> String.split("\n", trim: true)
    |> Enum.map(&String.split(&1, " "))
    |> Enum.find_value(fn
      [^key, v] ->
        case Integer.parse(v) do
          {i, ""} -> i
          _ -> nil
        end

      _ ->
        nil
    end)
  end

  {offset, mask} =
    case what do
      :code -> {1000, 0x11}
      :high -> {get.("highOffset") + 100, 0x01}
      :scale -> {get.("scaleOffset") + 2 * 7 + 1, 0x80}
    end

  case payload do
    <<head::binary-size(offset), byte, tail::binary>> ->
      Fs.must(Fs.write(Path.join(dir, "payload.bin"), <<head::binary, Bitwise.bxor(byte, mask), tail::binary>>))
      run.(bin, ["layout", dir, fmt])

    _ ->
      die.("layout: the payload of #{fmt} has no byte at #{offset}")
  end
end

controls =
  for {fmt, what} <- [{"t2", :code}, {"q4", :code}, {"q5", :code}, {"q5", :high}, {"q6", :high}, {"q8", :code},
                       {"t2", :scale}, {"q4", :scale}, {"q8", :scale}] do
    case corrupt.(fmt, what) do
      {:error, {1, out}} ->
        IO.puts("   ok    #{fmt} #{what}: " <> (out |> String.split("\n", trim: true) |> List.last() |> String.slice(0, 140)))
        true

      {:ok, out} ->
        IO.puts("   FAIL  #{fmt} #{what}: the gate passed a corrupted plane\n      " <> String.trim(out))
        false

      {:error, {status, out}} ->
        IO.puts("   FAIL  #{fmt} #{what}: exit #{status} is not a verdict\n      " <> String.trim(out))
        false
    end
  end

failed = Enum.count(verdicts, &(not &1)) + Enum.count(controls, &(not &1))
IO.puts("   #{Enum.count(verdicts, & &1)} of #{length(verdicts)} weights match, #{Enum.count(controls, & &1)} of #{length(controls)} controls failed as they must")

if failed == 0 do
  IO.puts("layout: every check passed")
else
  IO.puts("layout: #{failed} check(s) failed")
  System.halt(1)
end
