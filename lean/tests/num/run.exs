# SPDX-License-Identifier: Apache-2.0
# Ninfer.Num against the host compiler's hardware conversions.
#
#   elixir tests/num/run.exs          (from lean/)
#
# 1. `ninfer_num gen` writes the binary32 patterns to convert (every binary16 and bfloat16 rounding
#    boundary, both signs, and a random sample).
# 2. tests/num/oracle.cpp converts them with the compiler's `_Float16` and `__bf16`.
# 3. `ninfer_num check` compares the Lean conversions with that output and must exit 0.
# 4. Two planted-wrong conversions (ties rounded up, bfloat16 truncated) must each exit 1: a
#    comparison that passes a known-broken conversion is not a gate.
here = __DIR__
lean = Path.expand(Path.join(here, "../.."))
build = Path.join(lean, "build/num")
case File.mkdir_p(build) do
  :ok -> :ok
  {:error, reason} -> IO.puts(:stderr, "mkdir #{build}: #{inspect(reason)}"); System.halt(1)
end

run = fn cmd, args, opts ->
  case System.cmd(cmd, args, [stderr_to_stdout: true] ++ opts) do
    {out, 0} -> {:ok, out}
    {out, status} -> {:error, {status, out}}
  end
end

exe = fn name -> Path.join([lean, ".lake/build/bin", name]) end

indent = fn out ->
  out |> String.split("\n", trim: true) |> Enum.map_join("\n", fn l -> "      " <> l end)
end

steps = [
  {"build the checker", fn -> run.("lake", ["build", "ninfer_num"], cd: lean) end, :ok},
  {"write the inputs", fn -> run.(exe.("ninfer_num"), ["gen", Path.join(build, "inputs.txt")], []) end, :ok},
  {"build the oracle",
   fn ->
     run.("c++", ["-std=c++17", "-O1", "-ffp-contract=off", "-o", Path.join(build, "num_oracle"),
                  Path.join(here, "oracle.cpp")], [])
   end, :ok},
  {"run the oracle",
   fn ->
     run.(Path.join(build, "num_oracle"), [Path.join(build, "inputs.txt"), Path.join(build, "oracle.txt")], [])
   end, :ok},
  {"Lean agrees with the hardware",
   fn -> run.(exe.("ninfer_num"), ["check", Path.join(build, "oracle.txt")], []) end, :ok},
  {"control: ties rounded up must differ",
   fn -> run.(exe.("ninfer_num"), ["check", Path.join(build, "oracle.txt"), "tie-away"], []) end, :must_fail},
  {"control: truncated bfloat16 must differ",
   fn -> run.(exe.("ninfer_num"), ["check", Path.join(build, "oracle.txt"), "bf16-truncate"], []) end, :must_fail}
]

failed =
  Enum.reduce(steps, [], fn {name, fun, expect}, acc ->
    result = fun.()

    ok? =
      case {expect, result} do
        {:ok, {:ok, out}} ->
          IO.puts("ok    #{name}\n" <> indent.(out))
          true

        {:must_fail, {:error, {1, out}}} ->
          IO.puts("ok    #{name}\n" <> indent.(out))
          true

        {_, {:ok, out}} ->
          IO.puts("FAIL  #{name}: expected a failure, got a pass\n" <> indent.(out))
          false

        {_, {:error, {status, out}}} ->
          IO.puts("FAIL  #{name}: exit #{status}\n" <> indent.(out))
          false
      end

    if ok?, do: acc, else: [name | acc]
  end)

if failed == [] do
  IO.puts("num: every step passed")
else
  IO.puts("num: #{length(failed)} step(s) failed")
  System.halt(1)
end
