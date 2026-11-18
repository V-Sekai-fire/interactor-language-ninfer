# SPDX-License-Identifier: Apache-2.0
# The ternary linear kernels against the Lean reference model.
#
#   elixir tests/kernels/run.exs [--metal] [--only <regex>] [--skip-big] [--skip-mutants]    (from lean/)
#
# For every (case, variant) of `ninfer_linear plan`: `ninfer_linear write` makes the inputs and the
# expectation, a host runs the kernel over them, `ninfer_linear check` judges. Two hosts:
#
#   * cpu: the kernel as a DAG of executes (kernels/dag.exs). A kernel with group barriers is a DAG of
#     barrier-free processes; each is one run of its own C++ binary over the whole grid, in
#     topological order, and the shared arrays travel between runs as files. Every kernel runs
#     here, the 32-lane ones too. The first run of a case that has no spread of inputs wide enough
#     to matter is not trusted: the exact cases run the DAG a second time with every group's threads
#     in the opposite order and demand the same bytes. A process that depended on that order would
#     have a hazard between lanes.
#   * metal (--metal, macOS): the kernel with its barriers, on the GPU (tests/kernels/metal_host.swift).
#
# The `_f32` probes are emitted into build/ here, since they exist only for this gate.
#
# Controls, each of which must make the verdict FAIL (a gate that passes a known-broken kernel or
# input is not a gate):
#   * flip-row: the weights handed to the kernel have one row's signs flipped, the expectation not;
#   * faults planted in the emitted kernels (`emit_ninfer --mutant`): the lanes added in a plain
#     sequence, the illegal code `10` decoded as -2, the padding guard dropped, ties rounded away;
#   * faults planted in the DAG: every process in one (no barrier), two folds of the tree in one,
#     every edge reversed. The first two are also refused by the static check (`Dag.check`) before
#     anything is emitted, so they are run with --unchecked; the third is order-correct as far as the
#     static check can tell and only the model's verdict catches it;
#   * the premise: slangc's C++ target still refuses the fused kernel, the reason there is a DAG.
#
# Also: spirv-val on every committed SPIR-V module, and (macOS) the Metal compiler on every committed
# Metal source.
here = __DIR__
lean = Path.expand(Path.join(here, "../.."))
Code.require_file(Path.join(lean, "kernels/dag.exs"))
Code.require_file(Path.join(lean, "kernels/lib.exs"))
alias Ninfer.Fs
build = Path.join(lean, "build/kernels")
kdir = Path.join(lean, "kernels")
slangc = Fs.must(Ninfer.Slangc.standalone(System.get_env("SLANGC") || "slangc", Path.join(lean, "build/slangc")))
args = System.argv()
metal? = "--metal" in args
skip_big? = "--skip-big" in args

only =
  case Enum.find_index(args, &(&1 == "--only")) do
    nil -> nil
    i -> Fs.must(Fs.regex(Enum.at(args, i + 1)))
  end

bin = fn n -> Path.join(lean, ".lake/build/bin/#{n}") end

run = fn cmd, cmd_args, opts ->
  case System.cmd(cmd, cmd_args, [stderr_to_stdout: true] ++ opts) do
    {out, 0} -> {:ok, out}
    {out, status} -> {:error, {status, out}}
  end
end

die = fn msg -> IO.puts(:stderr, msg); System.halt(1) end

must = fn result, what ->
  case result do
    {:ok, out} -> out
    {:error, {status, out}} -> die.("#{what} failed (exit #{status}):\n#{out}")
  end
end

Fs.must(Fs.rm_rf(build))
Fs.must(Fs.mkdir_p(build))

IO.puts("== build ==")
must.(run.("lake", ["build", "ninfer_linear", "emit_ninfer"], cd: lean), "lake build")

IO.puts("== selftest ==")
IO.write(must.(run.(bin.("ninfer_linear"), ["selftest"], []), "ninfer_linear selftest"))

# --- hosts -----------------------------------------------------------------------------------

# Emit the kernels under a configuration into `dir`, compile every process of the selected kernels
# to C++ (or take the committed emissions) and link one host per process: `dir/host/<stem>`.
# Returns the kernels of the manifest.
build_hosts = fn dir, emit_flags, opts ->
  committed? = Keyword.get(opts, :committed, false)
  select = Keyword.get(opts, :select, fn _ -> true end)

  for d <- ["slang", "cpp", "host"], do: Fs.must(Fs.mkdir_p(Path.join(dir, d)))
  must.(run.(bin.("emit_ninfer"), [Path.join(dir, "slang"), "--glue"] ++ emit_flags, []),
        "emit_ninfer #{Enum.join(emit_flags, " ")}")

  kernels = Ninfer.Dag.parse_manifest(Fs.must(Fs.read(Path.join(dir, "slang/manifest.txt"))))
  chosen = kernels |> Enum.filter(fn {name, _} -> select.(name) end) |> Map.new()
  cpp_dir = if committed?, do: Path.join(kdir, "cpp"), else: Path.join(dir, "cpp")

  chosen
  |> Ninfer.Dag.stems()
  |> Task.async_stream(
    fn stem ->
      with :ok <-
             (if committed? do
                :ok
              else
                case run.(slangc, ["-target", "cpp", "-stage", "compute", "-entry", "main", "-o",
                                   "cpp/#{stem}_emit.cpp", "slang/#{stem}.slang"], cd: dir) do
                  {:ok, _} -> :ok
                  {:error, {_, out}} -> {:error, "slangc cpp #{stem}: #{out}"}
                end
              end),
           {:ok, _} <-
             run.("c++", ["-std=c++17", "-O2", "-ffp-contract=off", "-w", "-I", cpp_dir,
                          "-I", Path.join(dir, "slang"), Path.join(dir, "slang/#{stem}_glue.cpp"),
                          Path.join(here, "cpu_host.cpp"), "-o", Path.join(dir, "host/#{stem}")], []) do
        :ok
      else
        {:error, {_, out}} -> {:error, "c++ #{stem}: #{out}"}
        {:error, msg} -> {:error, msg}
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: 600_000
  )
  |> Enum.each(fn
    {:ok, :ok} -> :ok
    {:ok, {:error, msg}} -> die.(msg)
    {:exit, reason} -> die.("host build crashed: #{inspect(reason)}")
  end)

  kernels
end

IO.puts("== C++ hosts: the committed kernels, one per process ==")
prod = build_hosts.(Path.join(build, "prod"), [], committed: true)
IO.puts("   #{length(Ninfer.Dag.stems(prod))} processes of #{map_size(prod)} kernels")
IO.puts("== C++ hosts: the FP32 probes ==")
probe = build_hosts.(Path.join(build, "probe"), ["--probe"], select: &String.ends_with?(&1, "_f32"))
IO.puts("   #{length(Ninfer.Dag.stems(Map.filter(probe, fn {n, _} -> String.ends_with?(n, "_f32") end)))} processes")

# {kernels, host dir} of the configuration a kernel belongs to
config_for = fn kernel ->
  if String.ends_with?(kernel, "_f32"),
    do: {probe, Path.join(build, "probe/host")},
    else: {prod, Path.join(build, "prod/host")}
end

# :ok | {:error, reason}
run_cpu = fn kernel, script, variant, lanes ->
  {kernels, hosts} = config_for.(kernel)

  case Map.fetch(kernels, kernel) do
    {:ok, k} -> Ninfer.Dag.run(k, hosts, script, variant, lanes)
    :error -> {:error, "no kernel #{kernel} in the manifest"}
  end
end

judge = fn dir, variant -> run.(bin.("ninfer_linear"), ["check", dir, variant], []) end

# --- the Metal host --------------------------------------------------------------------------

metal_for =
  if metal? do
    IO.puts("== Metal host ==")
    must.(run.("elixir", [Path.join(here, "metal.exs"), "prepare", build], []), "metal prepare")
    fn kernel, script -> run.(Path.join(build, "metal/host"), [Path.join(build, "metal/lib"), kernel, script], []) end
  else
    nil
  end

# --- the plan --------------------------------------------------------------------------------

plan =
  must.(run.(bin.("ninfer_linear"), ["plan"], []), "ninfer_linear plan")
  |> String.split("\n", trim: true)
  |> Enum.map(&String.split(&1, " "))
  |> Enum.map(fn [case_id, kind, variant, kernel] -> %{case: case_id, kind: kind, variant: variant, kernel: kernel} end)
  |> Enum.filter(fn p -> only == nil or Regex.match?(only, p.case) end)
  |> Enum.reject(fn p ->
    skip_big? and p.kind == "crit" and
      p.case in ["c34816x5120t1", "c34816x5120t2", "c5120x6144t1", "c5120x6144t2", "c5120x6144t17",
                 "c5120x6144t65", "c5120x6144t129"]
  end)

cases = plan |> Enum.map(& &1.case) |> Enum.uniq()
IO.puts("== #{length(plan)} (case, variant) pairs over #{length(cases)} cases ==")

# The second run of the order gate: the same DAG, each group's threads last to first, the same bytes.
order_gate = fn p, script ->
  with {:ok, forward} <- Ninfer.Dag.outputs(script),
       :ok <- run_cpu.(p.kernel, script, p.variant, :reverse),
       {:ok, reverse} <- Ninfer.Dag.outputs(script) do
    if forward == reverse,
      do: {true, "the same bytes with each group's threads last to first"},
      else: {false, "the result depends on the order a group's threads run: a hazard between lanes"}
  else
    {:error, reason} -> {false, "did not run: #{reason}"}
  end
end

results =
  for c <- cases, reduce: [] do
    acc ->
      dir = Path.join(build, "case/#{c}")
      t0 = System.monotonic_time(:millisecond)
      must.(run.(bin.("ninfer_linear"), ["write", dir, c], []), "write #{c}")

      rs =
        for p <- Enum.filter(plan, &(&1.case == c)), host <- ["cpu"] ++ if(metal?, do: ["metal"], else: []) do
          script = Path.join(dir, "script.#{p.variant}.txt")

          # A host that wrote nothing must not be judged on a previous host's output.
          for stale <- Path.wildcard(Path.join(dir, "#{p.variant}.*.out")), do: Fs.must(Fs.rm(stale))

          ran =
            case host do
              "cpu" -> run_cpu.(p.kernel, script, p.variant, :forward)
              "metal" ->
                case metal_for.(p.kernel, script) do
                  {:ok, _} -> :ok
                  {:error, {_, out}} -> {:error, String.trim(out)}
                end
            end

          {ok?, text} =
            case ran do
              :ok ->
                case judge.(dir, p.variant) do
                  {:ok, out} -> {true, out}
                  {:error, {_, out}} -> {false, out}
                end

              {:error, reason} ->
                {false, "did not run: #{reason}"}
            end

          main = Map.merge(p, %{ok: ok?, text: String.trim(text), host: host})

          if host == "cpu" and ok? and p.kind != "crit" do
            {gate_ok?, gate_text} = order_gate.(p, script)
            [main, Map.merge(p, %{ok: gate_ok?, text: gate_text, host: "cpu/reversed"})]
          else
            [main]
          end
        end
        |> List.flatten()

      Fs.must(Fs.rm_rf(dir))
      secs = Float.round((System.monotonic_time(:millisecond) - t0) / 1000, 1)
      bad = Enum.count(rs, &(not &1.ok))
      IO.puts("   #{String.pad_trailing(c, 16)} #{length(rs)} runs, #{bad} failed (#{secs} s)")
      for r <- rs, not r.ok, do: IO.puts("      FAIL #{r.variant} [#{r.host}]: #{r.text}")
      acc ++ rs
  end

# --- controls --------------------------------------------------------------------------------

IO.puts("== controls: each verdict must be FAIL ==")

must_fail = fn name, verdict ->
  case verdict do
    {:error, {1, out}} ->
      IO.puts("   ok    #{name}: " <> (out |> String.split("\n", trim: true) |> List.first() |> String.slice(0, 150)))
      true

    {:ok, out} ->
      IO.puts("   FAIL  #{name}: the gate passed a broken kernel or input\n      " <> String.trim(out))
      false

    {:error, {status, out}} ->
      IO.puts("   FAIL  #{name}: exit #{status} is not a verdict\n      " <> String.trim(out))
      false
  end
end

# A control on one (case, variant): the fixture in `tag`, an optional input control, the kernels and
# hosts of a configuration, a lane order; the verdict of `ninfer_linear check`. A run that does not
# finish is not a verdict (status 2), so a crash cannot pass a control.
control = fn tag, case_id, variant, kernel, input_control, {kernels, hosts}, lanes ->
  dir = Path.join(build, "control/#{tag}")
  write_args = ["write", dir, case_id] ++ if(input_control, do: [input_control], else: [])
  must.(run.(bin.("ninfer_linear"), write_args, []), "write #{case_id} control")

  verdict =
    case Ninfer.Dag.run(Fs.must(Fs.fetch(kernels, kernel)), hosts, Path.join(dir, "script.#{variant}.txt"), variant, lanes) do
      :ok -> judge.(dir, variant)
      {:error, reason} -> {:error, {2, "did not run: #{reason}"}}
    end

  Fs.must(Fs.rm_rf(dir))
  verdict
end

flips =
  for {c, v, k} <- [{"e0", "a16_serial", "t2_a16_serial"}, {"e1", "a8_serial", "t2_a8_serial"},
                    {"e0", "a16_serial_f32", "t2_a16_serial_f32"}, {"e0", "a16_lanes", "t2_a16_lanes"},
                    {"e1", "a8_lanes_f32", "t2_a8_lanes_f32"},
                    {"q4-e0", "q4_a16_serial", "q4_a16_serial"}, {"q5-e0", "q5_a16_lanes", "q5_a16_lanes"},
                    {"q6-e0", "q6_a16_serial_f32", "q6_a16_serial_f32"}, {"q8-e0", "q8_a16_wide", "q8_a16_wide"},
                    # the accuracy gate can fail too: one wrong row of 130 is far outside the criterion
                    {"c130x5120t2", "a16_serial", "t2_a16_serial"}, {"q8-c130x5120t2", "q8_a16_serial", "q8_a16_serial"}] do
    must_fail.("flip-row #{c} #{v}", control.("flip-#{c}-#{v}", c, v, k, "flip-row", config_for.(k), :forward))
  end

# {mutant, extra emit flags, [{case, variant, kernel}]}
mutant_runs = [
  {"sequential-fold", [], [{"e0", "a16_serial_f32", "t2_a16_serial_f32"}, {"e0", "a16_lanes_f32", "t2_a16_lanes_f32"},
                           {"q4-e0", "q4_a16_serial_f32", "q4_a16_serial_f32"}, {"q8-e0", "q8_a16_lanes_f32", "q8_a16_lanes_f32"}]},
  {"decode-illegal", [], [{"e0", "a16_serial_f32", "t2_a16_serial_f32"}, {"e0", "a8_serial_f32", "t2_a8_serial_f32"},
                          {"e0", "a16_lanes_f32", "t2_a16_lanes_f32"}]},
  {"no-guard", [], [{"e1", "a16_serial_f32", "t2_a16_serial_f32"}, {"e1", "a16_lanes_f32", "t2_a16_lanes_f32"},
                    {"q4-e1", "q4_a16_serial_f32", "q4_a16_serial_f32"}, {"q8-e1", "q8_a16_lanes_f32", "q8_a16_lanes_f32"},
                    {"q5-e7", "q5_a16_serial_f32", "q5_a16_serial_f32"}]},
  {"tie-away", [], [{"q300t2", "a8_quantize", "t2_a8_quantize"}]},
  {"merge-all", ["--unchecked"], [{"e0", "a16_lanes_f32", "t2_a16_lanes_f32"}, {"e0", "a8_lanes_f32", "t2_a8_lanes_f32"},
                                 {"q6-e0", "q6_a16_lanes_f32", "q6_a16_lanes_f32"}]},
  {"merge-fold", ["--unchecked"], [{"e0", "a16_lanes_f32", "t2_a16_lanes_f32"}, {"q4-e0", "q4_a16_wide_f32", "q4_a16_wide_f32"}]},
  {"reverse-edges", [], [{"e0", "a16_lanes_f32", "t2_a16_lanes_f32"}, {"e0", "a8_lanes_f32", "t2_a8_lanes_f32"},
                         {"q5-e0", "q5_a16_lanes_f32", "q5_a16_lanes_f32"}]},
  # a code read as unsigned, the high plane left out, the nibbles of a byte swapped
  {"decode-unsigned", [], [{"q4-e0", "q4_a16_serial_f32", "q4_a16_serial_f32"}, {"q5-e0", "q5_a16_serial_f32", "q5_a16_serial_f32"},
                           {"q6-e0", "q6_a16_lanes_f32", "q6_a16_lanes_f32"}, {"q8-e0", "q8_a16_serial_f32", "q8_a16_serial_f32"}]},
  {"drop-high", [], [{"q5-e0", "q5_a16_serial_f32", "q5_a16_serial_f32"}, {"q6-e0", "q6_a16_serial_f32", "q6_a16_serial_f32"},
                     {"q6-e0", "q6_a16_wide_f32", "q6_a16_wide_f32"}]},
  {"swap-nibbles", [], [{"q4-e0", "q4_a16_lanes_f32", "q4_a16_lanes_f32"}, {"q5-e0", "q5_a16_serial_f32", "q5_a16_serial_f32"},
                        {"q6-e0", "q6_a16_serial_f32", "q6_a16_serial_f32"}]}
]

mutants =
  for {m, extra, runs} <- mutant_runs, "--skip-mutants" not in args do
    dir = Path.join(build, "mutant/#{m}")
    names = runs |> Enum.map(&elem(&1, 2)) |> Enum.uniq()
    flags = ["--mutant", m] ++ if(m == "tie-away", do: [], else: ["--probe"]) ++ extra
    kernels = build_hosts.(dir, flags, select: &(&1 in names))
    cfg = {kernels, Path.join(dir, "host")}

    for {c, v, k} <- runs do
      must_fail.("mutant #{m}: #{c} #{v}", control.("mut-#{m}-#{c}-#{v}", c, v, k, nil, cfg, :forward))
    end
  end
  |> List.flatten()

# The order gate must see a hazard the verdict of one order can miss: with two folds in one process
# the threads that run first read what the later ones have not yet written, in one order and not the
# other.
order_control =
  if "--skip-mutants" in args do
    []
  else
    dir = Path.join(build, "control/order-merge-fold")
    kernels = Ninfer.Dag.parse_manifest(Fs.must(Fs.read(Path.join(build, "mutant/merge-fold/slang/manifest.txt"))))
    hosts = Path.join(build, "mutant/merge-fold/host")
    k = Fs.must(Fs.fetch(kernels, "t2_a16_lanes_f32"))
    script = Path.join(dir, "script.a16_lanes_f32.txt")
    must.(run.(bin.("ninfer_linear"), ["write", dir, "e0"], []), "write e0 order control")

    outcome =
      with :ok <- Ninfer.Dag.run(k, hosts, script, "a16_lanes_f32", :forward),
           {:ok, forward} <- Ninfer.Dag.outputs(script),
           :ok <- Ninfer.Dag.run(k, hosts, script, "a16_lanes_f32", :reverse),
           {:ok, reverse} <- Ninfer.Dag.outputs(script) do
        if forward != reverse, do: :differs, else: :same
      else
        {:error, reason} -> {:error, reason}
      end

    Fs.must(Fs.rm_rf(dir))

    case outcome do
      :differs ->
        IO.puts("   ok    order gate: two folds in one process give different bytes forward and reversed")
        [true]

      :same ->
        IO.puts("   FAIL  order gate: two folds in one process give the same bytes in both orders")
        [false]

      {:error, reason} ->
        IO.puts("   FAIL  order gate: #{reason}")
        [false]
    end
  end

# `Dag.check` before anything is emitted. The faults that break a barrier are refused; the one that
# only reverses the order is, by design, not.
static =
  for {m, refused?} <- [{"merge-all", true}, {"merge-fold", true}, {"drop-edges", true}, {"reverse-edges", false}] do
    out_dir = Path.join(build, "static/#{m}")

    case {run.(bin.("emit_ninfer"), [out_dir, "--mutant", m], []), refused?} do
      {{:error, {1, msg}}, true} ->
        IO.puts("   ok    static check refuses #{m}: " <> (msg |> String.split("\n", trim: true) |> List.first() |> String.slice(0, 150)))
        true

      {{:ok, _}, false} ->
        IO.puts("   ok    static check accepts #{m} (only the model's verdict can tell it is out of order)")
        true

      {{:ok, _}, true} ->
        IO.puts("   FAIL  static check let #{m} through")
        false

      {{:error, {status, msg}}, _} ->
        IO.puts("   FAIL  static check on #{m}: exit #{status}\n      #{String.trim(msg)}")
        false
    end
  end

# The premise of the DAG: the C++ target refuses a kernel with a barrier. If slangc ever accepts it, the split is
# not needed for that target and this says so.
premise =
  case run.(slangc, ["-target", "cpp", "-stage", "compute", "-entry", "main", "-o", Path.join(build, "fused_lanes.cpp"),
                     "slang/t2_a16_lanes.slang"], cd: kdir) do
    {:error, {_, out}} ->
      if String.contains?(out, "E36107") do
        IO.puts("   ok    slangc's C++ target refuses the fused lanes kernel (E36107): the DAG is what runs it")
        [true]
      else
        IO.puts("   FAIL  slangc refused the fused lanes kernel for another reason:\n      #{String.trim(out)}")
        [false]
      end

    {:ok, _} ->
      IO.puts("   FAIL  slangc's C++ target now accepts the fused lanes kernel: the premise of the DAG changed")
      [false]
  end

# --- targets ---------------------------------------------------------------------------------

IO.puts("== targets: every committed kernel ==")

spv_ok =
  case System.find_executable("spirv-val") do
    nil ->
      IO.puts("   FAIL  spirv-val is not installed: an unchecked module reads as a pass")
      [false]

    val ->
      for f <- Path.wildcard(Path.join(kdir, "spv/*.spv")) do
        case run.(val, ["--target-env", "vulkan1.3", f], []) do
          {:ok, _} -> IO.puts("   ok    spirv-val #{Path.basename(f)}"); true
          {:error, {_, out}} -> IO.puts("   FAIL  spirv-val #{Path.basename(f)}: #{String.trim(out)}"); false
        end
      end
  end

metal_compile_ok =
  case :os.type() do
    {:unix, :darwin} ->
      for f <- Path.wildcard(Path.join(kdir, "metal/*.metal")) do
        air = Path.join(build, Path.basename(f, ".metal") <> ".air")
        case run.("xcrun", ["-sdk", "macosx", "metal", "-c", "-ffp-contract=off", "-fno-fast-math", f, "-o", air], []) do
          {:ok, _} -> IO.puts("   ok    metal -c #{Path.basename(f)}"); true
          {:error, {_, out}} -> IO.puts("   FAIL  metal -c #{Path.basename(f)}: #{String.trim(out)}"); false
        end
      end

    _ ->
      IO.puts("   (the Metal compiler runs on macOS only)")
      []
  end

# --- summary ---------------------------------------------------------------------------------

IO.puts("== summary ==")

for {{variant, host}, rs} <- results |> Enum.group_by(&{&1.variant, &1.host}) |> Enum.sort() do
  IO.puts("   #{String.pad_trailing(variant, 16)} [#{host}] #{Enum.count(rs, & &1.ok)}/#{length(rs)} cases")
end

controls = flips ++ mutants ++ order_control ++ static ++ premise
targets = spv_ok ++ metal_compile_ok
failed = Enum.count(results, &(not &1.ok)) + Enum.count(controls, &(not &1)) + Enum.count(targets, &(not &1))

IO.puts("   #{Enum.count(results, & &1.ok)} verdicts passed, #{Enum.count(controls, & &1)} of #{length(controls)} controls failed as they must, " <>
  "#{Enum.count(targets, & &1)} of #{length(targets)} target checks passed")

unless metal? do
  IO.puts("   NOT RUN  Metal: #{length(plan)} (case, variant) pairs ran on the C++ host only; --metal runs them on the GPU")
end

scope = if metal?, do: "", else: " (C++ host only, Metal not run)"

if failed == 0 do
  IO.puts("kernels: every check passed#{scope}")
else
  IO.puts("kernels: #{failed} check(s) failed")
  System.halt(1)
end
