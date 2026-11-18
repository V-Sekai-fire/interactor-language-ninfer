# SPDX-License-Identifier: Apache-2.0
# A kernel as a DAG of executes.
#
# A kernel with group barriers is a DAG of barrier-free processes (lean/Ninfer/Slang/Dag.lean). A
# target that cannot run a barrier runs the DAG: each process is one execute of its own host
# binary over the whole grid, in topological order, and what lived in `groupshared` is a file carried
# from one execute to the next. Nothing else crosses an edge, so a node is a process in the
# operating-system sense too: it reads its inputs from files, writes its outputs to files, and exits.
#
#   manifest   what `emit_ninfer` wrote (kernel, buf, scratch, proc and edge lines)
#   script     what `ninfer_t2 write` wrote for the fused kernel: grid, params, buffers by binding
#
# Every function returns :ok / {:ok, value} / {:error, reason}.
defmodule Ninfer.Dag do
  @moduledoc false

  # ---- manifest -----------------------------------------------------------------------------

  @doc "The kernels of a manifest, by name."
  def parse_manifest(text) do
    {kernels, _} =
      text
      |> String.split("\n", trim: true)
      |> Enum.reduce({%{}, nil}, fn line, {acc, cur} ->
        case String.split(line, " ") do
          ["kernel", name, "threads", t, "params", p, "procs", n] ->
            k = %{name: name, threads: String.to_integer(t), params: String.to_integer(p),
                  procs_n: String.to_integer(n), bufs: [], scratch: [], procs: [], edges: []}
            {Map.put(acc, name, k), name}

          ["buf", binding, name, elem, bytes, rw] ->
            b = %{binding: String.to_integer(binding), name: name, elem: elem,
                  bytes: String.to_integer(bytes), rw: rw == "rw"}
            {update_in(acc[cur].bufs, &(&1 ++ [b])), cur}

          ["scratch", name, elem, bytes, per_group] ->
            s = %{name: name, elem: elem, bytes: String.to_integer(bytes),
                  per_group: String.to_integer(per_group)}
            {update_in(acc[cur].scratch, &(&1 ++ [s])), cur}

          ["proc", i, stem, name] ->
            {update_in(acc[cur].procs, &(&1 ++ [%{index: String.to_integer(i), stem: stem, name: name}])), cur}

          ["edge", i, j] ->
            {update_in(acc[cur].edges, &(&1 ++ [{String.to_integer(i), String.to_integer(j)}])), cur}

          _ ->
            {acc, cur}
        end
      end)

    kernels
  end

  @doc "Every process stem of a manifest."
  def stems(kernels), do: kernels |> Map.values() |> Enum.flat_map(& &1.procs) |> Enum.map(& &1.stem) |> Enum.uniq()

  @doc """
  Process indices in execution order: the smallest ready index first. Computed here from the edges,
  not read off the manifest's order, so the runner is an independent reading of the DAG.
  """
  def order(%{procs: procs, edges: edges}) do
    indices = Enum.map(procs, & &1.index)

    case Enum.find(edges, fn {i, j} -> i not in indices or j not in indices or i == j end) do
      nil -> do_order(indices, edges, [])
      {i, j} -> {:error, "edge (#{i}, #{j}) names a process that is not there"}
    end
  end

  defp do_order([], _edges, acc), do: {:ok, Enum.reverse(acc)}

  defp do_order(remaining, edges, acc) do
    ready = Enum.filter(remaining, fn j -> not Enum.any?(edges, fn {i, k} -> k == j and i in remaining end) end)

    case ready do
      [] -> {:error, "the processes have a cycle"}
      [first | _] -> do_order(remaining -- [first], edges, [first | acc])
    end
  end

  # ---- script -------------------------------------------------------------------------------

  @doc "Grid, constant buffer and buffers of a host script."
  def parse_script(path) do
    case File.read(path) do
      {:error, reason} ->
        {:error, "cannot read #{path}: #{reason}"}

      {:ok, text} ->
        lines = text |> String.split("\n", trim: true) |> Enum.map(&String.split(&1, " "))

        grid = Enum.find_value(lines, fn l -> if hd(l) == "grid", do: tl(l) |> Enum.map(&String.to_integer/1) end)
        params = Enum.find_value(lines, [], fn l -> if hd(l) == "params", do: tl(l) |> Enum.map(&String.to_integer/1) end)

        bufs =
          for ["buf", binding, bytes, inp, out] <- lines do
            %{binding: String.to_integer(binding), bytes: String.to_integer(bytes), in: inp, out: out}
          end

        if grid && bufs != [], do: {:ok, %{grid: grid, params: params, bufs: bufs}}, else: {:error, "incomplete script #{path}"}
    end
  end

  # ---- running ------------------------------------------------------------------------------

  @poison_chunk 16 * 1024 * 1024

  # A shared array starts as 0xFF bytes (a NaN for a float): a slot no process wrote reads as NaN.
  defp write_poison(path, size) do
    case :file.open(path, [:write, :raw, :binary]) do
      {:error, reason} ->
        {:error, "cannot write #{path}: #{inspect(reason)}"}

      {:ok, f} ->
        result = write_chunks(f, size, :binary.copy(<<0xFF>>, min(max(size, 1), @poison_chunk)))
        :file.close(f)
        result
    end
  end

  defp write_chunks(_f, 0, _chunk), do: :ok

  defp write_chunks(f, left, chunk) do
    n = min(left, byte_size(chunk))

    case :file.write(f, binary_part(chunk, 0, n)) do
      :ok -> write_chunks(f, left - n, chunk)
      {:error, reason} -> {:error, "poison write failed: #{inspect(reason)}"}
    end
  end

  defp scratch_file(variant, s), do: "#{variant}.#{s.name}.scratch"

  defp proc_script(kernel, proc, sc, variant, first?) do
    [gx, gy, gz] = sc.grid
    n = length(sc.bufs)
    params = if kernel.scratch == [], do: sc.params, else: sc.params ++ [gx, gy]

    bufs =
      for b <- sc.bufs do
        # the first execute reads the input; after it a buffer the DAG writes lives in its out file
        inp = if first? or b.out == "-", do: b.in, else: b.out
        "buf #{b.binding} #{b.bytes} #{inp} #{b.out}"
      end

    scratch =
      for {s, k} <- Enum.with_index(kernel.scratch, 1),
          do: "buf #{n + k} #{s.bytes} #{scratch_file(variant, s)} #{scratch_file(variant, s)}"

    ["kernel #{proc.stem}", "grid #{gx} #{gy} #{gz}", "params #{Enum.join(params, " ")}"] ++ bufs ++ scratch
  end

  @doc """
  Run `kernel` over the fused kernel's `script` as a DAG of executes. `hosts` holds one binary per
  process stem. `lanes` is `:forward` or `:reverse`: the order each execute runs a group's threads.
  The buffers the script names as outputs hold the result when it returns.
  """
  def run(kernel, hosts, script_path, variant, lanes \\ :forward) do
    dir = Path.dirname(script_path)

    with {:ok, order} <- order(kernel),
         {:ok, sc} <- parse_script(script_path),
         :ok <- init_scratch(kernel, sc, dir, variant) do
      order
      |> Enum.with_index()
      |> Enum.reduce_while(:ok, fn {i, n}, :ok ->
        proc = Enum.find(kernel.procs, &(&1.index == i))
        path = Path.join(dir, "#{variant}.proc#{i}.txt")
        lines = Enum.join(proc_script(kernel, proc, sc, variant, n == 0), "\n") <> "\n"
        args = if lanes == :reverse, do: ["--lanes", "reverse", path], else: [path]

        with :ok <- File.write(path, lines),
             {_, 0} <- System.cmd(Path.join(hosts, proc.stem), args, stderr_to_stdout: true) do
          {:cont, :ok}
        else
          {out, status} when is_binary(out) -> {:halt, {:error, "#{proc.stem} exited #{status}: #{String.trim(out)}"}}
          {:error, reason} -> {:halt, {:error, "cannot write #{path}: #{inspect(reason)}"}}
        end
      end)
    end
  end

  defp init_scratch(kernel, sc, dir, variant) do
    [gx, gy, gz] = sc.grid

    Enum.reduce_while(kernel.scratch, :ok, fn s, :ok ->
      case write_poison(Path.join(dir, scratch_file(variant, s)), gx * gy * gz * s.per_group * s.bytes) do
        :ok -> {:cont, :ok}
        {:error, _} = e -> {:halt, e}
      end
    end)
  end

  @doc "The files a script writes: what a run leaves behind, by out name."
  def outputs(script_path) do
    with {:ok, sc} <- parse_script(script_path) do
      dir = Path.dirname(script_path)

      sc.bufs
      |> Enum.filter(&(&1.out != "-"))
      |> Enum.reduce_while({:ok, []}, fn b, {:ok, acc} ->
        case File.read(Path.join(dir, b.out)) do
          {:ok, data} -> {:cont, {:ok, acc ++ [{b.out, data}]}}
          {:error, reason} -> {:halt, {:error, "cannot read #{b.out}: #{inspect(reason)}"}}
        end
      end)
    end
  end
end
