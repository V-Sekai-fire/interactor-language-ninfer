# SPDX-License-Identifier: Apache-2.0
# Shared by kernels/gen.exs and tests/kernels/metal.exs.
defmodule Ninfer.Kernels do
  @moduledoc false

  @doc """
  slangc passes the HLSL `precise` qualifier through to Metal. MSL has no such variable qualifier
  (`precise` there is the namespace of the exact math functions), so `xcrun metal` rejects every
  declaration that carries it. The Metal build gets the same guarantee from its compile flags,
  `-ffp-contract=off -fno-fast-math`, so the qualifier is dropped from the Metal text.

  Only `precise float|half|double` is rewritten; `precise::` (the namespace) is left alone, and a
  `precise` that survives is an error rather than something to ship.
  """
  def strip_precise(text) do
    out = Regex.replace(~r/\bprecise\s+(float|half|double)\b/, text, "\\1")

    case Regex.run(~r/\bprecise\b(?!::)/, out) do
      nil -> {:ok, out}
      _ -> {:error, "a `precise` is left after stripping the qualifier"}
    end
  end
end

defmodule Ninfer.Fs do
  @moduledoc false
  # File, regex and map lookups that return tagged tuples, with the failing path in the error.
  # `must/1` is the end of a script: an error becomes one line on stderr and exit status 1, and a
  # success passes through.
  def mkdir_p(path), do: tag(File.mkdir_p(path), :mkdir_p, path)
  def rm_rf(path), do: tag(File.rm_rf(path), :rm_rf, path)
  def rm(path), do: tag(File.rm(path), :rm, path)
  def read(path), do: tag(File.read(path), :read, path)
  def write(path, data), do: tag(File.write(path, data), :write, path)
  def cp(from, to), do: tag(File.cp(from, to), :cp, from)
  def regex(source), do: tag(Regex.compile(source), :regex, source)
  def fetch(map, key), do: tag(Map.fetch(map, key), :fetch, key)
  def tmp_dir, do: tag(System.tmp_dir(), :tmp_dir, "")

  defp tag(:ok, _, _), do: :ok
  defp tag(:error, op, what), do: {:error, {op, what, :not_found}}
  defp tag(nil, op, what), do: {:error, {op, what, :unavailable}}
  defp tag({:ok, value}, _, _), do: {:ok, value}
  defp tag({:error, reason}, op, what), do: {:error, {op, what, reason}}
  defp tag({:error, reason, file}, op, _), do: {:error, {op, file, reason}}
  defp tag(value, _, _), do: {:ok, value}

  def must(:ok), do: :ok
  def must({:ok, value}), do: value

  def must({:error, reason}) do
    IO.puts(:stderr, "error: #{inspect(reason)}")
    System.halt(1)
  end
end

defmodule Ninfer.Slangc do
  @moduledoc false
  alias Ninfer.Fs

  # slangc writes `#include "<abs>/slang-cpp-prelude.h"` when an `include/` directory sits beside the
  # directory of its executable, and inlines the 4,000-line prelude when none does. The committed C++
  # holds the inlined form, so no host may get the other one. `standalone/2` copies the launcher into
  # <dir>/bin and links the install's `lib/` as <dir>/lib, with no `include/` beside it.
  def standalone(name, dir) do
    bin = Path.join([dir, "bin", "slangc"])

    with {:ok, exe} <- find(name),
         real = resolve(exe),
         {:ok, _} <- Fs.rm_rf(dir),
         :ok <- Fs.mkdir_p(Path.dirname(bin)),
         :ok <- Fs.cp(real, bin),
         :ok <- chmod(bin),
         :ok <- link_lib(Path.join(Path.dirname(Path.dirname(real)), "lib"), Path.join(dir, "lib")),
         :ok <- runs(bin) do
      {:ok, bin}
    end
  end

  defp find(name) do
    found = if String.contains?(name, "/"), do: name, else: System.find_executable(name)
    if is_binary(found) and File.exists?(found), do: {:ok, found}, else: {:error, {:slangc, name, :not_found}}
  end

  # A launcher on PATH may be a symbolic link.
  defp resolve(path) do
    case File.read_link(path) do
      {:ok, target} -> resolve(Path.expand(target, Path.dirname(path)))
      {:error, _} -> path
    end
  end

  defp chmod(path) do
    case File.chmod(path, 0o755) do
      :ok -> :ok
      {:error, reason} -> {:error, {:chmod, path, reason}}
    end
  end

  defp link_lib(lib, link) do
    if File.dir?(lib) do
      case File.ln_s(lib, link) do
        :ok -> :ok
        {:error, reason} -> {:error, {:ln_s, lib, reason}}
      end
    else
      :ok
    end
  end

  defp runs(bin) do
    case System.cmd(bin, ["-v"], stderr_to_stdout: true) do
      {_, 0} -> :ok
      {out, status} -> {:error, {:slangc, bin, {status, String.trim(out)}}}
    end
  end
end
