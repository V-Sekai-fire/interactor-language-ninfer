# Runs the generated NimbleParsec modules and prints results in the format of parsec_probe.mojo.
defmodule Helper do
  def count(rest, acc, context, _line, _offset), do: {rest, [length(acc)], context}
end

defmodule Ser do
  def term(i) when is_integer(i), do: "i#{i}"
  def term(b) when is_binary(b), do: "b" <> Base.encode16(b, case: :lower)
  def term(a) when is_atom(a), do: "a#{a}"
  def term(l) when is_list(l), do: "[" <> Enum.map_join(l, ",", &term/1) <> "]"
  def term(t) when is_tuple(t), do: "{" <> Enum.map_join(Tuple.to_list(t), ",", &term/1) <> "}"
  def hex(b), do: Base.encode16(b, case: :lower)

  def run(mod, input) do
    case mod.main(input) do
      {:ok, acc, rest, _ctx, {l, o}, off} -> "ok #{term(acc)} rest=#{hex(rest)} line=#{l},#{o} off=#{off}"
      {:error, reason, rest, _ctx, {l, o}, off} -> "err #{hex(reason)} rest=#{hex(rest)} line=#{l},#{o} off=#{off}"
    end
  end
end

[source, cases] = System.argv()
Code.require_file(source)
for {mod, inputs} <- cases |> File.read!() |> String.split("\n", trim: true) |> Enum.with_index() |> Enum.map(fn {line, i} -> {Module.concat([Generated, "P#{i}"]), JSON.decode!(line)} end) do
  IO.puts("case #{mod |> Module.split() |> List.last() |> String.trim_leading("P")}")
  for input <- inputs, do: IO.puts(Ser.run(mod, input))
end
