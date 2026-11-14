# NInfer in Mojo

Mojo ([mojolang.org](https://mojolang.org)) replaces the C++/CUDA inference core. Elixir keeps
the server: the OpenAI, Anthropic and llama.cpp HTTP surfaces run on the BEAM and call the Mojo
engine through NIFs built with [mojoler](https://github.com/thanos/mojoler). No C or C++ layer
sits between them.

```
HTTP clients ──► Elixir server (BEAM) ──mojoler NIF──► Mojo engine ──► CPU / GPU
                 routing, sessions,                    artifact, tokenizer,
                 streaming, scheduling                 kernels, sampling, KV
```

Rules for the split:

- Elixir receives terms; a native address stays inside the NIF (mojoler principle 1).
- Weights never cross the boundary. Mojo opens the `.ninfer` artifact; Elixir holds a resource.
- The Mojo packages are plain libraries with a test CLI, so each can be checked against the C++
  engine before the NIF wrapper exists.

## Port sequence (one PR each)

| # | Piece | C++ source | Status |
|---|---|---|---|
| 1 | Artifact reader: framing, directory, shards, exact range reads | `src/artifact/{framing,schema,reader}` | #7 |
| 2 | Formats, layouts, tensor geometry, binder | `src/artifact/{formats,layouts,binder}`, `src/core/weight*` | this PR (Host side; device placement waits for step 4) |
| 3 | Tokenizer and chat template | `src/text`, `src/models/qwen3_5/frontend` | |
| 4 | Materializer and host transcode | `src/artifact/{materializer,transcode}` | |
| 5 | Ternary and groupwise-int matmul kernels (Mojo GPU) | `src/ops` | |
| 6 | Gated DeltaNet layer, RV32IM in-weights computer | `src/ops`, `src/models/qwen3_5` | |
| 7 | Qwen3.5 model, KV cache, sampler, speculative decode | `src/models`, `src/runtime` | |
| 8 | mojoler NIF surface and the Elixir server | `src/serve` | |

Each PR keeps the C++ engine building and adds parity tests against it (or against the Python
reference in `tools/artifact`).

## Layout and tests

```
mojo/ninfer/core/       weight.mojo (QType, QuantLayout, WeightGeometry)
mojo/ninfer/artifact/   json framing schema reader formats layouts binder
mojo/tests/             dump_artifact.mojo  geometry_probe.mojo
                        test_artifact_reader.py  test_geometry.py
```

```sh
pixi run --manifest-path mojo/pixi.toml test
```

The tests write artifacts with the production writer (`tools/artifact`), single-file and
sharded, read them with the Mojo reader, and compare against the Python reference reader and
against the C++ validation messages. Verified on Mojo 1.2.0 (stable) and a 1.3 nightly.
