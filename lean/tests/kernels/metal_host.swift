// SPDX-License-Identifier: Apache-2.0
// Runs one Slang kernel, compiled to Metal, on the GPU over the files a script names.
//
//   metal_host <lib-dir> <kernel> <script>
//
// <lib-dir> holds <kernel>.metallib (its one entry is `main_0`) and manifest.txt, whose
// `kernel <name> threads <n> ...` lines give the threadgroup width. The script is the one
// cpu_host.cpp reads: `grid` is thread groups, `params` go to buffer(0) as bytes, and
// `buf <binding> <bytes per element> <in> <out|->` binds a buffer at that index. A buffer with an
// `out` name is written back after the dispatch. This file owns no judgement: `ninfer_t2 check` does.
import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("metal_host: " + message + "\n").utf8))
    exit(2)
}

let args = CommandLine.arguments
guard args.count == 4 else { fail("usage: metal_host <lib-dir> <kernel> <script>") }
let libDir = args[1]
let kernel = args[2]
let scriptPath = args[3]
let dir = (scriptPath as NSString).deletingLastPathComponent.isEmpty
    ? "." : (scriptPath as NSString).deletingLastPathComponent

func threadsOf(_ kernel: String) -> Int? {
    guard let text = try? String(contentsOfFile: libDir + "/manifest.txt", encoding: .utf8) else { return nil }
    for line in text.split(separator: "\n") {
        let w = line.split(separator: " ")
        if w.count >= 4, w[0] == "kernel", w[1] == kernel, w[2] == "threads", let n = Int(w[3]) { return n }
    }
    return nil
}

struct Buf {
    var binding: Int
    var data: Data
    var out: String
}

guard let script = try? String(contentsOfFile: scriptPath, encoding: .utf8) else { fail("cannot read \(scriptPath)") }
var grid: [Int] = []
var params: [UInt32] = []
var bufs: [Buf] = []
for line in script.split(separator: "\n") {
    let w = line.split(separator: " ").map(String.init)
    switch w.first ?? "" {
    case "grid":
        grid = w.dropFirst().compactMap { Int($0) }
    case "params":
        params = w.dropFirst().compactMap { UInt32($0) }
    case "buf":
        guard w.count == 5, let binding = Int(w[1]) else { fail("bad buf line: \(line)") }
        guard let data = FileManager.default.contents(atPath: dir + "/" + w[3]) else { fail("cannot read \(dir)/\(w[3])") }
        bufs.append(Buf(binding: binding, data: data, out: w[4]))
    default:
        break
    }
}
guard grid.count == 3, grid.allSatisfy({ $0 > 0 }), !bufs.isEmpty else { fail("incomplete script \(scriptPath)") }
guard let threads = threadsOf(kernel) else { fail("\(kernel) is not in \(libDir)/manifest.txt") }

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let library: MTLLibrary
do {
    library = try device.makeLibrary(URL: URL(fileURLWithPath: libDir + "/" + kernel + ".metallib"))
} catch {
    fail("cannot load \(kernel).metallib: \(error)")
}
guard let function = library.makeFunction(name: "main_0") else {
    fail("no main_0 in \(kernel).metallib (it has \(library.functionNames))")
}
let pipeline: MTLComputePipelineState
do {
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("cannot build the pipeline for \(kernel): \(error)")
}
guard threads <= pipeline.maxTotalThreadsPerThreadgroup else {
    fail("\(kernel) wants \(threads) threads, the device allows \(pipeline.maxTotalThreadsPerThreadgroup)")
}
guard let queue = device.makeCommandQueue(),
      let commands = queue.makeCommandBuffer(),
      let encoder = commands.makeComputeCommandEncoder() else { fail("cannot make a command buffer") }

encoder.setComputePipelineState(pipeline)
params.withUnsafeBytes { raw in
    encoder.setBytes(raw.baseAddress!, length: raw.count, index: 0)
}
var gpu: [MTLBuffer] = []
for b in bufs {
    guard let buffer = device.makeBuffer(length: max(b.data.count, 16), options: .storageModeShared) else {
        fail("cannot allocate \(b.data.count) bytes for binding \(b.binding)")
    }
    b.data.withUnsafeBytes { raw in
        if let base = raw.baseAddress { memcpy(buffer.contents(), base, raw.count) }
    }
    encoder.setBuffer(buffer, offset: 0, index: b.binding)
    gpu.append(buffer)
}
encoder.dispatchThreadgroups(
    MTLSize(width: grid[0], height: grid[1], depth: grid[2]),
    threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
encoder.endEncoding()
commands.commit()
commands.waitUntilCompleted()
guard commands.status == .completed else {
    fail("the dispatch of \(kernel) ended as status \(commands.status.rawValue): \(String(describing: commands.error))")
}
for (b, buffer) in zip(bufs, gpu) where b.out != "-" {
    let result = Data(bytes: buffer.contents(), count: b.data.count)
    do {
        try result.write(to: URL(fileURLWithPath: dir + "/" + b.out))
    } catch {
        fail("cannot write \(b.out): \(error)")
    }
}
