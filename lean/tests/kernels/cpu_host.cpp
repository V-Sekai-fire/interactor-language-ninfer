// SPDX-License-Identifier: Apache-2.0
// Runs one Slang module, compiled to C++, over the files a script names: one execute of a DAG.
//
//   cpu_host [--lanes forward|reverse] <script>
//
// The script (grammar in lean/T2Check.lean; `tests/kernels/dag.exs` writes one per process):
//   kernel <name>                                   (informational)
//   grid <gx> <gy> <gz>                             thread groups
//   params <u32> ...                                the constant buffer, one u32 each
//   buf <binding> <bytes per element> <in> <out|->  bindings 1..n, in order; `-` writes nothing back
// File names are relative to the script's directory. This file owns no judgement: it dispatches,
// and `ninfer_t2 check` decides. It is linked with one module's generated glue (`kernel_run`).
// `--lanes reverse` runs each group's threads from the last to the first.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

extern "C" void kernel_run(const uint32_t *params, uint8_t *const *bufs, const size_t *counts,
		uint32_t gx, uint32_t gy, uint32_t gz, int reverse);

struct Buf {
	unsigned binding = 0;
	size_t elem = 0;
	std::string out;
	std::vector<uint8_t> data;
};

static bool read_file(const std::string &path, std::vector<uint8_t> &out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	return true;
}

int main(int argc, char **argv) {
	int reverse = 0;
	int first = 1;
	if (argc == 4 && std::strcmp(argv[1], "--lanes") == 0) {
		if (std::strcmp(argv[2], "reverse") == 0) reverse = 1;
		else if (std::strcmp(argv[2], "forward") != 0) {
			std::fprintf(stderr, "cpu_host: --lanes takes forward or reverse, not %s\n", argv[2]);
			return 2;
		}
		first = 3;
	}
	if (argc - first != 1) {
		std::fprintf(stderr, "usage: cpu_host [--lanes forward|reverse] <script>\n");
		return 2;
	}
	std::string script = argv[first];
	std::string dir = script.find('/') == std::string::npos ? "." : script.substr(0, script.rfind('/'));
	std::ifstream in(script);
	if (!in) {
		std::fprintf(stderr, "cpu_host: cannot read %s\n", script.c_str());
		return 2;
	}
	uint32_t grid[3] = { 0, 0, 0 };
	std::vector<uint32_t> params;
	std::vector<Buf> bufs;
	for (std::string line; std::getline(in, line);) {
		std::istringstream ss(line);
		std::string word;
		ss >> word;
		if (word == "grid") {
			ss >> grid[0] >> grid[1] >> grid[2];
		} else if (word == "params") {
			for (uint32_t v; ss >> v;) params.push_back(v);
		} else if (word == "buf") {
			Buf b;
			std::string inName;
			ss >> b.binding >> b.elem >> inName >> b.out;
			if (!read_file(dir + "/" + inName, b.data)) {
				std::fprintf(stderr, "cpu_host: cannot read %s/%s\n", dir.c_str(), inName.c_str());
				return 2;
			}
			bufs.push_back(std::move(b));
		}
	}
	if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0 || bufs.empty()) {
		std::fprintf(stderr, "cpu_host: incomplete script %s\n", script.c_str());
		return 2;
	}
	std::vector<uint8_t *> ptrs;
	std::vector<size_t> counts;
	for (size_t i = 0; i < bufs.size(); ++i) {
		if (bufs[i].binding != i + 1) {
			std::fprintf(stderr, "cpu_host: bindings must be 1..n in order\n");
			return 2;
		}
		ptrs.push_back(bufs[i].data.data());
		counts.push_back(bufs[i].data.size() / bufs[i].elem);
	}
	kernel_run(params.data(), ptrs.data(), counts.data(), grid[0], grid[1], grid[2], reverse);
	for (auto &b : bufs) {
		if (b.out == "-") continue;
		std::ofstream o(dir + "/" + b.out, std::ios::binary);
		o.write(reinterpret_cast<const char *>(b.data.data()), (std::streamsize)b.data.size());
		if (!o) {
			std::fprintf(stderr, "cpu_host: cannot write %s\n", b.out.c_str());
			return 2;
		}
	}
	return 0;
}
