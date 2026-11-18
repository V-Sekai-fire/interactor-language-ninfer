// SPDX-License-Identifier: Apache-2.0
// The host compiler's own binary16 and bfloat16 conversions, as the oracle for Ninfer.Num.
//
//   num_oracle <inputs.txt> <out.txt>
//
// inputs.txt: one binary32 bit pattern per line, hex. out.txt gets, per input line,
//   n <f32 bits> <binary16 bits> <bfloat16 bits>      narrowing, round to nearest even
// and, for every one of the 65536 binary16 patterns,
//   w <binary16 bits> <binary32 bits>                  widening
// Built with -ffp-contract=off -fno-fast-math so nothing is folded to a different rounding.
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint32_t bits32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float from32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t bits16(_Float16 h) { uint16_t u; std::memcpy(&u, &h, 2); return u; }
static _Float16 half16(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return h; }
static uint16_t bitsbf(__bf16 b) { uint16_t u; std::memcpy(&u, &b, 2); return u; }

int main(int argc, char **argv) {
	if (argc != 3) { std::fprintf(stderr, "usage: num_oracle <inputs> <out>\n"); return 2; }
	FILE *in = std::fopen(argv[1], "r");
	FILE *out = std::fopen(argv[2], "w");
	if (!in || !out) { std::fprintf(stderr, "cannot open files\n"); return 2; }
	unsigned v;
	while (std::fscanf(in, "%x", &v) == 1) {
		float f = from32(v);
		std::fprintf(out, "n %08x %04x %04x\n", v, bits16((_Float16)f), bitsbf((__bf16)f));
	}
	for (unsigned h = 0; h < 65536; ++h)
		std::fprintf(out, "w %04x %08x\n", h, bits32((float)half16((uint16_t)h)));
	std::fclose(in);
	std::fclose(out);
	return 0;
}
