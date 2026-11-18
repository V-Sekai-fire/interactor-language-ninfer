// SPDX-License-Identifier: Apache-2.0
// The engine's own row-split weight, and what the engine's own decoder makes of it.
//
//   layout_oracle <t2|q4|q5|q6|q8> <n> <k> <seed> <hashed|coordinate> <dir>
//
// The weight is built by `quantized_weight::make_patterned_weight` and read back by
// `quantized_weight::logical_weight_fp64` and `dot_fp64` (tests/ops/quantized_weight.h): the code the
// engine's linear tests use as their oracle. Nothing is copied from it; the header is included, so
// the oracle cannot drift from the engine. `ninfer_linear layout` then holds the Lean model to it.
//
// Written to <dir>:
//   payload.bin   the engine's payload: code plane, high plane and scale plane at its offsets
//   layout.txt    `key value` lines: n k paddedK group codeBytes highOffset highBytes scaleOffset
//                 scaleBytes tokens
//   logical.bin   n * k doubles, row-major: logical_weight_fp64
//   x.bin         tokens * k float32, every value exactly a bfloat16 (three scales: 0.01, 1, 30)
//   dot.bin       tokens * n doubles, [token][row]: dot_fp64, the engine's FP64 reference
//
// Build with -ffp-contract=off: the reference sums `acc += w * x` step by step, and a fused
// multiply-add would round it differently from the Lean model.

#include "ops/quantized_weight.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qw = ninfer::test::quantized_weight;

namespace {

std::uint64_t mix(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// round to nearest even, then keep the top 16 bits: a float that is exactly a bfloat16
float round_bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, sizeof r);
    return r;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) { return false; }
    const bool ok = std::fwrite(data, 1, bytes, f) == bytes;
    return std::fclose(f) == 0 && ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: layout_oracle <t2|q4|q5|q6|q8> <n> <k> <seed> <hashed|coordinate> <dir>\n");
        return 2;
    }
    const std::string name = argv[1];
    ninfer::QType qtype;
    if (name == "t2") { qtype = ninfer::QType::T2_G128_FP16; }
    else if (name == "q4") { qtype = ninfer::QType::Q4_G64_FP16; }
    else if (name == "q5") { qtype = ninfer::QType::Q5_G64_FP16; }
    else if (name == "q6") { qtype = ninfer::QType::Q6_G64_FP16; }
    else if (name == "q8") { qtype = ninfer::QType::Q8_G32_FP16; }
    else {
        std::fprintf(stderr, "layout_oracle: unknown format %s\n", name.c_str());
        return 2;
    }
    const std::int32_t n    = std::atoi(argv[2]);
    const std::int32_t k    = std::atoi(argv[3]);
    const std::uint32_t seed = static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10));
    const std::string pattern = argv[5];
    const std::string dir     = argv[6];
    if (n <= 0 || k <= 0 || (pattern != "hashed" && pattern != "coordinate")) {
        std::fprintf(stderr, "layout_oracle: bad shape or pattern\n");
        return 2;
    }

    qw::PatternedWeightOptions options;
    options.row_split_scale = qw::RowSplitScalePattern::Small;
    options.row_split_codes = pattern == "hashed" ? qw::RowSplitCodePattern::Hashed
                                                  : qw::RowSplitCodePattern::Coordinate;
    const qw::PackedWeight packed = qw::make_patterned_weight(qtype, n, k, seed, options);

    // the engine's logical weights, as it decodes them
    std::vector<double> logical(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t col = 0; col < k; ++col) {
            logical[static_cast<std::size_t>(row) * k + col] = qw::logical_weight_fp64(packed, row, col);
        }
    }

    // activations: bfloat16-exact, three tokens at the scales the engine's tests see
    constexpr std::int32_t kTokens = 3;
    constexpr float kScale[kTokens] = {0.01F, 1.0F, 30.0F};
    std::vector<float> x(static_cast<std::size_t>(kTokens) * k);
    for (std::int32_t t = 0; t < kTokens; ++t) {
        for (std::int32_t c = 0; c < k; ++c) {
            const std::uint64_t h = mix(static_cast<std::uint64_t>(seed) * 1000003ULL +
                                        static_cast<std::uint64_t>(t) * 7919ULL + static_cast<std::uint64_t>(c));
            const float unit = static_cast<float>(h >> 40) / 16777216.0F - 0.5F;
            x[static_cast<std::size_t>(t) * k + c] = round_bf16(unit * 4.0F * kScale[t]);
        }
    }
    std::vector<double> dot(static_cast<std::size_t>(kTokens) * n);
    for (std::int32_t t = 0; t < kTokens; ++t) {
        for (std::int32_t row = 0; row < n; ++row) {
            dot[static_cast<std::size_t>(t) * n + row] =
                qw::dot_fp64(packed, row, x.data() + static_cast<std::size_t>(t) * k, k);
        }
    }

    char layout[512];
    std::snprintf(layout, sizeof layout,
                  "n %d\nk %d\npaddedK %d\ngroup %d\ncodeBytes %llu\nhighOffset %llu\nhighBytes %llu\n"
                  "scaleOffset %llu\nscaleBytes %llu\ntokens %d\n",
                  n, k, packed.weight.padded_shape[1], packed.weight.group_size,
                  static_cast<unsigned long long>(packed.code_plane_bytes),
                  static_cast<unsigned long long>(packed.high_plane_offset),
                  static_cast<unsigned long long>(packed.high_plane_bytes),
                  static_cast<unsigned long long>(packed.scale_plane_offset),
                  static_cast<unsigned long long>(packed.scale_plane_bytes), kTokens);
    const bool ok =
        write_file(dir + "/payload.bin", packed.payload.data(), packed.payload.size()) &&
        write_file(dir + "/layout.txt", layout, std::strlen(layout)) &&
        write_file(dir + "/logical.bin", logical.data(), logical.size() * sizeof(double)) &&
        write_file(dir + "/x.bin", x.data(), x.size() * sizeof(float)) &&
        write_file(dir + "/dot.bin", dot.data(), dot.size() * sizeof(double));
    if (!ok) {
        std::fprintf(stderr, "layout_oracle: could not write into %s\n", dir.c_str());
        return 1;
    }
    return 0;
}
