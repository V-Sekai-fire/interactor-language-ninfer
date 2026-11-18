struct Q6Params {
  uint k;
  uint groups;
  uint tokens;
  uint rowBegin;
  uint rows;
  uint outLd;
  uint outOffset;
  uint accumulate;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Q6Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint8_t> qdata;
[[vk::binding(2, 0)]]
StructuredBuffer<uint8_t> qhigh;
[[vk::binding(3, 0)]]
StructuredBuffer<uint16_t> scales;
[[vk::binding(4, 0)]]
StructuredBuffer<uint16_t> act;
[[vk::binding(5, 0)]]
RWStructuredBuffer<uint16_t> dst;

float bf16ToF32(uint h) {
  return asfloat((h << 16u));
}

uint f32ToBf16(float x) {
  uint w = asuint(x);
  if (((w & 2147483647u) > 2139095040u)) {
    return ((w >> 16u) | 64u);
  }
  return (((w + 32767u) + ((w >> 16u) & 1u)) >> 16u);
}

float f16ToF32(uint h) {
  uint sg = ((h & 32768u) << 16u);
  uint e = ((h >> 10u) & 31u);
  uint m = (h & 1023u);
  if ((e == 0u)) {
    if ((m == 0u)) {
      return asfloat(sg);
    }
    float mag = (float(m) * 5.9604645e-8f);
    return asfloat((asuint(mag) | sg));
  }
  if ((e == 31u)) {
    return asfloat(((sg | 2139095040u) | (m << 13u)));
  }
  return asfloat(((sg | ((e + 112u) << 23u)) | (m << 13u)));
}

int gDecode(uint byte, uint hbyte, uint lane, uint j) {
  uint hb = ((hbyte >> (((lane & 1u) * 4u) + (j * 2u))) & 3u);
  uint cu = (((byte >> (j * 4u)) & 15u) | (hb << 4u));
  return (int(cu) - (64 * int(((cu >> 5u) & 1u))));
}

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint idx = tid.x;
  if ((idx >= (params.rows * params.tokens))) {
    return;
  }
  uint loc = (idx % params.rows);
  uint t = (idx / params.rows);
  uint row = (params.rowBegin + loc);
  float lane[32];
  for (uint l = 0u; l < 32u; ++l) {
    lane[l] = 0.0f;
  }
  for (uint g = 0u; g < params.groups; ++g) {
    uint gi = ((row * params.groups) + g);
    float scale = f16ToF32(uint(scales[gi]));
    for (uint l = 0u; l < 32u; ++l) {
      uint byte = uint(qdata[((gi * 32u) + l)]);
      uint hbyte = uint(qhigh[((gi * 16u) + (l / 2u))]);
      uint base = ((g * 64u) + (l * 2u));
      float part0 = 0.0f;
      for (uint j = 0u; j < 2u; ++j) {
        if (((base + j) < params.k)) {
          float wf = float(gDecode(byte, hbyte, l, j));
          if ((t < params.tokens)) {
            part0 = (part0 + (wf * bf16ToF32(uint(act[((t * params.k) + (base + j))]))));
          }
        }
      }
      precise float pr0 = (part0 * scale);
      lane[l] = (lane[l] + pr0);
    }
  }
  for (uint l = 0u; l < 16u; ++l) {
    lane[l] = (lane[l] + lane[(l + 16u)]);
  }
  for (uint l = 0u; l < 8u; ++l) {
    lane[l] = (lane[l] + lane[(l + 8u)]);
  }
  for (uint l = 0u; l < 4u; ++l) {
    lane[l] = (lane[l] + lane[(l + 4u)]);
  }
  for (uint l = 0u; l < 2u; ++l) {
    lane[l] = (lane[l] + lane[(l + 2u)]);
  }
  for (uint l = 0u; l < 1u; ++l) {
    lane[l] = (lane[l] + lane[(l + 1u)]);
  }
  float tot = lane[0u];
  uint oi = ((t * params.outLd) + (params.outOffset + loc));
  if ((params.accumulate != 0u)) {
    tot = (tot + bf16ToF32(uint(dst[oi])));
  }
  dst[oi] = uint16_t(f32ToBf16(tot));
}
