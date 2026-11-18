struct Q5Params {
  uint k;
  uint groups;
  uint tokens;
  uint rowBegin;
  uint rows;
  uint outLd;
  uint outOffset;
  uint accumulate;
  uint gridX;
  uint gridY;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Q5Params> params;
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
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> sh;

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
  uint hb = ((hbyte >> (((lane & 3u) * 2u) + j)) & 1u);
  uint cu = (((byte >> (j * 4u)) & 15u) | (hb << 4u));
  return (int(cu) - (32 * int(((cu >> 4u) & 1u))));
}

[shader("compute")] [numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
  uint groupLinear = (gid.x + (params.gridX * (gid.y + (params.gridY * gid.z))));
  uint loc = gid.x;
  uint t0 = (gid.y * 1u);
  uint lane = gtid.x;
  uint row = (params.rowBegin + loc);
  float acc0 = 0.0f;
  for (uint g = 0u; g < params.groups; ++g) {
    uint gi = ((row * params.groups) + g);
    float scale = f16ToF32(uint(scales[gi]));
    uint byte = uint(qdata[((gi * 32u) + lane)]);
    uint hbyte = uint(qhigh[((gi * 8u) + (lane / 4u))]);
    uint base = ((g * 64u) + (lane * 2u));
    float part0 = 0.0f;
    for (uint j = 0u; j < 2u; ++j) {
      if (((base + j) < params.k)) {
        float wf = float(gDecode(byte, hbyte, lane, j));
        if (((t0 + 0u) < params.tokens)) {
          part0 = (part0 + (wf * bf16ToF32(uint(act[(((t0 + 0u) * params.k) + (base + j))]))));
        }
      }
    }
    precise float pr0 = (part0 * scale);
    acc0 = (acc0 + pr0);
  }
  sh[((groupLinear * 32u) + ((0u * 32u) + lane))] = acc0;
}
