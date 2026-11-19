struct Q4Params {
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
ConstantBuffer<Q4Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint8_t> qdata;
[[vk::binding(2, 0)]]
StructuredBuffer<uint16_t> scales;
[[vk::binding(3, 0)]]
StructuredBuffer<uint16_t> act;
[[vk::binding(4, 0)]]
RWStructuredBuffer<uint16_t> dst;
[[vk::binding(5, 0)]]
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
  uint cu = ((byte >> (j * 4u)) & 15u);
  return (int(cu) - (16 * int(((cu >> 3u) & 1u))));
}

[shader("compute")] [numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
  uint groupLinear = (gid.x + (params.gridX * (gid.y + (params.gridY * gid.z))));
  uint loc = gid.x;
  uint t0 = (gid.y * 4u);
  uint lane = gtid.x;
  uint row = (params.rowBegin + loc);
  if ((lane == 0u)) {
    if (((t0 + 0u) < params.tokens)) {
      float tot0 = sh[((groupLinear * 128u) + ((0u * 32u) + 0u))];
      uint t = (t0 + 0u);
      uint oi = ((t * params.outLd) + (params.outOffset + loc));
      if ((params.accumulate != 0u)) {
        tot0 = (tot0 + bf16ToF32(uint(dst[oi])));
      }
      dst[oi] = uint16_t(f32ToBf16(tot0));
    }
    if (((t0 + 1u) < params.tokens)) {
      float tot1 = sh[((groupLinear * 128u) + ((1u * 32u) + 0u))];
      uint t = (t0 + 1u);
      uint oi = ((t * params.outLd) + (params.outOffset + loc));
      if ((params.accumulate != 0u)) {
        tot1 = (tot1 + bf16ToF32(uint(dst[oi])));
      }
      dst[oi] = uint16_t(f32ToBf16(tot1));
    }
    if (((t0 + 2u) < params.tokens)) {
      float tot2 = sh[((groupLinear * 128u) + ((2u * 32u) + 0u))];
      uint t = (t0 + 2u);
      uint oi = ((t * params.outLd) + (params.outOffset + loc));
      if ((params.accumulate != 0u)) {
        tot2 = (tot2 + bf16ToF32(uint(dst[oi])));
      }
      dst[oi] = uint16_t(f32ToBf16(tot2));
    }
    if (((t0 + 3u) < params.tokens)) {
      float tot3 = sh[((groupLinear * 128u) + ((3u * 32u) + 0u))];
      uint t = (t0 + 3u);
      uint oi = ((t * params.outLd) + (params.outOffset + loc));
      if ((params.accumulate != 0u)) {
        tot3 = (tot3 + bf16ToF32(uint(dst[oi])));
      }
      dst[oi] = uint16_t(f32ToBf16(tot3));
    }
  }
}
