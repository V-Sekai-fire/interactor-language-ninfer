struct T2Params {
  uint k;
  uint groups;
  uint tokens;
  uint rowBegin;
  uint rows;
  uint outLd;
  uint outOffset;
  uint accumulate;
};

groupshared float sh[1][32];

[[vk::binding(0, 0)]]
ConstantBuffer<T2Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint8_t> qdata;
[[vk::binding(2, 0)]]
StructuredBuffer<uint16_t> scales;
[[vk::binding(3, 0)]]
StructuredBuffer<uint16_t> act;
[[vk::binding(4, 0)]]
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

int t2Decode(uint code) {
  uint mg = (code & 1u);
  uint sg = ((code >> 1u) & 1u);
  return (int(mg) - (2 * int((mg & sg))));
}

[shader("compute")] [numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
  uint loc = gid.x;
  uint t0 = (gid.y * 1u);
  uint lane = gtid.x;
  uint row = (params.rowBegin + loc);
  float acc0 = 0.0f;
  for (uint g = 0u; g < params.groups; ++g) {
    uint gi = ((row * params.groups) + g);
    float scale = f16ToF32(uint(scales[gi]));
    uint byte = uint(qdata[((gi * 32u) + lane)]);
    uint base = ((g * 128u) + (lane * 4u));
    float part0 = 0.0f;
    for (uint j = 0u; j < 4u; ++j) {
      if (((base + j) < params.k)) {
        float wf = float(t2Decode(((byte >> (j * 2u)) & 3u)));
        if (((t0 + 0u) < params.tokens)) {
          part0 = (part0 + (wf * bf16ToF32(uint(act[(((t0 + 0u) * params.k) + (base + j))]))));
        }
      }
    }
    precise float pr0 = (part0 * scale);
    acc0 = (acc0 + pr0);
  }
  sh[0u][lane] = acc0;
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 16u)) {
    sh[0u][lane] = (sh[0u][lane] + sh[0u][(lane + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 8u)) {
    sh[0u][lane] = (sh[0u][lane] + sh[0u][(lane + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 4u)) {
    sh[0u][lane] = (sh[0u][lane] + sh[0u][(lane + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 2u)) {
    sh[0u][lane] = (sh[0u][lane] + sh[0u][(lane + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane < 1u)) {
    sh[0u][lane] = (sh[0u][lane] + sh[0u][(lane + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((lane == 0u)) {
    if (((t0 + 0u) < params.tokens)) {
      float tot0 = sh[0u][0u];
      uint t = (t0 + 0u);
      uint oi = ((t * params.outLd) + (params.outOffset + loc));
      if ((params.accumulate != 0u)) {
        tot0 = (tot0 + bf16ToF32(uint(dst[oi])));
      }
      dst[oi] = uint16_t(f32ToBf16(tot0));
    }
  }
}
