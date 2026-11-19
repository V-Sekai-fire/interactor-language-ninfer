struct SiluParams {
  uint inter;
  uint total;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SiluParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint16_t> plane;
[[vk::binding(2, 0)]]
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

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint idx = tid.x;
  if ((idx >= params.total)) {
    return;
  }
  uint t = (idx / params.inter);
  uint r = (idx % params.inter);
  float gt = bf16ToF32(uint(plane[((t * (2u * params.inter)) + r)]));
  float up = bf16ToF32(uint(plane[(((t * (2u * params.inter)) + params.inter) + r)]));
  float sg = (gt / (1.0f + exp((-gt))));
  dst[idx] = uint16_t(f32ToBf16((sg * up)));
}
