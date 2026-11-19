struct QuantParams {
  uint k;
  uint groups;
  uint tokens;
};

[[vk::binding(0, 0)]]
ConstantBuffer<QuantParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint16_t> act;
[[vk::binding(2, 0)]]
RWStructuredBuffer<int8_t> codes;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint16_t> xscales;

float bf16ToF32(uint h) {
  return asfloat((h << 16u));
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

uint f32ToF16(float x) {
  uint w = asuint(x);
  uint sg = ((w >> 16u) & 32768u);
  uint e = ((w >> 23u) & 255u);
  uint m = (w & 8388607u);
  if ((e == 255u)) {
    if ((m == 0u)) {
      return (sg | 31744u);
    }
    return ((sg | 32256u) | (m >> 13u));
  }
  if ((e >= 143u)) {
    return (sg | 31744u);
  }
  if ((e >= 113u)) {
    uint h0 = (((e - 112u) << 10u) | (m >> 13u));
    uint rm = (m & 8191u);
    uint h = h0;
    if (((rm > 4096u) || ((rm == 4096u) && ((h0 & 1u) == 1u)))) {
      h = (h0 + 1u);
    }
    return (sg | h);
  }
  if ((e < 102u)) {
    return sg;
  }
  uint shf = (126u - e);
  uint sig = (m | 8388608u);
  uint q0 = (sig >> shf);
  uint rm = (sig & ((1u << shf) - 1u));
  uint mid = (1u << (shf - 1u));
  uint q = q0;
  if (((rm > mid) || ((rm == mid) && ((q0 & 1u) == 1u)))) {
    q = (q0 + 1u);
  }
  return (sg | q);
}

float roundEven(float x) {
  float f = floor(x);
  if (((x - f) == 0.5f)) {
    if (((floor((f * 0.5f)) * 2.0f) == f)) {
      return f;
    }
    return (f + 1.0f);
  }
  return round(x);
}

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint idx = tid.x;
  if ((idx >= (params.groups * params.tokens))) {
    return;
  }
  uint g = (idx % params.groups);
  uint t = (idx / params.groups);
  float amax = 0.0f;
  for (uint ii = 0u; ii < 128u; ++ii) {
    uint pos = ((g * 128u) + ii);
    float xv = 0.0f;
    if ((pos < params.k)) {
      xv = bf16ToF32(uint(act[((t * params.k) + pos)]));
    }
    amax = max(amax, abs(xv));
  }
  uint sh = f32ToF16((amax / 127.0f));
  xscales[((t * params.groups) + g)] = uint16_t(sh);
  float s = f16ToF32(sh);
  float inv = 0.0f;
  if ((s > 0.0f)) {
    inv = (1.0f / s);
  }
  for (uint ii = 0u; ii < 128u; ++ii) {
    uint pos = ((g * 128u) + ii);
    float xv = 0.0f;
    if ((pos < params.k)) {
      xv = bf16ToF32(uint(act[((t * params.k) + pos)]));
    }
    float q = roundEven((xv * inv));
    q = max((-127.0f), min(127.0f, q));
    codes[((((t * params.groups) * 128u) + (g * 128u)) + ii)] = int8_t(int(q));
  }
}
