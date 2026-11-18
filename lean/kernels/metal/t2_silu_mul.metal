#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 13 "slang/t2_silu_mul.slang"
float bf16ToF32_0(uint h_0)
{

#line 14
    return (as_type<float>((h_0 << 16U)));
}

uint f32ToBf16_0(float x_0)
{

#line 18
    uint w_0 = (as_type<uint>((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 20
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


#line 1
struct SiluParams_0
{
    uint inter_0;
    uint total_0;
};


#line 36
struct KernelContext_0
{
    SiluParams_0 constant* params_0;
    ushort device* plane_0;
    ushort device* dst_0;
};


#line 26
[[kernel]] void main_0(uint3 tid_0 [[thread_position_in_grid]], SiluParams_0 constant* params_1 [[buffer(0)]], ushort device* plane_1 [[buffer(1)]], ushort device* dst_1 [[buffer(2)]])
{

#line 26
    thread KernelContext_0 kernelContext_0;

#line 26
    (&kernelContext_0)->params_0 = params_1;

#line 26
    (&kernelContext_0)->plane_0 = plane_1;

#line 26
    (&kernelContext_0)->dst_0 = dst_1;
    uint idx_0 = tid_0.x;
    if(idx_0 >= (params_1->total_0))
    {

#line 29
        return;
    }
    uint t_0 = idx_0 / (&kernelContext_0)->params_0->inter_0;
    uint r_0 = idx_0 % (&kernelContext_0)->params_0->inter_0;
    float gt_0 = bf16ToF32_0(uint((&kernelContext_0)->plane_0[t_0 * (2U * (&kernelContext_0)->params_0->inter_0) + r_0]));


    *((&kernelContext_0)->dst_0+idx_0) = ushort(f32ToBf16_0(gt_0 / (1.0f + exp(- gt_0)) * bf16ToF32_0(uint((&kernelContext_0)->plane_0[t_0 * (2U * (&kernelContext_0)->params_0->inter_0) + (&kernelContext_0)->params_0->inter_0 + r_0]))));
    return;
}
