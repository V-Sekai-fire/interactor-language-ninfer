#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 54 "slang/t2_a8_serial.slang"
int t2Decode_0(uint code_0)
{

#line 55
    uint mg_0 = code_0 & 1U;

    return int(mg_0) - int(2) * int(mg_0 & ((code_0 >> 1U) & 1U));
}


#line 37
float f16ToF32_0(uint h_0)
{

#line 38
    uint sg_0 = (h_0 & 32768U) << 16U;
    uint e_0 = (h_0 >> 10U) & 31U;
    uint m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 42
        if(m_0 == 0U)
        {

#line 43
            return (as_type<float>((sg_0)));
        }

        return (as_type<float>(((as_type<uint>((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 49
        return (as_type<float>(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (as_type<float>(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}


#line 25
float bf16ToF32_0(uint h_1)
{

#line 26
    return (as_type<float>((h_1 << 16U)));
}

uint f32ToBf16_0(float x_0)
{

#line 30
    uint w_0 = (as_type<uint>((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 32
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


#line 1
struct T2Params_0
{
    uint k_0;
    uint groups_0;
    uint tokens_0;
    uint rowBegin_0;
    uint rows_0;
    uint outLd_0;
    uint outOffset_0;
    uint accumulate_0;
};


#line 89
struct KernelContext_0
{
    T2Params_0 constant* params_0;
    uchar device* qdata_0;
    char device* codes_0;
    ushort device* scales_0;
    ushort device* xscales_0;
    ushort device* dst_0;
};


#line 61
[[kernel]] void main_0(uint3 tid_0 [[thread_position_in_grid]], T2Params_0 constant* params_1 [[buffer(0)]], uchar device* qdata_1 [[buffer(1)]], char device* codes_1 [[buffer(3)]], ushort device* scales_1 [[buffer(2)]], ushort device* xscales_1 [[buffer(4)]], ushort device* dst_1 [[buffer(5)]])
{

#line 61
    thread KernelContext_0 kernelContext_0;

#line 61
    (&kernelContext_0)->params_0 = params_1;

#line 61
    (&kernelContext_0)->qdata_0 = qdata_1;

#line 61
    (&kernelContext_0)->codes_0 = codes_1;

#line 61
    (&kernelContext_0)->scales_0 = scales_1;

#line 61
    (&kernelContext_0)->xscales_0 = xscales_1;

#line 61
    (&kernelContext_0)->dst_0 = dst_1;
    uint idx_0 = tid_0.x;
    if(idx_0 >= (params_1->rows_0 * params_1->tokens_0))
    {

#line 64
        return;
    }
    uint loc_0 = idx_0 % params_1->rows_0;
    uint t_0 = idx_0 / params_1->rows_0;
    uint _S1 = (&kernelContext_0)->params_0->rowBegin_0 + loc_0;

#line 68
    float tot_0 = 0.0f;

#line 68
    uint g_0 = 0U;

    for(;;)
    {

#line 70
        if(g_0 < ((&kernelContext_0)->params_0->groups_0))
        {
        }
        else
        {

#line 70
            break;
        }

#line 71
        uint gi_0 = _S1 * (&kernelContext_0)->params_0->groups_0 + g_0;
        uint _S2 = gi_0 * 32U;
        uint _S3 = t_0 * (&kernelContext_0)->params_0->groups_0 * 128U + g_0 * 128U;

#line 73
        int dsum_0 = int(0);

#line 73
        uint b_0 = 0U;

        for(;;)
        {

#line 75
            if(b_0 < 32U)
            {
            }
            else
            {

#line 75
                break;
            }

#line 76
            uint _S4 = uint((&kernelContext_0)->qdata_0[_S2 + b_0]);

#line 76
            uint j_0 = 0U;
            for(;;)
            {

#line 77
                if(j_0 < 4U)
                {
                }
                else
                {

#line 77
                    break;
                }

#line 78
                int _S5 = dsum_0 + t2Decode_0((_S4 >> (j_0 * 2U)) & 3U) * int((&kernelContext_0)->codes_0[_S3 + b_0 * 4U + j_0]);

#line 77
                uint j_1 = j_0 + 1U;

#line 77
                dsum_0 = _S5;

#line 77
                j_0 = j_1;

#line 77
            }

#line 75
            b_0 = b_0 + 1U;

#line 75
        }

#line 83
        float p1_0 = float(dsum_0) * f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));
        float term_0 = p1_0 * f16ToF32_0(uint((&kernelContext_0)->xscales_0[t_0 * (&kernelContext_0)->params_0->groups_0 + g_0]));
        float _S6 = tot_0 + term_0;

#line 70
        uint g_1 = g_0 + 1U;

#line 70
        tot_0 = _S6;

#line 70
        g_0 = g_1;

#line 70
    }

#line 87
    uint oi_0 = t_0 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
    if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
    {

#line 88
        tot_0 = tot_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 88
    }


    *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(tot_0));
    return;
}
