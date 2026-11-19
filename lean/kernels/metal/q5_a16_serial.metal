#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 37 "slang/q5_a16_serial.slang"
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

int gDecode_0(uint byte_0, uint hbyte_0, uint lane_0, uint j_0)
{
    uint cu_0 = ((byte_0 >> (j_0 * 4U)) & 15U) | (((hbyte_0 >> ((lane_0 & 3U) * 2U + j_0)) & 1U) << 4U);
    return int(cu_0) - int(32) * int((cu_0 >> 4U) & 1U);
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
struct Q5Params_0
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


#line 111
struct KernelContext_0
{
    Q5Params_0 constant* params_0;
    ushort device* scales_0;
    uchar device* qdata_0;
    uchar device* qhigh_0;
    ushort device* act_0;
    ushort device* dst_0;
};


#line 61
[[kernel]] void main_0(uint3 tid_0 [[thread_position_in_grid]], Q5Params_0 constant* params_1 [[buffer(0)]], ushort device* scales_1 [[buffer(3)]], uchar device* qdata_1 [[buffer(1)]], uchar device* qhigh_1 [[buffer(2)]], ushort device* act_1 [[buffer(4)]], ushort device* dst_1 [[buffer(5)]])
{

#line 61
    float part0_0;

#line 61
    thread KernelContext_0 kernelContext_0;

#line 61
    (&kernelContext_0)->params_0 = params_1;

#line 61
    (&kernelContext_0)->scales_0 = scales_1;

#line 61
    (&kernelContext_0)->qdata_0 = qdata_1;

#line 61
    (&kernelContext_0)->qhigh_0 = qhigh_1;

#line 61
    (&kernelContext_0)->act_0 = act_1;

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
    thread array<float, int(32)> lane_1;

#line 69
    uint l_0 = 0U;
    for(;;)
    {

#line 70
        if(l_0 < 32U)
        {
        }
        else
        {

#line 70
            break;
        }

#line 71
        lane_1[l_0] = 0.0f;

#line 70
        l_0 = l_0 + 1U;

#line 70
    }

#line 70
    uint g_0 = 0U;


    for(;;)
    {

#line 73
        if(g_0 < ((&kernelContext_0)->params_0->groups_0))
        {
        }
        else
        {

#line 73
            break;
        }

#line 74
        uint gi_0 = _S1 * (&kernelContext_0)->params_0->groups_0 + g_0;
        float _S2 = f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));

#line 75
        l_0 = 0U;
        for(;;)
        {

#line 76
            if(l_0 < 32U)
            {
            }
            else
            {

#line 76
                break;
            }

#line 77
            uint _S3 = uint((&kernelContext_0)->qdata_0[gi_0 * 32U + l_0]);
            uint _S4 = uint((&kernelContext_0)->qhigh_0[gi_0 * 8U + l_0 / 4U]);
            uint _S5 = g_0 * 64U + l_0 * 2U;

#line 79
            part0_0 = 0.0f;

#line 79
            uint j_1 = 0U;

            for(;;)
            {

#line 81
                if(j_1 < 2U)
                {
                }
                else
                {

#line 81
                    break;
                }

#line 82
                uint _S6 = _S5 + j_1;

#line 82
                if(_S6 < ((&kernelContext_0)->params_0->k_0))
                {

#line 83
                    float wf_0 = float(gDecode_0(_S3, _S4, l_0, j_1));

#line 83
                    float part0_1;
                    if(t_0 < (params_1->tokens_0))
                    {

#line 84
                        part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[t_0 * (&kernelContext_0)->params_0->k_0 + _S6]));

#line 84
                    }
                    else
                    {

#line 84
                        part0_1 = part0_0;

#line 84
                    }

#line 84
                    part0_0 = part0_1;

#line 82
                }

#line 81
                j_1 = j_1 + 1U;

#line 81
            }

#line 89
            float pr0_0 = part0_0 * _S2;
            lane_1[l_0] = lane_1[l_0] + pr0_0;

#line 76
            l_0 = l_0 + 1U;

#line 76
        }

#line 73
        g_0 = g_0 + 1U;

#line 73
    }

#line 73
    l_0 = 0U;

#line 93
    for(;;)
    {

#line 93
        if(l_0 < 16U)
        {
        }
        else
        {

#line 93
            break;
        }

#line 94
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 16U];

#line 93
        l_0 = l_0 + 1U;

#line 93
    }

#line 93
    l_0 = 0U;


    for(;;)
    {

#line 96
        if(l_0 < 8U)
        {
        }
        else
        {

#line 96
            break;
        }

#line 97
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 8U];

#line 96
        l_0 = l_0 + 1U;

#line 96
    }

#line 96
    l_0 = 0U;


    for(;;)
    {

#line 99
        if(l_0 < 4U)
        {
        }
        else
        {

#line 99
            break;
        }

#line 100
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 4U];

#line 99
        l_0 = l_0 + 1U;

#line 99
    }

#line 99
    l_0 = 0U;


    for(;;)
    {

#line 102
        if(l_0 < 2U)
        {
        }
        else
        {

#line 102
            break;
        }

#line 103
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 2U];

#line 102
        l_0 = l_0 + 1U;

#line 102
    }

#line 102
    l_0 = 0U;


    for(;;)
    {

#line 105
        if(l_0 < 1U)
        {
        }
        else
        {

#line 105
            break;
        }

#line 106
        uint _S7 = l_0 + 1U;

#line 106
        lane_1[l_0] = lane_1[l_0] + lane_1[_S7];

#line 105
        l_0 = _S7;

#line 105
    }


    float tot_0 = lane_1[0U];
    uint oi_0 = t_0 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
    if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
    {

#line 110
        part0_0 = tot_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 110
    }
    else
    {

#line 110
        part0_0 = tot_0;

#line 110
    }


    *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(part0_0));
    return;
}
