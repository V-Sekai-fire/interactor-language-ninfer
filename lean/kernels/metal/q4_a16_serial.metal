#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 35 "slang/q4_a16_serial.slang"
float f16ToF32_0(uint h_0)
{

#line 36
    uint sg_0 = (h_0 & 32768U) << 16U;
    uint e_0 = (h_0 >> 10U) & 31U;
    uint m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 40
        if(m_0 == 0U)
        {

#line 41
            return (as_type<float>((sg_0)));
        }

        return (as_type<float>(((as_type<uint>((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 47
        return (as_type<float>(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (as_type<float>(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}

int gDecode_0(uint byte_0, uint hbyte_0, uint lane_0, uint j_0)
{

#line 53
    uint cu_0 = (byte_0 >> (j_0 * 4U)) & 15U;
    return int(cu_0) - int(16) * int((cu_0 >> 3U) & 1U);
}


#line 23
float bf16ToF32_0(uint h_1)
{

#line 24
    return (as_type<float>((h_1 << 16U)));
}

uint f32ToBf16_0(float x_0)
{

#line 28
    uint w_0 = (as_type<uint>((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 30
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


#line 1
struct Q4Params_0
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


#line 107
struct KernelContext_0
{
    Q4Params_0 constant* params_0;
    ushort device* scales_0;
    uchar device* qdata_0;
    ushort device* act_0;
    ushort device* dst_0;
};


#line 58
[[kernel]] void main_0(uint3 tid_0 [[thread_position_in_grid]], Q4Params_0 constant* params_1 [[buffer(0)]], ushort device* scales_1 [[buffer(2)]], uchar device* qdata_1 [[buffer(1)]], ushort device* act_1 [[buffer(3)]], ushort device* dst_1 [[buffer(4)]])
{

#line 58
    float part0_0;

#line 58
    thread KernelContext_0 kernelContext_0;

#line 58
    (&kernelContext_0)->params_0 = params_1;

#line 58
    (&kernelContext_0)->scales_0 = scales_1;

#line 58
    (&kernelContext_0)->qdata_0 = qdata_1;

#line 58
    (&kernelContext_0)->act_0 = act_1;

#line 58
    (&kernelContext_0)->dst_0 = dst_1;
    uint idx_0 = tid_0.x;
    if(idx_0 >= (params_1->rows_0 * params_1->tokens_0))
    {

#line 61
        return;
    }
    uint loc_0 = idx_0 % params_1->rows_0;
    uint t_0 = idx_0 / params_1->rows_0;
    uint _S1 = (&kernelContext_0)->params_0->rowBegin_0 + loc_0;
    thread array<float, int(32)> lane_1;

#line 66
    uint l_0 = 0U;
    for(;;)
    {

#line 67
        if(l_0 < 32U)
        {
        }
        else
        {

#line 67
            break;
        }

#line 68
        lane_1[l_0] = 0.0f;

#line 67
        l_0 = l_0 + 1U;

#line 67
    }

#line 67
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
        float _S2 = f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));

#line 72
        l_0 = 0U;
        for(;;)
        {

#line 73
            if(l_0 < 32U)
            {
            }
            else
            {

#line 73
                break;
            }

#line 74
            uint _S3 = uint((&kernelContext_0)->qdata_0[gi_0 * 32U + l_0]);
            uint _S4 = g_0 * 64U + l_0 * 2U;

#line 75
            part0_0 = 0.0f;

#line 75
            uint j_1 = 0U;

            for(;;)
            {

#line 77
                if(j_1 < 2U)
                {
                }
                else
                {

#line 77
                    break;
                }

#line 78
                uint _S5 = _S4 + j_1;

#line 78
                if(_S5 < ((&kernelContext_0)->params_0->k_0))
                {

#line 79
                    float wf_0 = float(gDecode_0(_S3, 0U, l_0, j_1));

#line 79
                    float part0_1;
                    if(t_0 < (params_1->tokens_0))
                    {

#line 80
                        part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[t_0 * (&kernelContext_0)->params_0->k_0 + _S5]));

#line 80
                    }
                    else
                    {

#line 80
                        part0_1 = part0_0;

#line 80
                    }

#line 80
                    part0_0 = part0_1;

#line 78
                }

#line 77
                j_1 = j_1 + 1U;

#line 77
            }

#line 85
            float pr0_0 = part0_0 * _S2;
            lane_1[l_0] = lane_1[l_0] + pr0_0;

#line 73
            l_0 = l_0 + 1U;

#line 73
        }

#line 70
        g_0 = g_0 + 1U;

#line 70
    }

#line 70
    l_0 = 0U;

#line 89
    for(;;)
    {

#line 89
        if(l_0 < 16U)
        {
        }
        else
        {

#line 89
            break;
        }

#line 90
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 16U];

#line 89
        l_0 = l_0 + 1U;

#line 89
    }

#line 89
    l_0 = 0U;


    for(;;)
    {

#line 92
        if(l_0 < 8U)
        {
        }
        else
        {

#line 92
            break;
        }

#line 93
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 8U];

#line 92
        l_0 = l_0 + 1U;

#line 92
    }

#line 92
    l_0 = 0U;


    for(;;)
    {

#line 95
        if(l_0 < 4U)
        {
        }
        else
        {

#line 95
            break;
        }

#line 96
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 4U];

#line 95
        l_0 = l_0 + 1U;

#line 95
    }

#line 95
    l_0 = 0U;


    for(;;)
    {

#line 98
        if(l_0 < 2U)
        {
        }
        else
        {

#line 98
            break;
        }

#line 99
        lane_1[l_0] = lane_1[l_0] + lane_1[l_0 + 2U];

#line 98
        l_0 = l_0 + 1U;

#line 98
    }

#line 98
    l_0 = 0U;


    for(;;)
    {

#line 101
        if(l_0 < 1U)
        {
        }
        else
        {

#line 101
            break;
        }

#line 102
        uint _S6 = l_0 + 1U;

#line 102
        lane_1[l_0] = lane_1[l_0] + lane_1[_S6];

#line 101
        l_0 = _S6;

#line 101
    }


    float tot_0 = lane_1[0U];
    uint oi_0 = t_0 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
    if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
    {

#line 106
        part0_0 = tot_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 106
    }
    else
    {

#line 106
        part0_0 = tot_0;

#line 106
    }


    *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(part0_0));
    return;
}
