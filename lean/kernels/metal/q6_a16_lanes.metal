#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 39 "slang/q6_a16_lanes.slang"
float f16ToF32_0(uint h_0)
{

#line 40
    uint sg_0 = (h_0 & 32768U) << 16U;
    uint e_0 = (h_0 >> 10U) & 31U;
    uint m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 44
        if(m_0 == 0U)
        {

#line 45
            return (as_type<float>((sg_0)));
        }

        return (as_type<float>(((as_type<uint>((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 51
        return (as_type<float>(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (as_type<float>(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}

int gDecode_0(uint byte_0, uint hbyte_0, uint lane_0, uint j_0)
{
    uint cu_0 = ((byte_0 >> (j_0 * 4U)) & 15U) | (((hbyte_0 >> ((lane_0 & 1U) * 4U + j_0 * 2U)) & 3U) << 4U);
    return int(cu_0) - int(64) * int((cu_0 >> 5U) & 1U);
}


#line 27
float bf16ToF32_0(uint h_1)
{

#line 28
    return (as_type<float>((h_1 << 16U)));
}

uint f32ToBf16_0(float x_0)
{

#line 32
    uint w_0 = (as_type<uint>((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 34
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


#line 1
struct Q6Params_0
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


#line 87
struct KernelContext_0
{
    Q6Params_0 constant* params_0;
    ushort device* scales_0;
    uchar device* qdata_0;
    uchar device* qhigh_0;
    ushort device* act_0;
    ushort device* dst_0;
    array<array<float, int(32)>, int(1)> threadgroup* sh_0;
};


#line 63
[[kernel]] void main_0(uint3 gid_0 [[threadgroup_position_in_grid]], uint3 gtid_0 [[thread_position_in_threadgroup]], Q6Params_0 constant* params_1 [[buffer(0)]], ushort device* scales_1 [[buffer(3)]], uchar device* qdata_1 [[buffer(1)]], uchar device* qhigh_1 [[buffer(2)]], ushort device* act_1 [[buffer(4)]], ushort device* dst_1 [[buffer(5)]])
{

#line 63
    float part0_0;

#line 63
    thread KernelContext_0 kernelContext_0;

#line 63
    (&kernelContext_0)->params_0 = params_1;

#line 63
    (&kernelContext_0)->scales_0 = scales_1;

#line 63
    (&kernelContext_0)->qdata_0 = qdata_1;

#line 63
    (&kernelContext_0)->qhigh_0 = qhigh_1;

#line 63
    (&kernelContext_0)->act_0 = act_1;

#line 63
    (&kernelContext_0)->dst_0 = dst_1;

#line 63
    threadgroup array<array<float, int(32)>, int(1)> sh_1;

#line 63
    (&kernelContext_0)->sh_0 = &sh_1;
    uint loc_0 = gid_0.x;
    uint _S1 = gid_0.y;
    uint lane_1 = gtid_0.x;
    uint _S2 = params_1->rowBegin_0 + loc_0;

#line 67
    float acc0_0 = 0.0f;

#line 67
    uint g_0 = 0U;

    for(;;)
    {

#line 69
        if(g_0 < ((&kernelContext_0)->params_0->groups_0))
        {
        }
        else
        {

#line 69
            break;
        }

#line 70
        uint gi_0 = _S2 * (&kernelContext_0)->params_0->groups_0 + g_0;
        float scale_0 = f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));
        uint _S3 = uint((&kernelContext_0)->qdata_0[gi_0 * 32U + lane_1]);
        uint _S4 = uint((&kernelContext_0)->qhigh_0[gi_0 * 16U + lane_1 / 2U]);
        uint _S5 = g_0 * 64U + lane_1 * 2U;

#line 74
        part0_0 = 0.0f;

#line 74
        uint j_1 = 0U;

        for(;;)
        {

#line 76
            if(j_1 < 2U)
            {
            }
            else
            {

#line 76
                break;
            }

#line 77
            uint _S6 = _S5 + j_1;

#line 77
            if(_S6 < ((&kernelContext_0)->params_0->k_0))
            {

#line 78
                float wf_0 = float(gDecode_0(_S3, _S4, lane_1, j_1));

#line 78
                float part0_1;
                if(_S1 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 79
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S1 * (&kernelContext_0)->params_0->k_0 + _S6]));

#line 79
                }
                else
                {

#line 79
                    part0_1 = part0_0;

#line 79
                }

#line 79
                part0_0 = part0_1;

#line 77
            }

#line 76
            j_1 = j_1 + 1U;

#line 76
        }

#line 84
        float pr0_0 = part0_0 * scale_0;
        float _S7 = acc0_0 + pr0_0;

#line 69
        uint g_1 = g_0 + 1U;

#line 69
        acc0_0 = _S7;

#line 69
        g_0 = g_1;

#line 69
    }

#line 87
    (*(&kernelContext_0)->sh_0)[0U][lane_1] = acc0_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 16U)
    {

#line 90
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][lane_1 + 16U];

#line 89
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 8U)
    {

#line 94
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][lane_1 + 8U];

#line 93
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 4U)
    {

#line 98
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][lane_1 + 4U];

#line 97
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 2U)
    {

#line 102
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][lane_1 + 2U];

#line 101
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 1U)
    {

#line 106
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][lane_1 + 1U];

#line 105
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 == 0U)
    {

#line 110
        if(_S1 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 111
            float tot0_0 = (*(&kernelContext_0)->sh_0)[0U][0U];

            uint oi_0 = _S1 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 114
                part0_0 = tot0_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 114
            }
            else
            {

#line 114
                part0_0 = tot0_0;

#line 114
            }


            *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(part0_0));

#line 110
        }

#line 109
    }

#line 120
    return;
}
