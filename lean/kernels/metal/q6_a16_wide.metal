#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 39 "slang/q6_a16_wide.slang"
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


#line 108
struct KernelContext_0
{
    Q6Params_0 constant* params_0;
    ushort device* scales_0;
    uchar device* qdata_0;
    uchar device* qhigh_0;
    ushort device* act_0;
    ushort device* dst_0;
    array<array<float, int(32)>, int(4)> threadgroup* sh_0;
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
    threadgroup array<array<float, int(32)>, int(4)> sh_1;

#line 63
    (&kernelContext_0)->sh_0 = &sh_1;
    uint loc_0 = gid_0.x;
    uint t0_0 = gid_0.y * 4U;
    uint lane_1 = gtid_0.x;
    uint _S1 = params_1->rowBegin_0 + loc_0;

#line 67
    float acc0_0 = 0.0f;

#line 67
    float acc1_0 = 0.0f;

#line 67
    float acc2_0 = 0.0f;

#line 67
    float acc3_0 = 0.0f;

#line 67
    uint g_0 = 0U;

#line 72
    for(;;)
    {

#line 72
        if(g_0 < ((&kernelContext_0)->params_0->groups_0))
        {
        }
        else
        {

#line 72
            break;
        }

#line 73
        uint gi_0 = _S1 * (&kernelContext_0)->params_0->groups_0 + g_0;
        float scale_0 = f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));
        uint _S2 = uint((&kernelContext_0)->qdata_0[gi_0 * 32U + lane_1]);
        uint _S3 = uint((&kernelContext_0)->qhigh_0[gi_0 * 16U + lane_1 / 2U]);
        uint _S4 = g_0 * 64U + lane_1 * 2U;

#line 77
        part0_0 = 0.0f;

#line 77
        float part1_0 = 0.0f;

#line 77
        float part2_0 = 0.0f;

#line 77
        float part3_0 = 0.0f;

#line 77
        uint j_1 = 0U;

#line 82
        for(;;)
        {

#line 82
            if(j_1 < 2U)
            {
            }
            else
            {

#line 82
                break;
            }

#line 83
            uint _S5 = _S4 + j_1;

#line 83
            if(_S5 < ((&kernelContext_0)->params_0->k_0))
            {

#line 84
                float wf_0 = float(gDecode_0(_S2, _S3, lane_1, j_1));

#line 84
                float part0_1;
                if(t0_0 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 85
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[t0_0 * (&kernelContext_0)->params_0->k_0 + _S5]));

#line 85
                }
                else
                {

#line 85
                    part0_1 = part0_0;

#line 85
                }


                uint _S6 = t0_0 + 1U;

#line 88
                float part1_1;

#line 88
                if(_S6 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 88
                    part1_1 = part1_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S6 * (&kernelContext_0)->params_0->k_0 + _S5]));

#line 88
                }
                else
                {

#line 88
                    part1_1 = part1_0;

#line 88
                }


                uint _S7 = t0_0 + 2U;

#line 91
                float part2_1;

#line 91
                if(_S7 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 91
                    part2_1 = part2_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S7 * (&kernelContext_0)->params_0->k_0 + _S5]));

#line 91
                }
                else
                {

#line 91
                    part2_1 = part2_0;

#line 91
                }


                uint _S8 = t0_0 + 3U;

#line 94
                float part3_1;

#line 94
                if(_S8 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 94
                    part3_1 = part3_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S8 * (&kernelContext_0)->params_0->k_0 + _S5]));

#line 94
                }
                else
                {

#line 94
                    part3_1 = part3_0;

#line 94
                }

#line 94
                part0_0 = part0_1;

#line 94
                part1_0 = part1_1;

#line 94
                part2_0 = part2_1;

#line 94
                part3_0 = part3_1;

#line 83
            }

#line 82
            j_1 = j_1 + 1U;

#line 82
        }

#line 99
        float pr0_0 = part0_0 * scale_0;
        float _S9 = acc0_0 + pr0_0;
        float pr1_0 = part1_0 * scale_0;
        float _S10 = acc1_0 + pr1_0;
        float pr2_0 = part2_0 * scale_0;
        float _S11 = acc2_0 + pr2_0;
        float pr3_0 = part3_0 * scale_0;
        float _S12 = acc3_0 + pr3_0;

#line 72
        uint g_1 = g_0 + 1U;

#line 72
        acc0_0 = _S9;

#line 72
        acc1_0 = _S10;

#line 72
        acc2_0 = _S11;

#line 72
        acc3_0 = _S12;

#line 72
        g_0 = g_1;

#line 72
    }

#line 108
    (*(&kernelContext_0)->sh_0)[0U][lane_1] = acc0_0;
    (*(&kernelContext_0)->sh_0)[1U][lane_1] = acc1_0;
    (*(&kernelContext_0)->sh_0)[2U][lane_1] = acc2_0;
    (*(&kernelContext_0)->sh_0)[3U][lane_1] = acc3_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 16U)
    {

#line 114
        uint _S13 = lane_1 + 16U;

#line 114
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S13];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S13];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S13];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S13];

#line 113
    }

#line 119
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 8U)
    {

#line 121
        uint _S14 = lane_1 + 8U;

#line 121
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S14];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S14];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S14];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S14];

#line 120
    }

#line 126
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 4U)
    {

#line 128
        uint _S15 = lane_1 + 4U;

#line 128
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S15];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S15];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S15];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S15];

#line 127
    }

#line 133
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 2U)
    {

#line 135
        uint _S16 = lane_1 + 2U;

#line 135
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S16];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S16];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S16];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S16];

#line 134
    }

#line 140
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 1U)
    {

#line 142
        uint _S17 = lane_1 + 1U;

#line 142
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S17];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S17];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S17];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S17];

#line 141
    }

#line 147
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 == 0U)
    {

#line 149
        if(t0_0 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 150
            float tot0_0 = (*(&kernelContext_0)->sh_0)[0U][0U];

            uint oi_0 = t0_0 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 153
                part0_0 = tot0_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 153
            }
            else
            {

#line 153
                part0_0 = tot0_0;

#line 153
            }


            *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(part0_0));

#line 149
        }

#line 158
        uint _S18 = t0_0 + 1U;

#line 158
        if(_S18 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 159
            float tot1_0 = (*(&kernelContext_0)->sh_0)[1U][0U];

            uint oi_1 = _S18 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 162
                part0_0 = tot1_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_1)));

#line 162
            }
            else
            {

#line 162
                part0_0 = tot1_0;

#line 162
            }


            *((&kernelContext_0)->dst_0+oi_1) = ushort(f32ToBf16_0(part0_0));

#line 158
        }

#line 167
        uint _S19 = t0_0 + 2U;

#line 167
        if(_S19 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 168
            float tot2_0 = (*(&kernelContext_0)->sh_0)[2U][0U];

            uint oi_2 = _S19 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 171
                part0_0 = tot2_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_2)));

#line 171
            }
            else
            {

#line 171
                part0_0 = tot2_0;

#line 171
            }


            *((&kernelContext_0)->dst_0+oi_2) = ushort(f32ToBf16_0(part0_0));

#line 167
        }

#line 176
        uint _S20 = t0_0 + 3U;

#line 176
        if(_S20 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 177
            float tot3_0 = (*(&kernelContext_0)->sh_0)[3U][0U];

            uint oi_3 = _S20 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 180
                part0_0 = tot3_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_3)));

#line 180
            }
            else
            {

#line 180
                part0_0 = tot3_0;

#line 180
            }


            *((&kernelContext_0)->dst_0+oi_3) = ushort(f32ToBf16_0(part0_0));

#line 176
        }

#line 148
    }

#line 186
    return;
}
