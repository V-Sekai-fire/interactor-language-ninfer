#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 37 "slang/t2_a16_wide.slang"
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

int t2Decode_0(uint code_0)
{

#line 55
    uint mg_0 = code_0 & 1U;

    return int(mg_0) - int(2) * int(mg_0 & ((code_0 >> 1U) & 1U));
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


#line 105
struct KernelContext_0
{
    T2Params_0 constant* params_0;
    ushort device* scales_0;
    uchar device* qdata_0;
    ushort device* act_0;
    ushort device* dst_0;
    array<array<float, int(32)>, int(4)> threadgroup* sh_0;
};


#line 61
[[kernel]] void main_0(uint3 gid_0 [[threadgroup_position_in_grid]], uint3 gtid_0 [[thread_position_in_threadgroup]], T2Params_0 constant* params_1 [[buffer(0)]], ushort device* scales_1 [[buffer(2)]], uchar device* qdata_1 [[buffer(1)]], ushort device* act_1 [[buffer(3)]], ushort device* dst_1 [[buffer(4)]])
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
    (&kernelContext_0)->act_0 = act_1;

#line 61
    (&kernelContext_0)->dst_0 = dst_1;

#line 61
    threadgroup array<array<float, int(32)>, int(4)> sh_1;

#line 61
    (&kernelContext_0)->sh_0 = &sh_1;
    uint loc_0 = gid_0.x;
    uint t0_0 = gid_0.y * 4U;
    uint lane_0 = gtid_0.x;
    uint _S1 = params_1->rowBegin_0 + loc_0;

#line 65
    float acc0_0 = 0.0f;

#line 65
    float acc1_0 = 0.0f;

#line 65
    float acc2_0 = 0.0f;

#line 65
    float acc3_0 = 0.0f;

#line 65
    uint g_0 = 0U;

#line 70
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
        float scale_0 = f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));
        uint _S2 = uint((&kernelContext_0)->qdata_0[gi_0 * 32U + lane_0]);
        uint _S3 = g_0 * 128U + lane_0 * 4U;

#line 74
        part0_0 = 0.0f;

#line 74
        float part1_0 = 0.0f;

#line 74
        float part2_0 = 0.0f;

#line 74
        float part3_0 = 0.0f;

#line 74
        uint j_0 = 0U;

#line 79
        for(;;)
        {

#line 79
            if(j_0 < 4U)
            {
            }
            else
            {

#line 79
                break;
            }

#line 80
            uint _S4 = _S3 + j_0;

#line 80
            if(_S4 < ((&kernelContext_0)->params_0->k_0))
            {

#line 81
                float wf_0 = float(t2Decode_0((_S2 >> (j_0 * 2U)) & 3U));

#line 81
                float part0_1;
                if(t0_0 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 82
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[t0_0 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 82
                }
                else
                {

#line 82
                    part0_1 = part0_0;

#line 82
                }


                uint _S5 = t0_0 + 1U;

#line 85
                float part1_1;

#line 85
                if(_S5 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 85
                    part1_1 = part1_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S5 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 85
                }
                else
                {

#line 85
                    part1_1 = part1_0;

#line 85
                }


                uint _S6 = t0_0 + 2U;

#line 88
                float part2_1;

#line 88
                if(_S6 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 88
                    part2_1 = part2_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S6 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 88
                }
                else
                {

#line 88
                    part2_1 = part2_0;

#line 88
                }


                uint _S7 = t0_0 + 3U;

#line 91
                float part3_1;

#line 91
                if(_S7 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 91
                    part3_1 = part3_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S7 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 91
                }
                else
                {

#line 91
                    part3_1 = part3_0;

#line 91
                }

#line 91
                part0_0 = part0_1;

#line 91
                part1_0 = part1_1;

#line 91
                part2_0 = part2_1;

#line 91
                part3_0 = part3_1;

#line 80
            }

#line 79
            j_0 = j_0 + 1U;

#line 79
        }

#line 96
        float pr0_0 = part0_0 * scale_0;
        float _S8 = acc0_0 + pr0_0;
        float pr1_0 = part1_0 * scale_0;
        float _S9 = acc1_0 + pr1_0;
        float pr2_0 = part2_0 * scale_0;
        float _S10 = acc2_0 + pr2_0;
        float pr3_0 = part3_0 * scale_0;
        float _S11 = acc3_0 + pr3_0;

#line 70
        uint g_1 = g_0 + 1U;

#line 70
        acc0_0 = _S8;

#line 70
        acc1_0 = _S9;

#line 70
        acc2_0 = _S10;

#line 70
        acc3_0 = _S11;

#line 70
        g_0 = g_1;

#line 70
    }

#line 105
    (*(&kernelContext_0)->sh_0)[0U][lane_0] = acc0_0;
    (*(&kernelContext_0)->sh_0)[1U][lane_0] = acc1_0;
    (*(&kernelContext_0)->sh_0)[2U][lane_0] = acc2_0;
    (*(&kernelContext_0)->sh_0)[3U][lane_0] = acc3_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_0 < 16U)
    {

#line 111
        uint _S12 = lane_0 + 16U;

#line 111
        (*(&kernelContext_0)->sh_0)[0U][lane_0] = (*(&kernelContext_0)->sh_0)[0U][lane_0] + (*(&kernelContext_0)->sh_0)[0U][_S12];
        (*(&kernelContext_0)->sh_0)[1U][lane_0] = (*(&kernelContext_0)->sh_0)[1U][lane_0] + (*(&kernelContext_0)->sh_0)[1U][_S12];
        (*(&kernelContext_0)->sh_0)[2U][lane_0] = (*(&kernelContext_0)->sh_0)[2U][lane_0] + (*(&kernelContext_0)->sh_0)[2U][_S12];
        (*(&kernelContext_0)->sh_0)[3U][lane_0] = (*(&kernelContext_0)->sh_0)[3U][lane_0] + (*(&kernelContext_0)->sh_0)[3U][_S12];

#line 110
    }

#line 116
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_0 < 8U)
    {

#line 118
        uint _S13 = lane_0 + 8U;

#line 118
        (*(&kernelContext_0)->sh_0)[0U][lane_0] = (*(&kernelContext_0)->sh_0)[0U][lane_0] + (*(&kernelContext_0)->sh_0)[0U][_S13];
        (*(&kernelContext_0)->sh_0)[1U][lane_0] = (*(&kernelContext_0)->sh_0)[1U][lane_0] + (*(&kernelContext_0)->sh_0)[1U][_S13];
        (*(&kernelContext_0)->sh_0)[2U][lane_0] = (*(&kernelContext_0)->sh_0)[2U][lane_0] + (*(&kernelContext_0)->sh_0)[2U][_S13];
        (*(&kernelContext_0)->sh_0)[3U][lane_0] = (*(&kernelContext_0)->sh_0)[3U][lane_0] + (*(&kernelContext_0)->sh_0)[3U][_S13];

#line 117
    }

#line 123
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_0 < 4U)
    {

#line 125
        uint _S14 = lane_0 + 4U;

#line 125
        (*(&kernelContext_0)->sh_0)[0U][lane_0] = (*(&kernelContext_0)->sh_0)[0U][lane_0] + (*(&kernelContext_0)->sh_0)[0U][_S14];
        (*(&kernelContext_0)->sh_0)[1U][lane_0] = (*(&kernelContext_0)->sh_0)[1U][lane_0] + (*(&kernelContext_0)->sh_0)[1U][_S14];
        (*(&kernelContext_0)->sh_0)[2U][lane_0] = (*(&kernelContext_0)->sh_0)[2U][lane_0] + (*(&kernelContext_0)->sh_0)[2U][_S14];
        (*(&kernelContext_0)->sh_0)[3U][lane_0] = (*(&kernelContext_0)->sh_0)[3U][lane_0] + (*(&kernelContext_0)->sh_0)[3U][_S14];

#line 124
    }

#line 130
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_0 < 2U)
    {

#line 132
        uint _S15 = lane_0 + 2U;

#line 132
        (*(&kernelContext_0)->sh_0)[0U][lane_0] = (*(&kernelContext_0)->sh_0)[0U][lane_0] + (*(&kernelContext_0)->sh_0)[0U][_S15];
        (*(&kernelContext_0)->sh_0)[1U][lane_0] = (*(&kernelContext_0)->sh_0)[1U][lane_0] + (*(&kernelContext_0)->sh_0)[1U][_S15];
        (*(&kernelContext_0)->sh_0)[2U][lane_0] = (*(&kernelContext_0)->sh_0)[2U][lane_0] + (*(&kernelContext_0)->sh_0)[2U][_S15];
        (*(&kernelContext_0)->sh_0)[3U][lane_0] = (*(&kernelContext_0)->sh_0)[3U][lane_0] + (*(&kernelContext_0)->sh_0)[3U][_S15];

#line 131
    }

#line 137
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_0 < 1U)
    {

#line 139
        uint _S16 = lane_0 + 1U;

#line 139
        (*(&kernelContext_0)->sh_0)[0U][lane_0] = (*(&kernelContext_0)->sh_0)[0U][lane_0] + (*(&kernelContext_0)->sh_0)[0U][_S16];
        (*(&kernelContext_0)->sh_0)[1U][lane_0] = (*(&kernelContext_0)->sh_0)[1U][lane_0] + (*(&kernelContext_0)->sh_0)[1U][_S16];
        (*(&kernelContext_0)->sh_0)[2U][lane_0] = (*(&kernelContext_0)->sh_0)[2U][lane_0] + (*(&kernelContext_0)->sh_0)[2U][_S16];
        (*(&kernelContext_0)->sh_0)[3U][lane_0] = (*(&kernelContext_0)->sh_0)[3U][lane_0] + (*(&kernelContext_0)->sh_0)[3U][_S16];

#line 138
    }

#line 144
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_0 == 0U)
    {

#line 146
        if(t0_0 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 147
            float tot0_0 = (*(&kernelContext_0)->sh_0)[0U][0U];

            uint oi_0 = t0_0 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 150
                part0_0 = tot0_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 150
            }
            else
            {

#line 150
                part0_0 = tot0_0;

#line 150
            }


            *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(part0_0));

#line 146
        }

#line 155
        uint _S17 = t0_0 + 1U;

#line 155
        if(_S17 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 156
            float tot1_0 = (*(&kernelContext_0)->sh_0)[1U][0U];

            uint oi_1 = _S17 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 159
                part0_0 = tot1_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_1)));

#line 159
            }
            else
            {

#line 159
                part0_0 = tot1_0;

#line 159
            }


            *((&kernelContext_0)->dst_0+oi_1) = ushort(f32ToBf16_0(part0_0));

#line 155
        }

#line 164
        uint _S18 = t0_0 + 2U;

#line 164
        if(_S18 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 165
            float tot2_0 = (*(&kernelContext_0)->sh_0)[2U][0U];

            uint oi_2 = _S18 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 168
                part0_0 = tot2_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_2)));

#line 168
            }
            else
            {

#line 168
                part0_0 = tot2_0;

#line 168
            }


            *((&kernelContext_0)->dst_0+oi_2) = ushort(f32ToBf16_0(part0_0));

#line 164
        }

#line 173
        uint _S19 = t0_0 + 3U;

#line 173
        if(_S19 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 174
            float tot3_0 = (*(&kernelContext_0)->sh_0)[3U][0U];

            uint oi_3 = _S19 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 177
                part0_0 = tot3_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_3)));

#line 177
            }
            else
            {

#line 177
                part0_0 = tot3_0;

#line 177
            }


            *((&kernelContext_0)->dst_0+oi_3) = ushort(f32ToBf16_0(part0_0));

#line 173
        }

#line 145
    }

#line 183
    return;
}
