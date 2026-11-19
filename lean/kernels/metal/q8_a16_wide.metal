#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 37 "slang/q8_a16_wide.slang"
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

#line 55
    return int(byte_0) - int(256) * int((byte_0 >> 7U) & 1U);
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
struct Q8Params_0
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


#line 103
struct KernelContext_0
{
    Q8Params_0 constant* params_0;
    ushort device* scales_0;
    uchar device* qdata_0;
    ushort device* act_0;
    ushort device* dst_0;
    array<array<float, int(32)>, int(4)> threadgroup* sh_0;
};


#line 59
[[kernel]] void main_0(uint3 gid_0 [[threadgroup_position_in_grid]], uint3 gtid_0 [[thread_position_in_threadgroup]], Q8Params_0 constant* params_1 [[buffer(0)]], ushort device* scales_1 [[buffer(2)]], uchar device* qdata_1 [[buffer(1)]], ushort device* act_1 [[buffer(3)]], ushort device* dst_1 [[buffer(4)]])
{

#line 59
    float part0_0;

#line 59
    thread KernelContext_0 kernelContext_0;

#line 59
    (&kernelContext_0)->params_0 = params_1;

#line 59
    (&kernelContext_0)->scales_0 = scales_1;

#line 59
    (&kernelContext_0)->qdata_0 = qdata_1;

#line 59
    (&kernelContext_0)->act_0 = act_1;

#line 59
    (&kernelContext_0)->dst_0 = dst_1;

#line 59
    threadgroup array<array<float, int(32)>, int(4)> sh_1;

#line 59
    (&kernelContext_0)->sh_0 = &sh_1;
    uint loc_0 = gid_0.x;
    uint t0_0 = gid_0.y * 4U;
    uint lane_1 = gtid_0.x;
    uint _S1 = params_1->rowBegin_0 + loc_0;

#line 63
    float acc0_0 = 0.0f;

#line 63
    float acc1_0 = 0.0f;

#line 63
    float acc2_0 = 0.0f;

#line 63
    float acc3_0 = 0.0f;

#line 63
    uint g_0 = 0U;

#line 68
    for(;;)
    {

#line 68
        if(g_0 < ((&kernelContext_0)->params_0->groups_0))
        {
        }
        else
        {

#line 68
            break;
        }

#line 69
        uint gi_0 = _S1 * (&kernelContext_0)->params_0->groups_0 + g_0;
        float scale_0 = f16ToF32_0(uint((&kernelContext_0)->scales_0[gi_0]));
        uint _S2 = uint((&kernelContext_0)->qdata_0[gi_0 * 32U + lane_1]);
        uint _S3 = g_0 * 32U + lane_1;

#line 72
        part0_0 = 0.0f;

#line 72
        float part1_0 = 0.0f;

#line 72
        float part2_0 = 0.0f;

#line 72
        float part3_0 = 0.0f;

#line 72
        uint j_1 = 0U;

#line 77
        for(;;)
        {

#line 77
            if(j_1 < 1U)
            {
            }
            else
            {

#line 77
                break;
            }

#line 78
            uint _S4 = _S3 + j_1;

#line 78
            if(_S4 < ((&kernelContext_0)->params_0->k_0))
            {

#line 79
                float wf_0 = float(gDecode_0(_S2, 0U, lane_1, j_1));

#line 79
                float part0_1;
                if(t0_0 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 80
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[t0_0 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 80
                }
                else
                {

#line 80
                    part0_1 = part0_0;

#line 80
                }


                uint _S5 = t0_0 + 1U;

#line 83
                float part1_1;

#line 83
                if(_S5 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 83
                    part1_1 = part1_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S5 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 83
                }
                else
                {

#line 83
                    part1_1 = part1_0;

#line 83
                }


                uint _S6 = t0_0 + 2U;

#line 86
                float part2_1;

#line 86
                if(_S6 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 86
                    part2_1 = part2_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S6 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 86
                }
                else
                {

#line 86
                    part2_1 = part2_0;

#line 86
                }


                uint _S7 = t0_0 + 3U;

#line 89
                float part3_1;

#line 89
                if(_S7 < ((&kernelContext_0)->params_0->tokens_0))
                {

#line 89
                    part3_1 = part3_0 + wf_0 * bf16ToF32_0(uint((&kernelContext_0)->act_0[_S7 * (&kernelContext_0)->params_0->k_0 + _S4]));

#line 89
                }
                else
                {

#line 89
                    part3_1 = part3_0;

#line 89
                }

#line 89
                part0_0 = part0_1;

#line 89
                part1_0 = part1_1;

#line 89
                part2_0 = part2_1;

#line 89
                part3_0 = part3_1;

#line 78
            }

#line 77
            j_1 = j_1 + 1U;

#line 77
        }

#line 94
        float pr0_0 = part0_0 * scale_0;
        float _S8 = acc0_0 + pr0_0;
        float pr1_0 = part1_0 * scale_0;
        float _S9 = acc1_0 + pr1_0;
        float pr2_0 = part2_0 * scale_0;
        float _S10 = acc2_0 + pr2_0;
        float pr3_0 = part3_0 * scale_0;
        float _S11 = acc3_0 + pr3_0;

#line 68
        uint g_1 = g_0 + 1U;

#line 68
        acc0_0 = _S8;

#line 68
        acc1_0 = _S9;

#line 68
        acc2_0 = _S10;

#line 68
        acc3_0 = _S11;

#line 68
        g_0 = g_1;

#line 68
    }

#line 103
    (*(&kernelContext_0)->sh_0)[0U][lane_1] = acc0_0;
    (*(&kernelContext_0)->sh_0)[1U][lane_1] = acc1_0;
    (*(&kernelContext_0)->sh_0)[2U][lane_1] = acc2_0;
    (*(&kernelContext_0)->sh_0)[3U][lane_1] = acc3_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 16U)
    {

#line 109
        uint _S12 = lane_1 + 16U;

#line 109
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S12];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S12];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S12];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S12];

#line 108
    }

#line 114
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 8U)
    {

#line 116
        uint _S13 = lane_1 + 8U;

#line 116
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S13];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S13];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S13];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S13];

#line 115
    }

#line 121
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 4U)
    {

#line 123
        uint _S14 = lane_1 + 4U;

#line 123
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S14];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S14];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S14];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S14];

#line 122
    }

#line 128
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 2U)
    {

#line 130
        uint _S15 = lane_1 + 2U;

#line 130
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S15];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S15];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S15];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S15];

#line 129
    }

#line 135
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 < 1U)
    {

#line 137
        uint _S16 = lane_1 + 1U;

#line 137
        (*(&kernelContext_0)->sh_0)[0U][lane_1] = (*(&kernelContext_0)->sh_0)[0U][lane_1] + (*(&kernelContext_0)->sh_0)[0U][_S16];
        (*(&kernelContext_0)->sh_0)[1U][lane_1] = (*(&kernelContext_0)->sh_0)[1U][lane_1] + (*(&kernelContext_0)->sh_0)[1U][_S16];
        (*(&kernelContext_0)->sh_0)[2U][lane_1] = (*(&kernelContext_0)->sh_0)[2U][lane_1] + (*(&kernelContext_0)->sh_0)[2U][_S16];
        (*(&kernelContext_0)->sh_0)[3U][lane_1] = (*(&kernelContext_0)->sh_0)[3U][lane_1] + (*(&kernelContext_0)->sh_0)[3U][_S16];

#line 136
    }

#line 142
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane_1 == 0U)
    {

#line 144
        if(t0_0 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 145
            float tot0_0 = (*(&kernelContext_0)->sh_0)[0U][0U];

            uint oi_0 = t0_0 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 148
                part0_0 = tot0_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_0)));

#line 148
            }
            else
            {

#line 148
                part0_0 = tot0_0;

#line 148
            }


            *((&kernelContext_0)->dst_0+oi_0) = ushort(f32ToBf16_0(part0_0));

#line 144
        }

#line 153
        uint _S17 = t0_0 + 1U;

#line 153
        if(_S17 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 154
            float tot1_0 = (*(&kernelContext_0)->sh_0)[1U][0U];

            uint oi_1 = _S17 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 157
                part0_0 = tot1_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_1)));

#line 157
            }
            else
            {

#line 157
                part0_0 = tot1_0;

#line 157
            }


            *((&kernelContext_0)->dst_0+oi_1) = ushort(f32ToBf16_0(part0_0));

#line 153
        }

#line 162
        uint _S18 = t0_0 + 2U;

#line 162
        if(_S18 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 163
            float tot2_0 = (*(&kernelContext_0)->sh_0)[2U][0U];

            uint oi_2 = _S18 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 166
                part0_0 = tot2_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_2)));

#line 166
            }
            else
            {

#line 166
                part0_0 = tot2_0;

#line 166
            }


            *((&kernelContext_0)->dst_0+oi_2) = ushort(f32ToBf16_0(part0_0));

#line 162
        }

#line 171
        uint _S19 = t0_0 + 3U;

#line 171
        if(_S19 < ((&kernelContext_0)->params_0->tokens_0))
        {

#line 172
            float tot3_0 = (*(&kernelContext_0)->sh_0)[3U][0U];

            uint oi_3 = _S19 * (&kernelContext_0)->params_0->outLd_0 + ((&kernelContext_0)->params_0->outOffset_0 + loc_0);
            if(((&kernelContext_0)->params_0->accumulate_0) != 0U)
            {

#line 175
                part0_0 = tot3_0 + bf16ToF32_0(uint(*((&kernelContext_0)->dst_0+oi_3)));

#line 175
            }
            else
            {

#line 175
                part0_0 = tot3_0;

#line 175
            }


            *((&kernelContext_0)->dst_0+oi_3) = ushort(f32ToBf16_0(part0_0));

#line 171
        }

#line 143
    }

#line 181
    return;
}
