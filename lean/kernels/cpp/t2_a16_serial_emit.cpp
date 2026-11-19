#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_a16_serial.slang"
struct T2Params_0
{
    uint32_t k_0;
    uint32_t groups_0;
    uint32_t tokens_0;
    uint32_t rowBegin_0;
    uint32_t rows_0;
    uint32_t outLd_0;
    uint32_t outOffset_0;
    uint32_t accumulate_0;
};


#line 108
struct GlobalParams_0
{
    T2Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
};


#line 108
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 35
static float f16ToF32_0(uint32_t h_0)
{

#line 36
    uint32_t sg_0 = (h_0 & 32768U) << 16U;
    uint32_t e_0 = (h_0 >> 10U) & 31U;
    uint32_t m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 40
        if(m_0 == 0U)
        {

#line 41
            return (U32_asfloat((sg_0)));
        }

        return (U32_asfloat(((F32_asuint((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 47
        return (U32_asfloat(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (U32_asfloat(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}

static int32_t t2Decode_0(uint32_t code_0)
{

#line 53
    uint32_t mg_0 = code_0 & 1U;

    return int32_t(mg_0) - int(2) * int32_t(mg_0 & ((code_0 >> 1U) & 1U));
}


#line 23
static float bf16ToF32_0(uint32_t h_1)
{

#line 24
    return (U32_asfloat((h_1 << 16U)));
}

static uint32_t f32ToBf16_0(float x_0)
{

#line 28
    uint32_t w_0 = (F32_asuint((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 30
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


#line 59
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 59
    float part0_0;

#line 59
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 59
    KernelContext_0 kernelContext_0;

#line 59
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t idx_0 = (_S2->groupID * Vector<uint32_t, 3> (64U, 1U, 1U) + _S2->groupThreadID).x;
    if(idx_0 >= ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rows_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
    {

#line 62
        return;
    }
    uint32_t loc_0 = idx_0 % (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rows_0;
    uint32_t t_0 = idx_0 / (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rows_0;
    uint32_t _S3 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rowBegin_0 + loc_0;
    FixedArray<float, 32>  lane_0;

#line 67
    uint32_t l_0 = 0U;
    for(;;)
    {

#line 68
        if(l_0 < 32U)
        {
        }
        else
        {

#line 68
            break;
        }

#line 69
        lane_0[l_0] = 0.0f;

#line 68
        l_0 = l_0 + 1U;

#line 68
    }

#line 68
    uint32_t g_0 = 0U;


    for(;;)
    {

#line 71
        if(g_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0))
        {
        }
        else
        {

#line 71
            break;
        }

#line 72
        uint32_t gi_0 = _S3 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0;
        float _S4 = f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->scales_0.Load(gi_0)));

#line 73
        l_0 = 0U;
        for(;;)
        {

#line 74
            if(l_0 < 32U)
            {
            }
            else
            {

#line 74
                break;
            }

#line 75
            uint32_t _S5 = uint32_t((&kernelContext_0)->globalParams_0->qdata_0.Load(gi_0 * 32U + l_0));
            uint32_t _S6 = g_0 * 128U + l_0 * 4U;

#line 76
            part0_0 = 0.0f;

#line 76
            uint32_t j_0 = 0U;

            for(;;)
            {

#line 78
                if(j_0 < 4U)
                {
                }
                else
                {

#line 78
                    break;
                }

#line 79
                uint32_t _S7 = _S6 + j_0;

#line 79
                if(_S7 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0))
                {

#line 80
                    float wf_0 = float(t2Decode_0((_S5 >> (j_0 * 2U)) & 3U));

#line 80
                    float part0_1;
                    if(t_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                    {

#line 81
                        part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S7)));

#line 81
                    }
                    else
                    {

#line 81
                        part0_1 = part0_0;

#line 81
                    }

#line 81
                    part0_0 = part0_1;

#line 79
                }

#line 78
                j_0 = j_0 + 1U;

#line 78
            }

#line 86
            precise float pr0_0 = part0_0 * _S4;
            lane_0[l_0] = lane_0[l_0] + pr0_0;

#line 74
            l_0 = l_0 + 1U;

#line 74
        }

#line 71
        g_0 = g_0 + 1U;

#line 71
    }

#line 71
    l_0 = 0U;

#line 90
    for(;;)
    {

#line 90
        if(l_0 < 16U)
        {
        }
        else
        {

#line 90
            break;
        }

#line 91
        lane_0[l_0] = lane_0[l_0] + lane_0[l_0 + 16U];

#line 90
        l_0 = l_0 + 1U;

#line 90
    }

#line 90
    l_0 = 0U;


    for(;;)
    {

#line 93
        if(l_0 < 8U)
        {
        }
        else
        {

#line 93
            break;
        }

#line 94
        lane_0[l_0] = lane_0[l_0] + lane_0[l_0 + 8U];

#line 93
        l_0 = l_0 + 1U;

#line 93
    }

#line 93
    l_0 = 0U;


    for(;;)
    {

#line 96
        if(l_0 < 4U)
        {
        }
        else
        {

#line 96
            break;
        }

#line 97
        lane_0[l_0] = lane_0[l_0] + lane_0[l_0 + 4U];

#line 96
        l_0 = l_0 + 1U;

#line 96
    }

#line 96
    l_0 = 0U;


    for(;;)
    {

#line 99
        if(l_0 < 2U)
        {
        }
        else
        {

#line 99
            break;
        }

#line 100
        lane_0[l_0] = lane_0[l_0] + lane_0[l_0 + 2U];

#line 99
        l_0 = l_0 + 1U;

#line 99
    }

#line 99
    l_0 = 0U;


    for(;;)
    {

#line 102
        if(l_0 < 1U)
        {
        }
        else
        {

#line 102
            break;
        }

#line 103
        uint32_t _S8 = l_0 + 1U;

#line 103
        lane_0[l_0] = lane_0[l_0] + lane_0[_S8];

#line 102
        l_0 = _S8;

#line 102
    }


    float tot_0 = lane_0[0U];
    uint32_t oi_0 = t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + loc_0);
    if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
    {

#line 107
        part0_0 = tot_0 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0])));

#line 107
    }
    else
    {

#line 107
        part0_0 = tot_0;

#line 107
    }


    *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0]) = uint16_t(f32ToBf16_0(part0_0));
    return;
}

// [numthreads(64, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(64, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t x = 0; x < 64; ++x)
    {
        threadInput.groupThreadID.x = x;
        _main_0(&threadInput, entryPointParams, globalParams);
    }
}
// [numthreads(64, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeVaryingInput vi = *varyingInput;
    ComputeVaryingInput groupVaryingInput = {};
    for (uint32_t z = vi.startGroupID.z; z < vi.endGroupID.z; ++z)
    {
        groupVaryingInput.startGroupID.z = z;
        for (uint32_t y = vi.startGroupID.y; y < vi.endGroupID.y; ++y)
        {
            groupVaryingInput.startGroupID.y = y;
            for (uint32_t x = vi.startGroupID.x; x < vi.endGroupID.x; ++x)
            {
                groupVaryingInput.startGroupID.x = x;
                main_0_Group(&groupVaryingInput, entryPointParams, globalParams);
            }
        }
    }
}
