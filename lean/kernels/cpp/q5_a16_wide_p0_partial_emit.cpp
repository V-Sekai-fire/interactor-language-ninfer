#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/q5_a16_wide_p0_partial.slang"
struct Q5Params_0
{
    uint32_t k_0;
    uint32_t groups_0;
    uint32_t tokens_0;
    uint32_t rowBegin_0;
    uint32_t rows_0;
    uint32_t outLd_0;
    uint32_t outOffset_0;
    uint32_t accumulate_0;
    uint32_t gridX_0;
    uint32_t gridY_0;
};


#line 111
struct GlobalParams_0
{
    Q5Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint8_t> qhigh_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> sh_0;
};


#line 111
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 41
static float f16ToF32_0(uint32_t h_0)
{

#line 42
    uint32_t sg_0 = (h_0 & 32768U) << 16U;
    uint32_t e_0 = (h_0 >> 10U) & 31U;
    uint32_t m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 46
        if(m_0 == 0U)
        {

#line 47
            return (U32_asfloat((sg_0)));
        }

        return (U32_asfloat(((F32_asuint((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 53
        return (U32_asfloat(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (U32_asfloat(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}

static int32_t gDecode_0(uint32_t byte_0, uint32_t hbyte_0, uint32_t lane_0, uint32_t j_0)
{
    uint32_t cu_0 = ((byte_0 >> (j_0 * 4U)) & 15U) | (((hbyte_0 >> ((lane_0 & 3U) * 2U + j_0)) & 1U) << 4U);
    return int32_t(cu_0) - int(32) * int32_t((cu_0 >> 4U) & 1U);
}


#line 29
static float bf16ToF32_0(uint32_t h_1)
{

#line 30
    return (U32_asfloat((h_1 << 16U)));
}


#line 65
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 65
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 65
    KernelContext_0 kernelContext_0;

#line 65
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t _S3 = _S2->groupID.x;

#line 66
    uint32_t _S4 = _S2->groupID.y;

#line 66
    uint32_t groupLinear_0 = _S3 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridX_0 * (_S4 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridY_0 * _S2->groupID.z);

    uint32_t _S5 = _S4 * 4U;
    uint32_t lane_1 = _S2->groupThreadID.x;
    uint32_t _S6 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rowBegin_0 + _S3;

#line 70
    float acc0_0 = 0.0f;

#line 70
    float acc1_0 = 0.0f;

#line 70
    float acc2_0 = 0.0f;

#line 70
    float acc3_0 = 0.0f;

#line 70
    uint32_t g_0 = 0U;

#line 75
    for(;;)
    {

#line 75
        if(g_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0))
        {
        }
        else
        {

#line 75
            break;
        }

#line 76
        uint32_t gi_0 = _S6 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0;
        float scale_0 = f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->scales_0.Load(gi_0)));
        uint32_t _S7 = uint32_t((&kernelContext_0)->globalParams_0->qdata_0.Load(gi_0 * 32U + lane_1));
        uint32_t _S8 = uint32_t((&kernelContext_0)->globalParams_0->qhigh_0.Load(gi_0 * 8U + lane_1 / 4U));
        uint32_t _S9 = g_0 * 64U + lane_1 * 2U;

#line 80
        float part0_0 = 0.0f;

#line 80
        float part1_0 = 0.0f;

#line 80
        float part2_0 = 0.0f;

#line 80
        float part3_0 = 0.0f;

#line 80
        uint32_t j_1 = 0U;

#line 85
        for(;;)
        {

#line 85
            if(j_1 < 2U)
            {
            }
            else
            {

#line 85
                break;
            }

#line 86
            uint32_t _S10 = _S9 + j_1;

#line 86
            if(_S10 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0))
            {

#line 87
                float wf_0 = float(gDecode_0(_S7, _S8, lane_1, j_1));

#line 87
                float part0_1;
                if(_S5 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                {

#line 88
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(_S5 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S10)));

#line 88
                }
                else
                {

#line 88
                    part0_1 = part0_0;

#line 88
                }


                uint32_t _S11 = _S5 + 1U;

#line 91
                float part1_1;

#line 91
                if(_S11 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                {

#line 91
                    part1_1 = part1_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(_S11 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S10)));

#line 91
                }
                else
                {

#line 91
                    part1_1 = part1_0;

#line 91
                }


                uint32_t _S12 = _S5 + 2U;

#line 94
                float part2_1;

#line 94
                if(_S12 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                {

#line 94
                    part2_1 = part2_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(_S12 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S10)));

#line 94
                }
                else
                {

#line 94
                    part2_1 = part2_0;

#line 94
                }


                uint32_t _S13 = _S5 + 3U;

#line 97
                float part3_1;

#line 97
                if(_S13 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                {

#line 97
                    part3_1 = part3_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(_S13 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S10)));

#line 97
                }
                else
                {

#line 97
                    part3_1 = part3_0;

#line 97
                }

#line 97
                part0_0 = part0_1;

#line 97
                part1_0 = part1_1;

#line 97
                part2_0 = part2_1;

#line 97
                part3_0 = part3_1;

#line 86
            }

#line 85
            j_1 = j_1 + 1U;

#line 85
        }

#line 102
        precise float pr0_0 = part0_0 * scale_0;
        float _S14 = acc0_0 + pr0_0;
        precise float pr1_0 = part1_0 * scale_0;
        float _S15 = acc1_0 + pr1_0;
        precise float pr2_0 = part2_0 * scale_0;
        float _S16 = acc2_0 + pr2_0;
        precise float pr3_0 = part3_0 * scale_0;
        float _S17 = acc3_0 + pr3_0;

#line 75
        uint32_t g_1 = g_0 + 1U;

#line 75
        acc0_0 = _S14;

#line 75
        acc1_0 = _S15;

#line 75
        acc2_0 = _S16;

#line 75
        acc3_0 = _S17;

#line 75
        g_0 = g_1;

#line 75
    }

#line 111
    uint32_t _S18 = groupLinear_0 * 128U;

#line 111
    *(&((&kernelContext_0)->globalParams_0->sh_0)[_S18 + lane_1]) = acc0_0;
    *(&((&kernelContext_0)->globalParams_0->sh_0)[_S18 + (32U + lane_1)]) = acc1_0;
    *(&((&kernelContext_0)->globalParams_0->sh_0)[_S18 + (64U + lane_1)]) = acc2_0;
    *(&((&kernelContext_0)->globalParams_0->sh_0)[_S18 + (96U + lane_1)]) = acc3_0;
    return;
}

// [numthreads(32, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(32, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t x = 0; x < 32; ++x)
    {
        threadInput.groupThreadID.x = x;
        _main_0(&threadInput, entryPointParams, globalParams);
    }
}
// [numthreads(32, 1, 1)]
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
