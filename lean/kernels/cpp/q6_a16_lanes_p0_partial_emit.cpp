#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/q6_a16_lanes_p0_partial.slang"
struct Q6Params_0
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


#line 90
struct GlobalParams_0
{
    Q6Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint8_t> qhigh_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> sh_0;
};


#line 90
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
    uint32_t cu_0 = ((byte_0 >> (j_0 * 4U)) & 15U) | (((hbyte_0 >> ((lane_0 & 1U) * 4U + j_0 * 2U)) & 3U) << 4U);
    return int32_t(cu_0) - int(64) * int32_t((cu_0 >> 5U) & 1U);
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


    uint32_t lane_1 = _S2->groupThreadID.x;
    uint32_t _S5 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rowBegin_0 + _S3;

#line 70
    float acc0_0 = 0.0f;

#line 70
    uint32_t g_0 = 0U;

    for(;;)
    {

#line 72
        if(g_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0))
        {
        }
        else
        {

#line 72
            break;
        }

#line 73
        uint32_t gi_0 = _S5 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0;
        float scale_0 = f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->scales_0.Load(gi_0)));
        uint32_t _S6 = uint32_t((&kernelContext_0)->globalParams_0->qdata_0.Load(gi_0 * 32U + lane_1));
        uint32_t _S7 = uint32_t((&kernelContext_0)->globalParams_0->qhigh_0.Load(gi_0 * 16U + lane_1 / 2U));
        uint32_t _S8 = g_0 * 64U + lane_1 * 2U;

#line 77
        float part0_0 = 0.0f;

#line 77
        uint32_t j_1 = 0U;

        for(;;)
        {

#line 79
            if(j_1 < 2U)
            {
            }
            else
            {

#line 79
                break;
            }

#line 80
            uint32_t _S9 = _S8 + j_1;

#line 80
            if(_S9 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0))
            {

#line 81
                float wf_0 = float(gDecode_0(_S6, _S7, lane_1, j_1));

#line 81
                float part0_1;
                if(_S4 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                {

#line 82
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(_S4 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S9)));

#line 82
                }
                else
                {

#line 82
                    part0_1 = part0_0;

#line 82
                }

#line 82
                part0_0 = part0_1;

#line 80
            }

#line 79
            j_1 = j_1 + 1U;

#line 79
        }

#line 87
        precise float pr0_0 = part0_0 * scale_0;
        float _S10 = acc0_0 + pr0_0;

#line 72
        uint32_t g_1 = g_0 + 1U;

#line 72
        acc0_0 = _S10;

#line 72
        g_0 = g_1;

#line 72
    }

#line 90
    *(&((&kernelContext_0)->globalParams_0->sh_0)[groupLinear_0 * 32U + lane_1]) = acc0_0;
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
