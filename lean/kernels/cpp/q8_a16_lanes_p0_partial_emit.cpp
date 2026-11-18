#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/q8_a16_lanes_p0_partial.slang"
struct Q8Params_0
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


#line 85
struct GlobalParams_0
{
    Q8Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> sh_0;
};


#line 85
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 39
static float f16ToF32_0(uint32_t h_0)
{

#line 40
    uint32_t sg_0 = (h_0 & 32768U) << 16U;
    uint32_t e_0 = (h_0 >> 10U) & 31U;
    uint32_t m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 44
        if(m_0 == 0U)
        {

#line 45
            return (U32_asfloat((sg_0)));
        }

        return (U32_asfloat(((F32_asuint((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 51
        return (U32_asfloat(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (U32_asfloat(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}

static int32_t gDecode_0(uint32_t byte_0, uint32_t hbyte_0, uint32_t lane_0, uint32_t j_0)
{

#line 57
    return int32_t(byte_0) - int(256) * int32_t((byte_0 >> 7U) & 1U);
}


#line 27
static float bf16ToF32_0(uint32_t h_1)
{

#line 28
    return (U32_asfloat((h_1 << 16U)));
}


#line 61
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 61
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 61
    KernelContext_0 kernelContext_0;

#line 61
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t _S3 = _S2->groupID.x;

#line 62
    uint32_t _S4 = _S2->groupID.y;

#line 62
    uint32_t groupLinear_0 = _S3 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridX_0 * (_S4 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridY_0 * _S2->groupID.z);


    uint32_t lane_1 = _S2->groupThreadID.x;
    uint32_t _S5 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rowBegin_0 + _S3;

#line 66
    float acc0_0 = 0.0f;

#line 66
    uint32_t g_0 = 0U;

    for(;;)
    {

#line 68
        if(g_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0))
        {
        }
        else
        {

#line 68
            break;
        }

#line 69
        uint32_t gi_0 = _S5 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0;
        float scale_0 = f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->scales_0.Load(gi_0)));
        uint32_t _S6 = uint32_t((&kernelContext_0)->globalParams_0->qdata_0.Load(gi_0 * 32U + lane_1));
        uint32_t _S7 = g_0 * 32U + lane_1;

#line 72
        float part0_0 = 0.0f;

#line 72
        uint32_t j_1 = 0U;

        for(;;)
        {

#line 74
            if(j_1 < 1U)
            {
            }
            else
            {

#line 74
                break;
            }

#line 75
            uint32_t _S8 = _S7 + j_1;

#line 75
            if(_S8 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0))
            {

#line 76
                float wf_0 = float(gDecode_0(_S6, 0U, lane_1, j_1));

#line 76
                float part0_1;
                if(_S4 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
                {

#line 77
                    part0_1 = part0_0 + wf_0 * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(_S4 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + _S8)));

#line 77
                }
                else
                {

#line 77
                    part0_1 = part0_0;

#line 77
                }

#line 77
                part0_0 = part0_1;

#line 75
            }

#line 74
            j_1 = j_1 + 1U;

#line 74
        }

#line 82
        precise float pr0_0 = part0_0 * scale_0;
        float _S9 = acc0_0 + pr0_0;

#line 68
        uint32_t g_1 = g_0 + 1U;

#line 68
        acc0_0 = _S9;

#line 68
        g_0 = g_1;

#line 68
    }

#line 85
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
