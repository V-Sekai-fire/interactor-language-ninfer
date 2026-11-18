#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_a8_lanes_p0_spread.slang"
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
    uint32_t gridX_0;
    uint32_t gridY_0;
};


#line 88
struct GlobalParams_0
{
    T2Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<int8_t> codes_0;
    StructuredBuffer<uint16_t> xscales_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> pg_0;
};


#line 88
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 58
static int32_t t2Decode_0(uint32_t code_0)
{

#line 59
    uint32_t mg_0 = code_0 & 1U;

    return int32_t(mg_0) - int(2) * int32_t(mg_0 & ((code_0 >> 1U) & 1U));
}


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
    uint32_t _S5 = _S3 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridX_0 * (_S4 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridY_0 * _S2->groupID.z);


    uint32_t _S6 = _S2->groupThreadID.x;
    uint32_t _S7 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rowBegin_0 + _S3;

#line 70
    uint32_t rd_0 = 0U;
    for(;;)
    {

#line 71
        if(rd_0 < 8U)
        {
        }
        else
        {

#line 71
            break;
        }

#line 72
        uint32_t g_0 = rd_0 * 32U + _S6;
        if(g_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0))
        {

#line 74
            uint32_t gi_0 = _S7 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0;
            uint32_t _S8 = gi_0 * 32U;
            uint32_t _S9 = _S4 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 * 128U + g_0 * 128U;

#line 76
            int32_t dsum_0 = int(0);

#line 76
            uint32_t b_0 = 0U;

            for(;;)
            {

#line 78
                if(b_0 < 32U)
                {
                }
                else
                {

#line 78
                    break;
                }

#line 79
                uint32_t _S10 = uint32_t((&kernelContext_0)->globalParams_0->qdata_0.Load(_S8 + b_0));

#line 79
                uint32_t j_0 = 0U;
                for(;;)
                {

#line 80
                    if(j_0 < 4U)
                    {
                    }
                    else
                    {

#line 80
                        break;
                    }

#line 81
                    int32_t _S11 = dsum_0 + t2Decode_0((_S10 >> (j_0 * 2U)) & 3U) * int32_t((&kernelContext_0)->globalParams_0->codes_0.Load(_S9 + b_0 * 4U + j_0));

#line 80
                    uint32_t j_1 = j_0 + 1U;

#line 80
                    dsum_0 = _S11;

#line 80
                    j_0 = j_1;

#line 80
                }

#line 78
                b_0 = b_0 + 1U;

#line 78
            }

#line 86
            precise float p1_0 = float(dsum_0) * f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->scales_0.Load(gi_0)));
            precise float term_0 = p1_0 * f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->xscales_0.Load(_S4 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0)));
            *(&((&kernelContext_0)->globalParams_0->pg_0)[_S5 * 256U + g_0]) = term_0;

#line 73
        }

#line 71
        rd_0 = rd_0 + 1U;

#line 71
    }

#line 91
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
