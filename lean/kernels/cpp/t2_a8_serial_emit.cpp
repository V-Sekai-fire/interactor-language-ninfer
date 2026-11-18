#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_a8_serial.slang"
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


#line 89
struct GlobalParams_0
{
    T2Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<int8_t> codes_0;
    StructuredBuffer<uint16_t> xscales_0;
    RWStructuredBuffer<uint16_t> dst_0;
};


#line 89
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 54
static int32_t t2Decode_0(uint32_t code_0)
{

#line 55
    uint32_t mg_0 = code_0 & 1U;

    return int32_t(mg_0) - int(2) * int32_t(mg_0 & ((code_0 >> 1U) & 1U));
}


#line 37
static float f16ToF32_0(uint32_t h_0)
{

#line 38
    uint32_t sg_0 = (h_0 & 32768U) << 16U;
    uint32_t e_0 = (h_0 >> 10U) & 31U;
    uint32_t m_0 = h_0 & 1023U;
    if(e_0 == 0U)
    {

#line 42
        if(m_0 == 0U)
        {

#line 43
            return (U32_asfloat((sg_0)));
        }

        return (U32_asfloat(((F32_asuint((float(m_0) * 5.9604644775390625e-08f))) | sg_0)));
    }
    if(e_0 == 31U)
    {

#line 49
        return (U32_asfloat(((sg_0 | 2139095040U) | (m_0 << 13U))));
    }
    return (U32_asfloat(((sg_0 | ((e_0 + 112U) << 23U)) | (m_0 << 13U))));
}


#line 25
static float bf16ToF32_0(uint32_t h_1)
{

#line 26
    return (U32_asfloat((h_1 << 16U)));
}

static uint32_t f32ToBf16_0(float x_0)
{

#line 30
    uint32_t w_0 = (F32_asuint((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 32
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
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
    uint32_t idx_0 = (_S2->groupID * Vector<uint32_t, 3> (64U, 1U, 1U) + _S2->groupThreadID).x;
    if(idx_0 >= ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rows_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
    {

#line 64
        return;
    }
    uint32_t loc_0 = idx_0 % (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rows_0;
    uint32_t t_0 = idx_0 / (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rows_0;
    uint32_t _S3 = (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->rowBegin_0 + loc_0;

#line 68
    float tot_0 = 0.0f;

#line 68
    uint32_t g_0 = 0U;

    for(;;)
    {

#line 70
        if(g_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0))
        {
        }
        else
        {

#line 70
            break;
        }

#line 71
        uint32_t gi_0 = _S3 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0;
        uint32_t _S4 = gi_0 * 32U;
        uint32_t _S5 = t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 * 128U + g_0 * 128U;

#line 73
        int32_t dsum_0 = int(0);

#line 73
        uint32_t b_0 = 0U;

        for(;;)
        {

#line 75
            if(b_0 < 32U)
            {
            }
            else
            {

#line 75
                break;
            }

#line 76
            uint32_t _S6 = uint32_t((&kernelContext_0)->globalParams_0->qdata_0.Load(_S4 + b_0));

#line 76
            uint32_t j_0 = 0U;
            for(;;)
            {

#line 77
                if(j_0 < 4U)
                {
                }
                else
                {

#line 77
                    break;
                }

#line 78
                int32_t _S7 = dsum_0 + t2Decode_0((_S6 >> (j_0 * 2U)) & 3U) * int32_t((&kernelContext_0)->globalParams_0->codes_0.Load(_S5 + b_0 * 4U + j_0));

#line 77
                uint32_t j_1 = j_0 + 1U;

#line 77
                dsum_0 = _S7;

#line 77
                j_0 = j_1;

#line 77
            }

#line 75
            b_0 = b_0 + 1U;

#line 75
        }

#line 83
        precise float p1_0 = float(dsum_0) * f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->scales_0.Load(gi_0)));
        precise float term_0 = p1_0 * f16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->xscales_0.Load(t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0)));
        float _S8 = tot_0 + term_0;

#line 70
        uint32_t g_1 = g_0 + 1U;

#line 70
        tot_0 = _S8;

#line 70
        g_0 = g_1;

#line 70
    }

#line 87
    uint32_t oi_0 = t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + loc_0);
    if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
    {

#line 88
        tot_0 = tot_0 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0])));

#line 88
    }


    *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0]) = uint16_t(f32ToBf16_0(tot_0));
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
