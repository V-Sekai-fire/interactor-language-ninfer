#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_a16_wide_p6_finish.slang"
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


#line 71
struct GlobalParams_0
{
    T2Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> sh_0;
};


#line 71
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 27
static float bf16ToF32_0(uint32_t h_0)
{

#line 28
    return (U32_asfloat((h_0 << 16U)));
}

static uint32_t f32ToBf16_0(float x_0)
{

#line 32
    uint32_t w_0 = (F32_asuint((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 34
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


#line 63
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 63
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 63
    KernelContext_0 kernelContext_0;

#line 63
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t _S3 = _S2->groupID.x;

#line 64
    uint32_t _S4 = _S2->groupID.y;

#line 64
    uint32_t groupLinear_0 = _S3 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridX_0 * (_S4 + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridY_0 * _S2->groupID.z);

    uint32_t t0_0 = _S4 * 4U;


    if((_S2->groupThreadID.x) == 0U)
    {

#line 69
        float tot3_0;
        if(t0_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
        {

#line 71
            float tot0_0 = *(&((&kernelContext_0)->globalParams_0->sh_0)[groupLinear_0 * 128U]);

            uint32_t oi_0 = t0_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + _S3);
            if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
            {

#line 74
                tot3_0 = tot0_0 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0])));

#line 74
            }
            else
            {

#line 74
                tot3_0 = tot0_0;

#line 74
            }


            *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0]) = uint16_t(f32ToBf16_0(tot3_0));

#line 70
        }

#line 79
        uint32_t _S5 = t0_0 + 1U;

#line 79
        if(_S5 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
        {

#line 80
            float tot1_0 = *(&((&kernelContext_0)->globalParams_0->sh_0)[groupLinear_0 * 128U + 32U]);

            uint32_t oi_1 = _S5 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + _S3);
            if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
            {

#line 83
                tot3_0 = tot1_0 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_1])));

#line 83
            }
            else
            {

#line 83
                tot3_0 = tot1_0;

#line 83
            }


            *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_1]) = uint16_t(f32ToBf16_0(tot3_0));

#line 79
        }

#line 88
        uint32_t _S6 = t0_0 + 2U;

#line 88
        if(_S6 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
        {

#line 89
            float tot2_0 = *(&((&kernelContext_0)->globalParams_0->sh_0)[groupLinear_0 * 128U + 64U]);

            uint32_t oi_2 = _S6 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + _S3);
            if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
            {

#line 92
                tot3_0 = tot2_0 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_2])));

#line 92
            }
            else
            {

#line 92
                tot3_0 = tot2_0;

#line 92
            }


            *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_2]) = uint16_t(f32ToBf16_0(tot3_0));

#line 88
        }

#line 97
        uint32_t _S7 = t0_0 + 3U;

#line 97
        if(_S7 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
        {

#line 98
            float tot3_1 = *(&((&kernelContext_0)->globalParams_0->sh_0)[groupLinear_0 * 128U + 96U]);

            uint32_t oi_3 = _S7 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + _S3);
            if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
            {

#line 101
                tot3_0 = tot3_1 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_3])));

#line 101
            }
            else
            {

#line 101
                tot3_0 = tot3_1;

#line 101
            }


            *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_3]) = uint16_t(f32ToBf16_0(tot3_0));

#line 97
        }

#line 69
    }

#line 107
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
