#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/q8_a16_lanes_p6_finish.slang"
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


#line 69
struct GlobalParams_0
{
    Q8Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> sh_0;
};


#line 69
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

#line 67
    if((_S2->groupThreadID.x) == 0U)
    {

#line 68
        if(_S4 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
        {

#line 69
            float tot0_0 = *(&((&kernelContext_0)->globalParams_0->sh_0)[groupLinear_0 * 32U]);

            uint32_t oi_0 = _S4 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outLd_0 + ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->outOffset_0 + _S3);

#line 71
            float tot0_1;
            if(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->accumulate_0) != 0U)
            {

#line 72
                tot0_1 = tot0_0 + bf16ToF32_0(uint32_t(*(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0])));

#line 72
            }
            else
            {

#line 72
                tot0_1 = tot0_0;

#line 72
            }


            *(&((&kernelContext_0)->globalParams_0->dst_0)[oi_0]) = uint16_t(f32ToBf16_0(tot0_1));

#line 68
        }

#line 67
    }

#line 78
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
