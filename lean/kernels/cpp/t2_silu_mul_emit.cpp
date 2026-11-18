#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_silu_mul.slang"
struct SiluParams_0
{
    uint32_t inter_0;
    uint32_t total_0;
};


#line 36
struct GlobalParams_0
{
    SiluParams_0* params_0;
    StructuredBuffer<uint16_t> plane_0;
    RWStructuredBuffer<uint16_t> dst_0;
};


#line 36
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 13
static float bf16ToF32_0(uint32_t h_0)
{

#line 14
    return (U32_asfloat((h_0 << 16U)));
}

static uint32_t f32ToBf16_0(float x_0)
{

#line 18
    uint32_t w_0 = (F32_asuint((x_0)));
    if((w_0 & 2147483647U) > 2139095040U)
    {

#line 20
        return (w_0 >> 16U) | 64U;
    }
    return (w_0 + 32767U + ((w_0 >> 16U) & 1U)) >> 16U;
}


void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 26
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 26
    KernelContext_0 kernelContext_0;

#line 26
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t idx_0 = (_S2->groupID * Vector<uint32_t, 3> (256U, 1U, 1U) + _S2->groupThreadID).x;
    if(idx_0 >= ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->total_0))
    {

#line 29
        return;
    }
    uint32_t t_0 = idx_0 / (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->inter_0;
    uint32_t r_0 = idx_0 % (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->inter_0;
    float gt_0 = bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->plane_0.Load(t_0 * (2U * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->inter_0) + r_0)));


    *(&((&kernelContext_0)->globalParams_0->dst_0)[idx_0]) = uint16_t(f32ToBf16_0(gt_0 / (1.0f + (F32_exp((- gt_0)))) * bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->plane_0.Load(t_0 * (2U * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->inter_0) + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->inter_0 + r_0)))));
    return;
}

// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t x = 0; x < 256; ++x)
    {
        threadInput.groupThreadID.x = x;
        _main_0(&threadInput, entryPointParams, globalParams);
    }
}
// [numthreads(256, 1, 1)]
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
