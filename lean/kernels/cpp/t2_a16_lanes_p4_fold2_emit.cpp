#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_a16_lanes_p4_fold2.slang"
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


#line 70
struct GlobalParams_0
{
    T2Params_0* params_0;
    StructuredBuffer<uint8_t> qdata_0;
    StructuredBuffer<uint16_t> scales_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<uint16_t> dst_0;
    RWStructuredBuffer<float> sh_0;
};


#line 70
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 63
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 63
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 63
    KernelContext_0 kernelContext_0;

#line 63
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t groupLinear_0 = _S2->groupID.x + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridX_0 * (_S2->groupID.y + (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->gridY_0 * _S2->groupID.z);


    uint32_t lane_0 = _S2->groupThreadID.x;

    if(lane_0 < 2U)
    {

#line 70
        uint32_t _S3 = groupLinear_0 * 32U;

#line 70
        uint32_t _S4 = _S3 + lane_0;

#line 70
        *(&((&kernelContext_0)->globalParams_0->sh_0)[_S4]) = *(&((&kernelContext_0)->globalParams_0->sh_0)[_S4]) + *(&((&kernelContext_0)->globalParams_0->sh_0)[_S3 + (lane_0 + 2U)]);

#line 69
    }


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
