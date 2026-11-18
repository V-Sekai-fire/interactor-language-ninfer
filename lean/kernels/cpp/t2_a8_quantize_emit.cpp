#include "slang-cpp-prelude.h"
#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/t2_a8_quantize.slang"
struct QuantParams_0
{
    uint32_t k_0;
    uint32_t groups_0;
    uint32_t tokens_0;
};


#line 104
struct GlobalParams_0
{
    QuantParams_0* params_0;
    StructuredBuffer<uint16_t> act_0;
    RWStructuredBuffer<int8_t> codes_0;
    RWStructuredBuffer<uint16_t> xscales_0;
};


#line 104
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 16
static float bf16ToF32_0(uint32_t h_0)
{

#line 17
    return (U32_asfloat((h_0 << 16U)));
}


#line 37
static uint32_t f32ToF16_0(float x_0)
{

#line 38
    uint32_t w_0 = (F32_asuint((x_0)));
    uint32_t sg_0 = (w_0 >> 16U) & 32768U;
    uint32_t e_0 = (w_0 >> 23U) & 255U;
    uint32_t m_0 = w_0 & 8388607U;
    if(e_0 == 255U)
    {

#line 43
        if(m_0 == 0U)
        {

#line 44
            return sg_0 | 31744U;
        }
        return (sg_0 | 32256U) | (m_0 >> 13U);
    }
    if(e_0 >= 143U)
    {

#line 49
        return sg_0 | 31744U;
    }

#line 49
    bool _S1;

#line 49
    uint32_t h_1;

    if(e_0 >= 113U)
    {

#line 52
        uint32_t h0_0 = ((e_0 - 112U) << 10U) | (m_0 >> 13U);
        uint32_t rm_0 = m_0 & 8191U;

        if(rm_0 > 4096U)
        {

#line 55
            _S1 = true;

#line 55
        }
        else
        {

#line 55
            if(rm_0 == 4096U)
            {

#line 55
                _S1 = (h0_0 & 1U) == 1U;

#line 55
            }
            else
            {

#line 55
                _S1 = false;

#line 55
            }

#line 55
        }

#line 55
        if(_S1)
        {

#line 55
            h_1 = h0_0 + 1U;

#line 55
        }
        else
        {

#line 55
            h_1 = h0_0;

#line 55
        }


        return sg_0 | h_1;
    }
    if(e_0 < 102U)
    {

#line 61
        return sg_0;
    }
    uint32_t shf_0 = 126U - e_0;
    uint32_t sig_0 = m_0 | 8388608U;
    uint32_t q0_0 = sig_0 >> shf_0;
    uint32_t rm_1 = sig_0 & ((1U << shf_0) - 1U);
    uint32_t mid_0 = 1U << (shf_0 - 1U);

    if(rm_1 > mid_0)
    {

#line 69
        _S1 = true;

#line 69
    }
    else
    {

#line 69
        if(rm_1 == mid_0)
        {

#line 69
            _S1 = (q0_0 & 1U) == 1U;

#line 69
        }
        else
        {

#line 69
            _S1 = false;

#line 69
        }

#line 69
    }

#line 69
    if(_S1)
    {

#line 69
        h_1 = q0_0 + 1U;

#line 69
    }
    else
    {

#line 69
        h_1 = q0_0;

#line 69
    }


    return sg_0 | h_1;
}


#line 20
static float f16ToF32_0(uint32_t h_2)
{

#line 21
    uint32_t sg_1 = (h_2 & 32768U) << 16U;
    uint32_t e_1 = (h_2 >> 10U) & 31U;
    uint32_t m_1 = h_2 & 1023U;
    if(e_1 == 0U)
    {

#line 25
        if(m_1 == 0U)
        {

#line 26
            return (U32_asfloat((sg_1)));
        }

        return (U32_asfloat(((F32_asuint((float(m_1) * 5.9604644775390625e-08f))) | sg_1)));
    }
    if(e_1 == 31U)
    {

#line 32
        return (U32_asfloat(((sg_1 | 2139095040U) | (m_1 << 13U))));
    }
    return (U32_asfloat(((sg_1 | ((e_1 + 112U) << 23U)) | (m_1 << 13U))));
}


#line 75
static float roundEven_0(float x_1)
{

#line 76
    float f_0 = (F32_floor((x_1)));
    if((x_1 - f_0) == 0.5f)
    {

#line 78
        if(((F32_floor((f_0 * 0.5f))) * 2.0f) == f_0)
        {

#line 79
            return f_0;
        }
        return f_0 + 1.0f;
    }
    return (F32_round((x_1)));
}


void _main_0(void* _S2, void* entryPointParams_0, void* globalParams_1)
{

#line 87
    float xv_0;

#line 87
    ComputeThreadVaryingInput * _S3 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S2));

#line 87
    KernelContext_0 kernelContext_0;

#line 87
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t idx_0 = (_S3->groupID * Vector<uint32_t, 3> (64U, 1U, 1U) + _S3->groupThreadID).x;
    if(idx_0 >= ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->tokens_0))
    {

#line 90
        return;
    }
    uint32_t g_0 = idx_0 % (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0;
    uint32_t t_0 = idx_0 / (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0;

#line 93
    float amax_0 = 0.0f;

#line 93
    uint32_t ii_0 = 0U;

    for(;;)
    {

#line 95
        if(ii_0 < 128U)
        {
        }
        else
        {

#line 95
            break;
        }

#line 96
        uint32_t pos_0 = g_0 * 128U + ii_0;

        if(pos_0 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0))
        {

#line 98
            xv_0 = bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + pos_0)));

#line 98
        }
        else
        {

#line 98
            xv_0 = 0.0f;

#line 98
        }


        float _S4 = (F32_max((amax_0), ((F32_abs((xv_0))))));

#line 95
        uint32_t ii_1 = ii_0 + 1U;

#line 95
        amax_0 = _S4;

#line 95
        ii_0 = ii_1;

#line 95
    }

#line 103
    uint32_t sh_0 = f32ToF16_0(amax_0 / 127.0f);
    *(&((&kernelContext_0)->globalParams_0->xscales_0)[t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 + g_0]) = uint16_t(sh_0);
    float s_0 = f16ToF32_0(sh_0);

#line 105
    float inv_0;

    if(s_0 > 0.0f)
    {

#line 107
        inv_0 = 1.0f / s_0;

#line 107
    }
    else
    {

#line 107
        inv_0 = 0.0f;

#line 107
    }

#line 107
    ii_0 = 0U;


    for(;;)
    {

#line 110
        if(ii_0 < 128U)
        {
        }
        else
        {

#line 110
            break;
        }

#line 111
        uint32_t _S5 = g_0 * 128U;

#line 111
        uint32_t pos_1 = _S5 + ii_0;

        if(pos_1 < ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0))
        {

#line 113
            xv_0 = bf16ToF32_0(uint32_t((&kernelContext_0)->globalParams_0->act_0.Load(t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->k_0 + pos_1)));

#line 113
        }
        else
        {

#line 113
            xv_0 = 0.0f;

#line 113
        }

#line 118
        *(&((&kernelContext_0)->globalParams_0->codes_0)[t_0 * (slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->groups_0 * 128U + _S5 + ii_0]) = int8_t(int32_t((F32_max((-127.0f), ((F32_min((127.0f), (roundEven_0(xv_0 * inv_0)))))))));

#line 110
        ii_0 = ii_0 + 1U;

#line 110
    }

#line 120
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
