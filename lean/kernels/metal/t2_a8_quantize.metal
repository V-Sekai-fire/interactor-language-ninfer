#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 16 "slang/t2_a8_quantize.slang"
float bf16ToF32_0(uint h_0)
{

#line 17
    return (as_type<float>((h_0 << 16U)));
}


#line 37
uint f32ToF16_0(float x_0)
{

#line 38
    uint w_0 = (as_type<uint>((x_0)));
    uint sg_0 = (w_0 >> 16U) & 32768U;
    uint e_0 = (w_0 >> 23U) & 255U;
    uint m_0 = w_0 & 8388607U;
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
    uint h_1;

    if(e_0 >= 113U)
    {

#line 52
        uint h0_0 = ((e_0 - 112U) << 10U) | (m_0 >> 13U);
        uint rm_0 = m_0 & 8191U;

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
    uint shf_0 = 126U - e_0;
    uint sig_0 = m_0 | 8388608U;
    uint q0_0 = sig_0 >> shf_0;
    uint rm_1 = sig_0 & ((1U << shf_0) - 1U);
    uint mid_0 = 1U << (shf_0 - 1U);

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
float f16ToF32_0(uint h_2)
{

#line 21
    uint sg_1 = (h_2 & 32768U) << 16U;
    uint e_1 = (h_2 >> 10U) & 31U;
    uint m_1 = h_2 & 1023U;
    if(e_1 == 0U)
    {

#line 25
        if(m_1 == 0U)
        {

#line 26
            return (as_type<float>((sg_1)));
        }

        return (as_type<float>(((as_type<uint>((float(m_1) * 5.9604644775390625e-08f))) | sg_1)));
    }
    if(e_1 == 31U)
    {

#line 32
        return (as_type<float>(((sg_1 | 2139095040U) | (m_1 << 13U))));
    }
    return (as_type<float>(((sg_1 | ((e_1 + 112U) << 23U)) | (m_1 << 13U))));
}


#line 75
float roundEven_0(float x_1)
{

#line 76
    float f_0 = floor(x_1);
    if((x_1 - f_0) == 0.5f)
    {

#line 78
        if((floor(f_0 * 0.5f) * 2.0f) == f_0)
        {

#line 79
            return f_0;
        }
        return f_0 + 1.0f;
    }
    return round(x_1);
}


#line 1
struct QuantParams_0
{
    uint k_0;
    uint groups_0;
    uint tokens_0;
};


#line 118
struct KernelContext_0
{
    QuantParams_0 constant* params_0;
    ushort device* act_0;
    ushort device* xscales_0;
    char device* codes_0;
};


#line 87
[[kernel]] void main_0(uint3 tid_0 [[thread_position_in_grid]], QuantParams_0 constant* params_1 [[buffer(0)]], ushort device* act_1 [[buffer(1)]], ushort device* xscales_1 [[buffer(3)]], char device* codes_1 [[buffer(2)]])
{

#line 87
    float xv_0;

#line 87
    thread KernelContext_0 kernelContext_0;

#line 87
    (&kernelContext_0)->params_0 = params_1;

#line 87
    (&kernelContext_0)->act_0 = act_1;

#line 87
    (&kernelContext_0)->xscales_0 = xscales_1;

#line 87
    (&kernelContext_0)->codes_0 = codes_1;
    uint idx_0 = tid_0.x;
    if(idx_0 >= (params_1->groups_0 * params_1->tokens_0))
    {

#line 90
        return;
    }
    uint g_0 = idx_0 % params_1->groups_0;
    uint t_0 = idx_0 / params_1->groups_0;

#line 93
    float amax_0 = 0.0f;

#line 93
    uint ii_0 = 0U;

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
        uint pos_0 = g_0 * 128U + ii_0;

        if(pos_0 < ((&kernelContext_0)->params_0->k_0))
        {

#line 98
            xv_0 = bf16ToF32_0(uint((&kernelContext_0)->act_0[t_0 * (&kernelContext_0)->params_0->k_0 + pos_0]));

#line 98
        }
        else
        {

#line 98
            xv_0 = 0.0f;

#line 98
        }


        float _S2 = max(amax_0, abs(xv_0));

#line 95
        uint ii_1 = ii_0 + 1U;

#line 95
        amax_0 = _S2;

#line 95
        ii_0 = ii_1;

#line 95
    }

#line 103
    uint sh_0 = f32ToF16_0(amax_0 / 127.0f);
    *((&kernelContext_0)->xscales_0+(t_0 * params_1->groups_0 + g_0)) = ushort(sh_0);
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
        uint _S3 = g_0 * 128U;

#line 111
        uint pos_1 = _S3 + ii_0;

        if(pos_1 < ((&kernelContext_0)->params_0->k_0))
        {

#line 113
            xv_0 = bf16ToF32_0(uint((&kernelContext_0)->act_0[t_0 * (&kernelContext_0)->params_0->k_0 + pos_1]));

#line 113
        }
        else
        {

#line 113
            xv_0 = 0.0f;

#line 113
        }

#line 118
        *((&kernelContext_0)->codes_0+(t_0 * params_1->groups_0 * 128U + _S3 + ii_0)) = char(int(max(-127.0f, min(127.0f, roundEven_0(xv_0 * inv_0)))));

#line 110
        ii_0 = ii_0 + 1U;

#line 110
    }

#line 120
    return;
}
