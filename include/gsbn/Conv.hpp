#ifndef __GSBN_CONV_HPP__
#define __GSBN_CONV_HPP__

#include "gsbn/Common.hpp"

namespace gsbn{

#define fp64 double
#define fp32 float
#define fp16 uint16_t

#define fx32 int32_t
#define fx16 int16_t
#define fx8 int8_t

#define ufx32 uint32_t
#define ufx16 uint16_t
#define ufx8 uint8_t

fp32 fp16_to_fp32(const fp16 in);
fp16 fp32_to_fp16(const fp32 in);

fp32 fx32_to_fp32(const fx32 in, uint32_t frac_bit);
fx32 fp32_to_fx32(const fp32 in, uint32_t frac_bit);
fp32 fx16_to_fp32(const fx16 in, uint32_t frac_bit);
fx16 fp32_to_fx16(const fp32 in, uint32_t frac_bit);
fp32 fx8_to_fp32(const fx8 in, uint32_t frac_bit);
fx8 fp32_to_fx8(const fp32 in, uint32_t frac_bit);

fp32 ufx32_to_fp32(const ufx32 in, uint32_t frac_bit);
ufx32 fp32_to_ufx32(const fp32 in, uint32_t frac_bit);
fp32 ufx16_to_fp32(const ufx16 in, uint32_t frac_bit);
ufx16 fp32_to_ufx16(const fp32 in, uint32_t frac_bit);
fp32 ufx8_to_fp32(const ufx8 in, uint32_t frac_bit);
ufx8 fp32_to_ufx8(const fp32 in, uint32_t frac_bit);

}

#endif
