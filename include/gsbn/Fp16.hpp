#ifndef __GSBN_FP16_HPP__
#define __GSBN_FP16_HPP__

#include "gsbn/Common.hpp"

namespace gsbn{

#define fp16 uint16_t
#define fp32 float
#define fp64 double

fp32 fp16_to_fp32(const fp16 in);
fp16 fp32_to_fp16(const fp32 in);

}

#endif
