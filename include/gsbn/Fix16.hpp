#ifndef __GSBN_FIX16_HPP__
#define __GSBN_FIX16_HPP__

#include "gsbn/Common.hpp"

namespace gsbn{


#define fix16_10 int16_t
#define fix16_15 int16_t
#define fix16 int16_t
#define fix16_to_fp32 fix16_10_to_fp32
#define fp32_to_fix16 fp32_to_fix16_10

/*
 * fix16_10 sign 1 bits, int 5 bits, fraction 10 bits
 * range: [-32, 32)
 * perceision: 0.0001
 */

float fix16_10_to_fp32(const fix16 in);
fix16 fp32_to_fix16_10(const float in);

float fix16_15_to_fp32(const fix16 in);
fix16 fp32_to_fix16_15(const float in);

}

#endif
