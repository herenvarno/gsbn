#ifndef __GSBN_FIX16_HPP__
#define __GSBN_FIX16_HPP__

#include "gsbn/Common.hpp"

namespace gsbn{


#define fix16_10 int16_t
#define fix16_15 int16_t
#define fix16 int16_t

/*
 * fix16_10 sign 1 bits, int x bits, fraction y bits, x+y=15
 */

float fix16_to_fp32(const fix16 in, int frac_bit);
fix16 fp32_to_fix16(const float in, int frac_bit);

}

#endif
