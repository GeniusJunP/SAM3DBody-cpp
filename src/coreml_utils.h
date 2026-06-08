#pragma once

#include <cstdint>

// Converts an IEEE 754 half-precision float (stored as uint16_t) to a single-precision float.
static inline float fsb_half_to_float(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x03ffu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out_f;
    __builtin_memcpy(&out_f, &f, sizeof(out_f));
    return out_f;
}

#ifdef __cplusplus
extern "C" {
#endif

void fsb_coreml_release_opaque(void* opaque);

#ifdef __cplusplus
}
#endif
