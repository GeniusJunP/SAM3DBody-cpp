#include "coreml_utils.h"

#ifndef __APPLE__
extern "C" {
void fsb_coreml_release_opaque(void* opaque) {
    (void)opaque;
}
}
#endif
