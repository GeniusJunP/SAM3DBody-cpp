#import <Foundation/Foundation.h>

extern "C" void fsb_coreml_release_opaque(void* opaque) {
    if (opaque) {
        CFRelease(opaque);
    }
}
