/*
 * Copyright 2026 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_XFB_ABI_H
#define KK_XFB_ABI_H

#include <stdbool.h>
#include <stdint.h>

#include "nir.h"

#define KK_XFB_BUFFER_COUNT 4
static_assert(KK_XFB_BUFFER_COUNT == MAX_XFB_BUFFERS,
              "KK XFB ABI must match NIR's transform-feedback limit");

/* Raw bound-buffer state. Shader-visible xfb_address is resolved to
 * buffer_address + binding_offset + current_offset exactly once; xfb_size is
 * the remaining byte range after those two offsets.
 */
struct kk_xfb_binding {
   uint64_t buffer_address;
   uint64_t range;
   uint64_t binding_offset;
   /* GPU address of the per-draw execution-time counter snapshot. Zero means
    * current_offset below is the static private-test ABI value.
    */
   uint64_t current_offset_addr;
   uint64_t current_offset;
   uint32_t valid;
   uint32_t _pad;
};

static inline bool
kk_xfb_binding_configure(struct kk_xfb_binding *binding,
                         uint64_t buffer_address, uint64_t range,
                         uint64_t binding_offset, uint64_t current_offset)
{
   *binding = (struct kk_xfb_binding){0};

   if (binding_offset > range || current_offset > range - binding_offset ||
       range - binding_offset - current_offset > UINT32_MAX ||
       buffer_address > UINT64_MAX - binding_offset - current_offset)
      return false;

   binding->buffer_address = buffer_address;
   binding->range = range;
   binding->binding_offset = binding_offset;
   binding->current_offset = current_offset;
   binding->valid = 1;
   return true;
}

/* Copy state into the draw ABI deterministically, including the disabled
 * case, so stale addresses from earlier draws cannot leak into a shader.
 */
static inline void
kk_xfb_binding_copy_to_draw(struct kk_xfb_binding *dst,
                            const struct kk_xfb_binding *src)
{
   *dst = src->valid ? *src : (struct kk_xfb_binding){0};
}

bool kk_nir_lower_xfb_sysvals(nir_shader *nir);

#endif /* KK_XFB_ABI_H */
