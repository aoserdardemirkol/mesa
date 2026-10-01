/*
 * Copyright 2026 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdio.h>

#include "kk_cmd_buffer.h"
#include "kk_xfb_abi.h"
#include "nir_builder.h"

struct intrinsic_counts {
   unsigned address;
   unsigned size;
   unsigned per_draw_ptr;
   unsigned global_load;
   unsigned raw_vertex_id;
   unsigned raw_vertex_offset;
   unsigned num_vertices;
   unsigned vertex_id;
   unsigned first_vertex;
};

static bool
count_intrinsics(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   struct intrinsic_counts *counts = data;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_xfb_address:
      counts->address++;
      break;
   case nir_intrinsic_load_xfb_size:
      counts->size++;
      break;
   case nir_intrinsic_load_per_draw_ptr_kk:
      counts->per_draw_ptr++;
      break;
   case nir_intrinsic_load_global_constant:
      counts->global_load++;
      break;
   case nir_intrinsic_load_raw_vertex_id:
      counts->raw_vertex_id++;
      break;
   case nir_intrinsic_load_raw_vertex_offset:
      counts->raw_vertex_offset++;
      break;
   case nir_intrinsic_load_num_vertices:
      counts->num_vertices++;
      break;
   case nir_intrinsic_load_vertex_id:
      counts->vertex_id++;
      break;
   case nir_intrinsic_load_first_vertex:
      counts->first_vertex++;
      break;
   default:
      break;
   }
   return false;
}

int
main(void)
{
   struct kk_xfb_binding binding;
   assert(kk_xfb_binding_configure(&binding, UINT64_C(0x123456789abc0000),
                                   8192, 256, 64));

   struct kk_per_draw_data per_draw = {0};
   kk_xfb_binding_copy_to_draw(&per_draw.xfb[2], &binding);
   assert(per_draw.xfb[2].valid == 1);
   assert(per_draw.xfb[2].buffer_address == UINT64_C(0x123456789abc0000));
   assert(per_draw.xfb[2].range == 8192);
   assert(per_draw.xfb[2].binding_offset == 256);
   assert(per_draw.xfb[2].current_offset == 64);

   struct kk_xfb_binding disabled = {0};
   kk_xfb_binding_copy_to_draw(&per_draw.xfb[1], &disabled);
   assert(per_draw.xfb[1].valid == 0);
   assert(per_draw.xfb[1].buffer_address == 0);
   assert(per_draw.xfb[1].range == 0);
   assert(per_draw.xfb[1].binding_offset == 0);
   assert(per_draw.xfb[1].current_offset == 0);

   struct kk_xfb_binding invalid;
   assert(!kk_xfb_binding_configure(&invalid, UINT64_C(0x100000000), 64,
                                    32, 40));
   assert(invalid.valid == 0 && invalid.buffer_address == 0 &&
          invalid.range == 0 && invalid.binding_offset == 0 &&
          invalid.current_offset == 0);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, NULL, "KK XFB per-draw ABI test");
   nir_load_xfb_address(&b, 64, .base = 2);
   nir_load_xfb_size(&b, .base = 2);
   b.shader->info.has_transform_feedback_varyings = true;
   nir_load_raw_vertex_id(&b);
   nir_load_raw_vertex_offset(&b);
   nir_load_num_vertices(&b);
   assert(kk_nir_lower_xfb_sysvals(b.shader));

   struct intrinsic_counts counts = {0};
   nir_shader_intrinsics_pass(b.shader, count_intrinsics, nir_metadata_all,
                              &counts);
   assert(counts.address == 0);
   assert(counts.size == 0);
   assert(counts.raw_vertex_id == 0);
   assert(counts.raw_vertex_offset == 0);
   assert(counts.num_vertices == 0);
   assert(counts.vertex_id == 1);
   assert(counts.first_vertex == 2);
   assert(counts.per_draw_ptr == 9);
   assert(counts.global_load == 9);
   ralloc_free(b.shader);

   puts("PASS: KK XFB per-draw address/range ABI and sysval lowering verified");
   return 0;
}
