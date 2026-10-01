/*
 * Copyright 2026 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "nir_builder.h"
#include "nir_xfb_info.h"
#include "kk_shader.h"

struct xfb_lowering_counts {
   unsigned store_output;
   unsigned store_global;
   unsigned bounded_stores;
   unsigned xfb_address;
   unsigned xfb_size;
   unsigned raw_vertex_id;
   unsigned num_vertices;
};

static bool
count_xfb_lowering(UNUSED nir_builder *b, nir_intrinsic_instr *intrin,
                   void *data)
{
   struct xfb_lowering_counts *counts = data;
   switch (intrin->intrinsic) {
   case nir_intrinsic_store_output:
      counts->store_output++;
      break;
   case nir_intrinsic_store_global:
      counts->store_global++;
      for (nir_cf_node *parent = intrin->instr.block->cf_node.parent;
           parent; parent = parent->parent) {
         if (parent->type == nir_cf_node_if) {
            counts->bounded_stores++;
            break;
         }
      }
      break;
   case nir_intrinsic_load_xfb_address:
      counts->xfb_address++;
      break;
   case nir_intrinsic_load_xfb_size:
      counts->xfb_size++;
      break;
   case nir_intrinsic_load_raw_vertex_id:
      counts->raw_vertex_id++;
      break;
   case nir_intrinsic_load_num_vertices:
      counts->num_vertices++;
      break;
   default:
      break;
   }
   return false;
}

static nir_shader *
make_vs(bool with_xfb)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "KK XFB metadata test");

   if (with_xfb) {
      nir_xfb_info *xfb = rzalloc_size(b.shader,
                                      nir_xfb_info_size(2));
      assert(xfb != NULL);
      b.shader->xfb_info = xfb;
      xfb->buffers_written = BITFIELD_BIT(0) | BITFIELD_BIT(1);
      xfb->streams_written = BITFIELD_BIT(0) | BITFIELD_BIT(2);
      xfb->buffers[0].stride = 32;
      xfb->buffers[0].varying_count = 1;
      xfb->buffers[1].stride = 48;
      xfb->buffers[1].varying_count = 1;
      xfb->buffer_to_stream[0] = 0;
      xfb->buffer_to_stream[1] = 2;
      xfb->output_count = 2;
      xfb->outputs[0] = (nir_xfb_output_info){
         .buffer = 0,
         .offset = 4,
         .location = 0,
         .component_mask = 0x3,
         .component_offset = 1,
      };
      xfb->outputs[1] = (nir_xfb_output_info){
         .buffer = 1,
         .offset = 16,
         .location = 1,
         .component_mask = 0xc,
         .component_offset = 2,
      };

      nir_store_output(&b, nir_imm_vec4(&b, 1.0, 2.0, 3.0, 4.0),
                       nir_imm_int(&b, 0),
                       .io_semantics = (struct nir_io_semantics){
                          .location = 0,
                          .num_slots = 1,
                       });
      nir_store_output(&b, nir_imm_vec4(&b, 5.0, 6.0, 7.0, 8.0),
                       nir_imm_int(&b, 0),
                       .io_semantics = (struct nir_io_semantics){
                          .location = 1,
                          .num_slots = 1,
                       });

      assert(nir_io_add_intrinsic_xfb_info(b.shader));
   }

   return b.shader;
}

int
main(void)
{
   nir_shader *nir = make_vs(true);
   struct kk_shader_info info = {0};
   assert(kk_shader_info_copy_xfb(&info, nir));
   assert(info.has_xfb_capture_metadata);
   assert(info.xfb_info.output_count == 2);
   assert(info.xfb_info.buffers_written == (BITFIELD_BIT(0) | BITFIELD_BIT(1)));
   assert(info.xfb_info.streams_written == (BITFIELD_BIT(0) | BITFIELD_BIT(2)));
   assert(info.xfb_info.buffer_to_stream[0] == 0);
   assert(info.xfb_info.buffer_to_stream[1] == 2);
   assert(info.xfb_info.buffers[0].stride == 32);
   assert(info.xfb_info.buffers[1].stride == 48);
   assert(info.xfb_stride[0] == 8);
   assert(info.xfb_stride[1] == 12);
   assert(info.xfb_info.outputs[0].buffer == 0);
   assert(info.xfb_info.outputs[0].offset == 4);
   assert(info.xfb_info.outputs[0].location == 0);
   assert(info.xfb_info.outputs[0].component_mask == 0x3);
   assert(info.xfb_info.outputs[0].component_offset == 1);
   assert(info.xfb_info.outputs[1].buffer == 1);
   assert(info.xfb_info.outputs[1].offset == 16);
   assert(info.xfb_info.outputs[1].location == 1);
   assert(info.xfb_info.outputs[1].component_mask == 0xc);
   assert(info.xfb_info.outputs[1].component_offset == 2);

   const nir_lower_xfb_to_stores_options xfb_options = {
      .address_format = nir_address_format_64bit_global,
      .keep_outputs = true,
      .bounds_check = true,
      .vertices_per_primitive = 3,
   };
   assert(nir_lower_xfb_to_stores(nir, &xfb_options));
   struct xfb_lowering_counts counts = {0};
   nir_shader_intrinsics_pass(nir, count_xfb_lowering, nir_metadata_all,
                              &counts);
   assert(counts.store_output == 2); /* Raster outputs stay live. */
   assert(counts.store_global == 2);
   assert(counts.bounded_stores == 2);
   assert(counts.xfb_address == 2);
   assert(counts.xfb_size == 2);
   assert(counts.raw_vertex_id == 6);
   assert(counts.num_vertices == 4);
   ralloc_free(nir);

   nir = make_vs(false);
   info = (struct kk_shader_info){0};
   assert(!kk_shader_info_copy_xfb(&info, nir));
   assert(!info.has_xfb_capture_metadata);
   for (unsigned i = 0; i < MAX_XFB_BUFFERS; i++)
      assert(info.xfb_stride[i] == 0);
   ralloc_free(nir);

   puts("PASS: KK VS XFB metadata survives compilation preparation");
   return 0;
}
