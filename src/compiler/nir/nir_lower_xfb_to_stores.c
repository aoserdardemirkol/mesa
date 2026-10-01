/*
 * Copyright (C) 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_xfb_info.h"

/*
 * Software transform feedback: lower store_output intrinsics that carry
 * io_xfb information (attached by nir_io_add_intrinsic_xfb_info) into
 * store_global writes at
 *
 *    xfb_address(buffer) + (instance_id * num_vertices + raw_vertex_id)
 *                          * stride + offset
 *
 * The driver provides the xfb_address and num_vertices system values.
 * load_vertex_id is rewritten to raw_vertex_id + raw_vertex_offset since
 * transform feedback programs consume the zero-based hardware vertex ID.
 *
 * options->address_format selects the format of the store_global address -
 * a plain 32-bit or 64-bit global. When options->keep_outputs is false, the
 * store_output is removed
 * once its captures are emitted - for shader variants that run as a separate
 * capture-only job. Drivers that rasterize and capture in the same draw keep
 * the outputs.
 */

struct lower_xfb_context {
   const nir_lower_xfb_to_stores_options *options;
   uint32_t record_extent_words[NIR_MAX_XFB_BUFFERS];
};

static void
lower_xfb_output(nir_builder *b, nir_intrinsic_instr *intr,
                 unsigned start_component, unsigned num_components,
                 unsigned buffer, unsigned offset_words,
                 const struct lower_xfb_context *ctx)
{
   unsigned address_bit_size =
      nir_address_format_bit_size(ctx->options->address_format);

   assert(buffer < MAX_XFB_BUFFERS);

   /* Transform feedback info in units of words, convert to bytes. */
   uint16_t stride = b->shader->info.xfb_stride[buffer] * 4;
   assert(stride != 0);

   uint16_t offset = offset_words * 4;

   nir_def *index = nir_iadd(
      b, nir_imul(b, nir_load_instance_id(b), nir_load_num_vertices(b)),
      nir_load_raw_vertex_id(b));

   BITSET_SET(b->shader->info.system_values_read,
              SYSTEM_VALUE_VERTEX_ID_ZERO_BASE);
   BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_INSTANCE_ID);

   nir_def *buf = nir_load_xfb_address(b, address_bit_size, .base = buffer);
   nir_def *word_offset = nir_iadd_imm(b, nir_imul_imm(b, index, stride),
                                       offset);
   nir_def *addr = nir_iadd(b, buf, nir_u2uN(b, word_offset, address_bit_size));

   nir_if *bound_if = NULL;
   if (ctx->options->bounds_check) {
      assert(ctx->record_extent_words[buffer] != 0);

      /* Compute the bound in 64 bits so an oversized vertex/instance index
       * cannot wrap around and pass the range test. All stores belonging to
       * one buffer use the full record extent, preventing partial records.
       */
      nir_def *instance_base = nir_imul(
         b, nir_u2u64(b, nir_load_instance_id(b)),
         nir_u2u64(b, nir_load_num_vertices(b)));
      nir_def *wide_index = nir_iadd(
         b, instance_base, nir_u2u64(b, nir_load_raw_vertex_id(b)));
      nir_def *last_record_index = wide_index;
      if (ctx->options->vertices_per_primitive > 0) {
         const unsigned primitive_vertices =
            ctx->options->vertices_per_primitive;
         nir_def *raw_vertex_id = nir_load_raw_vertex_id(b);
         nir_def *primitive_first = nir_imul_imm(
            b, nir_udiv_imm(b, raw_vertex_id, primitive_vertices),
            primitive_vertices);
         last_record_index = nir_iadd(
            b, nir_iadd(b, instance_base,
                        nir_u2u64(b, primitive_first)),
            nir_imm_int64(b, primitive_vertices - 1));
      }
      nir_def *record_end = nir_iadd_imm(
         b, nir_imul_imm(b, last_record_index, stride),
         ctx->record_extent_words[buffer] * 4);
      nir_def *range = nir_u2u64(
         b, nir_load_xfb_size(b, .base = buffer));
      bound_if = nir_push_if(b, nir_uge(b, range, record_end));
   }

   nir_def *src = intr->src[0].ssa;
   nir_component_mask_t mask = nir_component_mask(num_components);
   mask = (mask << start_component) >> nir_intrinsic_component(intr);
   nir_def *value = nir_channels(b, src, mask);
   nir_store_global(b, value, addr);
   if (bound_if)
      nir_pop_if(b, bound_if);
}

static bool
lower_xfb(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct lower_xfb_context *ctx = data;

   /* In transform feedback programs, vertex ID becomes zero-based, so apply
    * that lowering even on Valhall.
    */
   if (intr->intrinsic == nir_intrinsic_load_vertex_id) {
      b->cursor = nir_instr_remove(&intr->instr);

      nir_def *repl = nir_iadd(b, nir_load_raw_vertex_id(b),
                               nir_load_raw_vertex_offset(b));

      nir_def_rewrite_uses(&intr->def, repl);
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   bool progress = false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_io_xfb xfb = nir_intrinsic_io_xfb(intr);
   for (unsigned i = 0; i < 4; ++i) {
      if (!xfb.out[i].num_components)
         continue;

      lower_xfb_output(b, intr, i, xfb.out[i].num_components,
                        xfb.out[i].buffer, xfb.out[i].offset,
                        ctx);
      progress = true;
   }

   if (!ctx->options->keep_outputs)
      nir_instr_remove(&intr->instr);

   return progress;
}

bool
nir_lower_xfb_to_stores(nir_shader *nir, const nir_lower_xfb_to_stores_options *options)
{
   assert(options->address_format == nir_address_format_32bit_global ||
          options->address_format == nir_address_format_64bit_global);

   struct lower_xfb_context ctx = {.options = options};

   if (options->bounds_check) {
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_output)
                  continue;

               nir_io_xfb xfb = nir_intrinsic_io_xfb(intr);
               for (unsigned i = 0; i < 4; i++) {
                  if (!xfb.out[i].num_components)
                     continue;
                  unsigned buffer = xfb.out[i].buffer;
                  assert(buffer < NIR_MAX_XFB_BUFFERS);
                  ctx.record_extent_words[buffer] = MAX2(
                     ctx.record_extent_words[buffer],
                     xfb.out[i].offset + xfb.out[i].num_components);
               }
            }
         }
      }
   }

   return nir_shader_intrinsics_pass(nir, lower_xfb,
                                     nir_metadata_control_flow, &ctx);
}
