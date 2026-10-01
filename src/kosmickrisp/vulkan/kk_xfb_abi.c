/*
 * Copyright 2026 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_buffer.h"
#include "kk_xfb_abi.h"

#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"

static nir_def *
load_per_draw_field(nir_builder *b, uint32_t offset, uint8_t bit_size)
{
   nir_def *per_draw = nir_load_per_draw_ptr_kk(b, 1, 64);
   nir_def *addr = nir_iadd_imm(b, per_draw, offset);
   return nir_build_load_global_constant(b, 1, bit_size, addr,
                                         .align_mul = bit_size / 8,
                                         .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
load_xfb_current_offset(nir_builder *b, uint32_t binding_offset)
{
   nir_def *per_draw = nir_load_per_draw_ptr_kk(b, 1, 64);
   nir_def *static_offset_addr = nir_iadd_imm(
      b, per_draw,
      binding_offset + offsetof(struct kk_xfb_binding, current_offset));
   nir_def *static_offset = nir_build_load_global_constant(
      b, 1, 64, static_offset_addr, .align_mul = 8,
      .access = ACCESS_CAN_SPECULATE);

   nir_def *runtime_addr = load_per_draw_field(
      b, binding_offset + offsetof(struct kk_xfb_binding, current_offset_addr),
      64);
   nir_if *runtime = nir_push_if(b, nir_ine_imm(b, runtime_addr, 0));
   nir_def *runtime_offset = nir_u2u64(
      b, nir_build_load_global(b, 1, 32, runtime_addr, .align_mul = 4));
   nir_push_else(b, runtime);
   nir_def *constant_offset = static_offset;
   nir_pop_if(b, runtime);

   return nir_if_phi(b, runtime_offset, constant_offset);
}

static bool
lower_xfb_sysval(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const bool has_xfb_varyings = *(const bool *)data;

   if (intrin->intrinsic == nir_intrinsic_load_raw_vertex_id) {
      if (!has_xfb_varyings)
         return false;
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def *raw_id = nir_isub(b, nir_load_vertex_id(b),
                                 nir_load_first_vertex(b));
      nir_def_rewrite_uses(&intrin->def, raw_id);
      nir_instr_remove(&intrin->instr);
      return true;
   }

   if (intrin->intrinsic == nir_intrinsic_load_raw_vertex_offset) {
      if (!has_xfb_varyings)
         return false;
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def *offset = nir_load_first_vertex(b);
      nir_def_rewrite_uses(&intrin->def, offset);
      nir_instr_remove(&intrin->instr);
      return true;
   }

   if (intrin->intrinsic == nir_intrinsic_load_num_vertices) {
      if (!has_xfb_varyings)
         return false;
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def *per_draw = nir_load_per_draw_ptr_kk(b, 1, 64);
      nir_def *addr = nir_iadd_imm(
         b, per_draw, offsetof(struct kk_per_draw_data, vertex_count));
      nir_def *count = nir_build_load_global_constant(
         b, 1, intrin->def.bit_size, addr, .align_mul = 4,
         .access = ACCESS_CAN_SPECULATE);
      nir_def_rewrite_uses(&intrin->def, count);
      nir_instr_remove(&intrin->instr);
      return true;
   }

   if (intrin->intrinsic != nir_intrinsic_load_xfb_address &&
       intrin->intrinsic != nir_intrinsic_load_xfb_size)
      return false;

   const unsigned buffer = nir_intrinsic_base(intrin);
   assert(buffer < KK_XFB_BUFFER_COUNT);
   const uint32_t binding_offset =
      offsetof(struct kk_per_draw_data, xfb) +
      buffer * sizeof(struct kk_xfb_binding);
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *value;
   if (intrin->intrinsic == nir_intrinsic_load_xfb_address) {
      nir_def *base = load_per_draw_field(
         b, binding_offset + offsetof(struct kk_xfb_binding, buffer_address),
         64);
      nir_def *offset = load_per_draw_field(
         b, binding_offset + offsetof(struct kk_xfb_binding, binding_offset),
         64);
      nir_def *capture_offset = load_xfb_current_offset(b, binding_offset);
      value = nir_iadd(b, nir_iadd(b, base, offset), capture_offset);
      value = nir_u2uN(b, value, intrin->def.bit_size);
   } else {
      nir_def *range = load_per_draw_field(
         b, binding_offset + offsetof(struct kk_xfb_binding, range), 64);
      nir_def *offset = load_per_draw_field(
         b, binding_offset + offsetof(struct kk_xfb_binding, binding_offset),
         64);
      nir_def *capture_offset = load_xfb_current_offset(b, binding_offset);
      nir_def *usable_size = nir_isub(b, range, offset);
      nir_def *within_range = nir_uge(b, usable_size, capture_offset);
      nir_def *remaining = nir_isub(b, usable_size, capture_offset);
      value = nir_u2u32(b, nir_bcsel(b, within_range, remaining,
                                     nir_imm_int64(b, 0)));
   }

   nir_def_rewrite_uses(&intrin->def, value);
   nir_instr_remove(&intrin->instr);
   return true;
}

bool
kk_nir_lower_xfb_sysvals(nir_shader *nir)
{
   bool has_xfb_varyings = nir->info.has_transform_feedback_varyings;
   return nir_shader_intrinsics_pass(nir, lower_xfb_sysval,
                                     nir_metadata_control_flow,
                                     &has_xfb_varyings);
}
