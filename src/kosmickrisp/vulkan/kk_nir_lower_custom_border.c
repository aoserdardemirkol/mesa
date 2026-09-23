/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_private.h"

#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

#include "stdbool.h"

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_has_custom_border_color_agx,
                                  1, nir_type_bool1, false, false);
}

static nir_def *
unorm16_quantize(nir_builder *b, nir_def *x)
{
   nir_def *c = nir_fmin(b, nir_fmax(b, x, nir_imm_float(b, 0.0f)),
                         nir_imm_float(b, 1.0f));
   nir_def *p = nir_fmul_imm(b, c, 65535.0);
   nir_def *err = nir_ffma(b, c, nir_imm_float(b, 65535.0f), nir_fneg(b, p));
   nir_def *f = nir_ffloor(b, p);
   nir_def *frac = nir_fsub(b, p, f);
   nir_def *half = nir_imm_float(b, 0.5f);
   nir_def *up =
      nir_ior(b, nir_flt(b, half, frac),
              nir_iand(b, nir_feq(b, frac, half),
                       nir_fge(b, err, nir_imm_float(b, 0.0f))));
   nir_def *q = nir_fadd(b, f, nir_b2f32(b, up));
   return nir_bcsel(b, nir_fneu(b, x, x), x, q);
}

static nir_def *
depth_compare(nir_builder *b, nir_def *op, nir_def *unorm16, nir_def *depth,
              nir_def *ref)
{
   nir_def *d = nir_bcsel(b, unorm16, unorm16_quantize(b, depth), depth);
   nir_def *r = nir_bcsel(b, unorm16, unorm16_quantize(b, ref), ref);

   nir_def *res = nir_imm_false(b);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_LESS), nir_flt(b, r, d),
                   res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_EQUAL),
                   nir_feq(b, r, d), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_LESS_OR_EQUAL),
                   nir_fge(b, d, r), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_GREATER),
                   nir_flt(b, d, r), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_NOT_EQUAL),
                   nir_fneu(b, r, d), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_GREATER_OR_EQUAL),
                   nir_fge(b, r, d), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_ALWAYS),
                   nir_imm_true(b), res);
   return res;
}

static nir_def *
clone_with_comparator(nir_builder *b, nir_tex_instr *tex, nir_def *comparator)
{
   nir_tex_instr *clone =
      nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &clone->instr);
   int idx = nir_tex_instr_src_index(clone, nir_tex_src_comparator);
   nir_src_rewrite(&clone->src[idx].src, comparator);
   return &clone->def;
}

static bool
lower_shadow(nir_builder *b, nir_tex_instr *tex)
{
   int cmp_idx = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   if (cmp_idx < 0)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;

   const uint32_t fp_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;

   nir_push_if(b, has_custom);
   nir_def *replaced;
   {
      nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, clone_instr);
      nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
      tex_0->backend_flags |= KK_TEXTURE_FLAG_CLAMP_TO_0;
      nir_def *clamp_to_0 = &tex_0->def;
      const unsigned bit_size = clamp_to_0->bit_size;

      nir_def *border = nir_channel(
         b,
         nir_build_texture_query(b, tex_0, nir_texop_custom_border_color_agx,
                                 4, nir_type_float32, false, false),
         0);
      nir_def *state =
         nir_build_texture_query(b, tex_0, nir_texop_custom_border_color_agx,
                                 4, nir_type_uint32, false, false);
      nir_instr_as_tex(nir_def_instr(state))->backend_flags |=
         KK_TEXTURE_FLAG_COMPARE_STATE;
      nir_def *op = nir_channel(b, state, 0);
      nir_def *unorm16 = nir_ieq_imm(b, nir_channel(b, state, 1), 16);

      nir_def *comparator = tex->src[cmp_idx].src.ssa;
      nir_def *ref = nir_f2f32(b, comparator);
      nir_def *zero = nir_imm_float(b, 0.0f);
      nir_def *one = nir_imm_float(b, 1.0f);

      nir_def *q0 = depth_compare(b, op, unorm16, zero, ref);
      nir_def *q1 = depth_compare(b, op, unorm16, one, ref);
      nir_def *qb = depth_compare(b, op, unorm16, border, ref);

      nir_def *selected =
         nir_bcsel(b, nir_ieq(b, qb, q0), clamp_to_0, clamp_to_1);

      nir_push_if(b, nir_iand(b, nir_ieq(b, q0, q1), nir_ine(b, qb, q0)));
      nir_def *corrected;
      {
         nir_def *eq_op =
            nir_ior(b, nir_ieq_imm(b, op, VK_COMPARE_OP_EQUAL),
                    nir_ieq_imm(b, op, VK_COMPARE_OP_NOT_EQUAL));
         nir_def *probe = nir_bcsel(b, eq_op, zero, nir_imm_float(b, 0.5f));
         nir_def *probe_cmp = nir_f2fN(b, probe, comparator->bit_size);

         nir_def *probe_0 = clone_with_comparator(b, tex_0, probe_cmp);
         nir_def *probe_1 =
            clone_with_comparator(b, nir_instr_as_tex(orig), probe_cmp);

         const unsigned n = clamp_to_0->num_components;
         nir_def *sign = nir_fsub(
            b, nir_b2fN(b, depth_compare(b, op, unorm16, one, probe), bit_size),
            nir_b2fN(b, depth_compare(b, op, unorm16, zero, probe), bit_size));
         nir_def *beta = nir_fmul(b, nir_fsub(b, probe_1, probe_0),
                                  nir_replicate(b, sign, n));
         nir_def *delta = nir_fsub(b, nir_b2fN(b, qb, bit_size),
                                   nir_b2fN(b, q0, bit_size));
         corrected =
            nir_ffma(b, beta, nir_replicate(b, delta, n), clamp_to_0);
      }
      nir_pop_if(b, NULL);
      replaced = nir_if_phi(b, corrected, selected);
   }
   nir_pop_if(b, NULL);

   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   b->fp_math_ctrl = fp_math_ctrl;
   nir_def_replace(&tex->def, phi);
   return true;
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex))
      return false;

   if (tex->is_shadow)
      return lower_shadow(b, tex);

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced = NULL;
   {
      /* Sample again, this time with clamp-to-0 instead of clamp-to-1 */
      nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, clone_instr);

      nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
      nir_def *clamp_to_0 = &tex_0->def;

      tex_0->backend_flags |= KK_TEXTURE_FLAG_CLAMP_TO_0;

      /* Grab the border colour */
      nir_def *border = query_custom_border(b, tex_0);

      if (tex->op == nir_texop_tg4) {
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);
      }

      /* Combine together with the border */
      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {

         const uint32_t fp_math_ctrl = b->fp_math_ctrl;
         b->fp_math_ctrl = nir_fp_no_fast_math;
         nir_def *no_border = nir_ieq(b, clamp_to_0, clamp_to_1);
         nir_def *weight = nir_fsub(b, clamp_to_1, clamp_to_0);
         nir_def *blended = nir_ffma(b, weight, border, clamp_to_0);
         replaced = nir_bcsel(b, no_border, clamp_to_0, blended);
         b->fp_math_ctrl = fp_math_ctrl;
      } else {
         /* For integers, just select componentwise since there is no linear
          * filtering. Gathers also use this path since they are unfiltered in
          * each component.
          */
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_0, clamp_to_1), clamp_to_0,
                              border);
      }
   }
   nir_pop_if(b, NULL);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   nir_def_replace(&tex->def, phi);
   return true;
}

bool
kk_nir_lower_custom_border(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}
