/*
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"

#include "kk_query.h"

static inline global uint64_t *
query_report(global uint64_t *results, global uint16_t *oq_index,
             uint reports_per_query, uint query)
{
   /* For occlusion queries, results[] points to the device global heap. We
    * need to remap indices according to the query pool's allocation.
    */
   uint result_index = oq_index ? oq_index[query] : query;

   return results + (result_index * reports_per_query);
}

/**
 * Goes through a series of consecutive query indices in the given pool,
 * setting all element values to 0 and emitting them as available.
 */
KERNEL(1)
libkk_reset_query(global uint32_t *availability, global uint64_t *results,
                  global uint16_t *oq_index, uint32_t first_query,
                  uint16_t reports_per_query, int set_available)
{
   uint32_t query = first_query + cl_global_id.x;

   uint64_t value = 0;
   if (availability) {
      availability[query] = set_available;
   } else {
      value = set_available ? 0 : UINT64_MAX;
   }

   global uint64_t *report =
      query_report(results, oq_index, reports_per_query, query);

   /* XXX: is this supposed to happen on the begin? */
   for (unsigned j = 0; j < reports_per_query; ++j) {
      report[j] = value;
   }
}

KERNEL(1)
libkk_write_u32_array(global struct libkk_imm_write *write_array)
{
   uint id = cl_global_id.x;
   *(write_array[id].address) = write_array[id].value;
}

KERNEL(1)
libkk_write_u32(global uint32_t *address, uint32_t value)
{
   *address = value;
}

/* Internal transform-feedback session counters. The first four words are the
 * execution-time current byte offsets; the next four are per-draw snapshots
 * consumed by the vertex shader. Vulkan counter buffers serialize one u32.
 */
KERNEL(4)
libkk_xfb_counter_begin(global uint32_t *state,
                        global uint64_t *counter_addresses)
{
   uint binding = cl_global_id.x;
   uint64_t address = counter_addresses[binding];
   uint32_t offset = address ? *((global uint32_t *)address) : 0;
   state[binding] = offset;
   state[4 + binding] = offset;
}

KERNEL(1)
libkk_xfb_counter_prepare_draw(global uint32_t *state,
                               uint64_t usable_size, uint32_t stride,
                               uint32_t record_extent,
                               uint32_t vertex_count,
                               uint64_t query_report_addr)
{
   uint32_t current = state[0];
   state[4] = current;

   uint64_t remaining = current <= usable_size ? usable_size - current : 0;
   uint64_t complete_records = 0;
   if (stride && record_extent && remaining >= record_extent)
      complete_records = (remaining - record_extent) / stride + 1;

   uint64_t requested_triangles = vertex_count / 3;
   uint64_t capacity_triangles = complete_records / 3;
   uint64_t captured_vertices = min(requested_triangles,
                                    capacity_triangles) * 3;
   uint64_t advance = captured_vertices * stride;
   if (advance <= remaining && advance <= UINT32_MAX - current)
      state[0] = current + (uint32_t)advance;

   if (query_report_addr) {
      global uint64_t *query = (global uint64_t *)query_report_addr;
      query[0] += captured_vertices / 3;
      query[1] += requested_triangles;
   }
}

static inline uint64_t
libkk_count_segment(uint32_t topology, uint64_t count)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return count;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return count / 2;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return count > 1 ? count - 1 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return count / 3;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return count > 2 ? count - 2 : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      return count / 4;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
      return count > 3 ? count - 3 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      return count / 6;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
      return count > 4 ? count - 4 : 0;
   default:
      return 0;
   }
}

static inline uint32_t
libkk_read_index(global const uint8_t *indices, uint32_t index,
                 uint32_t index_size_B)
{
   if (index_size_B == 1)
      return indices[index];
   if (index_size_B == 2)
      return *((global const uint16_t *)indices + index);
   return *((global const uint32_t *)indices + index);
}

/* Count final stream primitives for VK_EXT_primitives_generated_query. The
 * draw descriptor is the exact direct/indirect command consumed by the final
 * raster draw (after KK tessellation/GS expansion and any restart unroll).
 * Indexed restart is counted by scanning actual indices, not by estimating
 * from indexCount. This query has no dependency on XFB state or capacity.
 */
KERNEL(1)
libkk_primitives_generated_add(global uint64_t *result,
                               uint64_t draw_addr,
                               uint64_t index_addr,
                               uint64_t index_range_B,
                               uint32_t topology,
                               uint32_t indexed,
                               uint32_t indirect,
                               uint32_t restart,
                               uint32_t restart_index,
                               uint32_t index_size_B)
{
   global const uint32_t *draw = (global const uint32_t *)draw_addr;
   uint32_t count = draw[0];
   uint32_t instances = draw[1];
   uint32_t first_index = 0;
   if (indexed && indirect)
      first_index = draw[2];

   uint64_t primitive_count = 0;
   if (!indexed || !restart) {
      primitive_count = libkk_count_segment(topology, count);
   } else {
      global const uint8_t *indices = (global const uint8_t *)index_addr;
      uint32_t segment = 0;
      uint32_t available = index_size_B ?
         (uint32_t)(index_range_B / index_size_B) : 0;
      for (uint32_t i = 0; i < count; ++i) {
         uint32_t at = first_index + i;
         if (at >= available)
            break;
         uint32_t value = libkk_read_index(indices, at, index_size_B);
         if (value == restart_index) {
            primitive_count += libkk_count_segment(topology, segment);
            segment = 0;
         } else {
            segment++;
         }
      }
      primitive_count += libkk_count_segment(topology, segment);
   }

   if (indexed && indirect)
      instances = draw[1];
   *result += primitive_count * instances;
}

KERNEL(4)
libkk_xfb_counter_end(global uint32_t *state,
                      global uint64_t *counter_addresses)
{
   uint binding = cl_global_id.x;
   uint64_t address = counter_addresses[binding];
   if (address)
      *((global uint32_t *)address) = state[binding];
}

KERNEL(1)
libkk_copy_queries(global uint32_t *availability, global uint64_t *results,
                   global uint16_t *oq_index, uint64_t dst_addr,
                   uint64_t dst_stride, uint32_t first_query,
                   VkQueryResultFlagBits flags, uint16_t reports_per_query)
{
   uint index = cl_group_id.x;
   uint64_t dst = dst_addr + (((uint64_t)index) * dst_stride);
   uint32_t query = first_query + index;

   bool available;
   if (availability)
      available = availability[query];
   else
      available = (results[query] != LIBKK_QUERY_UNAVAILABLE);

   if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
      /* For occlusion queries, results[] points to the device global heap. We
       * need to remap indices according to the query pool's allocation.
       */
      uint result_index = oq_index ? oq_index[query] : query;
      uint idx = result_index * reports_per_query;

      for (unsigned i = 0; i < reports_per_query; ++i) {
         vk_write_query(dst, i, flags, results[idx + i]);
      }
   }

   if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
      vk_write_query(dst, reports_per_query, flags, available);
   }
}
