/*
 * Copyright 2026 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */

#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_write_spv.h"

#define BUFFER_SIZE 64
#define XFB_CAPTURE_BUFFER_SIZE 128
#define CANARY_BYTE 0xcd

#include "xfb_capture_spv.h"

typedef VkResult(VKAPI_PTR *PFN_kk_test_record_xfb_abi_probe)(
   VkCommandBuffer, bool, VkDeviceAddress, VkDeviceSize, VkDeviceSize,
   VkDeviceSize, VkDeviceAddress, VkDeviceSize);
typedef VkResult(VKAPI_PTR *PFN_kk_test_set_xfb_capture_target)(
   VkCommandBuffer, VkDeviceAddress, VkDeviceSize, VkDeviceSize,
   VkDeviceSize);
typedef VkResult(VKAPI_PTR *PFN_kk_test_begin_xfb)(VkCommandBuffer,
                                                  VkDeviceAddress,
                                                  VkDeviceSize);
typedef VkResult(VKAPI_PTR *PFN_kk_test_end_xfb)(VkCommandBuffer,
                                                VkDeviceAddress,
                                                VkDeviceSize);

static bool vk_ok(VkResult result, const char *what);

static bool
run_xfb_capture_case(VkDevice device, VkQueue queue,
                     VkCommandPool command_pool, VkFence fence,
                     VkPipeline pipeline,
                     PFN_kk_test_set_xfb_capture_target set_target,
                     PFN_kk_test_begin_xfb begin_xfb,
                     PFN_kk_test_end_xfb end_xfb,
                     VkDeviceAddress xfb_address,
                     VkDeviceAddress counter_address,
                     uint8_t *mapped, uint32_t *mapped_counter,
                     VkDeviceSize range, unsigned expected_records)
{
   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &allocate_info,
                                       &command_buffer),
              "vkAllocateCommandBuffers(XFB capture)"))
      return false;

   memset(mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 8;
   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   bool passed = false;
   if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
              "vkBeginCommandBuffer(XFB capture)"))
      goto out;

   VkResult result = set_target(command_buffer, xfb_address, range, 8, 0);
   if (!vk_ok(result, "set private XFB capture target"))
      goto out;
   if (!vk_ok(begin_xfb(command_buffer, counter_address, 0),
              "resume private XFB session"))
      goto out;

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {1, 1}},
      .layerCount = 1,
   };
   vkCmdBeginRendering(command_buffer, &rendering_info);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRendering(command_buffer);
   if (!vk_ok(end_xfb(command_buffer, counter_address, 0),
              "end private XFB session"))
      goto out;
   if (!vk_ok(vkEndCommandBuffer(command_buffer),
              "vkEndCommandBuffer(XFB capture)"))
      goto out;

   if (!vk_ok(vkResetFences(device, 1, &fence),
              "vkResetFences(XFB capture)"))
      goto out;
   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   if (!vk_ok(vkQueueSubmit(queue, 1, &submit_info, fence),
              "vkQueueSubmit(XFB capture)") ||
       !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences(XFB capture)"))
      goto out;

   bool bytes_match = true;
   for (uint32_t byte = 0; byte < XFB_CAPTURE_BUFFER_SIZE; byte++) {
      uint8_t expected = CANARY_BYTE;
      if (byte >= 16 && byte < 16 + expected_records * 16) {
         const uint32_t word = (byte - 16) / 4;
         const uint32_t vertex = word / 4;
         const uint32_t component = word % 4;
         const float value = (float)(vertex * 4 + component + 1);
         uint32_t bits;
         memcpy(&bits, &value, sizeof(bits));
         expected = (bits >> (((byte - 16) % 4) * 8)) & 0xff;
      }
      if (mapped[byte] != expected) {
         fprintf(stderr,
                 "XFB capture range=%llu byte %u: expected 0x%02x, got "
                 "0x%02x\n",
                 (unsigned long long)range, byte, expected, mapped[byte]);
         bytes_match = false;
      }
   }
   if (!bytes_match)
      goto out;

   if (*mapped_counter != 8 + expected_records * 16) {
      fprintf(stderr,
              "XFB range=%llu: expected saved counter %u, got %u\n",
              (unsigned long long)range, 8 + expected_records * 16,
              *mapped_counter);
      goto out;
   }

   printf("PASS: XFB VS capture range=%llu captured %u vertex records; "
          "guards intact\n",
          (unsigned long long)range, expected_records);
   passed = true;


out:
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return passed;
}

static bool
submit_and_wait(VkDevice device, VkQueue queue, VkFence fence,
                VkCommandBuffer command_buffer, const char *label)
{
   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   return vk_ok(vkResetFences(device, 1, &fence), label) &&
          vk_ok(vkQueueSubmit(queue, 1, &submit_info, fence), label) &&
          vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), label);
}

static bool
record_single_xfb_draw(VkDevice device, VkCommandPool command_pool,
                       VkPipeline pipeline,
                       PFN_kk_test_set_xfb_capture_target set_target,
                       PFN_kk_test_begin_xfb begin_xfb,
                       PFN_kk_test_end_xfb end_xfb,
                       VkDeviceAddress xfb_address,
                       VkDeviceAddress resume_counter_address,
                       VkDeviceAddress save_counter_address,
                       VkCommandBuffer *command_buffer_out)
{
   VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &allocate_info,
                                       command_buffer_out),
              "vkAllocateCommandBuffers(XFB counter session)"))
      return false;

   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   VkCommandBuffer command_buffer = *command_buffer_out;
   if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
              "vkBeginCommandBuffer(XFB counter session)"))
      return false;

   if (!vk_ok(set_target(command_buffer, xfb_address, 128, 8, 0),
              "bind XFB counter session target") ||
       !vk_ok(begin_xfb(command_buffer, resume_counter_address, 0),
              "begin XFB counter session"))
      return false;

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {1, 1}},
      .layerCount = 1,
   };
   vkCmdBeginRendering(command_buffer, &rendering_info);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRendering(command_buffer);

   return vk_ok(end_xfb(command_buffer, save_counter_address, 0),
                "end XFB counter session") &&
          vk_ok(vkEndCommandBuffer(command_buffer),
                "vkEndCommandBuffer(XFB counter session)");
}

static bool
xfb_bytes_match(uint8_t *mapped, const uint32_t *start_offsets,
                unsigned capture_count)
{
   uint8_t expected[XFB_CAPTURE_BUFFER_SIZE];
   memset(expected, CANARY_BYTE, sizeof(expected));
   for (unsigned capture = 0; capture < capture_count; capture++) {
      for (unsigned vertex = 0; vertex < 3; vertex++) {
         for (unsigned component = 0; component < 4; component++) {
            const float value = (float)(vertex * 4 + component + 1);
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            for (unsigned byte = 0; byte < 4; byte++)
               expected[start_offsets[capture] + vertex * 16 + component * 4 +
                        byte] = (bits >> (byte * 8)) & 0xff;
         }
      }
   }

   for (unsigned i = 0; i < sizeof(expected); i++) {
      if (mapped[i] != expected[i]) {
         fprintf(stderr,
                 "XFB counter capture byte %u: expected 0x%02x, got 0x%02x\n",
                 i, expected[i], mapped[i]);
         return false;
      }
   }
   return true;
}

static bool
run_xfb_save_resume_test(VkDevice device, VkQueue queue,
                         VkCommandPool command_pool, VkFence fence,
                         VkPipeline pipeline,
                         PFN_kk_test_set_xfb_capture_target set_target,
                         PFN_kk_test_begin_xfb begin_xfb,
                         PFN_kk_test_end_xfb end_xfb,
                         VkDeviceAddress xfb_address,
                         VkDeviceAddress counter_address,
                         uint8_t *mapped, uint32_t *mapped_counter)
{
   memset(mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 0xdeadbeef;
   VkCommandBuffer first = VK_NULL_HANDLE;
   if (!record_single_xfb_draw(device, command_pool, pipeline, set_target,
                               begin_xfb, end_xfb, xfb_address, 0,
                               counter_address, &first) ||
       !submit_and_wait(device, queue, fence, first,
                        "submit XFB counter save"))
      return false;
   if (*mapped_counter != 3 * 16 ||
       !xfb_bytes_match(mapped, (uint32_t[]){8}, 1)) {
      fprintf(stderr, "XFB first submission did not save 3 * stride\n");
      return false;
   }
   printf("PASS: XFB submission A saved 4-byte counter=%u\n",
          *mapped_counter);
   vkFreeCommandBuffers(device, command_pool, 1, &first);

   VkCommandBuffer second = VK_NULL_HANDLE;
   if (!record_single_xfb_draw(device, command_pool, pipeline, set_target,
                               begin_xfb, end_xfb, xfb_address,
                               counter_address, counter_address, &second) ||
       !submit_and_wait(device, queue, fence, second,
                        "submit XFB counter resume"))
      return false;
   if (*mapped_counter != 6 * 16 ||
       !xfb_bytes_match(mapped, (uint32_t[]){8, 56}, 2)) {
      fprintf(stderr,
              "XFB resumed submission failed to append: counter=%u\n",
              *mapped_counter);
      return false;
   }
   printf("PASS: XFB submission B resumed at saved counter; counter=%u\n",
          *mapped_counter);
   vkFreeCommandBuffers(device, command_pool, 1, &second);
   return true;
}

static bool
run_xfb_command_buffer_replay_test(
   VkDevice device, VkQueue queue, VkCommandPool command_pool, VkFence fence,
   VkPipeline pipeline, PFN_kk_test_set_xfb_capture_target set_target,
   PFN_kk_test_begin_xfb begin_xfb, PFN_kk_test_end_xfb end_xfb,
   VkDeviceAddress xfb_address, VkDeviceAddress counter_address,
   uint8_t *mapped, uint32_t *mapped_counter)
{
   memset(mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 0;

   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   if (!record_single_xfb_draw(device, command_pool, pipeline, set_target,
                               begin_xfb, end_xfb, xfb_address,
                               counter_address, counter_address,
                               &command_buffer))
      return false;

   if (!submit_and_wait(device, queue, fence, command_buffer,
                        "submit XFB command buffer replay A") ||
       *mapped_counter != 3 * 16 ||
       !xfb_bytes_match(mapped, (uint32_t[]){8}, 1)) {
      fprintf(stderr, "XFB first replay submission failed: counter=%u\n",
              *mapped_counter);
      vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
      return false;
   }

   /* Reuse the same recorded command buffer. Its begin operation must load
    * the GPU counter again, and its bind/begin/draw/end order must be replayed
    * through the KK command queue. */
   if (!submit_and_wait(device, queue, fence, command_buffer,
                        "submit XFB command buffer replay B") ||
       *mapped_counter != 6 * 16 ||
       !xfb_bytes_match(mapped, (uint32_t[]){8, 56}, 2)) {
      fprintf(stderr, "XFB command-buffer replay failed: counter=%u\n",
              *mapped_counter);
      vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
      return false;
   }

   printf("PASS: XFB same-command-buffer replay resumed at GPU counter; "
          "counter=%u\n", *mapped_counter);
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return true;
}

static bool
run_xfb_execution_time_read_test(
   VkDevice device, VkQueue queue, VkCommandPool command_pool, VkFence fence,
   VkPipeline pipeline, PFN_kk_test_set_xfb_capture_target set_target,
   PFN_kk_test_begin_xfb begin_xfb, PFN_kk_test_end_xfb end_xfb,
   VkDeviceAddress xfb_address, VkDeviceAddress counter_address,
   uint8_t *mapped, uint32_t *mapped_counter)
{
   memset(mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 0;
   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   if (!record_single_xfb_draw(device, command_pool, pipeline, set_target,
                               begin_xfb, end_xfb, xfb_address,
                               counter_address, counter_address,
                               &command_buffer))
      return false;

   /* The GPU must observe this value at execution, although it was written
    * after the command buffer recorded its begin operation.
    */
   *mapped_counter = 16;
   if (!submit_and_wait(device, queue, fence, command_buffer,
                        "submit XFB late counter value") ||
       *mapped_counter != 64 ||
       !xfb_bytes_match(mapped, (uint32_t[]){24}, 1)) {
      fprintf(stderr,
              "XFB execution-time counter load failed: counter=%u\n",
              *mapped_counter);
      return false;
   }
   printf("PASS: XFB begin read counter value present at execution time\n");
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return true;
}

static bool
run_public_xfb_query_case(
   VkDevice device, VkQueue queue, VkCommandPool command_pool, VkFence fence,
   VkPipeline pipeline, VkBuffer xfb_buffer, VkBuffer counter_buffer,
   uint8_t *xfb_mapped, uint32_t *mapped_counter,
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb,
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb,
   PFN_vkCmdEndTransformFeedbackEXT end_xfb,
   PFN_vkCmdBeginQueryIndexedEXT begin_xfb_query,
   PFN_vkCmdEndQueryIndexedEXT end_xfb_query,
   VkQueryPool query_pool, VkDeviceSize capture_range, uint32_t draw_count,
   uint64_t expected_written, uint64_t expected_needed,
   uint32_t expected_counter, const uint32_t *capture_offsets,
   unsigned expected_capture_count, bool replay)
{
   memset(xfb_mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 0;

   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer),
              "vkAllocateCommandBuffers(public XFB)"))
      return false;

   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
              "vkBeginCommandBuffer(public XFB)"))
      goto fail;

   vkCmdResetQueryPool(command_buffer, query_pool, 0, 1);
   const VkDeviceSize bind_offset = 8;
   const VkBuffer buffers[] = {xfb_buffer};
   const VkDeviceSize offsets[] = {bind_offset};
   const VkDeviceSize sizes[] = {capture_range};
   bind_xfb(command_buffer, 0, 1, buffers, offsets, sizes);

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {1, 1}},
      .layerCount = 1,
   };
   vkCmdBeginRendering(command_buffer, &rendering_info);
   const VkBuffer counter_buffers[] = {counter_buffer};
   const VkDeviceSize counter_offsets[] = {32};
   /* The replay case records a GPU counter source. The first execution reads
    * zero; replay then reads the updated counter from the same buffer. */
   begin_xfb(command_buffer, 0, replay ? 1 : 0,
             replay ? counter_buffers : NULL,
             replay ? counter_offsets : NULL);
   begin_xfb_query(command_buffer, query_pool, 0, 0, 0);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   for (uint32_t i = 0; i < draw_count; i++)
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
   end_xfb_query(command_buffer, query_pool, 0, 0);
   end_xfb(command_buffer, 0, 1, counter_buffers, counter_offsets);
   vkCmdEndRendering(command_buffer);
   if (!vk_ok(vkEndCommandBuffer(command_buffer),
              "vkEndCommandBuffer(public XFB)"))
      goto fail;

   if (!submit_and_wait(device, queue, fence, command_buffer,
                        "submit public XFB query"))
      goto fail;

   uint64_t results[2] = {UINT64_MAX, UINT64_MAX};
   if (!vk_ok(vkGetQueryPoolResults(device, query_pool, 0, 1, sizeof(results),
                                    results, sizeof(results),
                                    VK_QUERY_RESULT_64_BIT |
                                       VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults(public XFB)"))
      goto fail;
   if (results[0] != expected_written || results[1] != expected_needed ||
       *mapped_counter != expected_counter ||
       !xfb_bytes_match(xfb_mapped, capture_offsets,
                        expected_capture_count)) {
      fprintf(stderr,
              "Public XFB query mismatch: written=%llu needed=%llu "
              "counter=%u (expected %llu/%llu/%u)\n",
              (unsigned long long)results[0],
              (unsigned long long)results[1], *mapped_counter,
              (unsigned long long)expected_written,
              (unsigned long long)expected_needed, expected_counter);
      goto fail;
   }

   if (replay) {
      /* Same public command buffer, with a new counter value visible only at
       * execution. Its recorded reset/bind/begin/query/draw/end stream must be
       * replayed, and the second triangle must append. */
      *mapped_counter = expected_counter;
      if (!submit_and_wait(device, queue, fence, command_buffer,
                           "replay public XFB query"))
         goto fail;
      results[0] = results[1] = UINT64_MAX;
      if (!vk_ok(vkGetQueryPoolResults(device, query_pool, 0, 1,
                                       sizeof(results), results,
                                       sizeof(results),
                                       VK_QUERY_RESULT_64_BIT |
                                          VK_QUERY_RESULT_WAIT_BIT),
                 "vkGetQueryPoolResults(public XFB replay)") ||
          results[0] != expected_written || results[1] != expected_needed ||
          *mapped_counter != 2 * expected_counter ||
          !xfb_bytes_match(xfb_mapped,
                           (uint32_t[]){capture_offsets[0], 56}, 2)) {
         fprintf(stderr,
                 "Public XFB command-buffer replay mismatch: "
                 "written=%llu needed=%llu counter=%u\n",
                 (unsigned long long)results[0],
                 (unsigned long long)results[1], *mapped_counter);
         goto fail;
      }
      printf("PASS: public XFB command-buffer replay counter=%u\n",
             *mapped_counter);
   }

   printf("PASS: public XFB query written=%llu needed=%llu counter=%u\n",
          (unsigned long long)results[0], (unsigned long long)results[1],
          *mapped_counter);
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return true;

fail:
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return false;
}

static bool
run_public_xfb_pause_resume_case(
   VkDevice device, VkQueue queue, VkCommandPool command_pool, VkFence fence,
   VkPipeline pipeline, VkBuffer xfb_buffer, VkBuffer counter_buffer,
   uint8_t *xfb_mapped, uint32_t *mapped_counter,
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb,
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb,
   PFN_vkCmdEndTransformFeedbackEXT end_xfb)
{
   memset(xfb_mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 0;

   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer),
              "vkAllocateCommandBuffers(public XFB pause/resume)"))
      return false;

   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
              "vkBeginCommandBuffer(public XFB pause/resume)"))
      goto fail;

   const VkDeviceSize bind_offset = 8;
   const VkBuffer buffers[] = {xfb_buffer};
   const VkDeviceSize offsets[] = {bind_offset};
   const VkDeviceSize sizes[] = {120};
   const VkBuffer counter_buffers[] = {counter_buffer};
   const VkDeviceSize counter_offsets[] = {32};
   bind_xfb(command_buffer, 0, 1, buffers, offsets, sizes);

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {1, 1}},
      .layerCount = 1,
   };
   begin_xfb(command_buffer, 0, 0, NULL, NULL);
   vkCmdBeginRendering(command_buffer, &rendering_info);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRendering(command_buffer);
   end_xfb(command_buffer, 0, 1, counter_buffers, counter_offsets);

   VkBufferMemoryBarrier2 counter_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
      .srcAccessMask =
         VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
      .dstAccessMask = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = counter_buffer,
      .offset = counter_offsets[0],
      .size = sizeof(uint32_t),
   };
   VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &counter_barrier,
   };
   vkCmdPipelineBarrier2(command_buffer, &dependency);

   begin_xfb(command_buffer, 0, 1, counter_buffers, counter_offsets);
   vkCmdBeginRendering(command_buffer, &rendering_info);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRendering(command_buffer);
   end_xfb(command_buffer, 0, 1, counter_buffers, counter_offsets);

   if (!vk_ok(vkEndCommandBuffer(command_buffer),
              "vkEndCommandBuffer(public XFB pause/resume)"))
      goto fail;
   if (!submit_and_wait(device, queue, fence, command_buffer,
                        "submit public XFB pause/resume"))
      goto fail;

   if (*mapped_counter != 96 ||
       !xfb_bytes_match(xfb_mapped, (uint32_t[]){8, 56}, 2)) {
      fprintf(stderr,
              "Public XFB counter barrier pause/resume failed: counter=%u\n",
              *mapped_counter);
      goto fail;
   }

   printf("PASS: public XFB counter barrier pause/resume appended; counter=%u\n",
          *mapped_counter);
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return true;

fail:
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return false;
}

static bool
run_public_pg_query_case(VkDevice device, VkQueue queue,
                         VkCommandPool command_pool, VkFence fence,
                         VkPipeline pipeline, VkQueryPool query_pool,
                         VkPrimitiveTopology topology, uint32_t vertex_count,
                         uint32_t instances, uint32_t draw_count,
                         uint64_t expected,
                         bool replay)
{
   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer),
              "vkAllocateCommandBuffers(PGQ)"))
      return false;

   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   bool passed = vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
                       "vkBeginCommandBuffer(PGQ)");
   if (passed) {
      vkCmdResetQueryPool(command_buffer, query_pool, 0, 1);
      VkRenderingInfo rendering_info = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {1, 1}},
         .layerCount = 1,
      };
      vkCmdBeginRendering(command_buffer, &rendering_info);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline);
      vkCmdBeginQuery(command_buffer, query_pool, 0, 0);
      for (uint32_t i = 0; i < draw_count; i++)
         vkCmdDraw(command_buffer, vertex_count, instances, 0, 0);
      vkCmdEndQuery(command_buffer, query_pool, 0);
      vkCmdEndRendering(command_buffer);
      passed = vk_ok(vkEndCommandBuffer(command_buffer),
                     "vkEndCommandBuffer(PGQ)");
   }

   for (unsigned submission = 0; passed && submission < (replay ? 2u : 1u);
        ++submission) {
      passed = submit_and_wait(device, queue, fence, command_buffer,
                               "submit public PGQ") ;
      uint64_t result = UINT64_MAX;
      if (passed)
         passed = vk_ok(vkGetQueryPoolResults(
                           device, query_pool, 0, 1, sizeof(result), &result,
                           sizeof(result), VK_QUERY_RESULT_64_BIT |
                                              VK_QUERY_RESULT_WAIT_BIT),
                        "vkGetQueryPoolResults(PGQ)");
      if (passed && result != expected) {
         fprintf(stderr,
                 "PGQ topology %u vertices=%u instances=%u: expected %llu, "
                 "got %llu\n",
                 topology, vertex_count, instances,
                 (unsigned long long)expected, (unsigned long long)result);
         passed = false;
      }
   }

   if (passed)
      printf("PASS: public PGQ topology=%u vertices=%u instances=%u result=%llu%s\n",
             topology, vertex_count, instances,
             (unsigned long long)expected, replay ? " replayed" : "");
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return passed;
}

static VkPipeline
create_pg_test_pipeline(VkDevice device, VkPrimitiveTopology topology)
{
   VkShaderModule shader = VK_NULL_HANDLE;
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkPipeline pipeline = VK_NULL_HANDLE;
   VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(kk_xfb_capture_spv),
      .pCode = kk_xfb_capture_spv,
   };
   if (vkCreateShaderModule(device, &shader_info, NULL, &shader) != VK_SUCCESS)
      goto out;
   VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   if (vkCreatePipelineLayout(device, &layout_info, NULL, &layout) !=
       VK_SUCCESS)
      goto out;
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 0,
   };
   const VkPipelineShaderStageCreateInfo stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = shader,
      .pName = "main",
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology,
   };
   const VkViewport viewport = {
      .x = 0.0f, .y = 0.0f, .width = 1.0f, .height = 1.0f,
      .minDepth = 0.0f, .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {{0, 0}, {1, 1}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rasterization = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
   };
   VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering, .stageCount = 1, .pStages = &stage,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend, .layout = layout, .basePipelineIndex = -1,
   };
   if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 NULL, &pipeline) != VK_SUCCESS)
      pipeline = VK_NULL_HANDLE;
out:
   if (layout)
      vkDestroyPipelineLayout(device, layout, NULL);
   if (shader)
      vkDestroyShaderModule(device, shader, NULL);
   return pipeline;
}

static bool
run_public_pg_query_tests(VkDevice device, VkQueue queue,
                          VkCommandPool command_pool, VkFence fence)
{
   VkQueryPool query_pool = VK_NULL_HANDLE;
   VkQueryPoolCreateInfo query_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
      .queryCount = 1,
   };
   if (!vk_ok(vkCreateQueryPool(device, &query_info, NULL, &query_pool),
              "vkCreateQueryPool(primitives generated)"))
      return false;

   const struct {
      VkPrimitiveTopology topology;
      uint32_t vertex_count;
      uint64_t expected;
   } cases[] = {
      {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 5, 5},
      {VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 5, 2},
      {VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, 5, 4},
      {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 7, 2},
      {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, 5, 3},
      {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, 5, 3},
   };
   const unsigned case_count = sizeof(cases) / sizeof(cases[0]);
   VkPipeline pipelines[sizeof(cases) / sizeof(cases[0])] = {0};
   bool passed = true;
   for (unsigned i = 0; i < case_count; i++) {
      pipelines[i] = create_pg_test_pipeline(device, cases[i].topology);
      if (!pipelines[i]) {
         fprintf(stderr, "Could not create PGQ topology pipeline %u\n",
                 cases[i].topology);
         passed = false;
         break;
      }
      passed = run_public_pg_query_case(
         device, queue, command_pool, fence, pipelines[i], query_pool,
         cases[i].topology, cases[i].vertex_count, 1, 1, cases[i].expected,
         cases[i].topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
      if (!passed)
         break;
   }
   for (unsigned i = 0; i < case_count; i++)
      if (pipelines[i])
         vkDestroyPipeline(device, pipelines[i], NULL);
   if (passed) {
      VkPipeline triangle_pipeline = create_pg_test_pipeline(
         device, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
      passed = triangle_pipeline &&
         run_public_pg_query_case(
            device, queue, command_pool, fence, triangle_pipeline, query_pool,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, 1, 2, 2, false);
      if (passed)
         passed = run_public_pg_query_case(
            device, queue, command_pool, fence, triangle_pipeline, query_pool,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, 2, 1, 2, false);
      if (triangle_pipeline)
         vkDestroyPipeline(device, triangle_pipeline, NULL);
   }
   vkDestroyQueryPool(device, query_pool, NULL);
   return passed;
}

static bool
run_public_pg_xfb_independence_case(
   VkDevice device, VkQueue queue, VkCommandPool command_pool, VkFence fence,
   VkPipeline pipeline, VkBuffer xfb_buffer, VkBuffer counter_buffer,
   uint8_t *xfb_mapped, uint32_t *mapped_counter,
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb,
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb,
   PFN_vkCmdEndTransformFeedbackEXT end_xfb,
   PFN_vkCmdBeginQueryIndexedEXT begin_xfb_query,
   PFN_vkCmdEndQueryIndexedEXT end_xfb_query,
   VkQueryPool xfb_query_pool, VkQueryPool pg_query_pool)
{
   memset(xfb_mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
   *mapped_counter = 0;
   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer),
              "vkAllocateCommandBuffers(PGQ/XFB distinction)"))
      return false;
   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   bool passed = vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
                       "vkBeginCommandBuffer(PGQ/XFB distinction)");
   if (passed) {
      vkCmdResetQueryPool(command_buffer, xfb_query_pool, 0, 1);
      vkCmdResetQueryPool(command_buffer, pg_query_pool, 0, 1);
      const VkBuffer buffers[] = {xfb_buffer};
      const VkDeviceSize offsets[] = {8};
      const VkDeviceSize sizes[] = {47};
      bind_xfb(command_buffer, 0, 1, buffers, offsets, sizes);
      VkRenderingInfo rendering_info = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {1, 1}},
         .layerCount = 1,
      };
      const VkBuffer counter_buffers[] = {counter_buffer};
      const VkDeviceSize counter_offsets[] = {32};
      vkCmdBeginRendering(command_buffer, &rendering_info);
      begin_xfb(command_buffer, 0, 0, NULL, NULL);
      vkCmdBeginQuery(command_buffer, pg_query_pool, 0, 0);
      begin_xfb_query(command_buffer, xfb_query_pool, 0, 0, 0);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
      end_xfb_query(command_buffer, xfb_query_pool, 0, 0);
      vkCmdEndQuery(command_buffer, pg_query_pool, 0);
      end_xfb(command_buffer, 0, 1, counter_buffers, counter_offsets);
      vkCmdEndRendering(command_buffer);
      passed = vk_ok(vkEndCommandBuffer(command_buffer),
                     "vkEndCommandBuffer(PGQ/XFB distinction)");
   }
   if (passed)
      passed = submit_and_wait(device, queue, fence, command_buffer,
                               "submit PGQ/XFB distinction");
   uint64_t xfb_results[2] = {UINT64_MAX, UINT64_MAX};
   uint64_t pg_result = UINT64_MAX;
   if (passed)
      passed = vk_ok(vkGetQueryPoolResults(
                        device, xfb_query_pool, 0, 1, sizeof(xfb_results),
                        xfb_results, sizeof(xfb_results),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                     "read XFB distinction query") &&
               vk_ok(vkGetQueryPoolResults(
                        device, pg_query_pool, 0, 1, sizeof(pg_result),
                        &pg_result, sizeof(pg_result),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                     "read PGQ distinction query");
   if (passed && (pg_result != 1 || xfb_results[0] != 0 ||
                  xfb_results[1] != 1 || *mapped_counter != 0)) {
      fprintf(stderr,
              "PGQ/XFB capacity distinction mismatch: PGQ=%llu XFB=%llu/%llu "
              "counter=%u\n",
              (unsigned long long)pg_result,
              (unsigned long long)xfb_results[0],
              (unsigned long long)xfb_results[1], *mapped_counter);
      passed = false;
   }
   for (unsigned i = 0; passed && i < XFB_CAPTURE_BUFFER_SIZE; i++) {
      if (xfb_mapped[i] != CANARY_BYTE) {
         fprintf(stderr, "PGQ/XFB distinction unexpectedly wrote byte %u\n", i);
         passed = false;
      }
   }
   if (passed)
      printf("PASS: PGQ=1 while insufficient XFB capacity reports "
             "written=0 needed=1\n");
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return passed;
}

static bool
run_public_xfb_api_tests(VkDevice device, VkQueue queue,
                         VkCommandPool command_pool, VkFence fence,
                         VkPipeline pipeline, VkBuffer xfb_buffer,
                         VkBuffer counter_buffer,
                         uint8_t *xfb_mapped, uint32_t *mapped_counter)
{
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb =
      (PFN_vkCmdBindTransformFeedbackBuffersEXT)vkGetDeviceProcAddr(
         device, "vkCmdBindTransformFeedbackBuffersEXT");
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb =
      (PFN_vkCmdBeginTransformFeedbackEXT)vkGetDeviceProcAddr(
         device, "vkCmdBeginTransformFeedbackEXT");
   PFN_vkCmdEndTransformFeedbackEXT end_xfb =
      (PFN_vkCmdEndTransformFeedbackEXT)vkGetDeviceProcAddr(
         device, "vkCmdEndTransformFeedbackEXT");
   PFN_vkCmdBeginQueryIndexedEXT begin_xfb_query =
      (PFN_vkCmdBeginQueryIndexedEXT)vkGetDeviceProcAddr(
         device, "vkCmdBeginQueryIndexedEXT");
   PFN_vkCmdEndQueryIndexedEXT end_xfb_query =
      (PFN_vkCmdEndQueryIndexedEXT)vkGetDeviceProcAddr(
         device, "vkCmdEndQueryIndexedEXT");
   if (!bind_xfb || !begin_xfb || !end_xfb || !begin_xfb_query ||
       !end_xfb_query) {
      fprintf(stderr, "Public XFB commands unavailable through device proc addr\n");
      return false;
   }

   VkQueryPool query_pool = VK_NULL_HANDLE;
   VkQueryPoolCreateInfo query_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT,
      .queryCount = 1,
   };
   if (!vk_ok(vkCreateQueryPool(device, &query_info, NULL, &query_pool),
              "vkCreateQueryPool(transform feedback stream)"))
      return false;

   VkQueryPool pg_query_pool = VK_NULL_HANDLE;
   VkQueryPoolCreateInfo pg_query_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
      .queryCount = 1,
   };
   if (!vk_ok(vkCreateQueryPool(device, &pg_query_info, NULL, &pg_query_pool),
              "vkCreateQueryPool(primitives generated public)")) {
      vkDestroyQueryPool(device, query_pool, NULL);
      return false;
   }

   const uint32_t one_capture[] = {8};
   const uint32_t two_captures[] = {8, 56};
   bool passed =
      run_public_xfb_pause_resume_case(
         device, queue, command_pool, fence, pipeline, xfb_buffer,
         counter_buffer, xfb_mapped, mapped_counter, bind_xfb, begin_xfb,
         end_xfb) &&
      run_public_xfb_query_case(
         device, queue, command_pool, fence, pipeline, xfb_buffer,
         counter_buffer, xfb_mapped, mapped_counter, bind_xfb, begin_xfb,
         end_xfb, begin_xfb_query, end_xfb_query, query_pool, 64, 1, 1, 1,
         48, one_capture, 1, false) &&
      run_public_xfb_query_case(
         device, queue, command_pool, fence, pipeline, xfb_buffer,
         counter_buffer, xfb_mapped, mapped_counter, bind_xfb, begin_xfb,
         end_xfb, begin_xfb_query, end_xfb_query, query_pool, 47, 1, 0, 1, 0,
         NULL, 0, false) &&
      run_public_xfb_query_case(
         device, queue, command_pool, fence, pipeline, xfb_buffer,
         counter_buffer, xfb_mapped, mapped_counter, bind_xfb, begin_xfb,
         end_xfb, begin_xfb_query, end_xfb_query, query_pool, 120, 2, 2, 2,
         96, two_captures, 2, false) &&
      run_public_xfb_query_case(
         device, queue, command_pool, fence, pipeline, xfb_buffer,
         counter_buffer, xfb_mapped, mapped_counter, bind_xfb, begin_xfb,
         end_xfb, begin_xfb_query, end_xfb_query, query_pool, 120, 1, 1, 1,
         48, one_capture, 1, true) &&
      run_public_pg_xfb_independence_case(
         device, queue, command_pool, fence, pipeline, xfb_buffer,
         counter_buffer, xfb_mapped, mapped_counter, bind_xfb, begin_xfb,
         end_xfb, begin_xfb_query, end_xfb_query, query_pool, pg_query_pool) &&
      run_public_pg_query_tests(device, queue, command_pool, fence);

   vkDestroyQueryPool(device, pg_query_pool, NULL);
   vkDestroyQueryPool(device, query_pool, NULL);
   return passed;
}

static bool
run_xfb_capture_tests(VkDevice device, VkQueue queue,
                      VkCommandPool command_pool, VkFence fence,
                      VkBuffer xfb_buffer, VkBuffer counter_buffer,
                      VkDeviceAddress xfb_address,
                      VkDeviceAddress counter_address,
                      uint8_t *mapped, uint32_t *mapped_counter,
                      PFN_kk_test_set_xfb_capture_target set_target,
                      PFN_kk_test_begin_xfb begin_xfb,
                      PFN_kk_test_end_xfb end_xfb)
{
   VkShaderModule shader = VK_NULL_HANDLE;
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkPipeline pipeline = VK_NULL_HANDLE;
   VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(kk_xfb_capture_spv),
      .pCode = kk_xfb_capture_spv,
   };
   bool passed = false;
   if (!vk_ok(vkCreateShaderModule(device, &shader_info, NULL, &shader),
              "vkCreateShaderModule(XFB capture)"))
      goto out;

   VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   if (!vk_ok(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
              "vkCreatePipelineLayout(XFB capture)"))
      goto out;

   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 0,
   };
   const VkPipelineShaderStageCreateInfo stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = shader,
      .pName = "main",
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .x = 0.0f, .y = 0.0f, .width = 1.0f, .height = 1.0f,
      .minDepth = 0.0f, .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {{0, 0}, {1, 1}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rasterization = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
   };
   VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 1,
      .pStages = &stage,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .layout = layout,
      .basePipelineIndex = -1,
   };
   if (!vk_ok(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                        &pipeline_info, NULL, &pipeline),
              "vkCreateGraphicsPipelines(XFB capture)"))
      goto out;

   passed =
      run_xfb_capture_case(device, queue, command_pool, fence, pipeline,
                           set_target, begin_xfb, end_xfb, xfb_address,
                           counter_address, mapped, mapped_counter, 80, 3) &&
      run_xfb_capture_case(device, queue, command_pool, fence, pipeline,
                           set_target, begin_xfb, end_xfb, xfb_address,
                           counter_address, mapped, mapped_counter, 64, 3) &&
      run_xfb_capture_case(device, queue, command_pool, fence, pipeline,
                           set_target, begin_xfb, end_xfb, xfb_address,
                           counter_address, mapped, mapped_counter, 63, 0) &&
      run_xfb_capture_case(device, queue, command_pool, fence, pipeline,
                           set_target, begin_xfb, end_xfb, xfb_address,
                           counter_address, mapped, mapped_counter, 47, 0);

   if (passed) {
      /* Active lifecycle appends two draws; before begin and after end must
       * leave the same GPU buffer untouched. Binding offset remains non-zero.
       */
      VkCommandBuffer command_buffer = VK_NULL_HANDLE;
      VkCommandBufferAllocateInfo allocate_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = command_pool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      if (!vk_ok(vkAllocateCommandBuffers(device, &allocate_info,
                                          &command_buffer),
                 "vkAllocateCommandBuffers(XFB lifecycle)")) {
         passed = false;
      } else {
         memset(mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
         VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
         };
         bool recorded = vk_ok(vkBeginCommandBuffer(command_buffer,
                                                    &begin_info),
                               "vkBeginCommandBuffer(XFB lifecycle)");
         if (recorded)
            recorded = vk_ok(set_target(command_buffer, xfb_address, 128, 8, 0),
                             "bind XFB lifecycle target");

         VkRenderingInfo rendering_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {1, 1}},
            .layerCount = 1,
         };
         if (recorded) {
            vkCmdBeginRendering(command_buffer, &rendering_info);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline);
            vkCmdDraw(command_buffer, 3, 1, 0, 0); /* inactive */

            recorded = vk_ok(begin_xfb(command_buffer, 0, 0),
                              "begin XFB lifecycle");
            if (recorded && begin_xfb(command_buffer, 0, 0) !=
                                VK_ERROR_VALIDATION_FAILED) {
               fprintf(stderr, "nested XFB begin was not rejected\n");
               recorded = false;
            }
         }

         if (recorded) {
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
         }
         if (recorded) {
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
         }
         if (recorded)
            recorded = vk_ok(end_xfb(command_buffer, counter_address, 0),
                             "end XFB lifecycle");
         if (recorded) {
            vkCmdDraw(command_buffer, 3, 1, 0, 0); /* inactive */
            vkCmdEndRendering(command_buffer);
            recorded = vk_ok(vkEndCommandBuffer(command_buffer),
                             "vkEndCommandBuffer(XFB lifecycle)");
         }
         if (recorded &&
             (!vk_ok(vkResetFences(device, 1, &fence),
                     "vkResetFences(XFB lifecycle)") ||
              !vk_ok(vkQueueSubmit(queue, 1,
                                   &(VkSubmitInfo){
                                      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = &command_buffer,
                                   }, fence),
                     "vkQueueSubmit(XFB lifecycle)") ||
              !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                     "vkWaitForFences(XFB lifecycle)")))
            recorded = false;

         if (recorded) {
            if (*mapped_counter != 6 * 16) {
               fprintf(stderr, "XFB append saved counter: expected 96, got %u\n",
                       *mapped_counter);
               recorded = false;
            }
            bool bytes_match = true;
            for (uint32_t byte = 0; byte < XFB_CAPTURE_BUFFER_SIZE; byte++) {
               uint8_t expected = CANARY_BYTE;
               if (byte >= 8 && byte < 8 + 6 * 16) {
                  const uint32_t word = (byte - 8) / 4;
                  const uint32_t vertex = (word / 4) % 3;
                  const uint32_t component = word % 4;
                  const float value = (float)(vertex * 4 + component + 1);
                  uint32_t bits;
                  memcpy(&bits, &value, sizeof(bits));
                  expected = (bits >> (((byte - 8) % 4) * 8)) & 0xff;
               }
               if (mapped[byte] != expected) {
                  fprintf(stderr,
                          "XFB lifecycle byte %u: expected 0x%02x, got 0x%02x\n",
                          byte, expected, mapped[byte]);
                  bytes_match = false;
               }
            }
            recorded = recorded && bytes_match;
         }

         if (recorded)
            printf("PASS: XFB begin/end gating, two-draw append, counters, and guards verified\n");
         else
            passed = false;
         vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
      }
   }

   if (passed) {
      /* A full first triangle consumes all whole-primitive capacity. The
       * second draw has room for one record only and must neither write nor
       * advance the byte counter.
       */
      VkCommandBuffer command_buffer = VK_NULL_HANDLE;
      VkCommandBufferAllocateInfo allocate_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = command_pool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      if (!vk_ok(vkAllocateCommandBuffers(device, &allocate_info,
                                          &command_buffer),
                 "vkAllocateCommandBuffers(XFB capacity lifecycle)")) {
         passed = false;
      } else {
         memset(mapped, CANARY_BYTE, XFB_CAPTURE_BUFFER_SIZE);
         VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
         };
         bool recorded = vk_ok(vkBeginCommandBuffer(command_buffer,
                                                    &begin_info),
                               "vkBeginCommandBuffer(XFB capacity lifecycle)");
         if (recorded)
            recorded = vk_ok(set_target(command_buffer, xfb_address, 80, 8, 0),
                             "bind XFB capacity lifecycle target");

         VkRenderingInfo rendering_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {1, 1}},
            .layerCount = 1,
         };
         if (recorded) {
            vkCmdBeginRendering(command_buffer, &rendering_info);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline);
            recorded = vk_ok(begin_xfb(command_buffer, 0, 0),
                              "begin XFB capacity lifecycle");
         }

         if (recorded) {
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
         }
         if (recorded) {
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
         }
         if (recorded)
            recorded = vk_ok(end_xfb(command_buffer, counter_address, 0),
                             "end XFB capacity lifecycle");
         if (recorded) {
            vkCmdEndRendering(command_buffer);
            recorded = vk_ok(vkEndCommandBuffer(command_buffer),
                             "vkEndCommandBuffer(XFB capacity lifecycle)");
         }
         if (recorded &&
             (!vk_ok(vkResetFences(device, 1, &fence),
                     "vkResetFences(XFB capacity lifecycle)") ||
              !vk_ok(vkQueueSubmit(queue, 1,
                                   &(VkSubmitInfo){
                                      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = &command_buffer,
                                   }, fence),
                     "vkQueueSubmit(XFB capacity lifecycle)") ||
              !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                     "vkWaitForFences(XFB capacity lifecycle)")))
            recorded = false;

         if (recorded) {
            if (*mapped_counter != 3 * 16) {
               fprintf(stderr,
                       "XFB insufficient-capacity saved counter: expected 48, "
                       "got %u\n", *mapped_counter);
               recorded = false;
            }
            bool bytes_match = true;
            for (uint32_t byte = 0; byte < XFB_CAPTURE_BUFFER_SIZE; byte++) {
               uint8_t expected = CANARY_BYTE;
               if (byte >= 8 && byte < 8 + 3 * 16) {
                  const uint32_t word = (byte - 8) / 4;
                  const uint32_t vertex = word / 4;
                  const uint32_t component = word % 4;
                  const float value = (float)(vertex * 4 + component + 1);
                  uint32_t bits;
                  memcpy(&bits, &value, sizeof(bits));
                  expected = (bits >> (((byte - 8) % 4) * 8)) & 0xff;
               }
               if (mapped[byte] != expected) {
                  fprintf(stderr,
                          "XFB capacity lifecycle byte %u: expected 0x%02x, "
                          "got 0x%02x\n",
                          byte, expected, mapped[byte]);
                  bytes_match = false;
               }
            }
            recorded = recorded && bytes_match;
         }

         if (recorded)
            printf("PASS: XFB insufficient-capacity draw writes no partial "
                   "primitive and leaves counter unchanged\n");
         else
            passed = false;
         vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
      }
   }

   if (passed)
      passed = run_xfb_save_resume_test(
                  device, queue, command_pool, fence, pipeline, set_target,
                  begin_xfb, end_xfb, xfb_address, counter_address, mapped,
                  mapped_counter) &&
               run_xfb_command_buffer_replay_test(
                  device, queue, command_pool, fence, pipeline, set_target,
                  begin_xfb, end_xfb, xfb_address, counter_address, mapped,
                  mapped_counter) &&
               run_xfb_execution_time_read_test(
                  device, queue, command_pool, fence, pipeline, set_target,
                  begin_xfb, end_xfb, xfb_address, counter_address, mapped,
                  mapped_counter) &&
               run_public_xfb_api_tests(
                  device, queue, command_pool, fence, pipeline, xfb_buffer,
                  counter_buffer, mapped, mapped_counter);

out:
   if (pipeline)
      vkDestroyPipeline(device, pipeline, NULL);
   if (layout)
      vkDestroyPipelineLayout(device, layout, NULL);
   if (shader)
      vkDestroyShaderModule(device, shader, NULL);
   return passed;
}

static bool
vk_ok(VkResult result, const char *what)
{
   if (result == VK_SUCCESS)
      return true;
   fprintf(stderr, "%s failed: VkResult %d\n", what, result);
   return false;
}

static bool
has_device_extension(VkPhysicalDevice physical, const char *name)
{
   uint32_t count = 0;
   if (!vk_ok(vkEnumerateDeviceExtensionProperties(physical, NULL, &count,
                                                   NULL),
              "vkEnumerateDeviceExtensionProperties(count)"))
      return false;
   VkExtensionProperties *extensions = calloc(count, sizeof(*extensions));
   if (!extensions)
      return false;
   bool found = false;
   if (vk_ok(vkEnumerateDeviceExtensionProperties(physical, NULL, &count,
                                                  extensions),
             "vkEnumerateDeviceExtensionProperties")) {
      for (uint32_t i = 0; i < count; i++)
         found |= strcmp(extensions[i].extensionName, name) == 0;
   }
   free(extensions);
   return found;
}

static bool
has_name(const char *haystack, const char *needle)
{
   return haystack && strstr(haystack, needle) != NULL;
}

static bool
run_xfb_abi_gpu_case(VkDevice device, VkQueue queue,
                     VkCommandPool command_pool, VkFence fence,
                     VkDeviceAddress xfb_address,
                     VkDeviceAddress diagnostic_address, uint8_t *mapped,
                     PFN_kk_test_record_xfb_abi_probe record_probe,
                     bool enabled, uint64_t expected_address,
                     uint32_t expected_size)
{
   VkCommandBuffer command_buffer = VK_NULL_HANDLE;
   VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &allocate_info,
                                       &command_buffer),
              "vkAllocateCommandBuffers(XFB ABI)"))
      return false;

   memset(mapped, CANARY_BYTE, BUFFER_SIZE);
   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   bool passed = false;
   if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
              "vkBeginCommandBuffer(XFB ABI)"))
      goto out;

   VkResult result = record_probe(
      command_buffer, enabled, xfb_address, 56, 8, 12, diagnostic_address,
      BUFFER_SIZE);
   if (!vk_ok(result, enabled ? "record enabled XFB ABI probe"
                              : "record disabled XFB ABI probe") ||
       !vk_ok(vkEndCommandBuffer(command_buffer),
              "vkEndCommandBuffer(XFB ABI)"))
      goto out;

   if (!vk_ok(vkResetFences(device, 1, &fence), "vkResetFences(XFB ABI)"))
      goto out;
   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   if (!vk_ok(vkQueueSubmit(queue, 1, &submit_info, fence),
              "vkQueueSubmit(XFB ABI)") ||
       !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences(XFB ABI)"))
      goto out;

   uint32_t words[3];
   memcpy(words, mapped, sizeof(words));
   const uint64_t actual_address = (uint64_t)words[0] |
                                   ((uint64_t)words[1] << 32);
   if (actual_address != expected_address || words[2] != expected_size) {
      fprintf(stderr,
              "XFB ABI %s: address expected 0x%016llx got 0x%016llx; "
              "size expected %u got %u\n",
              enabled ? "enabled" : "disabled",
              (unsigned long long)expected_address,
              (unsigned long long)actual_address, expected_size, words[2]);
      goto out;
   }

   printf("PASS: XFB ABI %s GPU readback address=0x%016llx size=%u\n",
          enabled ? "enabled" : "disabled",
          (unsigned long long)actual_address, words[2]);
   passed = true;

out:
   vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
   return passed;
}

static void
destroy_test_objects(VkInstance instance, VkDevice device, VkBuffer buffer,
                     VkDeviceMemory memory, VkBuffer xfb_buffer,
                     VkDeviceMemory xfb_memory,
                     VkDescriptorPool descriptor_pool,
                     VkDescriptorSetLayout descriptor_layout,
                     VkPipelineLayout pipeline_layout, VkPipeline pipeline,
                     VkShaderModule shader, VkCommandPool command_pool,
                     VkFence fence)
{
   if (device) {
      if (fence)
         vkDestroyFence(device, fence, NULL);
      if (command_pool)
         vkDestroyCommandPool(device, command_pool, NULL);
      if (pipeline)
         vkDestroyPipeline(device, pipeline, NULL);
      if (shader)
         vkDestroyShaderModule(device, shader, NULL);
      if (pipeline_layout)
         vkDestroyPipelineLayout(device, pipeline_layout, NULL);
      if (descriptor_pool)
         vkDestroyDescriptorPool(device, descriptor_pool, NULL);
      if (descriptor_layout)
         vkDestroyDescriptorSetLayout(device, descriptor_layout, NULL);
      if (buffer)
         vkDestroyBuffer(device, buffer, NULL);
      if (memory)
         vkFreeMemory(device, memory, NULL);
      if (xfb_buffer)
         vkDestroyBuffer(device, xfb_buffer, NULL);
      if (xfb_memory)
         vkFreeMemory(device, xfb_memory, NULL);
      vkDestroyDevice(device, NULL);
   }
   if (instance)
      vkDestroyInstance(instance, NULL);
}

int
main(void)
{
   VkInstance instance = VK_NULL_HANDLE;
   VkDevice device = VK_NULL_HANDLE;
   VkBuffer buffer = VK_NULL_HANDLE;
   VkDeviceMemory memory = VK_NULL_HANDLE;
   VkBuffer xfb_buffer = VK_NULL_HANDLE;
   VkDeviceMemory xfb_memory = VK_NULL_HANDLE;
   VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
   VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
   VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
   VkPipeline pipeline = VK_NULL_HANDLE;
   VkShaderModule shader = VK_NULL_HANDLE;
   VkCommandPool command_pool = VK_NULL_HANDLE;
   VkFence fence = VK_NULL_HANDLE;
   VkDeviceAddress buffer_address = 0;
   VkDeviceAddress xfb_address = 0;
   void *driver_library = NULL;
   uint8_t *mapped = NULL;
   uint8_t *xfb_mapped = NULL;
   int status = EXIT_FAILURE;

   VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "KosmicKrisp GPU write test",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };
   if (!vk_ok(vkCreateInstance(&instance_info, NULL, &instance),
              "vkCreateInstance"))
      goto out;

   uint32_t physical_count = 0;
   if (!vk_ok(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
              "vkEnumeratePhysicalDevices(count)"))
      goto out;
   if (!physical_count) {
      fprintf(stderr, "No Vulkan physical devices found\n");
      goto out;
   }

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      goto out;
   if (!vk_ok(vkEnumeratePhysicalDevices(instance, &physical_count,
                                         physical_devices),
              "vkEnumeratePhysicalDevices")) {
      free(physical_devices);
      goto out;
   }

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   VkPhysicalDeviceProperties physical_properties = {0};
   for (uint32_t i = 0; i < physical_count; ++i) {
      VkPhysicalDeviceDriverProperties driver_properties = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      };
      VkPhysicalDeviceProperties2 properties = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &driver_properties,
      };
      vkGetPhysicalDeviceProperties2(physical_devices[i], &properties);
      if (driver_properties.driverID == VK_DRIVER_ID_MESA_KOSMICKRISP &&
          has_name(driver_properties.driverName, "KosmicKrisp")) {
         physical = physical_devices[i];
         physical_properties = properties.properties;
         break;
      }
   }
   free(physical_devices);
   if (!physical) {
      fprintf(stderr, "Mesa KosmicKrisp physical device not found; refusing to test another driver\n");
      goto out;
   }
   printf("Using %s on %s\n", "KosmicKrisp", physical_properties.deviceName);

   if (!has_device_extension(physical, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME)) {
      fprintf(stderr, "VK_EXT_transform_feedback is not enumerated\n");
      goto out;
   }
   if (!has_device_extension(
          physical, VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME)) {
      fprintf(stderr, "VK_EXT_primitives_generated_query is not enumerated\n");
      goto out;
   }

   VkPhysicalDeviceTransformFeedbackFeaturesEXT transform_feedback_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
   };
   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pg_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
   };
   VkPhysicalDeviceTransformFeedbackPropertiesEXT transform_feedback_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &transform_feedback_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);
   VkPhysicalDeviceFeatures2 xfb_features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &transform_feedback_features,
   };
   transform_feedback_features.pNext = &pg_features;
   vkGetPhysicalDeviceFeatures2(physical, &xfb_features2);
   if (!transform_feedback_features.transformFeedback ||
       !pg_features.primitivesGeneratedQuery ||
       pg_features.primitivesGeneratedQueryWithRasterizerDiscard ||
       pg_features.primitivesGeneratedQueryWithNonZeroStreams ||
       transform_feedback_features.geometryStreams ||
       !transform_feedback_properties.transformFeedbackQueries ||
       transform_feedback_properties.maxTransformFeedbackStreams != 1 ||
       transform_feedback_properties.maxTransformFeedbackBuffers != 1 ||
       transform_feedback_properties.transformFeedbackStreamsLinesTriangles ||
       transform_feedback_properties.transformFeedbackRasterizationStreamSelect ||
       transform_feedback_properties.transformFeedbackDraw) {
      fprintf(stderr,
              "KosmicKrisp XFB feature/property contract is not conservative\n");
      goto out;
   }
   printf("PASS: host VK_EXT_transform_feedback=1 "
          "transformFeedback=%u geometryStreams=%u maxStreams=%u "
          "maxBuffers=%u transformFeedbackQueries=%u\n",
          transform_feedback_features.transformFeedback,
          transform_feedback_features.geometryStreams,
          transform_feedback_properties.maxTransformFeedbackStreams,
          transform_feedback_properties.maxTransformFeedbackBuffers,
          transform_feedback_properties.transformFeedbackQueries);
   printf("PASS: host VK_EXT_primitives_generated_query=1 "
          "primitivesGeneratedQuery=%u rasterizerDiscard=%u nonZeroStreams=%u "
          "pipelineStatisticsQuery=%u\n",
          pg_features.primitivesGeneratedQuery,
          pg_features.primitivesGeneratedQueryWithRasterizerDiscard,
          pg_features.primitivesGeneratedQueryWithNonZeroStreams,
          xfb_features2.features.pipelineStatisticsQuery);

   uint32_t queue_family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                             NULL);
   VkQueueFamilyProperties *queue_families =
      calloc(queue_family_count, sizeof(*queue_families));
   if (!queue_families)
      goto out;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                             queue_families);
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < queue_family_count; ++i) {
      const VkQueueFlags test_queue_flags =
         VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
      if (queue_families[i].queueCount &&
          (queue_families[i].queueFlags & test_queue_flags) ==
             test_queue_flags) {
         queue_family = i;
         break;
      }
   }
   free(queue_families);
   if (queue_family == UINT32_MAX) {
      fprintf(stderr, "KosmicKrisp has no graphics+compute queue\n");
      goto out;
   }

   const float queue_priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
   VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
   };
   VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
   };
   transform_feedback_features.pNext = &pg_features;
   pg_features.pNext = &bda_features;
   bda_features.pNext = &dynamic_rendering;
   VkPhysicalDeviceFeatures2 available_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &transform_feedback_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available_features);
   if (!bda_features.bufferDeviceAddress) {
      fprintf(stderr, "KosmicKrisp does not provide bufferDeviceAddress\n");
      goto out;
   }
   if (!dynamic_rendering.dynamicRendering) {
      fprintf(stderr, "KosmicKrisp does not provide dynamicRendering\n");
      goto out;
   }
   bda_features.bufferDeviceAddress = VK_TRUE;
   dynamic_rendering.dynamicRendering = VK_TRUE;
   transform_feedback_features.transformFeedback = VK_TRUE;
   transform_feedback_features.geometryStreams = VK_FALSE;
   pg_features.primitivesGeneratedQuery = VK_TRUE;
   pg_features.primitivesGeneratedQueryWithRasterizerDiscard = VK_FALSE;
   pg_features.primitivesGeneratedQueryWithNonZeroStreams = VK_FALSE;
   const char *device_extensions[] = {
      VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
      VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME,
   };
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &transform_feedback_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = device_extensions,
   };
   if (!vk_ok(vkCreateDevice(physical, &device_info, NULL, &device),
              "vkCreateDevice"))
      goto out;
   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = BUFFER_SIZE,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (!vk_ok(vkCreateBuffer(device, &buffer_info, NULL, &buffer),
              "vkCreateBuffer"))
      goto out;

   VkMemoryRequirements memory_requirements;
   vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   uint32_t memory_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
      if ((memory_requirements.memoryTypeBits & (1u << i)) &&
          (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         memory_type = i;
         break;
      }
   }
   if (memory_type == UINT32_MAX) {
      fprintf(stderr, "No host-visible coherent memory type for test buffer\n");
      goto out;
   }
   VkMemoryAllocateFlagsInfo address_flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &address_flags,
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex = memory_type,
   };
   if (!vk_ok(vkAllocateMemory(device, &memory_info, NULL, &memory),
              "vkAllocateMemory") ||
       !vk_ok(vkBindBufferMemory(device, buffer, memory, 0),
              "vkBindBufferMemory"))
      goto out;

   VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = buffer,
   };
   buffer_address = vkGetBufferDeviceAddress(device, &address_info);
   if (!buffer_address || buffer_address <= UINT32_MAX) {
      fprintf(stderr,
              "Expected a real >32-bit KosmicKrisp buffer address, got "
              "0x%016llx\n", (unsigned long long)buffer_address);
      goto out;
   }

   VkBufferCreateInfo xfb_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = XFB_CAPTURE_BUFFER_SIZE,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (!vk_ok(vkCreateBuffer(device, &xfb_buffer_info, NULL, &xfb_buffer),
              "vkCreateBuffer(XFB source)"))
      goto out;
   VkMemoryRequirements xfb_memory_requirements;
   vkGetBufferMemoryRequirements(device, xfb_buffer, &xfb_memory_requirements);
   uint32_t xfb_memory_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
      if ((xfb_memory_requirements.memoryTypeBits & (1u << i)) &&
          (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         xfb_memory_type = i;
         break;
      }
   }
   if (xfb_memory_type == UINT32_MAX) {
      fprintf(stderr, "No host-visible coherent XFB capture memory type\n");
      goto out;
   }
   VkMemoryAllocateInfo xfb_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &address_flags,
      .allocationSize = xfb_memory_requirements.size,
      .memoryTypeIndex = xfb_memory_type,
   };
   if (!vk_ok(vkAllocateMemory(device, &xfb_memory_info, NULL, &xfb_memory),
              "vkAllocateMemory(XFB source)") ||
       !vk_ok(vkBindBufferMemory(device, xfb_buffer, xfb_memory, 0),
              "vkBindBufferMemory(XFB source)"))
      goto out;
   VkBufferDeviceAddressInfo xfb_address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = xfb_buffer,
   };
   xfb_address = vkGetBufferDeviceAddress(device, &xfb_address_info);
   if (!xfb_address || xfb_address <= UINT32_MAX ||
       xfb_address == buffer_address) {
      fprintf(stderr,
              "Expected a distinct real >32-bit XFB source address, got "
              "0x%016llx (diagnostic 0x%016llx)\n",
              (unsigned long long)xfb_address,
              (unsigned long long)buffer_address);
      goto out;
   }

   if (!vk_ok(vkMapMemory(device, memory, 0, BUFFER_SIZE, 0,
                          (void **)&mapped), "vkMapMemory"))
      goto out;
   if (!vk_ok(vkMapMemory(device, xfb_memory, 0, XFB_CAPTURE_BUFFER_SIZE, 0,
                          (void **)&xfb_mapped),
              "vkMapMemory(XFB capture)"))
      goto out;
   memset(mapped, CANARY_BYTE, BUFFER_SIZE);

   VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   if (!vk_ok(vkCreateDescriptorSetLayout(device, &descriptor_layout_info,
                                          NULL, &descriptor_layout),
              "vkCreateDescriptorSetLayout"))
      goto out;

   VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptor_layout,
   };
   if (!vk_ok(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
                                     &pipeline_layout),
              "vkCreatePipelineLayout"))
      goto out;

   VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(kk_gpu_write_test_spv),
      .pCode = kk_gpu_write_test_spv,
   };
   if (!vk_ok(vkCreateShaderModule(device, &shader_info, NULL, &shader),
              "vkCreateShaderModule"))
      goto out;
   VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = shader,
         .pName = "main",
      },
      .layout = pipeline_layout,
   };
   if (!vk_ok(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &pipeline),
              "vkCreateComputePipelines"))
      goto out;

   VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
   };
   VkDescriptorPoolCreateInfo descriptor_pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   if (!vk_ok(vkCreateDescriptorPool(device, &descriptor_pool_info, NULL,
                                     &descriptor_pool),
              "vkCreateDescriptorPool"))
      goto out;
   VkDescriptorSet descriptor_set;
   VkDescriptorSetAllocateInfo descriptor_set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &descriptor_layout,
   };
   if (!vk_ok(vkAllocateDescriptorSets(device, &descriptor_set_info,
                                       &descriptor_set),
              "vkAllocateDescriptorSets"))
      goto out;
   VkDescriptorBufferInfo descriptor_buffer_info = {
      .buffer = buffer,
      .offset = 0,
      .range = BUFFER_SIZE,
   };
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor_buffer_info,
   };
   vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

   VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   if (!vk_ok(vkCreateCommandPool(device, &command_pool_info, NULL,
                                  &command_pool),
              "vkCreateCommandPool"))
      goto out;
   VkCommandBuffer command_buffer;
   VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &command_buffer_info,
                                       &command_buffer),
              "vkAllocateCommandBuffers"))
      goto out;
   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (!vk_ok(vkBeginCommandBuffer(command_buffer, &begin_info),
              "vkBeginCommandBuffer"))
      goto out;
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   vkCmdDispatch(command_buffer, 1, 1, 1);
   if (!vk_ok(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer"))
      goto out;

   VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   if (!vk_ok(vkCreateFence(device, &fence_info, NULL, &fence),
              "vkCreateFence"))
      goto out;
   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   if (!vk_ok(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit") ||
       !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences"))
      goto out;

   const struct {
      uint32_t word;
      uint32_t value;
   } expected_writes[] = {
      {4, 0x11223344u},
      {8, 0x55667788u},
      {12, 0x99aabbccu},
   };
   bool passed = true;
   for (uint32_t byte = 0; byte < BUFFER_SIZE; ++byte) {
      uint8_t expected = CANARY_BYTE;
      for (uint32_t i = 0;
           i < sizeof(expected_writes) / sizeof(expected_writes[0]); ++i) {
         uint32_t start = expected_writes[i].word * sizeof(uint32_t);
         if (byte >= start && byte < start + sizeof(uint32_t))
            expected = (expected_writes[i].value >> ((byte - start) * 8)) & 0xff;
      }
      if (mapped[byte] != expected) {
         fprintf(stderr, "byte %u: expected 0x%02x, got 0x%02x\n", byte,
                 expected, mapped[byte]);
         passed = false;
      }
   }
   if (!passed)
      goto out;

   printf("PASS: GPU writes, fence synchronization, readback, and canaries verified\n");

   const char *driver_path = getenv("VIMA_KK_DRIVER_LIBRARY");
   if (!driver_path || !*driver_path) {
      fprintf(stderr, "Set VIMA_KK_DRIVER_LIBRARY to the KosmicKrisp driver dylib\n");
      goto out;
   }
   driver_library = dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
   if (!driver_library) {
      fprintf(stderr, "dlopen(%s) failed: %s\n", driver_path, dlerror());
      goto out;
   }
   PFN_kk_test_record_xfb_abi_probe record_probe =
      (PFN_kk_test_record_xfb_abi_probe)dlsym(
         driver_library, "kk_test_record_xfb_abi_probe");
   PFN_kk_test_set_xfb_capture_target set_xfb_target =
      (PFN_kk_test_set_xfb_capture_target)dlsym(
         driver_library, "kk_test_set_xfb_capture_target");
   PFN_kk_test_begin_xfb begin_xfb =
      (PFN_kk_test_begin_xfb)dlsym(driver_library, "kk_test_begin_xfb");
   PFN_kk_test_end_xfb end_xfb =
      (PFN_kk_test_end_xfb)dlsym(driver_library, "kk_test_end_xfb");
   if (!record_probe) {
      fprintf(stderr, "KosmicKrisp XFB ABI test hook not exported: %s\n",
              dlerror());
      goto out;
   }
   if (!set_xfb_target) {
      fprintf(stderr, "KosmicKrisp XFB capture test hook not exported: %s\n",
              dlerror());
      goto out;
   }
   if (!begin_xfb || !end_xfb) {
      fprintf(stderr, "KosmicKrisp XFB lifecycle hooks not exported: %s\n",
              dlerror());
      goto out;
   }

   const uint64_t expected_address = xfb_address + 8 + 12;
   const uint32_t expected_size = 56 - 8 - 12;
   if (!run_xfb_abi_gpu_case(
          device, queue, command_pool, fence, xfb_address, buffer_address,
          mapped, record_probe, true, expected_address, expected_size) ||
       !run_xfb_abi_gpu_case(device, queue, command_pool, fence, xfb_address,
                             buffer_address, mapped, record_probe, false, 0,
                             0))
      goto out;

   VkDeviceAddress counter_address = buffer_address + 32;
   uint32_t *mapped_counter = (uint32_t *)(mapped + 32);
   if (!run_xfb_capture_tests(device, queue, command_pool, fence, xfb_buffer,
                              buffer, xfb_address, counter_address,
                              xfb_mapped, mapped_counter,
                              set_xfb_target, begin_xfb, end_xfb))
      goto out;

   status = EXIT_SUCCESS;

out:
   if (mapped && device && memory)
      vkUnmapMemory(device, memory);
   if (xfb_mapped && device && xfb_memory)
      vkUnmapMemory(device, xfb_memory);
   destroy_test_objects(instance, device, buffer, memory, xfb_buffer,
                        xfb_memory, descriptor_pool, descriptor_layout,
                        pipeline_layout, pipeline, shader,
                        command_pool, fence);
   if (driver_library)
      dlclose(driver_library);
   return status;
}
