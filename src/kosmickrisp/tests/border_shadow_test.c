/*
 * Copyright 2026 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 */

/*
 * Shadow (depth-compare) samplers with a border colour.
 *
 * A depth image is cleared to 0.5 and sampled through a CLAMP_TO_BORDER shadow sampler, from
 * inside the image and from outside it, so that the reference depth is compared once with a
 * texel and once with the border. The expected result is computed on the CPU from the Vulkan
 * rules: compareOp(Dref, D) with D the texel or the border value. Both D32_SFLOAT and D16_UNORM
 * images are covered, with a custom border colour and with the three standard ones.
 *
 * Run with VIMA_KK_DRIVER_LIBRARY unset or set, VK_DRIVER_FILES pointing at the KosmicKrisp ICD
 * and MESA_KK_EXPERIMENTAL=custom_border, which is what exposes VK_EXT_custom_border_color.
 */

#include <vulkan/vulkan.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "border_shadow_spv.h"

#define IMAGE_SIZE 4
#define DEPTH_VALUE 0.5f
#define MAX_CASES 64

static bool
vk_ok(VkResult result, const char *what)
{
   if (result == VK_SUCCESS)
      return true;
   fprintf(stderr, "FAIL: %s returned VkResult %d\n", what, (int)result);
   return false;
}

static uint32_t
find_memory(const VkPhysicalDeviceMemoryProperties *props, uint32_t bits,
            VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < props->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (props->memoryTypes[i].propertyFlags & want) == want)
         return i;
   }
   return UINT32_MAX;
}

struct border_config {
   const char *name;
   VkBorderColor color;
   float custom; /* the depth the border stands for */
   VkCompareOp op;
};

static float
border_depth(const struct border_config *config)
{
   switch (config->color) {
   case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
      return 1.0f;
   case VK_BORDER_COLOR_FLOAT_CUSTOM_EXT:
      return config->custom;
   default:
      return 0.0f; /* both black variants are depth 0 */
   }
}

static bool
compare(VkCompareOp op, float ref, float depth)
{
   switch (op) {
   case VK_COMPARE_OP_NEVER: return false;
   case VK_COMPARE_OP_LESS: return ref < depth;
   case VK_COMPARE_OP_EQUAL: return ref == depth;
   case VK_COMPARE_OP_LESS_OR_EQUAL: return ref <= depth;
   case VK_COMPARE_OP_GREATER: return ref > depth;
   case VK_COMPARE_OP_NOT_EQUAL: return ref != depth;
   case VK_COMPARE_OP_GREATER_OR_EQUAL: return ref >= depth;
   case VK_COMPARE_OP_ALWAYS: return true;
   default: return false;
   }
}

static const char *
op_name(VkCompareOp op)
{
   switch (op) {
   case VK_COMPARE_OP_LESS: return "LESS";
   case VK_COMPARE_OP_LESS_OR_EQUAL: return "LESS_OR_EQUAL";
   case VK_COMPARE_OP_GREATER: return "GREATER";
   case VK_COMPARE_OP_GREATER_OR_EQUAL: return "GREATER_OR_EQUAL";
   default: return "OTHER";
   }
}

int
main(void)
{
   int status = EXIT_FAILURE;
   VkInstance instance = VK_NULL_HANDLE;
   VkDevice device = VK_NULL_HANDLE;
   VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
   VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
   VkShaderModule shader = VK_NULL_HANDLE;
   VkPipeline pipeline = VK_NULL_HANDLE;
   VkDescriptorPool pool = VK_NULL_HANDLE;
   VkCommandPool command_pool = VK_NULL_HANDLE;
   VkBuffer cases_buffer = VK_NULL_HANDLE, results_buffer = VK_NULL_HANDLE;
   VkDeviceMemory cases_memory = VK_NULL_HANDLE, results_memory = VK_NULL_HANDLE;
   VkFence fence = VK_NULL_HANDLE;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "KosmicKrisp border shadow test",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   if (!vk_ok(vkCreateInstance(&instance_info, NULL, &instance), "vkCreateInstance"))
      goto out;

   uint32_t count = 0;
   vkEnumeratePhysicalDevices(instance, &count, NULL);
   VkPhysicalDevice *devices = calloc(count ? count : 1, sizeof(*devices));
   vkEnumeratePhysicalDevices(instance, &count, devices);
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceDriverProperties driver = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      };
      VkPhysicalDeviceProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &driver,
      };
      vkGetPhysicalDeviceProperties2(devices[i], &props);
      if (driver.driverID == VK_DRIVER_ID_MESA_KOSMICKRISP) {
         physical = devices[i];
         break;
      }
   }
   free(devices);
   if (!physical) {
      fprintf(stderr, "FAIL: no Mesa KosmicKrisp device; refusing to test another driver\n");
      goto out;
   }

   VkPhysicalDeviceCustomBorderColorFeaturesEXT custom_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &custom_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!custom_features.customBorderColors) {
      fprintf(stderr,
              "FAIL: VK_EXT_custom_border_color is not exposed; set MESA_KK_EXPERIMENTAL=custom_border\n");
      goto out;
   }
   const bool without_format = custom_features.customBorderColorWithoutFormat;
   custom_features.customBorderColors = VK_TRUE;

   VkPhysicalDeviceMemoryProperties memory_props;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_props);

   const float priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *extensions[] = {VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME};
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &custom_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   if (!vk_ok(vkCreateDevice(physical, &device_info, NULL, &device), "vkCreateDevice"))
      goto out;
   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);

   /* One shadow sampler, one buffer of cases in, one of results out. */
   VkDescriptorSetLayoutBinding bindings[3] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   VkDescriptorSetLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = bindings,
   };
   if (!vk_ok(vkCreateDescriptorSetLayout(device, &layout_info, NULL, &set_layout),
              "vkCreateDescriptorSetLayout"))
      goto out;
   VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   if (!vk_ok(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout),
              "vkCreatePipelineLayout"))
      goto out;
   VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(kk_border_shadow_test_spv),
      .pCode = kk_border_shadow_test_spv,
   };
   if (!vk_ok(vkCreateShaderModule(device, &shader_info, NULL, &shader),
              "vkCreateShaderModule"))
      goto out;
   VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main"},
      .layout = pipeline_layout,
   };
   if (!vk_ok(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL,
                                       &pipeline),
              "vkCreateComputePipelines"))
      goto out;

   VkDescriptorPoolSize pool_sizes[2] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
   };
   VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   if (!vk_ok(vkCreateDescriptorPool(device, &pool_info, NULL, &pool),
              "vkCreateDescriptorPool"))
      goto out;
   VkDescriptorSet set;
   VkDescriptorSetAllocateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   if (!vk_ok(vkAllocateDescriptorSets(device, &set_info, &set), "vkAllocateDescriptorSets"))
      goto out;

   VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = 0,
   };
   if (!vk_ok(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
              "vkCreateCommandPool"))
      goto out;
   VkCommandBuffer cmd;
   VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (!vk_ok(vkAllocateCommandBuffers(device, &cmd_info, &cmd), "vkAllocateCommandBuffers"))
      goto out;
   VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   if (!vk_ok(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence"))
      goto out;

   const VkDeviceSize cases_size = MAX_CASES * 4 * sizeof(float);
   const VkDeviceSize results_size = MAX_CASES * sizeof(float);
   VkBuffer *buffers[2] = {&cases_buffer, &results_buffer};
   VkDeviceMemory *memories[2] = {&cases_memory, &results_memory};
   const VkDeviceSize sizes[2] = {cases_size, results_size};
   void *mapped[2] = {NULL, NULL};
   for (int i = 0; i < 2; ++i) {
      VkBufferCreateInfo buffer_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = sizes[i],
         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      };
      if (!vk_ok(vkCreateBuffer(device, &buffer_info, NULL, buffers[i]), "vkCreateBuffer"))
         goto out;
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(device, *buffers[i], &req);
      uint32_t type = find_memory(&memory_props, req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (type == UINT32_MAX) {
         fprintf(stderr, "FAIL: no host-visible coherent memory\n");
         goto out;
      }
      VkMemoryAllocateInfo alloc = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = type,
      };
      if (!vk_ok(vkAllocateMemory(device, &alloc, NULL, memories[i]), "vkAllocateMemory") ||
          !vk_ok(vkBindBufferMemory(device, *buffers[i], *memories[i], 0), "vkBindBufferMemory") ||
          !vk_ok(vkMapMemory(device, *memories[i], 0, VK_WHOLE_SIZE, 0, &mapped[i]),
                 "vkMapMemory"))
         goto out;
   }
   float (*cases)[4] = mapped[0];
   float *results = mapped[1];

   /* What is sampled: from outside the image twice (two corners), and from inside it. */
   static const float border_refs[] = {0.0f, 0.1f, 0.25f, 0.4f, 0.6f, 0.75f, 0.9f, 1.0f};
   static const float inside_refs[] = {0.4f, 0.6f};
   struct sample_case {
      float u, v, ref;
      bool border;
   } list[MAX_CASES];
   unsigned n = 0;
   static const float outside[2][2] = {{-1.0f, -1.0f}, {2.0f, 2.0f}};
   for (int o = 0; o < 2; ++o)
      for (unsigned r = 0; r < sizeof(border_refs) / sizeof(*border_refs); ++r)
         list[n++] = (struct sample_case){outside[o][0], outside[o][1], border_refs[r], true};
   for (unsigned r = 0; r < sizeof(inside_refs) / sizeof(*inside_refs); ++r)
      list[n++] = (struct sample_case){0.5f, 0.5f, inside_refs[r], false};
   for (unsigned i = 0; i < n; ++i) {
      cases[i][0] = list[i].u;
      cases[i][1] = list[i].v;
      cases[i][2] = list[i].ref;
      cases[i][3] = 0.0f;
   }

   static const struct border_config configs[] = {
      {"custom 0.25", VK_BORDER_COLOR_FLOAT_CUSTOM_EXT, 0.25f, VK_COMPARE_OP_LESS_OR_EQUAL},
      {"custom 0.25", VK_BORDER_COLOR_FLOAT_CUSTOM_EXT, 0.25f, VK_COMPARE_OP_GREATER_OR_EQUAL},
      {"custom 0.75", VK_BORDER_COLOR_FLOAT_CUSTOM_EXT, 0.75f, VK_COMPARE_OP_GREATER},
      {"opaque white", VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE, 0.0f, VK_COMPARE_OP_LESS},
      {"opaque black", VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, 0.0f, VK_COMPARE_OP_GREATER},
      {"transparent black", VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, 0.0f,
       VK_COMPARE_OP_GREATER_OR_EQUAL},
   };
   static const struct {
      VkFormat format;
      const char *name;
      float texel; /* what the clear value reads back as */
   } formats[] = {
      {VK_FORMAT_D32_SFLOAT, "D32_SFLOAT", DEPTH_VALUE},
      {VK_FORMAT_D16_UNORM, "D16_UNORM", 32768.0f / 65535.0f},
   };

   unsigned failures = 0, runs = 0;
   for (unsigned f = 0; f < sizeof(formats) / sizeof(*formats); ++f) {
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory image_memory = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
      VkImageCreateInfo image_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = formats[f].format,
         .extent = {IMAGE_SIZE, IMAGE_SIZE, 1},
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      if (!vk_ok(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage"))
         goto out;
      VkMemoryRequirements req;
      vkGetImageMemoryRequirements(device, image, &req);
      uint32_t type = find_memory(&memory_props, req.memoryTypeBits, 0);
      VkMemoryAllocateInfo alloc = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = type,
      };
      if (type == UINT32_MAX ||
          !vk_ok(vkAllocateMemory(device, &alloc, NULL, &image_memory), "vkAllocateMemory(image)") ||
          !vk_ok(vkBindImageMemory(device, image, image_memory, 0), "vkBindImageMemory"))
         goto out;
      VkImageViewCreateInfo view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = formats[f].format,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
      };
      if (!vk_ok(vkCreateImageView(device, &view_info, NULL, &view), "vkCreateImageView"))
         goto out;

      /* Clear the depth to 0.5 and make it readable by the shader. */
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      vkResetCommandBuffer(cmd, 0);
      vkBeginCommandBuffer(cmd, &begin);
      VkImageMemoryBarrier to_clear = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = image,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_clear);
      VkClearDepthStencilValue clear = {.depth = DEPTH_VALUE, .stencil = 0};
      vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
                                  &to_clear.subresourceRange);
      VkImageMemoryBarrier to_read = to_clear;
      to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      to_read.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
                           &to_read);
      vkEndCommandBuffer(cmd);
      VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &cmd,
      };
      if (!vk_ok(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit(clear)") ||
          !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences"))
         goto out;
      vkResetFences(device, 1, &fence);

      for (unsigned c = 0; c < sizeof(configs) / sizeof(*configs); ++c) {
         const struct border_config *config = &configs[c];
         VkSamplerCustomBorderColorCreateInfoEXT custom = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
            .customBorderColor = {.float32 = {config->custom, config->custom, config->custom,
                                              config->custom}},
            .format = without_format ? VK_FORMAT_UNDEFINED : formats[f].format,
         };
         VkSamplerCreateInfo sampler_info = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = config->color == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT ? &custom : NULL,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .compareEnable = VK_TRUE,
            .compareOp = config->op,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = config->color,
         };
         VkSampler sampler;
         if (!vk_ok(vkCreateSampler(device, &sampler_info, NULL, &sampler), "vkCreateSampler"))
            goto out;

         VkDescriptorImageInfo image_descriptor = {
            .sampler = sampler,
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
         };
         VkDescriptorBufferInfo cases_descriptor = {cases_buffer, 0, cases_size};
         VkDescriptorBufferInfo results_descriptor = {results_buffer, 0, results_size};
         VkWriteDescriptorSet writes[3] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .pImageInfo = &image_descriptor},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .pBufferInfo = &cases_descriptor},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2,
             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .pBufferInfo = &results_descriptor},
         };
         vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

         for (unsigned i = 0; i < MAX_CASES; ++i)
            results[i] = -1.0f;
         vkResetCommandBuffer(cmd, 0);
         vkBeginCommandBuffer(cmd, &begin);
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
         vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                                 &set, 0, NULL);
         vkCmdDispatch(cmd, n, 1, 1);
         vkEndCommandBuffer(cmd);
         if (!vk_ok(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit(sample)") ||
             !vk_ok(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences"))
            goto out;
         vkResetFences(device, 1, &fence);

         unsigned bad = 0;
         for (unsigned i = 0; i < n; ++i) {
            const float depth = list[i].border ? border_depth(config) : formats[f].texel;
            const float expected = compare(config->op, list[i].ref, depth) ? 1.0f : 0.0f;
            if (fabsf(results[i] - expected) > 1e-4f) {
               if (bad < 4)
                  printf("  %s: %s %s uv=(%g,%g) ref=%g expected %g got %g\n",
                         formats[f].name, config->name, op_name(config->op), list[i].u,
                         list[i].v, list[i].ref, expected, results[i]);
               ++bad;
            }
         }
         ++runs;
         if (bad) {
            printf("FAIL: shadow border %s border=%s op=%s: %u of %u samples wrong\n",
                   formats[f].name, config->name, op_name(config->op), bad, n);
            failures += bad;
         } else {
            printf("PASS: shadow border %s border=%s op=%s (%u samples)\n", formats[f].name,
                   config->name, op_name(config->op), n);
         }
         vkDestroySampler(device, sampler, NULL);
      }

      vkDestroyImageView(device, view, NULL);
      vkDestroyImage(device, image, NULL);
      vkFreeMemory(device, image_memory, NULL);
   }

   printf("%s: shadow border test, %u runs, %u wrong samples\n", failures ? "FAIL" : "PASS",
          runs, failures);
   status = failures ? EXIT_FAILURE : EXIT_SUCCESS;

out:
   if (device) {
      vkDeviceWaitIdle(device);
      if (fence)
         vkDestroyFence(device, fence, NULL);
      if (command_pool)
         vkDestroyCommandPool(device, command_pool, NULL);
      if (cases_buffer)
         vkDestroyBuffer(device, cases_buffer, NULL);
      if (results_buffer)
         vkDestroyBuffer(device, results_buffer, NULL);
      if (cases_memory)
         vkFreeMemory(device, cases_memory, NULL);
      if (results_memory)
         vkFreeMemory(device, results_memory, NULL);
      if (pool)
         vkDestroyDescriptorPool(device, pool, NULL);
      if (pipeline)
         vkDestroyPipeline(device, pipeline, NULL);
      if (shader)
         vkDestroyShaderModule(device, shader, NULL);
      if (pipeline_layout)
         vkDestroyPipelineLayout(device, pipeline_layout, NULL);
      if (set_layout)
         vkDestroyDescriptorSetLayout(device, set_layout, NULL);
      vkDestroyDevice(device, NULL);
   }
   if (instance)
      vkDestroyInstance(instance, NULL);
   return status;
}
