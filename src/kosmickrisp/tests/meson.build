# Copyright 2026 LunarG, Inc.
# SPDX-License-Identifier: MIT

dep_vulkan_loader = dependency('vulkan', required : false)
if dep_vulkan_loader.found()
  executable(
    'kk-gpu-tests',
    ['gpu_write_test.c', 'gpu_write_spv.h', 'xfb_capture_spv.h'],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_vulkan_runtime_headers, dep_vulkan_loader],
    install : false,
    build_by_default : false,
  )
  executable(
    'kk-border-shadow-tests',
    ['border_shadow_test.c', 'border_shadow_spv.h'],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_vulkan_runtime_headers, dep_vulkan_loader],
    install : false,
    build_by_default : false,
  )
endif

kk_shader_metadata_tests = executable(
  'kk-shader-metadata-tests',
  'shader_metadata_test.c',
  include_directories : [inc_include, inc_src, include_directories('../vulkan')],
  dependencies : [idep_nir, idep_vulkan_runtime_headers,
                  idep_vulkan_util_headers],
  install : false,
  build_by_default : false,
)

test('kk-shader-metadata', kk_shader_metadata_tests)

kk_xfb_abi_tests = executable(
  'kk-xfb-abi-tests',
  ['xfb_abi_test.c', '../vulkan/kk_xfb_abi.c'],
  include_directories : [inc_include, inc_src, include_directories('../vulkan')],
  dependencies : [idep_nir, idep_vulkan_runtime_headers,
                  idep_vulkan_util_headers, idep_libkk_shaders_h],
  install : false,
  build_by_default : false,
)

test('kk-xfb-abi', kk_xfb_abi_tests)
