# Layla bindings for llama.cpp-omni: the vendored MiniCPM-V-Apps bridge and the
# minicpm_* C ABI on top of it. Layla's modules/minicpm/ios/CMakeLists.txt adds this
# directory after llama.cpp-omni itself, so the llama, mtmd and llama-common targets
# already exist. Nothing outside layla/ is modified for Layla.

if(NOT TARGET llama OR NOT TARGET mtmd OR NOT TARGET llama-common)
  message(FATAL_ERROR "layla/ needs the llama, mtmd and llama-common targets; add llama.cpp-omni first")
endif()

get_filename_component(OMNI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(MINICPM_BRIDGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/minicpm-v-apps")

# The bridge includes the public headers the way the MiniCPM-V app's llama.framework
# lays them out (<llama/llama.h>). Forwarding headers reproduce that layout so the
# vendored sources compile unchanged.
set(MINICPM_FRAMEWORK_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/framework-headers")
foreach(header
    "${OMNI_ROOT}/include/llama.h"
    "${OMNI_ROOT}/tools/mtmd/mtmd.h"
    "${OMNI_ROOT}/tools/mtmd/mtmd-helper.h")
  get_filename_component(header_name "${header}" NAME)
  file(GENERATE
    OUTPUT "${MINICPM_FRAMEWORK_HEADERS}/llama/${header_name}"
    CONTENT "#pragma once\n#include \"${header}\"\n")
endforeach()

add_library(minicpm_layla STATIC
  minicpm_capi.cpp
  "${MINICPM_BRIDGE_DIR}/MBMtmd.mm")

target_include_directories(minicpm_layla
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"          # minicpm_capi.h
  PRIVATE
    "${MINICPM_BRIDGE_DIR}"                # MBMtmd.h, MBPrewarmTable.h
    "${MINICPM_FRAMEWORK_HEADERS}"         # <llama/llama.h>, <llama/mtmd*.h>
    "${OMNI_ROOT}/vendor")                 # <nlohmann/json.hpp>

# The MiniCPM-V app builds the bridge as gnu++20 with ARC, and it relies on ARC to
# release the NSStrings it allocates.
target_compile_features(minicpm_layla PRIVATE cxx_std_20)
set_source_files_properties("${MINICPM_BRIDGE_DIR}/MBMtmd.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

# llama-common supplies the common_speculative_* symbols DSpark calls. The bridge
# declares those structs itself rather than including common.h (see the ABI asserts in
# MBMtmd.mm), so the two must come from the same llama.cpp-omni revision.
target_link_libraries(minicpm_layla
  PRIVATE
    llama
    mtmd
    llama-common
    "-framework Foundation"
    "-framework Metal")
