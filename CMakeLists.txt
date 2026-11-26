cmake_minimum_required(VERSION 3.25)

project(VulkanML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
foreach(config_type IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_CURRENT_BINARY_DIR}/bin")
endforeach()

option(VULKANML_BUILD_SHADERS "Compile changed GLSL compute shaders via glslc" ON)

add_compile_definitions(
    ENABLE_LOGGING=1
    ENABLE_SHADER_FUSION=1
)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -fdiagnostics-color=always -fno-var-tracking $<$<NOT:$<CONFIG:Debug>>:-O2>)
    if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-Wa,-mbig-obj)
    endif()
endif()

find_package(Vulkan REQUIRED)

find_path(VULKAN_SDK_INCLUDE_DIR
    NAMES vulkan/vulkan.h
    PATHS
        ${Vulkan_INCLUDE_DIRS}
        $ENV{VULKAN_SDK}/Include
)
if(NOT VULKAN_SDK_INCLUDE_DIR)
    message(FATAL_ERROR "Vulkan SDK headers (vulkan/vulkan.h) not found. Set VULKAN_SDK or Vulkan_INCLUDE_DIRS.")
endif()

find_library(SHADERC_LIB
    NAMES shaderc_shared shaderc_combined shaderc
    PATHS
        ${Vulkan_LIBRARY_DIR}
        $ENV{VULKAN_SDK}/Lib
)
if(NOT SHADERC_LIB)
    message(FATAL_ERROR "shaderc library not found. Set VULKAN_SDK or Vulkan_LIBRARY_DIR.")
endif()

find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        "${Vulkan_GLSLC_EXECUTABLE}"
        "$ENV{VULKAN_SDK}/Bin"
)

find_path(GLM_INCLUDE_DIR
    NAMES glm/glm.hpp
    PATHS
        "$ENV{GLM_DIR}/include"
        ${VULKAN_SDK_INCLUDE_DIR}
        ${Vulkan_INCLUDE_DIRS}
        "$ENV{VULKAN_SDK}/Include"
)
if(NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "GLM headers (glm/glm.hpp) not found. Set GLM_DIR or VULKAN_SDK.")
endif()

set(MUJOCO_DIR "" CACHE PATH "Path to MuJoCo installation")

find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/mujoco/include"
        "${MUJOCO_DIR}/include"
        "$ENV{MUJOCO_DIR}/include"
)

find_library(MUJOCO_LIB
    NAMES mujoco
    PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/mujoco/lib"
        "${MUJOCO_DIR}/lib"
        "$ENV{MUJOCO_DIR}/lib"
)

set(SHADER_STAMP_FILE "${CMAKE_CURRENT_BINARY_DIR}/compile_shaders.stamp")
if(VULKANML_BUILD_SHADERS AND GLSLC_EXECUTABLE)
    file(GLOB SHADER_COMP_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/compute_shader/comp/*.comp")
    if(SHADER_COMP_FILES)
        add_custom_command(
            OUTPUT "${SHADER_STAMP_FILE}"
            COMMAND ${CMAKE_COMMAND}
                -DGLSLC=${GLSLC_EXECUTABLE}
                -DCOMP_DIR=${CMAKE_CURRENT_SOURCE_DIR}/compute_shader/comp
                -DSPV_DIR=${CMAKE_CURRENT_SOURCE_DIR}/compute_shader/spv
                -DSTAMP_FILE=${SHADER_STAMP_FILE}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaders.cmake"
            DEPENDS ${SHADER_COMP_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaders.cmake"
            COMMENT "Compiling modified GLSL compute shaders"
            VERBATIM
        )
        add_custom_target(compile_shaders DEPENDS "${SHADER_STAMP_FILE}")
        list(LENGTH SHADER_COMP_FILES SHADER_COUNT)
        message(STATUS "[VulkanML] Configured incremental shader target compile_shaders (${SHADER_COUNT} shaders via ${GLSLC_EXECUTABLE})")
    endif()
elseif(VULKANML_BUILD_SHADERS)
    message(WARNING "[VulkanML] glslc not found; compile_shaders target disabled")
endif()

if(NOT DEFINED ACTIVE_FILE)
    set(ACTIVE_FILE "" CACHE STRING "Active source file to compile")
endif()

set(RESOLVED_ACTIVE_FILE "")

if(ACTIVE_FILE AND NOT ACTIVE_FILE MATCHES "^\\$\\{.*\\}$")
    if(EXISTS "${ACTIVE_FILE}")
        set(RESOLVED_ACTIVE_FILE "${ACTIVE_FILE}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ACTIVE_FILE}")
        set(RESOLVED_ACTIVE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${ACTIVE_FILE}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/${ACTIVE_FILE}")
        set(RESOLVED_ACTIVE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/apps/${ACTIVE_FILE}")
    endif()
endif()

if(NOT RESOLVED_ACTIVE_FILE)
    if(DEFINED ENV{ACTIVE_FILE} AND EXISTS "$ENV{ACTIVE_FILE}")
        set(RESOLVED_ACTIVE_FILE "$ENV{ACTIVE_FILE}")
    elseif(DEFINED ENV{VSCODE_ACTIVE_FILE} AND EXISTS "$ENV{VSCODE_ACTIVE_FILE}")
        set(RESOLVED_ACTIVE_FILE "$ENV{VSCODE_ACTIVE_FILE}")
    endif()
endif()

if(NOT RESOLVED_ACTIVE_FILE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/train_sllm.cpp")
        set(RESOLVED_ACTIVE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/apps/train_sllm.cpp")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp")
        set(RESOLVED_ACTIVE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp")
    endif()
endif()

set(ACTIVE_TARGET_NAME "")
if(RESOLVED_ACTIVE_FILE)
    file(RELATIVE_PATH ACTIVE_REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOLVED_ACTIVE_FILE}")
    if(ACTIVE_REL_PATH MATCHES "^apps[/\\\\]chexpert[/\\\\]")
        set(ACTIVE_TARGET_NAME "train_chexpert")
    elseif(ACTIVE_REL_PATH MATCHES "^tests[/\\\\]")
        set(ACTIVE_TARGET_NAME "vulkanml_tests")
    else()
        cmake_path(GET RESOLVED_ACTIVE_FILE STEM ACTIVE_TARGET_NAME)
    endif()
    message(STATUS "[VulkanML] Active target: ${ACTIVE_TARGET_NAME} (${RESOLVED_ACTIVE_FILE})")
endif()

file(GLOB_RECURSE ALL_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(VULKANML_CORE_SOURCES "")
set(CHEXLOCALIZE_SOURCES "")

foreach(source_file IN LISTS ALL_SRC_FILES)
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_SOURCE_DIR}" "${source_file}")
    if(rel_path MATCHES "^src[/\\\\]chexlocalize")
        list(APPEND CHEXLOCALIZE_SOURCES "${source_file}")
    else()
        list(APPEND VULKANML_CORE_SOURCES "${source_file}")
    endif()
endforeach()

if(VULKANML_CORE_SOURCES)
    add_library(vulkanml_core SHARED ${VULKANML_CORE_SOURCES})
    target_include_directories(vulkanml_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(vulkanml_core PRIVATE
        ${VULKAN_SDK_INCLUDE_DIR}
    )
    target_compile_definitions(vulkanml_core PUBLIC
        ENABLE_LOGGING=1
        ENABLE_SHADER_FUSION=1
    )
    target_compile_definitions(vulkanml_core PRIVATE
        VULKANML_EXPORTS
    )
    target_compile_features(vulkanml_core PUBLIC cxx_std_23)
    target_link_libraries(vulkanml_core PRIVATE Vulkan::Vulkan)
    target_link_libraries(vulkanml_core PRIVATE ${SHADERC_LIB})
    if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(vulkanml_core PRIVATE stdc++exp)
    endif()
    if(MINGW)
        target_link_options(vulkanml_core PRIVATE -Wl,--export-all-symbols)
    endif()
    if(TARGET compile_shaders)
        add_dependencies(vulkanml_core compile_shaders)
    endif()

    set_target_properties(vulkanml_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        OUTPUT_NAME "vulkanml_core"
        PREFIX ""
    )
    list(LENGTH VULKANML_CORE_SOURCES CORE_COUNT)
    message(STATUS "[VulkanML] Configured vulkanml_core as SHARED library (.dll) with ${CORE_COUNT} source files")
endif()

if(CHEXLOCALIZE_SOURCES)
    add_library(chexlocalize_ext STATIC ${CHEXLOCALIZE_SOURCES})
    target_include_directories(chexlocalize_ext PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(chexlocalize_ext PUBLIC ${GLM_INCLUDE_DIR})
    target_link_libraries(chexlocalize_ext PUBLIC vulkanml_core)
    if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(chexlocalize_ext PRIVATE stdc++exp)
    endif()
    list(LENGTH CHEXLOCALIZE_SOURCES CHEX_COUNT)
    message(STATUS "[VulkanML] Configured chexlocalize_ext as STATIC extension with ${CHEX_COUNT} source files")
endif()

function(add_project_executable target_name source_files is_default_build)
    if(is_default_build)
        add_executable(${target_name} ${source_files})
    else()
        add_executable(${target_name} EXCLUDE_FROM_ALL ${source_files})
    endif()

    if(TARGET vulkanml_core)
        target_link_libraries(${target_name} PRIVATE vulkanml_core)
    endif()

    if(target_name MATCHES "humanoid" OR target_name MATCHES "mujoco")
        if(MUJOCO_INCLUDE_DIR)
            target_include_directories(${target_name} PRIVATE ${MUJOCO_INCLUDE_DIR})
        endif()
        if(MUJOCO_LIB)
            target_link_libraries(${target_name} PRIVATE ${MUJOCO_LIB})
        endif()
        target_link_libraries(${target_name} PRIVATE
            glfw3
            opengl32
            gdi32
            comdlg32
        )
        set(MUJOCO_DLL_PATH "")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mujoco/bin/mujoco.dll")
            set(MUJOCO_DLL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mujoco/bin/mujoco.dll")
        elseif(MUJOCO_DIR AND EXISTS "${MUJOCO_DIR}/bin/mujoco.dll")
            set(MUJOCO_DLL_PATH "${MUJOCO_DIR}/bin/mujoco.dll")
        elseif(DEFINED ENV{MUJOCO_DIR} AND EXISTS "$ENV{MUJOCO_DIR}/bin/mujoco.dll")
            set(MUJOCO_DLL_PATH "$ENV{MUJOCO_DIR}/bin/mujoco.dll")
        endif()
        if(MUJOCO_DLL_PATH)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MUJOCO_DLL_PATH}"
                    "$<TARGET_FILE_DIR:${target_name}>/mujoco.dll"
                COMMENT "Copying mujoco.dll for ${target_name}"
            )
        endif()
    endif()

    if(target_name STREQUAL "train_chexpert" OR
       target_name STREQUAL "vulkanml_tests" OR
       target_name STREQUAL "test_bn" OR
       target_name STREQUAL "test_nnunet")
        if(TARGET chexlocalize_ext)
            target_link_libraries(${target_name} PRIVATE chexlocalize_ext)
        endif()
    endif()

    if(target_name STREQUAL "vulkanml_tests" OR target_name STREQUAL "benchmark_ternary")
        target_include_directories(${target_name} PRIVATE ${VULKAN_SDK_INCLUDE_DIR})
    endif()

    if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(${target_name} PRIVATE stdc++exp)
    endif()

    set_target_properties(${target_name} PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endfunction()

file(GLOB APP_CPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp")
set(CREATED_TARGETS "")

foreach(source_file IN LISTS APP_CPP_SOURCES)
    cmake_path(GET source_file STEM current_stem)
    set(IS_DEFAULT FALSE)
    if(current_stem STREQUAL ACTIVE_TARGET_NAME)
        set(IS_DEFAULT TRUE)
    endif()
    add_project_executable(${current_stem} "${source_file}" ${IS_DEFAULT})
    list(APPEND CREATED_TARGETS ${current_stem})
endforeach()

file(GLOB CHEXPERT_APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/apps/chexpert/*.cpp")
if(CHEXPERT_APP_SOURCES)
    set(IS_CHEX_DEFAULT FALSE)
    if(ACTIVE_TARGET_NAME STREQUAL "train_chexpert")
        set(IS_CHEX_DEFAULT TRUE)
    endif()
    add_project_executable("train_chexpert" "${CHEXPERT_APP_SOURCES}" ${IS_CHEX_DEFAULT})
    list(APPEND CREATED_TARGETS "train_chexpert")
endif()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
if(TEST_SOURCES)
    add_project_executable("vulkanml_tests" "${TEST_SOURCES}" TRUE)
    set_target_properties(vulkanml_tests PROPERTIES OUTPUT_NAME "test")
    list(APPEND CREATED_TARGETS "vulkanml_tests")
endif()

if(RESOLVED_ACTIVE_FILE AND ACTIVE_TARGET_NAME AND NOT ACTIVE_TARGET_NAME IN_LIST CREATED_TARGETS)
    add_project_executable(${ACTIVE_TARGET_NAME} "${RESOLVED_ACTIVE_FILE}" TRUE)
    list(APPEND CREATED_TARGETS ${ACTIVE_TARGET_NAME})
endif()

if(ACTIVE_TARGET_NAME AND TARGET ${ACTIVE_TARGET_NAME})
    add_custom_target(active_file
        DEPENDS ${ACTIVE_TARGET_NAME}
        COMMENT "Building active target: ${ACTIVE_TARGET_NAME}"
    )
endif()

message(STATUS "[VulkanML] Available executable targets: ${CREATED_TARGETS}")

include(CTest)
if(BUILD_TESTING AND TARGET vulkanml_tests)
    set(VULKANML_TEST_SUITES
        cpu_core
        gpu_core
        engine_systems
        ternary_quant
        vision_seg
        heavy_unet
        regressions_w2_w5
        regressions_w6_w7
        perf_phases
        dx_infra
        layer_upgrades
        sealed_read
        weight_loader
    )
    foreach(suite_name IN LISTS VULKANML_TEST_SUITES)
        add_test(
            NAME ${suite_name}
            COMMAND vulkanml_tests --suite ${suite_name}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endforeach()
    set_tests_properties(heavy_unet PROPERTIES LABELS "heavy")
endif()

include(CMakePackageConfigHelpers)

if(TARGET vulkanml_core)
    install(TARGETS vulkanml_core
        EXPORT VulkanMLTargets
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
    install(EXPORT VulkanMLTargets
        FILE VulkanMLTargets.cmake
        NAMESPACE VulkanML::
        DESTINATION lib/cmake/VulkanML
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VulkanMLConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/VulkanMLConfig.cmake"
        INSTALL_DESTINATION lib/cmake/VulkanML
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/VulkanMLConfigVersion.cmake"
        VERSION 1.0.0
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/VulkanMLConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/VulkanMLConfigVersion.cmake"
        DESTINATION lib/cmake/VulkanML
    )
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include
        PATTERN "chexlocalize" EXCLUDE
        PATTERN "environment" EXCLUDE
        PATTERN "ops" EXCLUDE
        PATTERN "engine" EXCLUDE
        PATTERN "tensor_impl.h" EXCLUDE
        PATTERN "*_tensor_impl.h" EXCLUDE
    )
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/data_type.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/loss_scaler.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/async_data_pipeline.h"
        DESTINATION include/engine
    )
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compute_shader/
        DESTINATION compute_shader
    )
    if(WIN32)
        find_file(SHADERC_SHARED_DLL
            NAMES shaderc_shared.dll
            PATHS
                "$ENV{VULKAN_SDK}/Bin"
                "${VULKAN_SDK_INCLUDE_DIR}/../Bin"
        )
        if(SHADERC_SHARED_DLL)
            install(FILES "${SHADERC_SHARED_DLL}" DESTINATION bin)
        endif()
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/user_preferences.json")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/user_preferences.json" DESTINATION .)
    endif()
endif()
