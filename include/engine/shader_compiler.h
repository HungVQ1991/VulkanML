#pragma once

#include <cstdint>
#include <shaderc/shaderc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"

#ifndef ENABLE_SHADER_DEBUG_LOGS
#define ENABLE_SHADER_DEBUG_LOGS 0
#endif

#if ENABLE_SHADER_DEBUG_LOGS
#define SHADER_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define SHADER_LOG_DEBUG(msg) ((void)0)
#endif

class Shader_Compiler
{
private:
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

public:
    Shader_Compiler()
    {
        SHADER_LOG_DEBUG("Shader_Compiler::Shader_Compiler: Initializing shader compiler");
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    }

    std::vector<std::uint32_t> compileGlslToSpv(const std::string &glsl_code, const std::string &shader_name)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
        glsl_code,
        shaderc_compute_shader,
        shader_name.c_str(),
        options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        Logger::logMessage("Shader_Compiler::compileGlslToSpv: " + module.GetErrorMessage(), LOG_ERROR, true);
        throw std::runtime_error("SPIR-V compilation failed: " + module.GetErrorMessage());
    }

    return {module.cbegin(), module.cend()};
}
};