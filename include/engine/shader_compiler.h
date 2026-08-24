#pragma once

#include <cstdint>
#include <format>
#include <shaderc/shaderc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"

class Shader_Compiler
{
private:
    shaderc::Compiler compiler;
    shaderc::CompileOptions compile_options;

public:
    Shader_Compiler()
    {
        Logger::logMessage("Shader_Compiler::Shader_Compiler: Initializing shader compiler",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        compile_options.SetOptimizationLevel(shaderc_optimization_level_performance);
        compile_options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        compile_options.SetTargetSpirv(shaderc_spirv_version_1_5);
    }

    [[nodiscard]] const shaderc::Compiler &getCompiler() const noexcept
    {
        return compiler;
    }

    [[nodiscard]] shaderc::Compiler &getCompiler() noexcept
    {
        return compiler;
    }

    [[nodiscard]] const shaderc::CompileOptions &getCompileOptions() const noexcept
    {
        return compile_options;
    }

    [[nodiscard]] shaderc::CompileOptions &getCompileOptions() noexcept
    {
        return compile_options;
    }

    std::vector<std::uint32_t> compileGlslToSpirv(const std::string &_glsl_code, const std::string &_shader_name = "compute_shader") const
    {
        shaderc::SpvCompilationResult compilation_result = compiler.CompileGlslToSpv(
            _glsl_code,
            shaderc_compute_shader,
            _shader_name.c_str(),
            compile_options);

        if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            Logger::logMessage(std::format("Shader_Compiler::compileGlslToSpirv: {}", compilation_result.GetErrorMessage()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("SPIR-V compilation failed: " + compilation_result.GetErrorMessage());
        }

        return {compilation_result.cbegin(), compilation_result.cend()};
    }
};