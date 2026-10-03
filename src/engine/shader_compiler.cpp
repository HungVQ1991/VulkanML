#include "engine/shader_compiler.h"

#include <format>
#include <stdexcept>

#include "helper/logger.h"

Shader_Compiler::Shader_Compiler()
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

std::vector<uint32_t> Shader_Compiler::compileGlslToSpirv(const std::string &_glsl_code, const std::string &_shader_name) const
{
    shaderc::SpvCompilationResult compilation_result = compiler.CompileGlslToSpv(
        _glsl_code,
        shaderc_compute_shader,
        _shader_name.c_str(),
        compile_options);

    if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        Logger::logMessage(Input_Format{"Shader_Compiler::compileGlslToSpirv failed for {}: {}\nSource:\n{}", _shader_name, compilation_result.GetErrorMessage(), _glsl_code},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("SPIR-V compilation failed: " + compilation_result.GetErrorMessage());
    }

    return {compilation_result.cbegin(), compilation_result.cend()};
}

void Shader_Compiler::setCompileOptions(const shaderc::CompileOptions &_options)
{
    compile_options.~CompileOptions();
    ::new (&compile_options) shaderc::CompileOptions(_options);
}
