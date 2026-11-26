#include "engine/shader_compiler.h"

#include <chrono>
#include <format>
#include <stdexcept>
#include <vulkan/vulkan.h>

#include "helper/logger.h"

Shader_Compiler::Shader_Compiler(uint32_t _vulkan_api_version)
    : compile_options(std::make_unique<shaderc::CompileOptions>())
{
    if (!compiler.IsValid())
    {
        Logger::logMessage("Shader_Compiler::Shader_Compiler: Failed to initialize shaderc::Compiler",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Shader_Compiler::Shader_Compiler: shaderc compiler is invalid");
    }

    Logger::logMessage("Shader_Compiler::Shader_Compiler: Initializing shader compiler",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    compile_options->SetOptimizationLevel(shaderc_optimization_level_performance);

    const uint32_t target_api = (_vulkan_api_version == 0) ? VK_API_VERSION_1_3 : _vulkan_api_version;
    if (target_api >= VK_API_VERSION_1_3)
    {
        compile_options->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        compile_options->SetTargetSpirv(shaderc_spirv_version_1_5);
    }
    else if (target_api >= VK_API_VERSION_1_2)
    {
        compile_options->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
        compile_options->SetTargetSpirv(shaderc_spirv_version_1_5);
    }
    else if (target_api >= VK_API_VERSION_1_1)
    {
        compile_options->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
        compile_options->SetTargetSpirv(shaderc_spirv_version_1_3);
    }
    else
    {
        compile_options->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
        compile_options->SetTargetSpirv(shaderc_spirv_version_1_0);
    }
}

std::vector<uint32_t> Shader_Compiler::compileGlslToSpirv(const std::string &_glsl_code, const std::string &_shader_name) const
{
    if (_glsl_code.empty())
    {
        throw std::invalid_argument("Shader_Compiler::compileGlslToSpirv: GLSL source code must not be empty");
    }

    std::lock_guard<std::mutex> lock(compiler_mutex);
    const auto compile_start = std::chrono::high_resolution_clock::now();
    shaderc::SpvCompilationResult compilation_result = compiler.CompileGlslToSpv(
        _glsl_code,
        shaderc_compute_shader,
        _shader_name.c_str(),
        *compile_options);

    if (compilation_result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        Logger::logMessage(Input_Format{"Shader_Compiler::compileGlslToSpirv failed for {}: {}\nSource:\n{}", _shader_name, compilation_result.GetErrorMessage(), _glsl_code},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("SPIR-V compilation failed: " + compilation_result.GetErrorMessage());
    }

    if (compilation_result.GetNumWarnings() > 0)
    {
        Logger::logMessage(Input_Format{"Shader_Compiler::compileGlslToSpirv warnings ({}) for {}: {}",
                                        compilation_result.GetNumWarnings(), _shader_name, compilation_result.GetErrorMessage()},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }

    std::vector<uint32_t> spirv(compilation_result.cbegin(), compilation_result.cend());
    if (spirv.empty())
    {
        throw std::runtime_error("Shader_Compiler::compileGlslToSpirv: Compiled SPIR-V output is empty");
    }
    const double compile_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - compile_start)
                                  .count();
    Logger::logMessage(Input_Format{"Shader_Compiler::compileGlslToSpirv: '{}' -> {} bytes in {:.2f} ms",
                                    _shader_name, spirv.size() * sizeof(uint32_t), compile_ms},
                       Log_Level::LOG_DEBUG,
                       false,
                       0,
                       Log_Feature::PIPELINE_CACHE | Log_Feature::SHADER_GENERATION);
    return spirv;
}

void Shader_Compiler::setCompileOptions(const shaderc::CompileOptions &_options)
{
    std::lock_guard<std::mutex> lock(compiler_mutex);
    if (&_options == compile_options.get())
    {
        return;
    }
    auto cloned = std::make_unique<shaderc::CompileOptions>(_options);
    compile_options = std::move(cloned);
}
