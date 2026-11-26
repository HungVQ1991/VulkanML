#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>

class Shader_Compiler
{
private:
    shaderc::Compiler compiler;
    std::unique_ptr<shaderc::CompileOptions> compile_options;
    mutable std::mutex compiler_mutex;

public:
    explicit Shader_Compiler(uint32_t _vulkan_api_version = 0);

    Shader_Compiler(const Shader_Compiler &) = delete;
    Shader_Compiler &operator=(const Shader_Compiler &) = delete;
    Shader_Compiler(Shader_Compiler &&) = delete;
    Shader_Compiler &operator=(Shader_Compiler &&) = delete;

    std::vector<uint32_t> compileGlslToSpirv(const std::string &_glsl_code, const std::string &_shader_name = "compute_shader") const;

    const shaderc::CompileOptions &getCompileOptions() const noexcept { return *compile_options; }
    const shaderc::Compiler &getCompiler() const noexcept { return compiler; }

    void setCompileOptions(const shaderc::CompileOptions &_options);
};
