#pragma once

#include <cstdint>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>

class Shader_Compiler
{
private:
    shaderc::Compiler compiler;
    shaderc::CompileOptions compile_options;

public:
    Shader_Compiler();

    std::vector<uint32_t> compileGlslToSpirv(const std::string &_glsl_code, const std::string &_shader_name = "compute_shader") const;

    const shaderc::CompileOptions &getCompileOptions() const noexcept { return compile_options; }
    shaderc::CompileOptions &getCompileOptions() noexcept { return compile_options; }
    const shaderc::Compiler &getCompiler() const noexcept { return compiler; }
    shaderc::Compiler &getCompiler() noexcept { return compiler; }

    void setCompileOptions(const shaderc::CompileOptions &_options);
};