#pragma once

#include <format>
#include <sstream>
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

class Shader_Generator
{
private:
    std::uint32_t group_x;
    std::uint32_t group_y;
    std::uint32_t group_z;
    std::uint32_t current_binding = 0;
    std::uint32_t var_counter = 0;

    std::ostringstream header_stream;
    std::ostringstream bindings_stream;
    std::ostringstream shared_memory_stream;
    std::ostringstream body_stream;

    bool use_subgroup = false;

public:
    Shader_Generator(std::uint32_t gx, std::uint32_t gy = 1, std::uint32_t gz = 1)
        : group_x(gx), group_y(gy), group_z(gz)
    {
        SHADER_LOG_DEBUG(std::format("Shader_Generator::Shader_Generator: Initializing generator with local_size ({}, {}, {})", gx, gy, gz));
        header_stream << "#version 450\n";
        header_stream << std::format("layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n", group_x, group_y, group_z);
    }

    void enableSubgroupOperations()
    {
        SHADER_LOG_DEBUG("Shader_Generator::enableSubgroupOperations: Enabling subgroup operations");
        use_subgroup = true;
    }

    std::string addBuffer(std::uint32_t binding_index, const std::string &buffer_name = "")
    {
        if (current_binding >= 32)
        {
            throw std::runtime_error("Shader_Generator: Exceeded maximum of 16 bindings.");
        }
        current_binding++;

        std::string name = buffer_name.empty() ? std::format("buf_{}", binding_index) : buffer_name;
        SHADER_LOG_DEBUG(std::format("Shader_Generator::addBuffer: Added buffer binding {} with name '{}'", binding_index, name));
        bindings_stream << std::format("layout(std430, binding = {}) coherent buffer Buffer_{} {{ float {}[]; }};\n",
                                       binding_index, binding_index, name);
        return name;
    }

    void setPushConstants(const std::string &struct_definition)
    {
        bindings_stream << "layout(push_constant) uniform PushConstants {\n";
        bindings_stream << struct_definition << "\n";
        bindings_stream << "} pc;\n\n";
    }

    std::string addSharedMemory(std::uint32_t size, const std::string &prefix = "shared_mem")
    {
        std::string name = std::format("{}_{}", prefix, var_counter++);
        SHADER_LOG_DEBUG(std::format("Shader_Generator::addSharedMemory: Added shared memory array '{}' of size {}", name, size));
        shared_memory_stream << std::format("shared float {}[{}];\n", name, size);
        return name;
    }

    std::string getUniqueVar(const std::string &prefix = "val")
    {
        return std::format("{}_{}", prefix, var_counter++);
    }

    void addLogicSnippet(const std::string &snippet)
    {
        body_stream << snippet << "\n";
    }

    std::string build() const
    {
        SHADER_LOG_DEBUG("Shader_Generator::build: Building GLSL shader code");
        std::ostringstream final_shader;
        final_shader << header_stream.str();

        if (use_subgroup)
        {
            final_shader << "#extension GL_KHR_shader_subgroup_arithmetic : enable\n";
            final_shader << "#extension GL_KHR_shader_subgroup_basic : enable\n\n";
        }

        final_shader << bindings_stream.str();
        final_shader << shared_memory_stream.str();

        final_shader << "void main() {\n";
        final_shader << "    uint global_id = gl_GlobalInvocationID.x;\n";
        final_shader << "    uint local_id = gl_LocalInvocationID.x;\n";

        if (use_subgroup)
        {
            final_shader << "    uint subgroup_id = gl_SubgroupID;\n";
            final_shader << "    uint subgroup_local_id = gl_SubgroupInvocationID;\n";
        }

        final_shader << body_stream.str();
        final_shader << "}\n";

        return final_shader.str();
    }
};