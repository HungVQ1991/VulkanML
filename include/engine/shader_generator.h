#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"

enum class Buffer_Access
{
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
};

struct Specialization_Constant_Entry
{
    std::uint32_t constant_id;
    std::string name;
    std::string type_name;
    std::string default_val;
};

class Specialization_Map_Builder
{
private:
    std::vector<VkSpecializationMapEntry> entries;
    std::vector<std::uint8_t> data;

public:
    template <typename T>
    void addConstant(std::uint32_t _constant_id, const T &_value)
    {
        std::size_t offset = data.size();
        std::size_t size = sizeof(T);

        entries.push_back(VkSpecializationMapEntry{
            .constantID = _constant_id,
            .offset = static_cast<std::uint32_t>(offset),
            .size = size});

        const auto *byte_pointer = reinterpret_cast<const std::uint8_t *>(&_value);
        data.insert(data.end(), byte_pointer, byte_pointer + size);
    }

    [[nodiscard]] VkSpecializationInfo build() const noexcept
    {
        return VkSpecializationInfo{
            .mapEntryCount = static_cast<std::uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = data.size(),
            .pData = data.data()};
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return entries.empty();
    }

    void clear() noexcept
    {
        entries.clear();
        data.clear();
    }
};

class Shader_Generator
{
private:
    std::uint32_t group_x;
    std::uint32_t group_y;
    std::uint32_t group_z;
    std::uint32_t current_binding = 0;
    std::uint32_t var_counter = 0;

    std::ostringstream header_stream;
    std::ostringstream specialization_stream;
    std::ostringstream bindings_stream;
    std::ostringstream shared_memory_stream;
    std::ostringstream body_stream;

    bool is_subgroup_enabled = false;
    bool is_control_flow_enabled = false;
    std::vector<Specialization_Constant_Entry> spec_constants;

public:
    Shader_Generator(std::uint32_t _group_x, std::uint32_t _group_y = 1, std::uint32_t _group_z = 1)
        : group_x(_group_x), group_y(_group_y), group_z(_group_z)
    {
        Logger::logMessage(std::format("Shader_Generator::Shader_Generator: Initializing generator with local_size ({}, {}, {})", _group_x, _group_y, _group_z),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }

    void enableSubgroupOperations()
    {
        Logger::logMessage("Shader_Generator::enableSubgroupOperations: Enabling subgroup operations",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        is_subgroup_enabled = true;
    }

    void enableControlFlowAttributes()
    {
        Logger::logMessage("Shader_Generator::enableControlFlowAttributes: Enabling control flow attributes",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        is_control_flow_enabled = true;
    }

    void addSpecializationConstant(std::uint32_t _constant_id,
                                   const std::string &_name,
                                   const std::string &_type_name = "uint",
                                   const std::string &_default_value = "0")
    {
        Logger::logMessage(std::format("Shader_Generator::addSpecializationConstant: Added constant_id {} with name '{}' type '{}' default '{}'",
                                       _constant_id, _name, _type_name, _default_value),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        spec_constants.push_back(Specialization_Constant_Entry{
            .constant_id = _constant_id,
            .name = _name,
            .type_name = _type_name,
            .default_val = _default_value});

        specialization_stream << std::format("layout(constant_id = {}) const {} {} = {};\n",
                                             _constant_id, _type_name, _name, _default_value);
    }

    [[nodiscard]] const std::vector<Specialization_Constant_Entry> &getSpecializationConstants() const noexcept
    {
        return spec_constants;
    }

    std::string addBuffer(std::uint32_t _binding_index,
                          const std::string &_buffer_name = "",
                          const std::string &_type_name = "float",
                          Buffer_Access _access = Buffer_Access::READ_WRITE)
    {
        if (current_binding >= 32)
        {
            throw std::runtime_error("Shader_Generator: Exceeded maximum of 32 bindings.");
        }
        current_binding++;

        std::string qualifier;
        switch (_access)
        {
        case Buffer_Access::READ_ONLY:
            qualifier = "readonly";
            break;
        case Buffer_Access::WRITE_ONLY:
            qualifier = "writeonly";
            break;
        case Buffer_Access::READ_WRITE:
            qualifier = "coherent";
            break;
        }

        std::string name = _buffer_name.empty() ? std::format("buf_{}", _binding_index) : _buffer_name;
        Logger::logMessage(std::format("Shader_Generator::addBuffer: Added buffer binding {} with name '{}' of type '{}' and access '{}'",
                                       _binding_index, name, _type_name, qualifier),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);

        bindings_stream << std::format("layout(std430, binding = {}) {} buffer Buffer_{} {{ {} {}[]; }};\n",
                                       _binding_index, qualifier, _binding_index, _type_name, name);
        return name;
    }

    void setPushConstants(const std::string &_struct_definition)
    {
        bindings_stream << "layout(push_constant) uniform PushConstants {\n";
        bindings_stream << _struct_definition << "\n";
        bindings_stream << "} pc;\n\n";
    }

    std::string addSharedMemory(std::uint32_t _size, const std::string &_prefix = "shared_mem", const std::string &_type_name = "float")
    {
        std::string name = std::format("{}_{}", _prefix, var_counter++);
        Logger::logMessage(std::format("Shader_Generator::addSharedMemory: Added shared memory array '{}' of size {} of type '{}'", name, _size, _type_name),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        shared_memory_stream << std::format("shared {} {}[{}];\n", _type_name, name, _size);
        return name;
    }

    void addSharedMemoryRaw(const std::string &_declaration)
    {
        shared_memory_stream << _declaration << "\n";
    }

    std::string getUniqueVar(const std::string &_prefix = "val")
    {
        return std::format("{}_{}", _prefix, var_counter++);
    }

    void addLogicSnippet(const std::string &_snippet)
    {
        body_stream << _snippet << "\n";
    }

    std::string build() const
    {
        Logger::logMessage("Shader_Generator::build: Building GLSL shader code",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        std::ostringstream final_shader;
        final_shader << "#version 450\n";

        if (is_subgroup_enabled)
        {
            final_shader << "#extension GL_KHR_shader_subgroup_arithmetic : enable\n";
            final_shader << "#extension GL_KHR_shader_subgroup_basic : enable\n";
        }

        if (is_control_flow_enabled)
        {
            final_shader << "#extension GL_EXT_control_flow_attributes : enable\n";
        }

        final_shader << "\n";
        final_shader << std::format("layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n", group_x, group_y, group_z);
        final_shader << specialization_stream.str();
        if (!spec_constants.empty())
        {
            final_shader << "\n";
        }
        final_shader << bindings_stream.str();
        final_shader << shared_memory_stream.str();

        final_shader << "void main() {\n";
        final_shader << "    uint global_id = gl_GlobalInvocationID.x;\n";
        final_shader << "    uint local_id = gl_LocalInvocationID.x;\n";

        if (is_subgroup_enabled)
        {
            final_shader << "    uint subgroup_id = gl_SubgroupID;\n";
            final_shader << "    uint subgroup_local_id = gl_SubgroupInvocationID;\n";
        }

        final_shader << body_stream.str();
        final_shader << "}\n";

        return final_shader.str();
    }
};