#include "engine/shader_generator.h"

#include <format>
#include <stdexcept>

#include "helper/logger.h"

Shader_Generator::Shader_Generator(uint32_t _group_x, uint32_t _group_y, uint32_t _group_z, const std::string &_default_data_type)
    : group_x(_group_x), group_y(_group_y), group_z(_group_z), default_data_type(_default_data_type)
{
    if (_group_x == 0 || _group_y == 0 || _group_z == 0)
    {
        throw std::invalid_argument("Shader_Generator::Shader_Generator: Workgroup dimensions must all be >= 1");
    }
    if (default_data_type == "float16_t")
    {
        is_float16_enabled = true;
    }
    Logger::logMessage(Input_Format{"Shader_Generator::Shader_Generator: Initializing generator with local_size ({}, {}, {}) and default type '{}'", _group_x, _group_y, _group_z, _default_data_type},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
}

void Shader_Generator::enableSubgroupOperations()
{
    Logger::logMessage("Shader_Generator::enableSubgroupOperations: Enabling subgroup operations",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    is_subgroup_enabled = true;
}

void Shader_Generator::enableControlFlowAttributes()
{
    Logger::logMessage("Shader_Generator::enableControlFlowAttributes: Enabling control flow attributes",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    is_control_flow_enabled = true;
}

void Shader_Generator::addSpecializationConstant(uint32_t _constant_id,
                                               const std::string &_name,
                                               const std::string &_type_name,
                                               const std::string &_default_value)
{
    if (_type_name == "float16_t")
    {
        is_float16_enabled = true;
    }
    Logger::logMessage(Input_Format{"Shader_Generator::addSpecializationConstant: Added constant_id {} with name '{}' type '{}' default '{}'",
                                    _constant_id, _name, _type_name, _default_value},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    spec_constants.push_back(Specialization_Constant_Entry{
        .constant_id = _constant_id,
        .name = _name,
        .type_name = _type_name,
        .default_val = _default_value});
}

std::string Shader_Generator::addBuffer(uint32_t _binding_index,
                                        const std::string &_buffer_name,
                                        const std::string &_type_name,
                                        Buffer_Access _access)
{
    if (_binding_index >= 32)
    {
        throw std::invalid_argument(std::format("Shader_Generator::addBuffer: Binding index {} exceeds maximum of 32.", _binding_index));
    }
    if (used_bindings.test(_binding_index))
    {
        throw std::invalid_argument(std::format("Shader_Generator::addBuffer: Duplicate binding index {}.", _binding_index));
    }
    used_bindings.set(_binding_index);
    current_binding = static_cast<uint32_t>(used_bindings.count());

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
        qualifier = "";
        break;
    }

    std::string resolved_type = _type_name.empty() ? default_data_type : _type_name;
    if (resolved_type == "float16_t")
    {
        is_float16_enabled = true;
    }
    std::string name = _buffer_name.empty() ? std::format("buf_{}", _binding_index) : _buffer_name;
    Logger::logMessage(Input_Format{"Shader_Generator::addBuffer: Added buffer binding {} with name '{}' of type '{}' and access '{}'",
                                    _binding_index, name, resolved_type, qualifier.empty() ? "read_write" : qualifier},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::string qual_prefix = qualifier.empty() ? "" : (qualifier + " ");
    bindings_stream << std::format("layout(std430, binding = {}) {}buffer Buffer_{} {{ {} {}[]; }};\n",
                                   _binding_index, qual_prefix, _binding_index, resolved_type, name);
    return name;
}

void Shader_Generator::setPushConstants(const std::string &_struct_definition)
{
    if (has_push_constants)
    {
        throw std::logic_error("Shader_Generator::setPushConstants: Push constants block already defined");
    }
    has_push_constants = true;
    bindings_stream << "layout(push_constant) uniform PushConstants {\n";
    bindings_stream << _struct_definition << "\n";
    bindings_stream << "} pc;\n\n";
}

std::string Shader_Generator::addSharedMemory(uint32_t _size, const std::string &_prefix, const std::string &_type_name)
{
    if (_size == 0)
    {
        throw std::invalid_argument("Shader_Generator::addSharedMemory: Shared memory size must be >= 1");
    }
    std::string resolved_type = _type_name.empty() ? default_data_type : _type_name;
    if (resolved_type == "float16_t")
    {
        is_float16_enabled = true;
    }
    const size_t element_bytes = (resolved_type == "float16_t") ? 2ULL : 4ULL;
    total_shared_memory_bytes += static_cast<size_t>(_size) * element_bytes;
    if (total_shared_memory_bytes > 16384ULL)
    {
        Logger::logMessage(Input_Format{"Shader_Generator::addSharedMemory: Total shared memory ({} bytes) exceeds minimum guaranteed 16384-byte Vulkan limit",
                                        total_shared_memory_bytes},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
    std::string name = std::format("{}_{}", _prefix, var_counter++);
    Logger::logMessage(Input_Format{"Shader_Generator::addSharedMemory: Added shared memory array '{}' of size {} of type '{}'", name, _size, resolved_type},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    shared_memory_stream << std::format("shared {} {}[{}];\n", resolved_type, name, _size);
    return name;
}

void Shader_Generator::addSharedMemoryRaw(const std::string &_declaration)
{
    if (_declaration.find("float16_t") != std::string::npos)
    {
        is_float16_enabled = true;
    }
    shared_memory_stream << _declaration << "\n";
}

std::string Shader_Generator::getUniqueVar(const std::string &_prefix)
{
    return std::format("{}_{}", _prefix, var_counter++);
}

void Shader_Generator::addLogicSnippet(const std::string &_snippet)
{
    body_stream << _snippet << "\n";
}

std::string Shader_Generator::build() const
{
    Logger::logMessage("Shader_Generator::build: Building GLSL shader code",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    std::ostringstream final_shader;
    final_shader << "#version 450\n";

    if (is_cooperative_matrix_enabled)
    {
        final_shader << "#extension GL_KHR_cooperative_matrix : enable\n";
        final_shader << "#extension GL_KHR_memory_scope_semantics : enable\n";
    }

    if (is_float16_enabled || is_cooperative_matrix_enabled)
    {
        final_shader << "#extension GL_EXT_shader_16bit_storage : enable\n";
        final_shader << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n";
    }

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
    for (const auto &entry : spec_constants)
    {
        final_shader << std::format("layout(constant_id = {}) const {} {} = {};\n",
                                    entry.constant_id, entry.type_name, entry.name, entry.default_val);
    }
    if (!spec_constants.empty())
    {
        final_shader << "\n";
    }
    final_shader << bindings_stream.str();
    final_shader << shared_memory_stream.str();

    final_shader << "void main() {\n";
    if (group_y == 1 && group_z == 1)
    {
        final_shader << "    uint global_id = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * gl_WorkGroupSize.x + gl_LocalInvocationID.x;\n";
    }
    else
    {
        final_shader << "    uint global_id = gl_GlobalInvocationID.x;\n";
    }
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
