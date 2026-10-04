#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

extern bool is_coop;

enum class Buffer_Access
{
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
};

struct Specialization_Constant_Entry
{
    uint32_t constant_id;
    std::string name;
    std::string type_name;
    std::string default_val;

    const std::string &getDefaultVal() const noexcept { return default_val; }
    const std::string &getTypeName() const noexcept { return type_name; }
    const std::string &getName() const noexcept { return name; }
    uint32_t getConstantId() const noexcept { return constant_id; }

    void setDefaultVal(const std::string &_default_val) { default_val = _default_val; }
    void setTypeName(const std::string &_type_name) { type_name = _type_name; }
    void setName(const std::string &_name) { name = _name; }
    void setConstantId(uint32_t _constant_id) noexcept { constant_id = _constant_id; }
};

class Specialization_Map_Builder
{
private:
    std::vector<VkSpecializationMapEntry> entries;
    std::vector<std::uint8_t> data;

public:
    template <typename T>
    void addConstant(uint32_t _constant_id, const T &_value)
    {
        size_t offset = data.size();
        size_t size = sizeof(T);

        entries.push_back(VkSpecializationMapEntry{
            .constantID = _constant_id,
            .offset = static_cast<uint32_t>(offset),
            .size = size});

        const auto *byte_pointer = reinterpret_cast<const std::uint8_t *>(&_value);
        data.insert(data.end(), byte_pointer, byte_pointer + size);
    }

    VkSpecializationInfo build() const noexcept
    {
        return VkSpecializationInfo{
            .mapEntryCount = static_cast<uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = data.size(),
            .pData = data.data()};
    }

    bool empty() const noexcept { return entries.empty(); }
    void clear() noexcept
    {
        entries.clear();
        data.clear();
    }

    const std::vector<VkSpecializationMapEntry> &getEntries() const noexcept { return entries; }
    const std::vector<std::uint8_t> &getData() const noexcept { return data; }

    void setEntries(const std::vector<VkSpecializationMapEntry> &_entries) { entries = _entries; }
    void setData(const std::vector<std::uint8_t> &_data) { data = _data; }
};

class Shader_Generator
{
private:
    uint32_t group_x;
    uint32_t group_y;
    uint32_t group_z;
    uint32_t current_binding = 0;
    uint32_t var_counter = 0;
    std::string default_data_type = "float";

    std::ostringstream header_stream;
    std::ostringstream specialization_stream;
    std::ostringstream bindings_stream;
    std::ostringstream shared_memory_stream;
    std::ostringstream body_stream;

    bool is_subgroup_enabled = false;
    bool is_control_flow_enabled = false;
    bool is_float16_enabled = false;
    std::vector<Specialization_Constant_Entry> spec_constants;

public:
    Shader_Generator(uint32_t _group_x, uint32_t _group_y = 1, uint32_t _group_z = 1, const std::string &_default_data_type = "float");

    void enableFloat16() noexcept { is_float16_enabled = true; }
    void enableSubgroupOperations();
    void enableControlFlowAttributes();

    void addSpecializationConstant(uint32_t _constant_id,
                                   const std::string &_name,
                                   const std::string &_type_name = "uint",
                                   const std::string &_default_value = "0");

    std::string addBuffer(uint32_t _binding_index,
                          const std::string &_buffer_name = "",
                          const std::string &_type_name = "",
                          Buffer_Access _access = Buffer_Access::READ_WRITE);

    void setPushConstants(const std::string &_struct_definition);

    std::string addSharedMemory(uint32_t _size, const std::string &_prefix = "shared_mem", const std::string &_type_name = "");
    void addSharedMemoryRaw(const std::string &_declaration);

    std::string getUniqueVar(const std::string &_prefix = "val");
    void addLogicSnippet(const std::string &_snippet);

    std::string build() const;

    const std::string &getDefaultDataType() const noexcept { return default_data_type; }
    void setDefaultDataType(const std::string &_type_name) noexcept { default_data_type = _type_name; }
    const std::vector<Specialization_Constant_Entry> &getSpecializationConstants() const noexcept { return spec_constants; }
    void setSpecializationConstants(const std::vector<Specialization_Constant_Entry> &_constants) { spec_constants = _constants; }
    uint32_t getCurrentBinding() const noexcept { return current_binding; }
    uint32_t getVarCounter() const noexcept { return var_counter; }
    uint32_t getGroupX() const noexcept { return group_x; }
    uint32_t getGroupY() const noexcept { return group_y; }
    uint32_t getGroupZ() const noexcept { return group_z; }
    void setCurrentBinding(uint32_t _binding) noexcept { current_binding = _binding; }
    void setVarCounter(uint32_t _counter) noexcept { var_counter = _counter; }
    void setGroupX(uint32_t _group_x) noexcept { group_x = _group_x; }
    void setGroupY(uint32_t _group_y) noexcept { group_y = _group_y; }
    void setGroupZ(uint32_t _group_z) noexcept { group_z = _group_z; }
    bool isControlFlowEnabled() const noexcept { return is_control_flow_enabled; }
    bool isSubgroupEnabled() const noexcept { return is_subgroup_enabled; }
    bool isFloat16Enabled() const noexcept { return is_float16_enabled; }
    void setControlFlowEnabled(bool _enabled) noexcept { is_control_flow_enabled = _enabled; }
    void setSubgroupEnabled(bool _enabled) noexcept { is_subgroup_enabled = _enabled; }
    void setFloat16Enabled(bool _enabled) noexcept { is_float16_enabled = _enabled; }
};