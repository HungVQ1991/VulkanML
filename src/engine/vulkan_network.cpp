#include "engine/vulkan_network.h"
#include "engine/pipeline_cache_manager.h"
#include "engine/vulkan_context.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/resource_path.h"

std::vector<uint32_t> Vulkan_Network::readSpirvFile(const std::string &_file_path) const
{
    Logger::logMessage(Input_Format{"Vulkan_Network::readSpirvFile: Reading SPIR-V file: {}", _file_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::ifstream file_stream(_file_path, std::ios::ate | std::ios::binary);
    if (!file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::readSpirvFile: Failed to open SPIR-V file: {}", _file_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to open SPIR-V file: " + _file_path);
    }

    std::streampos end_pos = file_stream.tellg();
    if (end_pos <= 0)
    {
        throw std::runtime_error("Invalid or empty SPIR-V file: " + _file_path);
    }

    size_t file_size = static_cast<size_t>(end_pos);
    if (file_size % sizeof(uint32_t) != 0)
    {
        throw std::runtime_error("SPIR-V file size is not a multiple of 4 bytes: " + _file_path);
    }

    std::vector<uint32_t> buffer_data(file_size / sizeof(uint32_t));
    file_stream.seekg(0, std::ios::beg);
    if (!file_stream.read(reinterpret_cast<char *>(buffer_data.data()), static_cast<std::streamsize>(file_size)))
    {
        throw std::runtime_error("Failed to read complete SPIR-V file: " + _file_path);
    }
    file_stream.close();

    return buffer_data;
}

VkShaderModule Vulkan_Network::createShaderModule(const std::vector<uint32_t> &_code_buffer) const
{
    if (_code_buffer.empty())
    {
        throw std::runtime_error("Vulkan_Network::createShaderModule: Empty SPIR-V buffer");
    }

    Logger::logMessage(Input_Format{"Vulkan_Network::createShaderModule: Creating shader module of size {} bytes", _code_buffer.size() * sizeof(uint32_t)},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    VkShaderModuleCreateInfo create_information{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = _code_buffer.size() * sizeof(uint32_t),
        .pCode = _code_buffer.data()};

    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &create_information, nullptr, &shader_module) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Network::createShaderModule: Failed to create shader module",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to create shader module");
    }

    return shader_module;
}

uint32_t Vulkan_Network::reflectSpirvPushConstantBytes(const std::vector<uint32_t> &_code_buffer) noexcept
{
    if (_code_buffer.size() < 5 || _code_buffer[0] != 0x07230203u)
    {
        return 0u;
    }

    enum class Type_Kind : uint8_t
    {
        SCALAR,
        VECTOR,
        MATRIX,
        ARRAY,
        STRUCT
    };
    struct Type_Desc
    {
        Type_Kind kind = Type_Kind::SCALAR;
        uint32_t scalar_bytes = 0;
        uint32_t elem_or_col_type = 0;
        uint32_t count_or_len_id = 0;
        std::vector<uint32_t> members;
    };

    std::unordered_map<uint32_t, Type_Desc> types;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> pointer_types;
    std::vector<uint32_t> push_constant_ptr_types;
    std::unordered_map<uint64_t, uint32_t> member_offsets;
    std::unordered_map<uint32_t, uint32_t> array_strides;
    std::unordered_map<uint32_t, uint32_t> matrix_strides;

    constexpr uint32_t OP_TYPE_INT = 21u;
    constexpr uint32_t OP_TYPE_FLOAT = 22u;
    constexpr uint32_t OP_TYPE_VECTOR = 23u;
    constexpr uint32_t OP_TYPE_MATRIX = 24u;
    constexpr uint32_t OP_TYPE_ARRAY = 28u;
    constexpr uint32_t OP_TYPE_STRUCT = 30u;
    constexpr uint32_t OP_TYPE_POINTER = 32u;
    constexpr uint32_t OP_CONSTANT = 43u;
    constexpr uint32_t OP_VARIABLE = 59u;
    constexpr uint32_t OP_DECORATE = 71u;
    constexpr uint32_t OP_MEMBER_DECORATE = 72u;
    constexpr uint32_t DEC_ARRAY_STRIDE = 6u;
    constexpr uint32_t DEC_MATRIX_STRIDE = 7u;
    constexpr uint32_t DEC_OFFSET = 35u;
    constexpr uint32_t SC_PUSH_CONSTANT = 9u;

    size_t i = 5;
    while (i < _code_buffer.size())
    {
        const uint32_t header = _code_buffer[i];
        const uint32_t op = header & 0xFFFFu;
        const uint32_t wc = header >> 16u;
        if (wc == 0u || i + wc > _code_buffer.size())
        {
            break;
        }
        if (op == OP_TYPE_INT && wc >= 4u)
        {
            types[_code_buffer[i + 1]] = Type_Desc{.kind = Type_Kind::SCALAR, .scalar_bytes = _code_buffer[i + 2] / 8u};
        }
        else if (op == OP_TYPE_FLOAT && wc >= 3u)
        {
            types[_code_buffer[i + 1]] = Type_Desc{.kind = Type_Kind::SCALAR, .scalar_bytes = _code_buffer[i + 2] / 8u};
        }
        else if (op == OP_TYPE_VECTOR && wc >= 4u)
        {
            types[_code_buffer[i + 1]] = Type_Desc{
                .kind = Type_Kind::VECTOR,
                .elem_or_col_type = _code_buffer[i + 2],
                .count_or_len_id = _code_buffer[i + 3]};
        }
        else if (op == OP_TYPE_MATRIX && wc >= 4u)
        {
            types[_code_buffer[i + 1]] = Type_Desc{
                .kind = Type_Kind::MATRIX,
                .elem_or_col_type = _code_buffer[i + 2],
                .count_or_len_id = _code_buffer[i + 3]};
        }
        else if (op == OP_TYPE_ARRAY && wc >= 4u)
        {
            types[_code_buffer[i + 1]] = Type_Desc{
                .kind = Type_Kind::ARRAY,
                .elem_or_col_type = _code_buffer[i + 2],
                .count_or_len_id = _code_buffer[i + 3]};
        }
        else if (op == OP_TYPE_STRUCT && wc >= 2u)
        {
            Type_Desc desc{.kind = Type_Kind::STRUCT};
            desc.members.assign(_code_buffer.begin() + static_cast<ptrdiff_t>(i + 2),
                                _code_buffer.begin() + static_cast<ptrdiff_t>(i + wc));
            types[_code_buffer[i + 1]] = std::move(desc);
        }
        else if (op == OP_TYPE_POINTER && wc >= 4u)
        {
            pointer_types[_code_buffer[i + 1]] = {_code_buffer[i + 2], _code_buffer[i + 3]};
        }
        else if (op == OP_CONSTANT && wc >= 4u)
        {
            constants[_code_buffer[i + 2]] = _code_buffer[i + 3];
        }
        else if (op == OP_VARIABLE && wc >= 4u)
        {
            if (_code_buffer[i + 3] == SC_PUSH_CONSTANT)
            {
                push_constant_ptr_types.push_back(_code_buffer[i + 1]);
            }
        }
        else if (op == OP_DECORATE && wc >= 4u)
        {
            if (_code_buffer[i + 2] == DEC_ARRAY_STRIDE)
            {
                array_strides[_code_buffer[i + 1]] = _code_buffer[i + 3];
            }
            else if (_code_buffer[i + 2] == DEC_MATRIX_STRIDE)
            {
                matrix_strides[_code_buffer[i + 1]] = _code_buffer[i + 3];
            }
        }
        else if (op == OP_MEMBER_DECORATE && wc >= 5u)
        {
            if (_code_buffer[i + 3] == DEC_OFFSET)
            {
                const uint64_t key = (static_cast<uint64_t>(_code_buffer[i + 1]) << 32u) |
                                     static_cast<uint64_t>(_code_buffer[i + 2]);
                member_offsets[key] = _code_buffer[i + 4];
            }
        }
        i += wc;
    }

    auto compute_type_size = [&](auto &self, uint32_t type_id, int depth) -> uint32_t
    {
        if (depth > 16)
        {
            return 0u;
        }
        auto it = types.find(type_id);
        if (it == types.end())
        {
            return 0u;
        }
        const Type_Desc &desc = it->second;
        switch (desc.kind)
        {
        case Type_Kind::SCALAR:
            return desc.scalar_bytes;
        case Type_Kind::VECTOR:
            return self(self, desc.elem_or_col_type, depth + 1) * desc.count_or_len_id;
        case Type_Kind::MATRIX:
        {
            auto ms_it = matrix_strides.find(type_id);
            const uint32_t col_stride = (ms_it != matrix_strides.end())
                ? ms_it->second
                : self(self, desc.elem_or_col_type, depth + 1);
            return col_stride * desc.count_or_len_id;
        }
        case Type_Kind::ARRAY:
        {
            auto len_it = constants.find(desc.count_or_len_id);
            if (len_it == constants.end())
            {
                return 0u;
            }
            auto as_it = array_strides.find(type_id);
            const uint32_t elem_stride = (as_it != array_strides.end())
                ? as_it->second
                : self(self, desc.elem_or_col_type, depth + 1);
            return len_it->second * elem_stride;
        }
        case Type_Kind::STRUCT:
        {
            uint32_t max_end = 0u;
            for (uint32_t m_idx = 0; m_idx < static_cast<uint32_t>(desc.members.size()); ++m_idx)
            {
                const uint64_t key = (static_cast<uint64_t>(type_id) << 32u) | static_cast<uint64_t>(m_idx);
                auto off_it = member_offsets.find(key);
                const uint32_t offset = (off_it != member_offsets.end()) ? off_it->second : 0u;
                max_end = std::max(max_end, offset + self(self, desc.members[m_idx], depth + 1));
            }
            return max_end;
        }
        }
        return 0u;
    };

    uint32_t max_bytes = 0u;
    for (uint32_t ptr_type_id : push_constant_ptr_types)
    {
        auto ptr_it = pointer_types.find(ptr_type_id);
        if (ptr_it != pointer_types.end())
        {
            max_bytes = std::max(max_bytes, compute_type_size(compute_type_size, ptr_it->second.second, 0));
        }
    }
    return max_bytes;
}

uint32_t Vulkan_Network::getPipelinePushConstantSize(Compute_Pipeline _pipeline) const noexcept
{
    const size_t pipeline_index = static_cast<size_t>(_pipeline);
    if (pipeline_index >= pipeline_push_constant_sizes.size())
    {
        return 0u;
    }
    return pipeline_push_constant_sizes[pipeline_index].load(std::memory_order_acquire);
}

VkPipeline Vulkan_Network::createComputePipeline(const std::string &_shader_path, bool is_coop, const uint32_t *_spec_constant_u32, uint32_t *_out_push_constant_bytes) const
{
    Logger::logMessage(Input_Format{"Vulkan_Network::createComputePipeline: Creating compute pipeline for shader: {}", _shader_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::vector<uint32_t> shader_code = readSpirvFile(_shader_path);
    if (_out_push_constant_bytes != nullptr)
    {
        *_out_push_constant_bytes = reflectSpirvPushConstantBytes(shader_code);
    }
    VkShaderModule shader_module = createShaderModule(shader_code);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_subgroup_size_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = 32};

    bool require_subgroup = is_coop && (context != nullptr && context->canRequireSubgroupSize(32));
    bool require_full_subgroups = require_subgroup && context->isComputeFullSubgroupsSupported();

    uint32_t spec_value = (_spec_constant_u32 != nullptr) ? *_spec_constant_u32 : 0u;
    VkSpecializationMapEntry spec_entry{
        .constantID = 0,
        .offset = 0,
        .size = sizeof(uint32_t)};
    VkSpecializationInfo spec_info{
        .mapEntryCount = 1,
        .pMapEntries = &spec_entry,
        .dataSize = sizeof(uint32_t),
        .pData = &spec_value};

    VkPipelineShaderStageCreateInfo shader_stage_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = require_subgroup ? &required_subgroup_size_create_information : nullptr,
        .flags = require_full_subgroups ? static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : VkPipelineShaderStageCreateFlags{0},
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = (_spec_constant_u32 != nullptr) ? &spec_info : nullptr};

    VkComputePipelineCreateInfo pipeline_create_information{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = shader_stage_create_information,
        .layout = pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1};

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult pipeline_res = VK_ERROR_INITIALIZATION_FAILED;
    if (shared_pipeline_cache_manager != nullptr)
    {
        pipeline_res = shared_pipeline_cache_manager->createComputePipelineWithCache(pipeline_create_information, &pipeline);
    }
    else
    {
        VkPipelineCache pipeline_cache_handle = (context != nullptr) ? context->getPipelineCache() : VK_NULL_HANDLE;
        pipeline_res = vkCreateComputePipelines(device, pipeline_cache_handle, 1, &pipeline_create_information, nullptr, &pipeline);
    }
    if (pipeline_res != VK_SUCCESS)
    {
        vkDestroyShaderModule(device, shader_module, nullptr);
        Logger::logMessage(Input_Format{"Vulkan_Network::createComputePipeline: Failed to create compute pipeline for shader: {}", _shader_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to create compute pipeline");
    }

    vkDestroyShaderModule(device, shader_module, nullptr);
    return pipeline;
}

bool Vulkan_Network::requiresFloat16(Compute_Pipeline _pipeline) noexcept
{
    if (_pipeline == Compute_Pipeline::FAKEQUANT_TERNARY)
    {
        return true;
    }
    std::string_view name = magic_enum::enum_name(_pipeline);
    return name.find("FP16") != std::string_view::npos;
}

bool Vulkan_Network::requiresCooperativeMatrix(Compute_Pipeline _pipeline) noexcept
{
    std::string_view name = magic_enum::enum_name(_pipeline);
    if (name.find("COOPMAT") != std::string_view::npos)
    {
        return true;
    }
    return (_pipeline == Compute_Pipeline::FLASH_ATTENTION_FP16 ||
            _pipeline == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_FP16 ||
            _pipeline == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16 ||
            _pipeline == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16);
}

VkPipeline Vulkan_Network::createPipelineForEnum(Compute_Pipeline _pipeline, const uint32_t *_spec_constant_u32) const
{
    if (requiresFloat16(_pipeline) && context != nullptr && !context->isFloat16Enabled())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::createPipelineForEnum: Skipping float16 pipeline {} because device does not have float16 enabled", magic_enum::enum_name(_pipeline)},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return VK_NULL_HANDLE;
    }

    bool is_coop_pipeline = requiresCooperativeMatrix(_pipeline);
    if (is_coop_pipeline && context != nullptr && !context->isCooperativeMatrixEnabled())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::createPipelineForEnum: Skipping cooperative matrix pipeline {} because device does not have cooperative matrix enabled", magic_enum::enum_name(_pipeline)},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return VK_NULL_HANDLE;
    }

    std::string shader_name = std::string(magic_enum::enum_name(_pipeline));
    std::string lower_shader_name = shader_name;
    std::ranges::transform(lower_shader_name, lower_shader_name.begin(), [](unsigned char character)
                           { return static_cast<char>(std::tolower(character)); });
    const std::string shader_path = (std::filesystem::path(pipeline_folder) / (lower_shader_name + ".spv")).generic_string();

    try
    {
        uint32_t reflected_pc_bytes = 0u;
        VkPipeline created = createComputePipeline(shader_path, is_coop_pipeline, _spec_constant_u32, &reflected_pc_bytes);
        const size_t pipeline_index = static_cast<size_t>(_pipeline);
        if (created != VK_NULL_HANDLE && pipeline_index < pipeline_push_constant_sizes.size())
        {
            pipeline_push_constant_sizes[pipeline_index].store(reflected_pc_bytes, std::memory_order_release);
        }
        return created;
    }
    catch (const std::exception &e)
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::createPipelineForEnum: Failed to compile pipeline {}: {}", shader_name, e.what()},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return VK_NULL_HANDLE;
    }
}

void Vulkan_Network::cleanUp() noexcept
{
    if (device == VK_NULL_HANDLE)
    {
        for (size_t i = 0; i < pipelines.size(); ++i)
        {
            pipeline_compile_failed[i].store(false, std::memory_order_release);
            pipeline_push_constant_sizes[i].store(0u, std::memory_order_release);
        }
        specialized_compile_failed.clear();
        specialized_pipelines.clear();
        return;
    }

    Logger::logMessage("Vulkan_Network::cleanUp: Cleaning up Vulkan network pipelines and layouts",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);

    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        pipeline_compile_failed[i].store(false, std::memory_order_release);
        pipeline_push_constant_sizes[i].store(0u, std::memory_order_release);
        VkPipeline pipeline = pipelines[i].exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
    }
    for (auto &[key, pipeline] : specialized_pipelines)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
    }
    specialized_pipelines.clear();
    specialized_compile_failed.clear();

    if (pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        pipeline_layout = VK_NULL_HANDLE;
    }

    if (descriptor_set_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        descriptor_set_layout = VK_NULL_HANDLE;
    }
}

Vulkan_Network::Vulkan_Network(const Vulkan_Context &_context, const std::string &_pipeline_folder)
    : context(&_context),
      pipeline_folder(Resource_Path::resolveDirectory(_pipeline_folder, "VULKANML_SHADER_DIR", "compute_shader/spv")),
      device(_context.getDevice())
{
    Logger::logMessage(Input_Format{"Vulkan_Network::Vulkan_Network: Initializing Vulkan Network with shader folder: {}", pipeline_folder},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SHADER_GENERATION);

    struct Cleanup_Guard
    {
        Vulkan_Network *network_instance;
        bool is_active = true;

        explicit Cleanup_Guard(Vulkan_Network *_network_instance)
            : network_instance(_network_instance) {}

        ~Cleanup_Guard()
        {
            if (is_active)
            {
                network_instance->cleanUp();
            }
        }
    } guard{this};

    VkDescriptorSetLayoutBinding layout_bindings[DESCRIPTOR_BINDINGS_COUNT]{};
    for (uint32_t i = 0; i < DESCRIPTOR_BINDINGS_COUNT; ++i)
    {
        layout_bindings[i] = VkDescriptorSetLayoutBinding{
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr};
    }

    VkDescriptorSetLayoutCreateFlags layout_flags = 0;
    if (context != nullptr && context->isPushDescriptorEnabled())
    {
        layout_flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        is_push_descriptor_active = true;
        Logger::logMessage("Vulkan_Network::Vulkan_Network: Configured descriptor set layout with VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR",
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
    }

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_information{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = layout_flags,
        .bindingCount = DESCRIPTOR_BINDINGS_COUNT,
        .pBindings = layout_bindings};

    if (vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_information, nullptr, &descriptor_set_layout) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create descriptor set layout",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    VkPushConstantRange push_constant_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128};

    VkPipelineLayoutCreateInfo pipeline_layout_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range};

    if (vkCreatePipelineLayout(device, &pipeline_layout_create_information, nullptr, &pipeline_layout) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create pipeline layout",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
        throw std::runtime_error("Failed to create pipeline layout");
    }

    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        pipelines[i].store(VK_NULL_HANDLE, std::memory_order_relaxed);
        pipeline_compile_failed[i].store(false, std::memory_order_relaxed);
        pipeline_push_constant_sizes[i].store(0u, std::memory_order_relaxed);
    }

    guard.is_active = false;
}

Vulkan_Network::Vulkan_Network(Vulkan_Network &&other) noexcept
    : context(other.context),
      shared_pipeline_cache_manager(other.shared_pipeline_cache_manager),
      pipeline_folder(std::move(other.pipeline_folder)),
      device(other.device),
      descriptor_set_layout(other.descriptor_set_layout),
      pipeline_layout(other.pipeline_layout),
      is_push_descriptor_active(other.is_push_descriptor_active)
{
    std::lock_guard<std::mutex> lock(other.pipeline_mutex);
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        pipelines[i].store(other.pipelines[i].exchange(VK_NULL_HANDLE, std::memory_order_acq_rel), std::memory_order_release);
        pipeline_compile_failed[i].store(other.pipeline_compile_failed[i].exchange(false, std::memory_order_acq_rel), std::memory_order_release);
        pipeline_push_constant_sizes[i].store(other.pipeline_push_constant_sizes[i].exchange(0u, std::memory_order_acq_rel), std::memory_order_release);
    }
    specialized_pipelines = std::move(other.specialized_pipelines);
    specialized_compile_failed = std::move(other.specialized_compile_failed);
    other.specialized_pipelines.clear();
    other.specialized_compile_failed.clear();
    other.context = nullptr;
    other.shared_pipeline_cache_manager = nullptr;
    other.device = VK_NULL_HANDLE;
    other.descriptor_set_layout = VK_NULL_HANDLE;
    other.pipeline_layout = VK_NULL_HANDLE;
    other.is_push_descriptor_active = false;
}

Vulkan_Network &Vulkan_Network::operator=(Vulkan_Network &&other) noexcept
{
    if (this != &other)
    {
        std::scoped_lock lock(pipeline_mutex, other.pipeline_mutex);
        cleanUp();
        context = other.context;
        shared_pipeline_cache_manager = other.shared_pipeline_cache_manager;
        pipeline_folder = std::move(other.pipeline_folder);
        device = other.device;
        descriptor_set_layout = other.descriptor_set_layout;
        pipeline_layout = other.pipeline_layout;
        is_push_descriptor_active = other.is_push_descriptor_active;
        for (size_t i = 0; i < pipelines.size(); ++i)
        {
            pipelines[i].store(other.pipelines[i].exchange(VK_NULL_HANDLE, std::memory_order_acq_rel), std::memory_order_release);
            pipeline_compile_failed[i].store(other.pipeline_compile_failed[i].exchange(false, std::memory_order_acq_rel), std::memory_order_release);
            pipeline_push_constant_sizes[i].store(other.pipeline_push_constant_sizes[i].exchange(0u, std::memory_order_acq_rel), std::memory_order_release);
        }
        specialized_pipelines = std::move(other.specialized_pipelines);
        specialized_compile_failed = std::move(other.specialized_compile_failed);
        other.specialized_pipelines.clear();
        other.specialized_compile_failed.clear();

        other.context = nullptr;
        other.shared_pipeline_cache_manager = nullptr;
        other.device = VK_NULL_HANDLE;
        other.descriptor_set_layout = VK_NULL_HANDLE;
        other.pipeline_layout = VK_NULL_HANDLE;
        other.is_push_descriptor_active = false;
    }
    return *this;
}

Vulkan_Network::~Vulkan_Network()
{
    cleanUp();
}

VkPipeline Vulkan_Network::getPipeline(Compute_Pipeline _pipeline) const
{
    size_t pipeline_index = static_cast<size_t>(_pipeline);
    if (pipeline_index >= pipelines.size())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::getPipeline: Pipeline index out of bounds ({})", pipeline_index},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);
        return VK_NULL_HANDLE;
    }

    if (requiresFloat16(_pipeline) && context != nullptr && !context->isFloat16Enabled())
    {
        return VK_NULL_HANDLE;
    }
    if (requiresCooperativeMatrix(_pipeline) && context != nullptr && !context->isCooperativeMatrixEnabled())
    {
        return VK_NULL_HANDLE;
    }

    VkPipeline cached_pipeline = pipelines[pipeline_index].load(std::memory_order_acquire);
    if (cached_pipeline != VK_NULL_HANDLE)
    {
        return cached_pipeline;
    }

    if (pipeline_compile_failed[pipeline_index].load(std::memory_order_acquire))
    {
        return VK_NULL_HANDLE;
    }

    std::lock_guard<std::mutex> lock(pipeline_mutex);

    cached_pipeline = pipelines[pipeline_index].load(std::memory_order_relaxed);
    if (cached_pipeline != VK_NULL_HANDLE)
    {
        return cached_pipeline;
    }

    if (pipeline_compile_failed[pipeline_index].load(std::memory_order_relaxed))
    {
        return VK_NULL_HANDLE;
    }

    VkPipeline new_pipeline = createPipelineForEnum(_pipeline);
    if (new_pipeline == VK_NULL_HANDLE)
    {
        pipeline_compile_failed[pipeline_index].store(true, std::memory_order_release);
        return VK_NULL_HANDLE;
    }

    pipelines[pipeline_index].store(new_pipeline, std::memory_order_release);
    return new_pipeline;
}

VkPipeline Vulkan_Network::getSpecializedPipeline(Compute_Pipeline _pipeline, uint32_t _spec_constant_u32) const
{
    if (_spec_constant_u32 == 0u)
    {
        return getPipeline(_pipeline);
    }

    size_t pipeline_index = static_cast<size_t>(_pipeline);
    if (pipeline_index >= pipelines.size())
    {
        return VK_NULL_HANDLE;
    }
    if (requiresFloat16(_pipeline) && context != nullptr && !context->isFloat16Enabled())
    {
        return VK_NULL_HANDLE;
    }
    if (requiresCooperativeMatrix(_pipeline) && context != nullptr && !context->isCooperativeMatrixEnabled())
    {
        return VK_NULL_HANDLE;
    }

    const uint64_t key = (static_cast<uint64_t>(pipeline_index) << 32u) | static_cast<uint64_t>(_spec_constant_u32);
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (auto it = specialized_pipelines.find(key); it != specialized_pipelines.end())
    {
        return it->second;
    }
    if (specialized_compile_failed.contains(key))
    {
        return VK_NULL_HANDLE;
    }

    VkPipeline created = createPipelineForEnum(_pipeline, &_spec_constant_u32);
    if (created == VK_NULL_HANDLE)
    {
        specialized_compile_failed.insert(key);
        return VK_NULL_HANDLE;
    }
    specialized_pipelines.emplace(key, created);
    return created;
}

void Vulkan_Network::setPipelines(const std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> &_pipelines)
{
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        VkPipeline old_pipe = pipelines[i].exchange(_pipelines[i], std::memory_order_acq_rel);
        if (device != VK_NULL_HANDLE && old_pipe != VK_NULL_HANDLE && old_pipe != _pipelines[i])
        {
            vkDestroyPipeline(device, old_pipe, nullptr);
        }
        pipeline_compile_failed[i].store(false, std::memory_order_release);
        pipeline_push_constant_sizes[i].store(0u, std::memory_order_release);
    }
    for (auto &[key, old_pipe] : specialized_pipelines)
    {
        if (device != VK_NULL_HANDLE && old_pipe != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, old_pipe, nullptr);
        }
    }
    specialized_pipelines.clear();
    specialized_compile_failed.clear();
}

void Vulkan_Network::setPipelineFolder(const std::string &_pipeline_folder)
{
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (_pipeline_folder.empty())
    {
        Logger::logMessage("Vulkan_Network::setPipelineFolder: Attempted to set empty folder path",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return;
    }
    const std::string resolved_folder = Resource_Path::resolveDirectory(_pipeline_folder, nullptr, "compute_shader/spv");
    if (pipeline_folder == resolved_folder)
    {
        return;
    }
    Logger::logMessage(Input_Format{"Vulkan_Network::setPipelineFolder: Changing compute shader folder to {}", resolved_folder},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        VkPipeline old_pipe = pipelines[i].exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
        if (device != VK_NULL_HANDLE && old_pipe != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, old_pipe, nullptr);
        }
        pipeline_compile_failed[i].store(false, std::memory_order_release);
        pipeline_push_constant_sizes[i].store(0u, std::memory_order_release);
    }
    for (auto &[key, old_pipe] : specialized_pipelines)
    {
        if (device != VK_NULL_HANDLE && old_pipe != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, old_pipe, nullptr);
        }
    }
    specialized_pipelines.clear();
    specialized_compile_failed.clear();
    pipeline_folder = resolved_folder;
}

void Vulkan_Network::setPipelineCacheManager(const Pipeline_Cache_Manager *_manager) noexcept
{
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    shared_pipeline_cache_manager = _manager;
}

void Vulkan_Network::setPipelineLayout(VkPipelineLayout _pipeline_layout)
{
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (pipeline_layout == _pipeline_layout)
    {
        return;
    }
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
        for (size_t i = 0; i < pipelines.size(); ++i)
        {
            pipeline_compile_failed[i].store(false, std::memory_order_release);
            pipeline_push_constant_sizes[i].store(0u, std::memory_order_release);
            VkPipeline old_pipe = pipelines[i].exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
            if (old_pipe != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, old_pipe, nullptr);
            }
        }
        for (auto &[key, old_pipe] : specialized_pipelines)
        {
            if (old_pipe != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, old_pipe, nullptr);
            }
        }
        specialized_pipelines.clear();
        specialized_compile_failed.clear();
        if (pipeline_layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        }
    }
    pipeline_layout = _pipeline_layout;
}

void Vulkan_Network::setDescriptorSetLayout(VkDescriptorSetLayout _layout)
{
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (descriptor_set_layout == _layout)
    {
        return;
    }
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
        for (size_t i = 0; i < pipelines.size(); ++i)
        {
            pipeline_compile_failed[i].store(false, std::memory_order_release);
            pipeline_push_constant_sizes[i].store(0u, std::memory_order_release);
            VkPipeline old_pipe = pipelines[i].exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
            if (old_pipe != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, old_pipe, nullptr);
            }
        }
        for (auto &[key, old_pipe] : specialized_pipelines)
        {
            if (old_pipe != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, old_pipe, nullptr);
            }
        }
        specialized_pipelines.clear();
        specialized_compile_failed.clear();
        if (descriptor_set_layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        }
    }
    descriptor_set_layout = _layout;
}

void Vulkan_Network::setDevice(VkDevice _device)
{
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (device == _device)
    {
        return;
    }
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }
    cleanUp();
    device = _device;
}
