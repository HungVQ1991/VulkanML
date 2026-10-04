#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vulkan_network.h"

enum class Operation_Class
{
    ELEMENTWISE,
    MATRIX_2D,
    TENSOR_3D,
    STANDALONE,
    OPERATION_CLASS_END
};

struct Snippet_Metadata
{
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    uint32_t shared_memory_size = 0;
    uint32_t cooperative_shared_memory_size = 0;
    bool is_writing_multiple_elements = false;
    bool is_cooperative_matrix_support = false;
    std::vector<uint32_t> accumulator_output_indices;
    std::vector<uint32_t> persistent_output_indices;
    std::string glsl_template;
    std::string cooperative_glsl_template;
    std::string index_expression;
    Operation_Class operation_class = Operation_Class::STANDALONE;
};

class Shader_Dictionary
{
private:
    std::array<Snippet_Metadata, static_cast<size_t>(Compute_Pipeline::COMPUTE_PIPELINE_END)> metadata_table;

public:
    explicit Shader_Dictionary(const std::string &_file_path = "compute_shader/shader_dictionary.json");
    ~Shader_Dictionary() = default;

    Shader_Dictionary(const Shader_Dictionary &) = delete;
    Shader_Dictionary &operator=(const Shader_Dictionary &) = delete;

    Shader_Dictionary(Shader_Dictionary &&_other) noexcept = default;
    Shader_Dictionary &operator=(Shader_Dictionary &&_other) noexcept = default;

    static const Shader_Dictionary &getInstance();
    static const Shader_Dictionary &getInstance(const std::string &_file_path);

    void loadFromFile(const std::string &_file_path);

    const std::array<Snippet_Metadata, static_cast<size_t>(Compute_Pipeline::COMPUTE_PIPELINE_END)> &getMetadataTable() const noexcept { return metadata_table; }
    const Snippet_Metadata &getSnippetMetadata(Compute_Pipeline _pipeline) const noexcept { return metadata_table[static_cast<size_t>(_pipeline)]; }
    const Snippet_Metadata &getMetadata(Compute_Pipeline _pipeline) const noexcept { return metadata_table[static_cast<size_t>(_pipeline)]; }
    const std::string &getGlslTemplate(Compute_Pipeline _pipeline, bool _use_cooperative_matrix) const noexcept
    {
        const auto &meta = metadata_table[static_cast<size_t>(_pipeline)];
        if (_use_cooperative_matrix && meta.is_cooperative_matrix_support && !meta.cooperative_glsl_template.empty())
        {
            return meta.cooperative_glsl_template;
        }
        return meta.glsl_template;
    }
    uint32_t getSharedMemorySize(Compute_Pipeline _pipeline, bool _use_cooperative_matrix) const noexcept
    {
        const auto &meta = metadata_table[static_cast<size_t>(_pipeline)];
        if (_use_cooperative_matrix && meta.is_cooperative_matrix_support)
        {
            return meta.cooperative_shared_memory_size;
        }
        return meta.shared_memory_size;
    }
    bool hasMetadata(Compute_Pipeline _pipeline) const noexcept
    {
        size_t pipeline_index = static_cast<size_t>(_pipeline);
        return pipeline_index < metadata_table.size() && !metadata_table[pipeline_index].glsl_template.empty();
    }

    void setMetadataTable(const std::array<Snippet_Metadata, static_cast<size_t>(Compute_Pipeline::COMPUTE_PIPELINE_END)> &_table) noexcept { metadata_table = _table; }
    void setMetadata(Compute_Pipeline _pipeline, const Snippet_Metadata &_metadata) noexcept { metadata_table[static_cast<size_t>(_pipeline)] = _metadata; }
};