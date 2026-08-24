#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/json.hpp"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
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
    std::uint32_t input_count = 0;
    std::uint32_t output_count = 0;
    std::uint32_t shared_memory_size = 0;
    bool is_writing_multiple_elements = false;
    std::vector<std::uint32_t> accumulator_output_indices;
    std::vector<std::uint32_t> persistent_output_indices;
    std::string glsl_template;
    std::string index_expression;
    Operation_Class operation_class = Operation_Class::STANDALONE;
};

class Shader_Dictionary
{
private:
    using json = nlohmann::json;
    std::array<Snippet_Metadata, Compute_Pipeline::COMPUTE_PIPELINE_END> metadata_table;

public:
    explicit Shader_Dictionary(const std::string &_file_path = "compute_shader/shader_dictionary.json")
    {
        Logger::logMessage("Shader_Dictionary::Shader_Dictionary: Initializing shader dictionary",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        loadFromFile(_file_path);
    }

    ~Shader_Dictionary() = default;

    Shader_Dictionary(const Shader_Dictionary &) = delete;
    Shader_Dictionary &operator=(const Shader_Dictionary &) = delete;

    Shader_Dictionary(Shader_Dictionary &&other) noexcept = default;
    Shader_Dictionary &operator=(Shader_Dictionary &&other) noexcept = default;

    static const Shader_Dictionary &getInstance(const std::string &_file_path = "compute_shader/shader_dictionary.json")
    {
        static Shader_Dictionary instance(_file_path);
        return instance;
    }

    void loadFromFile(const std::string &_file_path)
    {
        Logger::logMessage(std::format("Shader_Dictionary::loadFromFile: Loading shader metadata from {}", _file_path),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);

        std::ifstream file_stream(_file_path);
        if (!file_stream.is_open())
        {
            Logger::logMessage(std::format("Shader_Dictionary::loadFromFile: Failed to open {}", _file_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("Failed to open shader dictionary file");
        }

        json root_json;
        file_stream >> root_json;

        for (std::size_t pipeline_index = 0; pipeline_index < Compute_Pipeline::COMPUTE_PIPELINE_END; ++pipeline_index)
        {
            auto pipeline_enum = static_cast<Compute_Pipeline>(pipeline_index);
            std::string pipeline_name = std::string(magic_enum::enum_name(pipeline_enum));

            if (root_json.contains(pipeline_name))
            {
                const auto &json_entry = root_json[pipeline_name];

                Operation_Class parsed_operation_class = Operation_Class::STANDALONE;
                std::string operation_class_string = json_entry.value("op_class", "STANDALONE");

                if (operation_class_string == "MATRIX_2D")
                {
                    parsed_operation_class = Operation_Class::MATRIX_2D;
                }
                else if (operation_class_string == "ELEMENTWISE")
                {
                    parsed_operation_class = Operation_Class::ELEMENTWISE;
                }
                else if (operation_class_string == "TENSOR_3D")
                {
                    parsed_operation_class = Operation_Class::TENSOR_3D;
                }
                else if (operation_class_string == "STANDALONE" || operation_class_string == "REDUCTION")
                {
                    parsed_operation_class = Operation_Class::STANDALONE;
                }

                metadata_table[pipeline_index] = Snippet_Metadata{
                    .input_count = json_entry.value("input_count", 0u),
                    .output_count = json_entry.value("output_count", 0u),
                    .shared_memory_size = json_entry.value("shared_mem_size", 0u),
                    .is_writing_multiple_elements = json_entry.value("writes_multiple_elements", false),
                    .accumulator_output_indices = json_entry.value("accumulator_output_indices", std::vector<std::uint32_t>{}),
                    .persistent_output_indices = json_entry.value("persistent_outputs", std::vector<std::uint32_t>{}),
                    .glsl_template = json_entry.value("code", ""),
                    .index_expression = json_entry.value("index_expr", ""),
                    .operation_class = parsed_operation_class};
            }
        }
    }

    [[nodiscard]] const Snippet_Metadata &getMetadata(Compute_Pipeline _pipeline) const noexcept
    {
        return metadata_table[static_cast<std::size_t>(_pipeline)];
    }

    [[nodiscard]] const Snippet_Metadata &getSnippetMetadata(Compute_Pipeline _pipeline) const noexcept
    {
        return metadata_table[static_cast<std::size_t>(_pipeline)];
    }

    [[nodiscard]] const std::array<Snippet_Metadata, Compute_Pipeline::COMPUTE_PIPELINE_END> &getMetadataTable() const noexcept
    {
        return metadata_table;
    }

    [[nodiscard]] bool hasMetadata(Compute_Pipeline _pipeline) const noexcept
    {
        std::size_t pipeline_index = static_cast<std::size_t>(_pipeline);
        return pipeline_index < metadata_table.size() && !metadata_table[pipeline_index].glsl_template.empty();
    }
};