#include "engine/shader_dictionary.h"

#include <format>
#include <fstream>
#include <stdexcept>

#include "helper/json.hpp"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"

Shader_Dictionary::Shader_Dictionary(const std::string &_file_path)
{
    Logger::logMessage("Shader_Dictionary::Shader_Dictionary: Initializing shader dictionary",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    loadFromFile(_file_path);
}

const Shader_Dictionary &Shader_Dictionary::getInstance()
{
    static Shader_Dictionary instance("compute_shader/shader_dictionary.json");
    return instance;
}

const Shader_Dictionary &Shader_Dictionary::getInstance(const std::string &_file_path)
{
    static Shader_Dictionary instance(_file_path);
    return instance;
}

void Shader_Dictionary::loadFromFile(const std::string &_file_path)
{
    Logger::logMessage(Input_Format{"Shader_Dictionary::loadFromFile: Loading shader metadata from {}", _file_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::ifstream file_stream(_file_path);
    if (!file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Shader_Dictionary::loadFromFile: Failed to open {}", _file_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to open shader dictionary file");
    }

    nlohmann::json root_json;
    file_stream >> root_json;

    for (size_t pipeline_index = 0; pipeline_index < static_cast<size_t>(Compute_Pipeline::COMPUTE_PIPELINE_END); ++pipeline_index)
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
                .cooperative_shared_memory_size = json_entry.value("cooperative_shared_mem_size", 0u),
                .is_writing_multiple_elements = json_entry.value("writes_multiple_elements", false),
                .is_cooperative_matrix_support = json_entry.value("supports_cooperative_matrix", false),
                .accumulator_output_indices = json_entry.value("accumulator_output_indices", std::vector<uint32_t>{}),
                .persistent_output_indices = json_entry.value("persistent_outputs", std::vector<uint32_t>{}),
                .glsl_template = json_entry.value("code", ""),
                .cooperative_glsl_template = json_entry.value("cooperative_matrix_code", ""),
                .index_expression = json_entry.value("index_expr", ""),
                .operation_class = parsed_operation_class};
        }
    }
}
