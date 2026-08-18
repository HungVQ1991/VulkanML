#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/json.hpp"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "vulkan_network.h"

#ifndef ENABLE_SHADER_DEBUG_LOGS
#define ENABLE_SHADER_DEBUG_LOGS 0
#endif

#if ENABLE_SHADER_DEBUG_LOGS
#define SHADER_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define SHADER_LOG_DEBUG(msg) ((void)0)
#endif

enum class Op_Class
{
    ELEMENTWISE,
    MATRIX_2D,
    TENSOR_3D,
    STANDALONE,
    OP_CLASS_END
};

struct Snippet_Metadata
{
    std::uint32_t input_count = 0;
    std::uint32_t output_count = 0;
    std::uint32_t shared_mem_size = 0;
    bool writes_multiple_elements = false;
    std::vector<std::uint32_t> accumulator_output_indices;
    std::vector<std::uint32_t> persistent_outputs;
    std::string glsl_template;
    std::string index_expr;
    Op_Class op_class = Op_Class::STANDALONE;
};

class Shader_Dictionary
{
private:
    using json = nlohmann::json;
    std::array<Snippet_Metadata, Compute_Pipeline::COMPUTE_PIPELINE_END> lookup_table;

public:
    explicit Shader_Dictionary(const std::string &file_path = "compute_shader/shader_dictionary.json")
    {
        SHADER_LOG_DEBUG("Shader_Dictionary::Shader_Dictionary: Initializing shader dictionary");
        loadFromFile(file_path);
    }
    ~Shader_Dictionary() = default;

    Shader_Dictionary(const Shader_Dictionary &) = delete;
    Shader_Dictionary &operator=(const Shader_Dictionary &) = delete;

    static const Shader_Dictionary &getInstance(const std::string &file_path = "compute_shader/shader_dictionary.json")
    {
        static Shader_Dictionary instance(file_path);
        return instance;
    }

    void loadFromFile(const std::string &file_path)
    {
        SHADER_LOG_DEBUG("Shader_Dictionary::loadFromFile: Loading shader metadata from " + file_path);
        std::ifstream file_stream(file_path);
        if (!file_stream.is_open())
        {
            Logger::logMessage("Shader_Dictionary::loadFromFile: Failed to open " + file_path, LOG_ERROR, true);
            throw std::runtime_error("Failed to open shader dictionary file");
        }

        json root_json;
        file_stream >> root_json;

        for (std::size_t i = 0; i < Compute_Pipeline::COMPUTE_PIPELINE_END; ++i)
        {
            auto op_enum = static_cast<Compute_Pipeline>(i);
            std::string enum_name = std::string(magic_enum::enum_name(op_enum));

            if (root_json.contains(enum_name))
            {
                const auto &entry = root_json[enum_name];

                Op_Class parsed_class = Op_Class::STANDALONE;
                std::string class_str = entry.value("op_class", "STANDALONE");
                if (class_str == "MATRIX_2D")
                {
                    parsed_class = Op_Class::MATRIX_2D;
                }
                else if (class_str == "ELEMENTWISE")
                {
                    parsed_class = Op_Class::ELEMENTWISE;
                }
                else if (class_str == "TENSOR_3D")
                {
                    parsed_class = Op_Class::TENSOR_3D;
                }
                else if (class_str == "STANDALONE" || class_str == "REDUCTION")
                {
                    parsed_class = Op_Class::STANDALONE;
                }

                lookup_table[i] = Snippet_Metadata{
                    .input_count = entry.value("input_count", 0u),
                    .output_count = entry.value("output_count", 0u),
                    .shared_mem_size = entry.value("shared_mem_size", 0u),
                    .writes_multiple_elements = entry.value("writes_multiple_elements", false),
                    .accumulator_output_indices = entry.value("accumulator_output_indices", std::vector<std::uint32_t>{}),
                    .persistent_outputs = entry.value("persistent_outputs", std::vector<std::uint32_t>{}),
                    .glsl_template = entry.value("code", ""),
                    .index_expr = entry.value("index_expr", ""),
                    .op_class = parsed_class};
            }
        }
    }

    const Snippet_Metadata &getMetadata(Compute_Pipeline pipeline) const
    {
        return lookup_table[static_cast<std::size_t>(pipeline)];
    }
};