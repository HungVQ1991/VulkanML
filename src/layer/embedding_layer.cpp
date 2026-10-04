#include "layer/embedding_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"


void Embedding_Layer::initializeWeights()
{
        std::vector<float> weight_data(vocab_size * embedding_dim);
        float stddev = 0.02f;
        std::mt19937 gen(1337);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (float &val : weight_data)
        {
            val = dist(gen);
        }

        weight = Tensor(Shape{ vocab_size, embedding_dim }, weight_data, execution_target);
        weight_gradient = Tensor(Shape{ vocab_size, embedding_dim }, execution_target);
        weight_gradient.zero();
    }

Embedding_Layer::Embedding_Layer(size_t _vocab_size, size_t _embedding_dim, Execution_Target _execution_target, Data_Type _data_type)
    : vocab_size(_vocab_size),
          embedding_dim(_embedding_dim),
          execution_target(_execution_target),
          data_type(_data_type),
          output_tensor(_execution_target)
{
        output_tensor.setDataType(_data_type);
        initializeWeights();
    }

Tensor Embedding_Layer::forward(const std::vector<int32_t> &token_ids) const
{
        if (token_ids.empty())
        {
            return Tensor(Shape{ 0, embedding_dim }, data_type, execution_target);
        }
        std::vector<float> idx_float(token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i)
        {
            idx_float[i] = static_cast<float>(token_ids[i]);
        }
        last_indices_tensor = Tensor(Shape{ token_ids.size() }, std::move(idx_float), execution_target);
        output_tensor.setDataType(data_type);
        output_tensor.reshape(Shape{ token_ids.size(), embedding_dim });
        weight.embeddingForward(last_indices_tensor, output_tensor);
        return output_tensor;
    }

Tensor Embedding_Layer::forward(const std::vector<std::vector<int32_t>> &batch_token_ids) const
{
        if (batch_token_ids.empty() || batch_token_ids[0].empty())
        {
            return Tensor(Shape{ 0, 0, embedding_dim }, data_type, execution_target);
        }
        size_t B = batch_token_ids.size();
        size_t S = batch_token_ids[0].size();
        std::vector<float> idx_float(B * S);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t s = 0; s < S; ++s)
            {
                idx_float[b * S + s] = static_cast<float>(batch_token_ids[b][s]);
            }
        }
        last_indices_tensor = Tensor(Shape{ B, S }, std::move(idx_float), execution_target);
        output_tensor.setDataType(data_type);
        output_tensor.reshape(Shape{ B, S, embedding_dim });
        weight.embeddingForward(last_indices_tensor, output_tensor);
        return output_tensor;
    }

Tensor Embedding_Layer::forward(const Tensor &_input_tensor)
{
        if (_input_tensor.getTotalElements() == 0)
        {
            return Tensor(Shape{ 0, embedding_dim }, data_type, execution_target);
        }
        last_indices_tensor = _input_tensor;
        output_tensor.setDataType(data_type);
        output_tensor.reshape(Shape{ _input_tensor.getTotalElements(), embedding_dim });
        weight.embeddingForward(last_indices_tensor, output_tensor);
        if (_input_tensor.getShape().getRank() == 2)
        {
            output_tensor.reshape(Shape{ _input_tensor.getShape()[0], _input_tensor.getShape()[1], embedding_dim });
        }
        return output_tensor;
    }

Tensor Embedding_Layer::backward(const Tensor &_output_gradient)
{
        if (last_indices_tensor.getTotalElements() == 0 || _output_gradient.getTotalElements() == 0)
        {
            return Tensor(0, 0);
        }

        Tensor flat_grad = _output_gradient;
        if (flat_grad.getShape().getRank() == 3)
        {
            flat_grad.reshape(Shape{ flat_grad.getTotalElements() / embedding_dim, embedding_dim });
        }

        weight.embeddingBackward(last_indices_tensor, flat_grad, weight_gradient);
        return Tensor(0, 0);
    }

void Embedding_Layer::resetGradient()
{
        weight_gradient.zero();
    }

void Embedding_Layer::resetGradients()
{
        resetGradient();
    }

std::vector<std::pair<Tensor *, Tensor *>> Embedding_Layer::getParametersAndGradients()
{
        return { { &weight, &weight_gradient } };
    }

void Embedding_Layer::setExecutionTarget(Execution_Target _execution_target)
{
        logChangeExecutionTarget(_execution_target);
        execution_target = _execution_target;
        weight.setExecutionTarget(_execution_target);
        weight_gradient.setExecutionTarget(_execution_target);
        output_tensor = Tensor(_execution_target);
        output_tensor.setDataType(data_type);
    }

void Embedding_Layer::setMixedPrecision(bool _enable) noexcept
{
        ILayer::setMixedPrecision(_enable);
        data_type = _enable ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        output_tensor.setDataType(data_type);
        weight.invalidateFp16Cache();
    }

void Embedding_Layer::invalidateWeightCache() noexcept
{
        weight.invalidateFp16Cache();
    }

std::unique_ptr<ILayer> Embedding_Layer::clone() const
{
        auto copy = std::make_unique<Embedding_Layer>(vocab_size, embedding_dim, execution_target, data_type);
        copy->weight = weight;
        copy->weight_gradient = weight_gradient;
        return copy;
    }

void Embedding_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        uint64_t v_size = vocab_size;
        uint64_t e_dim = embedding_dim;
        _output_file_stream.write(reinterpret_cast<const char *>(&v_size), sizeof(v_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&e_dim), sizeof(e_dim));
    }

void Embedding_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        weight.saveTensor(_output_file_stream);
    }

void Embedding_Layer::loadInference(std::ifstream &_input_file_stream)
{
        weight = Tensor::loadTensor(_input_file_stream, execution_target);
        weight.invalidateFp16Cache();
    }

void Embedding_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        weight.saveTensor(_output_file_stream);
        weight_gradient.saveTensor(_output_file_stream);
    }

void Embedding_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        weight = Tensor::loadTensor(_input_file_stream, execution_target);
        weight_gradient = Tensor::loadTensor(_input_file_stream, execution_target);
        weight.invalidateFp16Cache();
    }

std::function<float(std::mt19937&)> Embedding_Layer::getPopulationParameterInitializer(size_t) const
{
        float stddev = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
        return [stddev](std::mt19937 &gen) {
            std::normal_distribution<float> dist(0.0f, stddev);
            return dist(gen);
        };
    }

size_t Embedding_Layer::getParameterCount() const noexcept
{ return vocab_size * embedding_dim; }

const Tensor & Embedding_Layer::getWeight() const noexcept
{ return weight; }

Tensor & Embedding_Layer::getWeight() noexcept
{ return weight; }
