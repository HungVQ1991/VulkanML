#include "layer/embedding_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
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
#include "ops/llm/embedding.h"

void Embedding_Layer::initializeWeights()
{
        std::vector<float> weight_data(vocab_size * embedding_dim);
        float stddev = 0.02f;
        std::random_device rd;
        std::mt19937 gen(rd());
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
        if (_vocab_size == 0 || _embedding_dim == 0)
        {
            throw std::invalid_argument("Embedding_Layer: vocab_size and embedding_dim must be > 0");
        }
        if (_data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Embedding_Layer: TERNARY_INT2 is not supported");
        }
        if (_data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        output_tensor.setDataType(_data_type);
        initializeWeights();
    }

Tensor Embedding_Layer::forward(const std::vector<int32_t> &token_ids)
{
        if (token_ids.empty())
        {
            last_indices_tensor = Tensor(execution_target);
            return Tensor(Shape{ 0, embedding_dim }, data_type, execution_target);
        }
        std::vector<float> idx_float(token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i)
        {
            idx_float[i] = static_cast<float>(token_ids[i]);
        }
        last_indices_tensor = Tensor(Shape{ token_ids.size() }, std::move(idx_float), execution_target);
        const bool use_fp16 = (data_type == Data_Type::FLOAT16 || is_mixed_precision_enabled);
        ops::llm::embeddingForward(weight, last_indices_tensor, output_tensor, use_fp16);
        return output_tensor;
    }

Tensor Embedding_Layer::forward(const std::vector<std::vector<int32_t>> &batch_token_ids)
{
        if (batch_token_ids.empty())
        {
            last_indices_tensor = Tensor(execution_target);
            return Tensor(Shape{ 0, 0, embedding_dim }, data_type, execution_target);
        }
        size_t B = batch_token_ids.size();
        size_t S = batch_token_ids[0].size();
        for (size_t b = 1; b < B; ++b)
        {
            if (batch_token_ids[b].size() != S)
            {
                throw std::invalid_argument("Embedding_Layer::forward: Jagged batch_token_ids sequences are not allowed");
            }
        }
        if (S == 0)
        {
            last_indices_tensor = Tensor(execution_target);
            return Tensor(Shape{ B, 0, embedding_dim }, data_type, execution_target);
        }
        std::vector<float> idx_float(B * S);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t s = 0; s < S; ++s)
            {
                idx_float[b * S + s] = static_cast<float>(batch_token_ids[b][s]);
            }
        }
        last_indices_tensor = Tensor(Shape{ B, S }, std::move(idx_float), execution_target);
        const bool use_fp16 = (data_type == Data_Type::FLOAT16 || is_mixed_precision_enabled);
        ops::llm::embeddingForward(weight, last_indices_tensor, output_tensor, use_fp16);
        return output_tensor;
    }

Tensor Embedding_Layer::forward(const Tensor &_input_tensor)
{
        if (!_input_tensor.getImplementation())
        {
            throw std::invalid_argument("Embedding_Layer::forward: Input tensor has null implementation");
        }
        if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Embedding_Layer::forward: TERNARY_INT2 indices are not supported");
        }
        if (_input_tensor.getTotalElements() == 0)
        {
            last_indices_tensor = Tensor(execution_target);
            if (_input_tensor.getShape().getRank() == 2)
            {
                return Tensor(Shape{ _input_tensor.getShape()[0], _input_tensor.getShape()[1], embedding_dim }, data_type, execution_target);
            }
            return Tensor(Shape{ 0, embedding_dim }, data_type, execution_target);
        }
        if (_input_tensor.getExecutionTarget() != execution_target)
        {
            last_indices_tensor = Tensor(_input_tensor.getShape(), _input_tensor.getData(), execution_target);
        }
        else
        {
            last_indices_tensor = _input_tensor;
        }
        const bool use_fp16 = (data_type == Data_Type::FLOAT16 || is_mixed_precision_enabled);
        ops::llm::embeddingForward(weight, last_indices_tensor, output_tensor, use_fp16);
        return output_tensor;
    }

Tensor Embedding_Layer::backward(const Tensor &_output_gradient)
{
        if (embedding_dim == 0)
        {
            throw std::invalid_argument("Embedding_Layer::backward: embedding_dim must be > 0");
        }
        if (!_output_gradient.getImplementation())
        {
            throw std::invalid_argument("Embedding_Layer::backward: Output gradient has null implementation");
        }
        if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Embedding_Layer::backward: TERNARY_INT2 gradient is not supported");
        }
        if (_output_gradient.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("Embedding_Layer::backward: Execution_Target mismatch");
        }
        if (last_indices_tensor.getTotalElements() == 0 || _output_gradient.getTotalElements() == 0 || !is_trainable)
        {
            return Tensor(0, 0, execution_target);
        }

        const size_t num_tokens = last_indices_tensor.getTotalElements();
        const Shape &grad_shape = _output_gradient.getShape();
        const size_t grad_rank = grad_shape.getRank();
        if (grad_rank == 0 || grad_shape[grad_rank - 1] != embedding_dim ||
            _output_gradient.getTotalElements() != num_tokens * embedding_dim)
        {
            throw std::invalid_argument("Embedding_Layer::backward: Output gradient shape mismatch");
        }

        Tensor flat_grad = (grad_rank == 2)
            ? _output_gradient
            : _output_gradient.view(Shape{ num_tokens, embedding_dim });

        const bool eff_accum = is_accumulated || accumulate_on_backward_with_tied;
        ops::llm::embeddingBackward(weight, last_indices_tensor, flat_grad, weight_gradient, eff_accum);
        return Tensor(0, 0, execution_target);
    }

void Embedding_Layer::resetGradient()
{
        last_indices_tensor = Tensor(execution_target);
    }

void Embedding_Layer::resetGradients()
{
        resetGradient();
        weight_gradient.zero();
    }

std::vector<std::pair<Tensor *, Tensor *>> Embedding_Layer::getParametersAndGradients()
{
        if (!is_trainable)
        {
            return {};
        }
        return { { &weight, &weight_gradient } };
    }

void Embedding_Layer::setExecutionTarget(Execution_Target _execution_target)
{
        if (execution_target == _execution_target)
        {
            return;
        }
        logChangeExecutionTarget(_execution_target);
        execution_target = _execution_target;
        weight.setExecutionTarget(_execution_target);
        weight_gradient.setExecutionTarget(_execution_target);
        last_indices_tensor.setExecutionTarget(_execution_target);
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
        copy->weight = weight.clone();
        copy->weight_gradient = weight_gradient.clone();
        copyBaseStateTo(*copy);
        copy->setAccumulateForTiedWeights(accumulate_on_backward_with_tied);
        return copy;
    }

void Embedding_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::saveConfiguration: Output stream is not open or writable");
        }
        if (vocab_size > std::numeric_limits<uint32_t>::max() ||
            embedding_dim > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Embedding_Layer::saveConfiguration: Dimension exceeds UINT32_MAX");
        }
        uint64_t v_size = vocab_size;
        uint64_t e_dim = embedding_dim;
        _output_file_stream.write(reinterpret_cast<const char *>(&v_size), sizeof(v_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&e_dim), sizeof(e_dim));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::saveConfiguration: Failed writing configuration");
        }
    }

void Embedding_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::saveInference: Output stream is not open or writable");
        }
        weight.saveTensor(_output_file_stream);
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::saveInference: Failed writing weight tensor");
        }
    }

void Embedding_Layer::loadInference(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::loadInference: Input stream is not open or readable");
        }
        Tensor loaded_weight = Tensor::loadTensor(_input_file_stream, execution_target);
        if (!_input_file_stream)
        {
            throw std::runtime_error("Embedding_Layer::loadInference: Failed reading weight tensor");
        }
        if (!loaded_weight.getImplementation() ||
            loaded_weight.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_weight.getShape() != Shape{ vocab_size, embedding_dim })
        {
            throw std::runtime_error("Embedding_Layer::loadInference: Loaded weight shape or dtype mismatch");
        }
        weight = std::move(loaded_weight);
        weight.invalidateFp16Cache();
    }

void Embedding_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::saveCheckpoint: Output stream is not open or writable");
        }
        weight.saveTensor(_output_file_stream);
        weight_gradient.saveTensor(_output_file_stream);
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::saveCheckpoint: Failed writing checkpoint state");
        }
    }

void Embedding_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("Embedding_Layer::loadCheckpoint: Input stream is not open or readable");
        }
        Tensor loaded_weight = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_grad = Tensor::loadTensor(_input_file_stream, execution_target);
        if (!_input_file_stream)
        {
            throw std::runtime_error("Embedding_Layer::loadCheckpoint: Failed reading checkpoint state");
        }
        const Shape expected_shape{ vocab_size, embedding_dim };
        if (!loaded_weight.getImplementation() ||
            loaded_weight.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_weight.getShape() != expected_shape ||
            !loaded_grad.getImplementation() ||
            loaded_grad.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_grad.getShape() != expected_shape)
        {
            throw std::runtime_error("Embedding_Layer::loadCheckpoint: Loaded tensor shape or dtype mismatch");
        }
        weight = std::move(loaded_weight);
        weight_gradient = std::move(loaded_grad);
        weight.invalidateFp16Cache();
    }

size_t Embedding_Layer::getParameterCount() const noexcept
{ return vocab_size * embedding_dim; }

const Tensor & Embedding_Layer::getWeight() const noexcept
{ return weight; }

Tensor & Embedding_Layer::getWeight() noexcept
{ return weight; }
