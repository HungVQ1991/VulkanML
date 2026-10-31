#include "math/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "helper/logger.h"
#include "ops/optimizer/adam_step.h"
#include "ops/optimizer/sgd_step.h"
#include "ops/loss/loss_ops.h"
#include "ops/vision/conv2d.h"
#include "ops/vision/pooling.h"
#include "ops/normalization/batchnorm.h"
#include "ops/normalization/rmsnorm.h"
#include "ops/activation/activation.h"
#include "ops/llm/rope.h"
#include "ops/llm/swiglu.h"
#include "ops/llm/embedding.h"
#include "ops/llm/attention.h"
#include "ops/linear/linear.h"


Tensor::Tensor(Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(0, 0);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(0, 0);
        }
    }

Tensor::Tensor(size_t rows, size_t columns, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(rows, columns);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(rows, columns);
        }
    }

Tensor::Tensor(size_t rows, size_t columns, const std::vector<float> &host_data, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(rows, columns, host_data);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(rows, columns, host_data);
        }
    }

Tensor::Tensor(size_t rows, size_t columns, std::vector<float> &&host_data, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(rows, columns, std::move(host_data));
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(rows, columns, std::move(host_data));
        }
    }

Tensor::Tensor(Shape shape, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(shape);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(shape);
        }
    }

Tensor::Tensor(Shape shape, const std::vector<float> &host_data, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(shape, host_data);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(shape, host_data);
        }
    }

Tensor::Tensor(Shape shape, Data_Type type, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(shape, type);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(shape, type);
        }
    }

Tensor::Tensor(Shape shape, const std::vector<float> &host_data, Data_Type type, Execution_Target target)
    : execution_target(target)
{
        if (execution_target == Execution_Target::CPU)
        {
            auto cpu_impl = std::make_shared<Cpu_Tensor_Impl>(shape, type);
            cpu_impl->uploadData(host_data);
            implementation = cpu_impl;
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(shape, host_data, type);
        }
    }

void Tensor::to(Data_Type target_type, Tensor &output) const
{
        if (!output.implementation || output.getExecutionTarget() != execution_target || output.getShape() != getShape() || output.getDataType() != target_type)
        {
            output = Tensor(getShape(), target_type, execution_target);
        }
        implementation->to(target_type, *output.implementation);
    }

Tensor Tensor::to(Data_Type target_type) const
{
        if (implementation->getDataType() == target_type)
        {
            return clone();
        }
        Tensor result(getShape(), target_type, execution_target);
        implementation->to(target_type, *result.implementation);
        return result;
    }

Tensor Tensor::toFp16() const
{
        return to(Data_Type::FLOAT16);
    }

Tensor Tensor::toFp32() const
{
        return to(Data_Type::FLOAT32);
    }

Tensor::Tensor(std::initializer_list<size_t> shape_list, Execution_Target target)
    : Tensor(Shape(shape_list), target)
{
    }

Tensor::Tensor(std::shared_ptr<Tensor_Impl> impl, Execution_Target target)
    : implementation(std::move(impl)), execution_target(target)
{
    }

void Tensor::initializeShape(size_t rows, size_t columns)
{
        if (implementation->getRows() == rows && implementation->getColumns() == columns)
        {
            return;
        }
        implementation->reshape(rows, columns);
    }

void Tensor::reshape(Shape new_shape)
{
        implementation->reshape(new_shape);
    }

Tensor Tensor::permute(const std::vector<size_t> &axes_permutation) const
{
        Tensor result(execution_target);
        implementation->permute(axes_permutation, *result.implementation);
        return result;
    }

Tensor Tensor::slice(size_t axis, size_t start, size_t length) const
{
        Tensor result(execution_target);
        implementation->slice(axis, start, length, *result.implementation);
        return result;
    }

void Tensor::updateSlice(size_t axis, size_t start, const Tensor &source)
{
        implementation->updateSlice(axis, start, *source.implementation);
    }

Tensor Tensor::gatherRows(const std::vector<int32_t> &indices) const
{
        size_t S = indices.size();
        size_t D = getColumns();
        Tensor result(Shape{ S, D }, getDataType(), execution_target);
        implementation->gatherRows(indices, *result.implementation);
        return result;
    }

Tensor Tensor::contiguous() const
{
        if (implementation->isContiguous() && implementation->getByteOffset() == 0)
        {
            return *this;
        }
        Tensor result(execution_target);
        implementation->contiguous(*result.implementation);
        return result;
    }

void Tensor::ensureOutputTarget(Tensor &output) const
{
        if (output.getExecutionTarget() != execution_target)
        {
            output = Tensor(execution_target);
        }
    }

void Tensor::contiguous(Tensor &output) const
{
        ensureOutputTarget(output);
        implementation->contiguous(*output.implementation);
    }

void Tensor::matmul(const Tensor &other, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->matmul(*other.implementation, *output.implementation);
}

void Tensor::matdiv(const Tensor &other, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->matdiv(*other.implementation, *output.implementation);
}

void Tensor::add(const Tensor &other, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->add(*other.implementation, *output.implementation);
}

void Tensor::sub(const Tensor &other, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->sub(*other.implementation, *output.implementation);
}

void Tensor::mulScalar(float scalar, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->mulScalar(scalar, *output.implementation);
}

void Tensor::divScalar(float scalar, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->divScalar(scalar, *output.implementation);
}

void Tensor::hadamardMul(const Tensor &other, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->hadamardMul(*other.implementation, *output.implementation);
}

void Tensor::hadamardDiv(const Tensor &other, Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->hadamardDiv(*other.implementation, *output.implementation);
}

void Tensor::transpose(Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->transpose(*output.implementation);
}

void Tensor::inverse(Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->inverse(*output.implementation);
}

void Tensor::normalize(Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    implementation->normalize(*output.implementation);
}

void Tensor::relu(Tensor &output) const
{
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
    }
    ensureOutputTarget(output);
    ops::activation::relu(*this, output);
}

void Tensor::reluBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); ops::activation::reluBackward(*this, output_gradient, input_gradient); }

void Tensor::gelu(Tensor &output) const
{ ensureOutputTarget(output); ops::activation::gelu(*this, output); }

void Tensor::geluBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); ops::activation::geluBackward(*this, output_gradient, input_gradient); }

void Tensor::sigmoid(Tensor &output) const
{ ensureOutputTarget(output); ops::activation::sigmoid(*this, output); }

void Tensor::sigmoidBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); ops::activation::sigmoidBackward(*this, output_gradient, input_gradient); }

void Tensor::softmax(Tensor &output) const
{ ensureOutputTarget(output); ops::activation::softmax(*this, output); }

void Tensor::softmaxBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); ops::activation::softmaxBackward(*this, output_gradient, input_gradient); }

void Tensor::matmulAdd(const Tensor &other, const Tensor &biases, Tensor &output) const
{ ensureOutputTarget(output); implementation->matmulAdd(*other.implementation, *biases.implementation, *output.implementation); }

void Tensor::sgdUpdate(const Tensor &gradient, float learning_rate, float max_gradient, float inv_scale)
{
    ops::optimizer::sgdStep(*this, gradient, learning_rate, max_gradient, inv_scale);
}

void Tensor::adamUpdate(const Tensor &gradient, const Tensor &first_moment, const Tensor &second_moment, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay)
{
    ops::optimizer::adamStep(*this, gradient, const_cast<Tensor &>(first_moment), const_cast<Tensor &>(second_moment), learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
}

void Tensor::conv2d(const Tensor &weights, const Tensor &biases, Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor *scratch) const
{
        ops::vision::conv2d(*this, weights, biases, output,
                            input_height, input_width, input_channels, output_channels, kernel_size, stride, padding,
                            scratch);
    }

void Tensor::conv2dBackwardInput(const Tensor &weights, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
        ops::vision::conv2dBackwardInput(*this, weights, input_gradient,
                                         input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
    }

void Tensor::conv2dBackwardWeight(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor *im2col_scratch) const
{
        ops::vision::conv2dBackwardWeight(*this, output_gradient, weight_gradient, bias_gradient,
                                          input_height, input_width, input_channels, output_height, output_width, output_channels,
                                          kernel_size, stride, padding,
                                          im2col_scratch);
    }

void Tensor::maxpool2d(Tensor &output, Tensor &output_mask, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
        ops::vision::maxpool2d(*this, output, output_mask,
                               input_height, input_width, channels, kernel_size, stride, padding);
    }

void Tensor::maxpool2dBackward(const Tensor &mask, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t output_height, uint32_t output_width, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
        ops::vision::maxpool2dBackward(*this, mask, input_gradient,
                                       input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
    }

void Tensor::globalAvgPool2d(Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const
{
        ops::vision::globalAvgPool2d(*this, output, input_height, input_width, channels);
    }

void Tensor::globalAvgPool2dBackward(Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const
{
        ops::vision::globalAvgPool2dBackward(*this, input_gradient, input_height, input_width, channels);
    }

void Tensor::batchNormForward(const Tensor &gamma, const Tensor &beta, Tensor &running_mean, Tensor &running_variance, Tensor &batch_mean, Tensor &batch_variance, Tensor &normalized_input, Tensor &output, float epsilon, float momentum, bool is_training) const
{
    ops::normalization::batchNormForward(*this, gamma, beta, running_mean, running_variance, batch_mean, batch_variance, normalized_input, output, epsilon, momentum, is_training);
}

void Tensor::batchNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input, Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, float epsilon) const
{
    ops::normalization::batchNormBackward(*this, output_gradient, gamma, batch_variance, normalized_input, gamma_gradient, beta_gradient, input_gradient, epsilon);
}

void Tensor::rmsNormForward(const Tensor &gamma, Tensor &inv_rms, Tensor &output, float epsilon) const
{
    ops::normalization::rmsNormForward(*this, gamma, inv_rms, output, epsilon);
}

void Tensor::rmsNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &inv_rms, Tensor &gamma_gradient, Tensor &input_gradient, bool accumulate_gamma) const
{
    ops::normalization::rmsNormBackward(*this, output_gradient, gamma, inv_rms, gamma_gradient, input_gradient, accumulate_gamma);
}

void Tensor::applyRoPE(Tensor &output, uint32_t seq_len, uint32_t head_dim, int direction, float base, uint32_t num_heads, uint32_t mode) const
{
    ops::llm::applyRoPE(*this, output, seq_len, head_dim, direction, base, num_heads, mode);
}

void Tensor::swigluForward(const Tensor &b, Tensor &output) const
{
    ops::llm::swigluForward(*this, b, output);
}

void Tensor::swigluBackward(const Tensor &output_gradient, const Tensor &b, Tensor &grad_a, Tensor &grad_b) const
{
    ops::llm::swigluBackward(output_gradient, *this, b, grad_a, grad_b);
}

void Tensor::fusedSwiGLUForward(Tensor &output) const
{
    ops::llm::fusedSwiGLUForward(*this, output);
}

void Tensor::fusedSwiGLUBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
    ops::llm::fusedSwiGLUBackward(*this, output_gradient, input_gradient);
}

void Tensor::flashAttentionForward(const Tensor &k, const Tensor &v, Tensor &output, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, Tensor *l_stats) const
{
    ops::llm::flashAttentionForward(*this, k, v, output, num_heads, seq_len, head_dim, is_causal, scale, l_stats);
}

void Tensor::flashAttentionBackward(const Tensor &k, const Tensor &v, const Tensor &o, const Tensor &do_grad, Tensor &dq, Tensor &dk, Tensor &dv, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, const Tensor *l_stats) const
{
    ops::llm::flashAttentionBackward(*this, k, v, o, do_grad, dq, dk, dv, num_heads, seq_len, head_dim, is_causal, scale, l_stats);
}

void Tensor::embeddingForward(const Tensor &indices, Tensor &output) const
{
    ops::llm::embeddingForward(*this, indices, output);
}

Tensor Tensor::embeddingForward(const Tensor &indices) const
{
    Tensor result(execution_target);
    embeddingForward(indices, result);
    return result;
}

void Tensor::embeddingForward(const std::vector<int32_t>& indices, Tensor& output) const
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        const auto& gpu_impl = Gpu_Tensor_Impl::castToGpu(*implementation);
        auto indices_buf = gpu_impl.getOrUploadIndicesBuffer(indices);
        Shape idx_shape{ indices.size() };
        auto idx_impl = std::make_shared<Gpu_Tensor_Impl>(idx_shape, idx_shape.computeContiguousStrides(), indices_buf, 0, Data_Type::FLOAT32);
        Tensor idx_tensor(std::move(idx_impl), Execution_Target::VULKAN_GPU);
        ops::llm::embeddingForward(*this, idx_tensor, output);
        return;
    }

    std::vector<float> idx_float(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        idx_float[i] = static_cast<float>(indices[i]);
    }
    Tensor idx_tensor(Shape{ indices.size() }, std::move(idx_float), execution_target);
    ops::llm::embeddingForward(*this, idx_tensor, output);
}

Tensor Tensor::embeddingForward(const std::vector<int32_t> &indices) const
{
    Tensor result(execution_target);
    embeddingForward(indices, result);
    return result;
}

void Tensor::embeddingBackward(const Tensor &indices, const Tensor &output_gradient, Tensor &weight_gradient) const
{
    ops::llm::embeddingBackward(*this, indices, output_gradient, weight_gradient);
}

void Tensor::embeddingBackward(const std::vector<int32_t>& indices, const Tensor& output_gradient, Tensor& weight_gradient) const
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        const auto& gpu_impl = Gpu_Tensor_Impl::castToGpu(*implementation);
        auto indices_buf = gpu_impl.getOrUploadIndicesBuffer(indices);
        Shape idx_shape{ indices.size() };
        auto idx_impl = std::make_shared<Gpu_Tensor_Impl>(idx_shape, idx_shape.computeContiguousStrides(), indices_buf, 0, Data_Type::FLOAT32);
        Tensor idx_tensor(std::move(idx_impl), Execution_Target::VULKAN_GPU);
        ops::llm::embeddingBackward(*this, idx_tensor, output_gradient, weight_gradient);
        return;
    }

    std::vector<float> idx_float(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        idx_float[i] = static_cast<float>(indices[i]);
    }
    Tensor idx_tensor(Shape{ indices.size() }, std::move(idx_float), execution_target);
    ops::llm::embeddingBackward(*this, idx_tensor, output_gradient, weight_gradient);
}

void Tensor::singleTokenAttentionForward(const Tensor &k, const Tensor &v, Tensor &output, size_t num_heads, size_t head_dim, size_t total_seq_len) const
{
    ops::llm::singleTokenAttentionForward(*this, k, v, output, num_heads, head_dim, total_seq_len);
}

Tensor Tensor::singleTokenAttentionForward(const Tensor &k, const Tensor &v, size_t num_heads, size_t head_dim, size_t total_seq_len) const
{
    Tensor result(execution_target);
    singleTokenAttentionForward(k, v, result, num_heads, head_dim, total_seq_len);
    return result;
}


float Tensor::fusedCrossEntropyLoss(const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens) const
{
    return ops::loss::fusedCrossEntropy(*this, targets, d_logits, valid_tokens);
}

float Tensor::fusedCrossEntropyLoss(const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens) const
{
    return ops::loss::fusedCrossEntropy(*this, targets, d_logits, valid_tokens);
}

void Tensor::linearForward(const Tensor &weights, const Tensor &biases, Tensor &output) const
{
    ops::linear::linearForward(*this, weights, biases, output);
}

void Tensor::linearBackwardInput(const Tensor &weights, Tensor &input_gradient) const
{
    ops::linear::linearBackwardInput(*this, weights, input_gradient);
}

void Tensor::linearBackwardWeightBias(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate) const
{
    ops::linear::linearBackwardWeightBias(*this, output_gradient, weight_gradient, bias_gradient, accumulate);
}

void Tensor::linearBackwardWeightAdam(const Tensor &output_gradient, Tensor &weights, Tensor &first_moment, Tensor &second_moment, Tensor &bias_gradient, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay) const
{
    ensureOutputTarget(weights);
    ensureOutputTarget(first_moment);
    ensureOutputTarget(second_moment);
    ensureOutputTarget(bias_gradient);
    Tensor temp_w_grad(weights.getShape(), weights.getDataType(), weights.getExecutionTarget());
    linearBackwardWeightBias(output_gradient, temp_w_grad, bias_gradient, false);
    ops::optimizer::adamStep(weights, temp_w_grad, first_moment, second_moment, learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
}

void Tensor::batchNorm2dForward(const Tensor &gamma, const Tensor &beta, Tensor &running_mean, Tensor &running_variance, Tensor &batch_mean, Tensor &batch_variance, Tensor &normalized_input, Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon, float momentum, bool is_training) const
{
    ops::normalization::batchNorm2dForward(*this, gamma, beta, running_mean, running_variance, batch_mean, batch_variance, normalized_input, output, input_height, input_width, input_channels, epsilon, momentum, is_training);
}

void Tensor::batchNorm2dBackward(const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input, Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const
{
    ops::normalization::batchNorm2dBackward(*this, gamma, batch_variance, normalized_input, gamma_gradient, beta_gradient, input_gradient, input_height, input_width, input_channels, epsilon);
}

void Tensor::cceLoss(const Tensor &target, Tensor &output, float epsilon) const
{
    ops::loss::cce(*this, target, output, epsilon);
}

void Tensor::mseLoss(const Tensor &target, Tensor &output) const
{
    ops::loss::mse(*this, target, output);
}

void Tensor::maeLoss(const Tensor &target, Tensor &output) const
{
    ops::loss::mae(*this, target, output);
}

void Tensor::bceLoss(const Tensor &target, Tensor &output, float epsilon) const
{
    ops::loss::bce(*this, target, output, epsilon);
}

void Tensor::huberLoss(const Tensor &target, Tensor &output, float delta) const
{
    ops::loss::huber(*this, target, output, delta);
}

Tensor Tensor::operator*(const Tensor &other) const
{
        Tensor result(execution_target);
        matmul(other, result);
        return result;
    }

Tensor Tensor::operator/(const Tensor &other) const
{
        Tensor result(execution_target);
        matdiv(other, result);
        return result;
    }

Tensor Tensor::operator+(const Tensor &other) const
{
        Tensor result(execution_target);
        add(other, result);
        return result;
    }

Tensor Tensor::operator-(const Tensor &other) const
{
        Tensor result(execution_target);
        sub(other, result);
        return result;
    }

Tensor Tensor::operator*(float scalar) const
{
        Tensor result(execution_target);
        mulScalar(scalar, result);
        return result;
    }

Tensor Tensor::operator/(float scalar) const
{
        Tensor result(execution_target);
        divScalar(scalar, result);
        return result;
    }

Tensor Tensor::hadamardMul(const Tensor &other) const
{
        Tensor result(execution_target);
        hadamardMul(other, result);
        return result;
    }

Tensor Tensor::hadamardDiv(const Tensor &other) const
{
        Tensor result(execution_target);
        hadamardDiv(other, result);
        return result;
    }

Tensor Tensor::transpose() const
{
        Tensor result(execution_target);
        transpose(result);
        return result;
    }

Tensor Tensor::inverse() const
{
        Tensor result(execution_target);
        inverse(result);
        return result;
    }

Tensor Tensor::normalize() const
{
        Tensor result(execution_target);
        normalize(result);
        return result;
    }

Tensor Tensor::relu() const
{
        Tensor result(execution_target);
        relu(result);
        return result;
    }

Tensor Tensor::reluBackward(const Tensor &output_gradient) const
{
        Tensor result(execution_target);
        reluBackward(output_gradient, result);
        return result;
    }

Tensor Tensor::gelu() const
{
        Tensor result(execution_target);
        gelu(result);
        return result;
    }

Tensor Tensor::geluBackward(const Tensor &output_gradient) const
{
        Tensor result(execution_target);
        geluBackward(output_gradient, result);
        return result;
    }

Tensor Tensor::sigmoid() const
{
        Tensor result(execution_target);
        sigmoid(result);
        return result;
    }

Tensor Tensor::sigmoidBackward(const Tensor &output_gradient) const
{
        Tensor result(execution_target);
        sigmoidBackward(output_gradient, result);
        return result;
    }

Tensor Tensor::softmax() const
{
        Tensor result(execution_target);
        softmax(result);
        return result;
    }

Tensor Tensor::softmaxBackward(const Tensor &output_gradient) const
{
        Tensor result(execution_target);
        softmaxBackward(output_gradient, result);
        return result;
    }

Tensor Tensor::matmulAdd(const Tensor &other, const Tensor &biases) const
{
        Tensor result(execution_target);
        matmulAdd(other, biases, result);
        return result;
    }

void Tensor::concatenateColumns(const Tensor &other, Tensor &output) const
{
        implementation->concatenateColumns(*other.implementation, *output.implementation);
    }

Tensor Tensor::concatenateColumns(const Tensor &other) const
{
        Tensor result(execution_target);
        concatenateColumns(other, result);
        return result;
    }

void Tensor::concatenateRows(const Tensor &other, Tensor &output) const
{
        implementation->concatenateRows(*other.implementation, *output.implementation);
    }

Tensor Tensor::concatenateRows(const Tensor &other) const
{
        Tensor result(execution_target);
        concatenateRows(other, result);
        return result;
    }

void Tensor::splitColumns(size_t split_index, Tensor &result_left, Tensor &result_right) const
{
        implementation->splitColumns(split_index, *result_left.implementation, *result_right.implementation);
    }

std::pair<Tensor, Tensor> Tensor::splitColumns(size_t split_index) const
{
        Tensor result_left(execution_target);
        Tensor result_right(execution_target);
        splitColumns(split_index, result_left, result_right);
        return {std::move(result_left), std::move(result_right)};
    }

void Tensor::splitRows(size_t split_index, Tensor &result_up, Tensor &result_down) const
{
        implementation->splitRows(split_index, *result_up.implementation, *result_down.implementation);
    }

std::pair<Tensor, Tensor> Tensor::splitRows(size_t split_index) const
{
        Tensor result_up(execution_target);
        Tensor result_down(execution_target);
        splitRows(split_index, result_up, result_down);
        return {std::move(result_up), std::move(result_down)};
    }

float Tensor::getScalar() const
{
        const auto host_data = getData();
        float sum = 0.0F;
        for (float val : host_data)
        {
            sum += val;
        }
        return sum;
    }

void Tensor::print(size_t max_display_rows, size_t max_display_columns) const
{
        const auto data_vector = getData();
        size_t print_rows = std::min(getRows(), max_display_rows);
        size_t print_cols = std::min(getColumns(), max_display_columns);
        std::cout << "Tensor " << getShape().toString() << ":\n";
        for (size_t r = 0; r < print_rows; ++r)
        {
            std::cout << "  [ ";
            for (size_t c = 0; c < print_cols; ++c)
            {
                std::cout << std::format("{:8.4f} ", data_vector[r * getColumns() + c]);
            }
            if (getColumns() > print_cols)
            {
                std::cout << "... ";
            }
            std::cout << "]\n";
        }
        if (getRows() > print_rows)
        {
            std::cout << "  ...\n";
        }
    }

void Tensor::saveTensor(std::ofstream &output_file_stream) const
{
        if (!output_file_stream.is_open())
        {
            throw std::runtime_error("Tensor::saveTensor: Output stream is not open");
        }

        output_file_stream.write(reinterpret_cast<const char *>(&TENSOR_MAGIC_HEADER), sizeof(TENSOR_MAGIC_HEADER));

        uint32_t rank = static_cast<uint32_t>(getRank());
        output_file_stream.write(reinterpret_cast<const char *>(&rank), sizeof(rank));

        const auto &dims = getShape();
        for (size_t i = 0; i < rank; ++i)
        {
            uint32_t dim_val = static_cast<uint32_t>(dims[i]);
            output_file_stream.write(reinterpret_cast<const char *>(&dim_val), sizeof(dim_val));
        }

        std::vector<float> host_data = getData();
        output_file_stream.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
    }

Tensor Tensor::loadTensor(std::ifstream &input_file_stream, Execution_Target target)
{
        if (!input_file_stream.is_open())
        {
            throw std::runtime_error("Tensor::loadTensor: Input stream is not open");
        }

        uint32_t first_header_field = 0;
        input_file_stream.read(reinterpret_cast<char *>(&first_header_field), sizeof(first_header_field));

        if (first_header_field == TENSOR_MAGIC_HEADER || first_header_field == 0xFFFFFFFF)
        {
            uint32_t rank = 0;
            input_file_stream.read(reinterpret_cast<char *>(&rank), sizeof(rank));

            std::vector<size_t> dims(rank);
            size_t total = 1;
            for (size_t i = 0; i < rank; ++i)
            {
                uint32_t dim_val = 0;
                input_file_stream.read(reinterpret_cast<char *>(&dim_val), sizeof(dim_val));
                dims[i] = static_cast<size_t>(dim_val);
                total *= dims[i];
            }

            std::vector<float> host_data(total);
            input_file_stream.read(reinterpret_cast<char *>(host_data.data()), static_cast<std::streamsize>(total * sizeof(float)));
            return Tensor(Shape(dims), std::move(host_data), target);
        }

        uint32_t rows_count = first_header_field;
        uint32_t columns_count = 0;
        input_file_stream.read(reinterpret_cast<char *>(&columns_count), sizeof(columns_count));
        std::vector<float> host_data(rows_count * columns_count);
        input_file_stream.read(reinterpret_cast<char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
        return Tensor(rows_count, columns_count, std::move(host_data), target);
    }

void Tensor::saveMatrix(std::ofstream &output_file_stream) const
{
        saveTensor(output_file_stream);
    }

Tensor Tensor::loadMatrix(std::ifstream &input_file_stream, Execution_Target target)
{
        return loadTensor(input_file_stream, target);
    }

Tensor Tensor::clone() const
{
        return Tensor(getShape(), getData(), getDataType(), execution_target);
    }

void Tensor::fill(float value)
{
        implementation->fill(value);
    }

void Tensor::zero()
{
        implementation->zero();
    }

void Tensor::invalidateFp16Cache() noexcept
{
        if (implementation)
        {
            implementation->invalidateFp16Cache();
        }
    }

void Tensor::prewarmFp16Cache()
{
        if (implementation)
        {
            implementation->prewarmFp16Cache();
        }
    }

std::vector<float> Tensor::getData() const
{ return implementation->getData(); }

void Tensor::uploadData(const std::vector<float> &host_data)
{ implementation->uploadData(host_data); }

void Tensor::setDataType(Data_Type _type) noexcept
{ implementation->setDataType(_type); }

void Tensor::logFp16Stats(std::string_view tensor_name, Log_Level level) const
{
        std::vector<float> host_data = getData();
        Logger::logFp16TensorStats(tensor_name, host_data, getTotalElements(), getDataType() == Data_Type::FLOAT16 ? "FLOAT16" : "FLOAT32", level);
    }
