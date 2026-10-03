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
{ ensureOutputTarget(output); implementation->matmul(*other.implementation, *output.implementation); }

void Tensor::matdiv(const Tensor &other, Tensor &output) const
{ ensureOutputTarget(output); implementation->matdiv(*other.implementation, *output.implementation); }

void Tensor::add(const Tensor &other, Tensor &output) const
{ ensureOutputTarget(output); implementation->add(*other.implementation, *output.implementation); }

void Tensor::sub(const Tensor &other, Tensor &output) const
{ ensureOutputTarget(output); implementation->sub(*other.implementation, *output.implementation); }

void Tensor::mulScalar(float scalar, Tensor &output) const
{ ensureOutputTarget(output); implementation->mulScalar(scalar, *output.implementation); }

void Tensor::divScalar(float scalar, Tensor &output) const
{ ensureOutputTarget(output); implementation->divScalar(scalar, *output.implementation); }

void Tensor::hadamardMul(const Tensor &other, Tensor &output) const
{ ensureOutputTarget(output); implementation->hadamardMul(*other.implementation, *output.implementation); }

void Tensor::hadamardDiv(const Tensor &other, Tensor &output) const
{ ensureOutputTarget(output); implementation->hadamardDiv(*other.implementation, *output.implementation); }

void Tensor::transpose(Tensor &output) const
{ ensureOutputTarget(output); implementation->transpose(*output.implementation); }

void Tensor::inverse(Tensor &output) const
{ ensureOutputTarget(output); implementation->inverse(*output.implementation); }

void Tensor::normalize(Tensor &output) const
{ ensureOutputTarget(output); implementation->normalize(*output.implementation); }

void Tensor::relu(Tensor &output) const
{ ensureOutputTarget(output); implementation->relu(*output.implementation); }

void Tensor::reluBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); implementation->reluBackward(*output_gradient.implementation, *input_gradient.implementation); }

void Tensor::gelu(Tensor &output) const
{ ensureOutputTarget(output); implementation->gelu(*output.implementation); }

void Tensor::geluBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); implementation->geluBackward(*output_gradient.implementation, *input_gradient.implementation); }

void Tensor::softmax(Tensor &output) const
{ ensureOutputTarget(output); implementation->softmax(*output.implementation); }

void Tensor::softmaxBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{ ensureOutputTarget(input_gradient); implementation->softmaxBackward(*output_gradient.implementation, *input_gradient.implementation); }

void Tensor::matmulAdd(const Tensor &other, const Tensor &biases, Tensor &output) const
{ ensureOutputTarget(output); implementation->matmulAdd(*other.implementation, *biases.implementation, *output.implementation); }

void Tensor::sgdUpdate(const Tensor &gradient, float learning_rate, float max_gradient, float inv_scale)
{
        implementation->sgdUpdate(*gradient.implementation, learning_rate, max_gradient, inv_scale);
    }

void Tensor::adamUpdate(const Tensor &gradient, const Tensor &first_moment, const Tensor &second_moment, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay)
{
        implementation->adamUpdate(*gradient.implementation,
                                   *first_moment.implementation,
                                   *second_moment.implementation,
                                   learning_rate,
                                   beta1,
                                   beta2,
                                   epsilon,
                                   timestep,
                                   max_gradient,
                                   inv_scale,
                                   weight_decay);
    }

void Tensor::conv2d(const Tensor &weights, const Tensor &biases, Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor *scratch) const
{
        implementation->conv2d(*weights.implementation, *biases.implementation, *output.implementation,
                               input_height, input_width, input_channels, output_channels, kernel_size, stride, padding,
                               scratch ? scratch->implementation.get() : nullptr);
    }

void Tensor::conv2dBackwardInput(const Tensor &weights, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
        implementation->conv2dBackwardInput(*weights.implementation, *input_gradient.implementation,
                                            input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
    }

void Tensor::conv2dBackwardWeight(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor *im2col_scratch) const
{
        implementation->conv2dBackwardWeight(*output_gradient.implementation, *weight_gradient.implementation, *bias_gradient.implementation,
                                             input_height, input_width, input_channels, output_height, output_width, output_channels,
                                             kernel_size, stride, padding,
                                             im2col_scratch ? im2col_scratch->implementation.get() : nullptr);
    }

void Tensor::maxpool2d(Tensor &output, Tensor &output_mask, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
        implementation->maxpool2d(*output.implementation, *output_mask.implementation,
                                  input_height, input_width, channels, kernel_size, stride, padding);
    }

void Tensor::maxpool2dBackward(const Tensor &mask, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t output_height, uint32_t output_width, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
        implementation->maxpool2dBackward(*mask.implementation, *input_gradient.implementation,
                                          input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
    }

void Tensor::globalAvgPool2d(Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const
{
        implementation->globalAvgPool2d(*output.implementation, input_height, input_width, channels);
    }

void Tensor::globalAvgPool2dBackward(Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const
{
        implementation->globalAvgPool2dBackward(*input_gradient.implementation, input_height, input_width, channels);
    }

void Tensor::batchNormForward(const Tensor &gamma, const Tensor &beta, Tensor &running_mean, Tensor &running_variance, Tensor &batch_mean, Tensor &batch_variance, Tensor &normalized_input, Tensor &output, float epsilon, float momentum, bool is_training) const
{
        implementation->batchNormForward(*gamma.implementation, *beta.implementation,
                                         *running_mean.implementation, *running_variance.implementation,
                                         *batch_mean.implementation, *batch_variance.implementation,
                                         *normalized_input.implementation, *output.implementation,
                                         epsilon, momentum, is_training);
    }

void Tensor::batchNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input, Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, float epsilon) const
{
        implementation->batchNormBackward(*output_gradient.implementation, *gamma.implementation,
                                          *batch_variance.implementation, *normalized_input.implementation,
                                          *gamma_gradient.implementation, *beta_gradient.implementation,
                                          *input_gradient.implementation, epsilon);
    }

void Tensor::rmsNormForward(const Tensor &gamma, Tensor &inv_rms, Tensor &output, float epsilon) const
{
        ensureOutputTarget(inv_rms);
        ensureOutputTarget(output);
        implementation->rmsNormForward(*gamma.implementation, *inv_rms.implementation, *output.implementation, epsilon);
    }

void Tensor::rmsNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &inv_rms, Tensor &gamma_gradient, Tensor &input_gradient, bool accumulate_gamma) const
{
        ensureOutputTarget(gamma_gradient);
        ensureOutputTarget(input_gradient);
        implementation->rmsNormBackward(*output_gradient.implementation, *gamma.implementation, *inv_rms.implementation,
                                        *gamma_gradient.implementation, *input_gradient.implementation, accumulate_gamma);
    }

void Tensor::applyRoPE(Tensor &output, uint32_t seq_len, uint32_t head_dim, int direction, float base, uint32_t num_heads, uint32_t mode) const
{
        ensureOutputTarget(output);
        implementation->applyRoPE(*output.implementation, seq_len, head_dim, direction, base, num_heads, mode);
    }

void Tensor::swigluForward(const Tensor &b, Tensor &output) const
{
        ensureOutputTarget(output);
        implementation->swigluForward(*b.implementation, *output.implementation);
    }

void Tensor::swigluBackward(const Tensor &output_gradient, const Tensor &b, Tensor &grad_a, Tensor &grad_b) const
{
        ensureOutputTarget(grad_a);
        ensureOutputTarget(grad_b);
        implementation->swigluBackward(*output_gradient.implementation, *b.implementation, *grad_a.implementation, *grad_b.implementation);
    }

void Tensor::fusedSwiGLUForward(Tensor &output) const
{
        ensureOutputTarget(output);
        implementation->fusedSwiGLUForward(*output.implementation);
    }

void Tensor::fusedSwiGLUBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
        ensureOutputTarget(input_gradient);
        implementation->fusedSwiGLUBackward(*output_gradient.implementation, *input_gradient.implementation);
    }

void Tensor::flashAttentionForward(const Tensor &k, const Tensor &v, Tensor &output, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, Tensor *l_stats) const
{
        ensureOutputTarget(output);
        if (l_stats)
        {
            ensureOutputTarget(*l_stats);
        }
        implementation->flashAttentionForward(*k.implementation, *v.implementation, *output.implementation,
                                              num_heads, seq_len, head_dim, is_causal, scale,
                                              l_stats ? l_stats->implementation.get() : nullptr);
    }

void Tensor::flashAttentionBackward(const Tensor &k, const Tensor &v, const Tensor &o, const Tensor &do_grad, Tensor &dq, Tensor &dk, Tensor &dv, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, const Tensor *l_stats) const
{
        ensureOutputTarget(dq);
        ensureOutputTarget(dk);
        ensureOutputTarget(dv);
        implementation->flashAttentionBackward(*k.implementation, *v.implementation,
                                               *o.implementation, *do_grad.implementation,
                                               *dq.implementation, *dk.implementation, *dv.implementation,
                                               num_heads, seq_len, head_dim, is_causal, scale,
                                               l_stats ? l_stats->implementation.get() : nullptr);
    }

void Tensor::embeddingForward(const Tensor &indices, Tensor &output) const
{
        ensureOutputTarget(output);
        implementation->embeddingForward(*indices.implementation, *output.implementation);
    }

Tensor Tensor::embeddingForward(const Tensor &indices) const
{
        Tensor result(execution_target);
        embeddingForward(indices, result);
        return result;
    }

void Tensor::embeddingForward(const std::vector<int32_t> &indices, Tensor &output) const
{
        std::vector<float> idx_float(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) idx_float[i] = static_cast<float>(indices[i]);
        Tensor idx_tensor(Shape{ indices.size() }, std::move(idx_float), execution_target);
        embeddingForward(idx_tensor, output);
    }

Tensor Tensor::embeddingForward(const std::vector<int32_t> &indices) const
{
        Tensor result(execution_target);
        embeddingForward(indices, result);
        return result;
    }

void Tensor::embeddingBackward(const Tensor &indices, const Tensor &output_gradient, Tensor &weight_gradient) const
{
        ensureOutputTarget(weight_gradient);
        implementation->embeddingBackward(*indices.implementation, *output_gradient.implementation, *weight_gradient.implementation);
    }

void Tensor::embeddingBackward(const std::vector<int32_t> &indices, const Tensor &output_gradient, Tensor &weight_gradient) const
{
        std::vector<float> idx_float(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) idx_float[i] = static_cast<float>(indices[i]);
        Tensor idx_tensor(Shape{ indices.size() }, std::move(idx_float), execution_target);
        embeddingBackward(idx_tensor, output_gradient, weight_gradient);
    }

void Tensor::singleTokenAttentionForward(const Tensor &k, const Tensor &v, Tensor &output, size_t num_heads, size_t head_dim, size_t total_seq_len) const
{
        implementation->singleTokenAttentionForward(*k.implementation, *v.implementation, *output.implementation,
                                                    num_heads, head_dim, total_seq_len);
    }

Tensor Tensor::singleTokenAttentionForward(const Tensor &k, const Tensor &v, size_t num_heads, size_t head_dim, size_t total_seq_len) const
{
        Tensor result(execution_target);
        singleTokenAttentionForward(k, v, result, num_heads, head_dim, total_seq_len);
        return result;
    }

float Tensor::fusedCrossEntropyLoss(const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens) const
{
        if (d_logits.getExecutionTarget() != execution_target || d_logits.getDataType() != getDataType())
        {
            d_logits = Tensor(getShape(), getDataType(), execution_target);
        }
        return implementation->fusedCrossEntropyLoss(*targets.implementation, *d_logits.implementation, valid_tokens);
    }

float Tensor::fusedCrossEntropyLoss(const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens) const
{
        if (d_logits.getExecutionTarget() != execution_target || d_logits.getDataType() != getDataType())
        {
            d_logits = Tensor(getShape(), getDataType(), execution_target);
        }
        size_t S = (getShape().getRank() == 3) ? (getShape()[0] * getShape()[1]) : getRows();
        std::vector<float> tgt_float(S, -100.0f);
        uint32_t counted_valid = 0;
        size_t V = (getShape().getRank() == 3) ? getShape()[2] : getColumns();
        for (size_t i = 0; i < std::min(S, targets.size()); ++i)
        {
            tgt_float[i] = static_cast<float>(targets[i]);
            if (targets[i] >= 0 && static_cast<size_t>(targets[i]) < V)
            {
                counted_valid++;
            }
        }
        uint32_t effective_valid = (valid_tokens > 0) ? valid_tokens : counted_valid;
        Tensor tgt_tensor(Shape{ S }, std::move(tgt_float), execution_target);
        return implementation->fusedCrossEntropyLoss(*tgt_tensor.implementation, *d_logits.implementation, effective_valid);
    }

void Tensor::linearForward(const Tensor &weights, const Tensor &biases, Tensor &output) const
{
        ensureOutputTarget(output);
        implementation->linearForward(*weights.implementation, *biases.implementation, *output.implementation);
    }

void Tensor::linearBackwardInput(const Tensor &weights, Tensor &input_gradient) const
{
        ensureOutputTarget(input_gradient);
        implementation->linearBackwardInput(*weights.implementation, *input_gradient.implementation);
    }

void Tensor::linearBackwardWeightBias(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate) const
{
        ensureOutputTarget(weight_gradient);
        ensureOutputTarget(bias_gradient);
        implementation->linearBackwardWeightBias(*output_gradient.implementation, *weight_gradient.implementation, *bias_gradient.implementation, accumulate);
    }

void Tensor::linearBackwardWeightAdam(const Tensor &output_gradient, Tensor &weights, Tensor &first_moment, Tensor &second_moment, Tensor &bias_gradient, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay) const
{
        ensureOutputTarget(weights);
        ensureOutputTarget(first_moment);
        ensureOutputTarget(second_moment);
        ensureOutputTarget(bias_gradient);
        implementation->linearBackwardWeightAdam(*output_gradient.implementation, *weights.implementation, *first_moment.implementation, *second_moment.implementation, *bias_gradient.implementation,
                                                 learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
    }

void Tensor::batchNorm2dForward(const Tensor &gamma, const Tensor &beta, Tensor &running_mean, Tensor &running_variance, Tensor &batch_mean, Tensor &batch_variance, Tensor &normalized_input, Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon, float momentum, bool is_training) const
{
        implementation->batchNorm2dForward(*gamma.implementation, *beta.implementation,
                                           *running_mean.implementation, *running_variance.implementation,
                                           *batch_mean.implementation, *batch_variance.implementation,
                                           *normalized_input.implementation, *output.implementation,
                                           input_height, input_width, input_channels, epsilon, momentum, is_training);
    }

void Tensor::batchNorm2dBackward(const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input, Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const
{
        implementation->batchNorm2dBackward(*gamma.implementation, *batch_variance.implementation,
                                            *normalized_input.implementation, *gamma_gradient.implementation,
                                            *beta_gradient.implementation, *input_gradient.implementation,
                                            input_height, input_width, input_channels, epsilon);
    }

void Tensor::cceLoss(const Tensor &target, Tensor &output, float epsilon) const
{
        implementation->cceLoss(*target.implementation, *output.implementation, epsilon);
    }

void Tensor::mseLoss(const Tensor &target, Tensor &output) const
{
        implementation->mseLoss(*target.implementation, *output.implementation);
    }

void Tensor::maeLoss(const Tensor &target, Tensor &output) const
{
        implementation->maeLoss(*target.implementation, *output.implementation);
    }

void Tensor::bceLoss(const Tensor &target, Tensor &output, float epsilon) const
{
        implementation->bceLoss(*target.implementation, *output.implementation, epsilon);
    }

void Tensor::huberLoss(const Tensor &target, Tensor &output, float delta) const
{
        implementation->huberLoss(*target.implementation, *output.implementation, delta);
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
