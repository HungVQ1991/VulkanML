#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cpu_tensor_impl.h"
#include "gpu_tensor_impl.h"
#include "helper/logger.h"
#include "shape.h"
#include "tensor_impl.h"

enum class Execution_Target
{
    CPU,
    VULKAN_GPU
};

class Tensor
{
private:
    std::shared_ptr<Tensor_Impl> implementation;
    Execution_Target execution_target;

    static constexpr uint32_t TENSOR_MAGIC_HEADER = 0x7FFFFFFF;

public:
    explicit Tensor(Execution_Target target = Execution_Target::CPU)
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

    Tensor(size_t rows, size_t columns, Execution_Target target = Execution_Target::CPU)
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

    Tensor(size_t rows, size_t columns, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU)
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

    Tensor(size_t rows, size_t columns, std::vector<float> &&host_data, Execution_Target target = Execution_Target::CPU)
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

    explicit Tensor(Shape shape, Execution_Target target = Execution_Target::CPU)
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

    Tensor(Shape shape, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU)
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

    Tensor(Shape shape, Data_Type type, Execution_Target target = Execution_Target::CPU)
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

    Tensor(Shape shape, const std::vector<float> &host_data, Data_Type type, Execution_Target target = Execution_Target::CPU)
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

    void to(Data_Type target_type, Tensor &output) const
    {
        if (!output.implementation || output.getExecutionTarget() != execution_target || output.getShape() != getShape() || output.getDataType() != target_type)
        {
            output = Tensor(getShape(), target_type, execution_target);
        }
        implementation->to(target_type, *output.implementation);
    }

    Tensor to(Data_Type target_type) const
    {
        if (implementation->getDataType() == target_type)
        {
            return clone();
        }
        Tensor result(getShape(), target_type, execution_target);
        implementation->to(target_type, *result.implementation);
        return result;
    }

    Tensor toFp16() const
    {
        return to(Data_Type::FLOAT16);
    }

    Tensor toFp32() const
    {
        return to(Data_Type::FLOAT32);
    }

    Tensor(std::initializer_list<size_t> shape_list, Execution_Target target = Execution_Target::CPU)
        : Tensor(Shape(shape_list), target)
    {
    }

    explicit Tensor(std::shared_ptr<Tensor_Impl> impl, Execution_Target target = Execution_Target::CPU)
        : implementation(std::move(impl)), execution_target(target)
    {
    }

    ~Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    void initializeShape(size_t rows, size_t columns)
    {
        if (implementation->getRows() == rows && implementation->getColumns() == columns)
        {
            return;
        }
        implementation->reshape(rows, columns);
    }

    void initShape(size_t rows, size_t columns)
    {
        initializeShape(rows, columns);
    }

    void reshape(Shape new_shape)
    {
        implementation->reshape(new_shape);
    }

    Tensor permute(const std::vector<size_t> &axes_permutation) const
    {
        Tensor result(execution_target);
        implementation->permute(axes_permutation, *result.implementation);
        return result;
    }

    Tensor slice(size_t axis, size_t start, size_t length) const
    {
        Tensor result(execution_target);
        implementation->slice(axis, start, length, *result.implementation);
        return result;
    }

    void updateSlice(size_t axis, size_t start, const Tensor &source)
    {
        implementation->updateSlice(axis, start, *source.implementation);
    }

    Tensor gatherRows(const std::vector<int32_t> &indices) const
    {
        size_t S = indices.size();
        size_t D = getColumns();
        Tensor result(Shape{ S, D }, getDataType(), execution_target);
        implementation->gatherRows(indices, *result.implementation);
        return result;
    }

    Tensor contiguous() const
    {
        if (implementation->isContiguous() && implementation->getByteOffset() == 0)
        {
            return *this;
        }
        Tensor result(execution_target);
        implementation->contiguous(*result.implementation);
        return result;
    }

    void ensureOutputTarget(Tensor &output) const
    {
        if (output.getExecutionTarget() != execution_target)
        {
            output = Tensor(execution_target);
        }
    }

    void contiguous(Tensor &output) const
    {
        ensureOutputTarget(output);
        implementation->contiguous(*output.implementation);
    }

    void matmul(const Tensor &other, Tensor &output) const { ensureOutputTarget(output); implementation->matmul(*other.implementation, *output.implementation); }
    void matdiv(const Tensor &other, Tensor &output) const { ensureOutputTarget(output); implementation->matdiv(*other.implementation, *output.implementation); }
    void add(const Tensor &other, Tensor &output) const { ensureOutputTarget(output); implementation->add(*other.implementation, *output.implementation); }
    void sub(const Tensor &other, Tensor &output) const { ensureOutputTarget(output); implementation->sub(*other.implementation, *output.implementation); }
    void mulScalar(float scalar, Tensor &output) const { ensureOutputTarget(output); implementation->mulScalar(scalar, *output.implementation); }
    void divScalar(float scalar, Tensor &output) const { ensureOutputTarget(output); implementation->divScalar(scalar, *output.implementation); }
    void hadamardMul(const Tensor &other, Tensor &output) const { ensureOutputTarget(output); implementation->hadamardMul(*other.implementation, *output.implementation); }
    void hadamardDiv(const Tensor &other, Tensor &output) const { ensureOutputTarget(output); implementation->hadamardDiv(*other.implementation, *output.implementation); }
    void transpose(Tensor &output) const { ensureOutputTarget(output); implementation->transpose(*output.implementation); }
    void inverse(Tensor &output) const { ensureOutputTarget(output); implementation->inverse(*output.implementation); }
    void normalize(Tensor &output) const { ensureOutputTarget(output); implementation->normalize(*output.implementation); }
    void relu(Tensor &output) const { ensureOutputTarget(output); implementation->relu(*output.implementation); }
    void reluBackward(const Tensor &output_gradient, Tensor &input_gradient) const { ensureOutputTarget(input_gradient); implementation->reluBackward(*output_gradient.implementation, *input_gradient.implementation); }
    void gelu(Tensor &output) const { ensureOutputTarget(output); implementation->gelu(*output.implementation); }
    void geluBackward(const Tensor &output_gradient, Tensor &input_gradient) const { ensureOutputTarget(input_gradient); implementation->geluBackward(*output_gradient.implementation, *input_gradient.implementation); }
    void softmax(Tensor &output) const { ensureOutputTarget(output); implementation->softmax(*output.implementation); }
    void softmaxBackward(const Tensor &output_gradient, Tensor &input_gradient) const { ensureOutputTarget(input_gradient); implementation->softmaxBackward(*output_gradient.implementation, *input_gradient.implementation); }
    void matmulAdd(const Tensor &other, const Tensor &biases, Tensor &output) const { ensureOutputTarget(output); implementation->matmulAdd(*other.implementation, *biases.implementation, *output.implementation); }

    void sgdUpdate(const Tensor &gradient, float learning_rate, float max_gradient = 0.0F, float inv_scale = 1.0F)
    {
        implementation->sgdUpdate(*gradient.implementation, learning_rate, max_gradient, inv_scale);
    }

    void adamUpdate(const Tensor &gradient,
                    const Tensor &first_moment,
                    const Tensor &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    size_t timestep,
                    float max_gradient = 1.0F,
                    float inv_scale = 1.0F,
                    float weight_decay = 0.0F)
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

    void conv2d(const Tensor &weights, const Tensor &biases, Tensor &output,
                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                uint32_t output_channels, uint32_t kernel_size,
                uint32_t stride, uint32_t padding,
                Tensor *scratch = nullptr) const
    {
        implementation->conv2d(*weights.implementation, *biases.implementation, *output.implementation,
                               input_height, input_width, input_channels, output_channels, kernel_size, stride, padding,
                               scratch ? scratch->implementation.get() : nullptr);
    }

    void conv2dBackwardInput(const Tensor &weights, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                             uint32_t kernel_size, uint32_t stride, uint32_t padding) const
    {
        implementation->conv2dBackwardInput(*weights.implementation, *input_gradient.implementation,
                                            input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
    }

    void conv2dBackwardWeight(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient,
                              uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                              uint32_t kernel_size, uint32_t stride, uint32_t padding,
                              Tensor *im2col_scratch = nullptr) const
    {
        implementation->conv2dBackwardWeight(*output_gradient.implementation, *weight_gradient.implementation, *bias_gradient.implementation,
                                             input_height, input_width, input_channels, output_height, output_width, output_channels,
                                             kernel_size, stride, padding,
                                             im2col_scratch ? im2col_scratch->implementation.get() : nullptr);
    }

    void maxpool2d(Tensor &output, Tensor &output_mask,
                   uint32_t input_height, uint32_t input_width, uint32_t channels,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding) const
    {
        implementation->maxpool2d(*output.implementation, *output_mask.implementation,
                                  input_height, input_width, channels, kernel_size, stride, padding);
    }

    void maxpool2dBackward(const Tensor &mask, Tensor &input_gradient,
                           uint32_t input_height, uint32_t input_width, uint32_t channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) const
    {
        implementation->maxpool2dBackward(*mask.implementation, *input_gradient.implementation,
                                          input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
    }

    void globalAvgPool2d(Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const
    {
        implementation->globalAvgPool2d(*output.implementation, input_height, input_width, channels);
    }

    void globalAvgPool2dBackward(Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const
    {
        implementation->globalAvgPool2dBackward(*input_gradient.implementation, input_height, input_width, channels);
    }

    void batchNormForward(const Tensor &gamma, const Tensor &beta,
                          Tensor &running_mean, Tensor &running_variance,
                          Tensor &batch_mean, Tensor &batch_variance,
                          Tensor &normalized_input, Tensor &output,
                          float epsilon, float momentum, bool is_training) const
    {
        implementation->batchNormForward(*gamma.implementation, *beta.implementation,
                                         *running_mean.implementation, *running_variance.implementation,
                                         *batch_mean.implementation, *batch_variance.implementation,
                                         *normalized_input.implementation, *output.implementation,
                                         epsilon, momentum, is_training);
    }

    void batchNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input,
                           Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, float epsilon) const
    {
        implementation->batchNormBackward(*output_gradient.implementation, *gamma.implementation,
                                          *batch_variance.implementation, *normalized_input.implementation,
                                          *gamma_gradient.implementation, *beta_gradient.implementation,
                                          *input_gradient.implementation, epsilon);
    }

    void rmsNormForward(const Tensor &gamma, Tensor &inv_rms, Tensor &output, float epsilon = 1e-5f) const
    {
        ensureOutputTarget(inv_rms);
        ensureOutputTarget(output);
        implementation->rmsNormForward(*gamma.implementation, *inv_rms.implementation, *output.implementation, epsilon);
    }

    void rmsNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &inv_rms,
                         Tensor &gamma_gradient, Tensor &input_gradient, bool accumulate_gamma = false) const
    {
        ensureOutputTarget(gamma_gradient);
        ensureOutputTarget(input_gradient);
        implementation->rmsNormBackward(*output_gradient.implementation, *gamma.implementation, *inv_rms.implementation,
                                        *gamma_gradient.implementation, *input_gradient.implementation, accumulate_gamma);
    }

    void applyRoPE(Tensor &output, uint32_t seq_len, uint32_t head_dim, int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0) const
    {
        ensureOutputTarget(output);
        implementation->applyRoPE(*output.implementation, seq_len, head_dim, direction, base, num_heads, mode);
    }

    void swigluForward(const Tensor &b, Tensor &output) const
    {
        ensureOutputTarget(output);
        implementation->swigluForward(*b.implementation, *output.implementation);
    }

    void swigluBackward(const Tensor &output_gradient, const Tensor &b, Tensor &grad_a, Tensor &grad_b) const
    {
        ensureOutputTarget(grad_a);
        ensureOutputTarget(grad_b);
        implementation->swigluBackward(*output_gradient.implementation, *b.implementation, *grad_a.implementation, *grad_b.implementation);
    }

    void fusedSwiGLUForward(Tensor &output) const
    {
        ensureOutputTarget(output);
        implementation->fusedSwiGLUForward(*output.implementation);
    }

    void fusedSwiGLUBackward(const Tensor &output_gradient, Tensor &input_gradient) const
    {
        ensureOutputTarget(input_gradient);
        implementation->fusedSwiGLUBackward(*output_gradient.implementation, *input_gradient.implementation);
    }

    void flashAttentionForward(const Tensor &k, const Tensor &v, Tensor &output,
                               uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                               bool is_causal = false, float scale = 0.0f,
                               Tensor *l_stats = nullptr) const
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

    void flashAttentionBackward(const Tensor &k, const Tensor &v,
                                const Tensor &o, const Tensor &do_grad,
                                Tensor &dq, Tensor &dk, Tensor &dv,
                                uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                                bool is_causal = false, float scale = 0.0f,
                                const Tensor *l_stats = nullptr) const
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

    void embeddingForward(const Tensor &indices, Tensor &output) const
    {
        ensureOutputTarget(output);
        implementation->embeddingForward(*indices.implementation, *output.implementation);
    }

    Tensor embeddingForward(const Tensor &indices) const
    {
        Tensor result(execution_target);
        embeddingForward(indices, result);
        return result;
    }

    void embeddingForward(const std::vector<int32_t> &indices, Tensor &output) const
    {
        std::vector<float> idx_float(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) idx_float[i] = static_cast<float>(indices[i]);
        Tensor idx_tensor(Shape{ indices.size() }, std::move(idx_float), execution_target);
        embeddingForward(idx_tensor, output);
    }

    Tensor embeddingForward(const std::vector<int32_t> &indices) const
    {
        Tensor result(execution_target);
        embeddingForward(indices, result);
        return result;
    }

    void embeddingBackward(const Tensor &indices, const Tensor &output_gradient, Tensor &weight_gradient) const
    {
        ensureOutputTarget(weight_gradient);
        implementation->embeddingBackward(*indices.implementation, *output_gradient.implementation, *weight_gradient.implementation);
    }

    void embeddingBackward(const std::vector<int32_t> &indices, const Tensor &output_gradient, Tensor &weight_gradient) const
    {
        std::vector<float> idx_float(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) idx_float[i] = static_cast<float>(indices[i]);
        Tensor idx_tensor(Shape{ indices.size() }, std::move(idx_float), execution_target);
        embeddingBackward(idx_tensor, output_gradient, weight_gradient);
    }

    void singleTokenAttentionForward(const Tensor &k, const Tensor &v, Tensor &output,
                                     size_t num_heads, size_t head_dim, size_t total_seq_len) const
    {
        implementation->singleTokenAttentionForward(*k.implementation, *v.implementation, *output.implementation,
                                                    num_heads, head_dim, total_seq_len);
    }

    Tensor singleTokenAttentionForward(const Tensor &k, const Tensor &v,
                                       size_t num_heads, size_t head_dim, size_t total_seq_len) const
    {
        Tensor result(execution_target);
        singleTokenAttentionForward(k, v, result, num_heads, head_dim, total_seq_len);
        return result;
    }

    float fusedCrossEntropyLoss(const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens = 0) const
    {
        if (d_logits.getExecutionTarget() != execution_target || d_logits.getDataType() != getDataType())
        {
            d_logits = Tensor(getShape(), getDataType(), execution_target);
        }
        return implementation->fusedCrossEntropyLoss(*targets.implementation, *d_logits.implementation, valid_tokens);
    }

    float fusedCrossEntropyLoss(const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens = 0) const
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

    void linearForward(const Tensor &weights, const Tensor &biases, Tensor &output) const
    {
        ensureOutputTarget(output);
        implementation->linearForward(*weights.implementation, *biases.implementation, *output.implementation);
    }

    void linearBackwardInput(const Tensor &weights, Tensor &input_gradient) const
    {
        ensureOutputTarget(input_gradient);
        implementation->linearBackwardInput(*weights.implementation, *input_gradient.implementation);
    }

    void linearBackwardWeightBias(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate = false) const
    {
        ensureOutputTarget(weight_gradient);
        ensureOutputTarget(bias_gradient);
        implementation->linearBackwardWeightBias(*output_gradient.implementation, *weight_gradient.implementation, *bias_gradient.implementation, accumulate);
    }

    void linearBackwardWeightAdam(const Tensor &output_gradient, Tensor &weights, Tensor &first_moment, Tensor &second_moment, Tensor &bias_gradient,
                                  float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient = 1.0F, float inv_scale = 1.0F, float weight_decay = 0.0F) const
    {
        ensureOutputTarget(weights);
        ensureOutputTarget(first_moment);
        ensureOutputTarget(second_moment);
        ensureOutputTarget(bias_gradient);
        implementation->linearBackwardWeightAdam(*output_gradient.implementation, *weights.implementation, *first_moment.implementation, *second_moment.implementation, *bias_gradient.implementation,
                                                 learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
    }

    void batchNorm2dForward(const Tensor &gamma, const Tensor &beta,
                            Tensor &running_mean, Tensor &running_variance,
                            Tensor &batch_mean, Tensor &batch_variance,
                            Tensor &normalized_input, Tensor &output,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const
    {
        implementation->batchNorm2dForward(*gamma.implementation, *beta.implementation,
                                           *running_mean.implementation, *running_variance.implementation,
                                           *batch_mean.implementation, *batch_variance.implementation,
                                           *normalized_input.implementation, *output.implementation,
                                           input_height, input_width, input_channels, epsilon, momentum, is_training);
    }

    void batchNorm2dBackward(const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input,
                             Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const
    {
        implementation->batchNorm2dBackward(*gamma.implementation, *batch_variance.implementation,
                                            *normalized_input.implementation, *gamma_gradient.implementation,
                                            *beta_gradient.implementation, *input_gradient.implementation,
                                            input_height, input_width, input_channels, epsilon);
    }

    void cceLoss(const Tensor &target, Tensor &output, float epsilon = 1e-7F) const
    {
        implementation->cceLoss(*target.implementation, *output.implementation, epsilon);
    }

    void mseLoss(const Tensor &target, Tensor &output) const
    {
        implementation->mseLoss(*target.implementation, *output.implementation);
    }

    void maeLoss(const Tensor &target, Tensor &output) const
    {
        implementation->maeLoss(*target.implementation, *output.implementation);
    }

    void bceLoss(const Tensor &target, Tensor &output, float epsilon = 1e-7F) const
    {
        implementation->bceLoss(*target.implementation, *output.implementation, epsilon);
    }

    void huberLoss(const Tensor &target, Tensor &output, float delta = 1.0F) const
    {
        implementation->huberLoss(*target.implementation, *output.implementation, delta);
    }

    Tensor operator*(const Tensor &other) const
    {
        Tensor result(execution_target);
        matmul(other, result);
        return result;
    }

    Tensor operator/(const Tensor &other) const
    {
        Tensor result(execution_target);
        matdiv(other, result);
        return result;
    }

    Tensor operator+(const Tensor &other) const
    {
        Tensor result(execution_target);
        add(other, result);
        return result;
    }

    Tensor operator-(const Tensor &other) const
    {
        Tensor result(execution_target);
        sub(other, result);
        return result;
    }

    Tensor operator*(float scalar) const
    {
        Tensor result(execution_target);
        mulScalar(scalar, result);
        return result;
    }

    Tensor operator/(float scalar) const
    {
        Tensor result(execution_target);
        divScalar(scalar, result);
        return result;
    }

    Tensor hadamardMul(const Tensor &other) const
    {
        Tensor result(execution_target);
        hadamardMul(other, result);
        return result;
    }

    Tensor hadamardDiv(const Tensor &other) const
    {
        Tensor result(execution_target);
        hadamardDiv(other, result);
        return result;
    }

    Tensor transpose() const
    {
        Tensor result(execution_target);
        transpose(result);
        return result;
    }

    Tensor inverse() const
    {
        Tensor result(execution_target);
        inverse(result);
        return result;
    }

    Tensor normalize() const
    {
        Tensor result(execution_target);
        normalize(result);
        return result;
    }

    Tensor relu() const
    {
        Tensor result(execution_target);
        relu(result);
        return result;
    }

    Tensor reluBackward(const Tensor &output_gradient) const
    {
        Tensor result(execution_target);
        reluBackward(output_gradient, result);
        return result;
    }

    Tensor gelu() const
    {
        Tensor result(execution_target);
        gelu(result);
        return result;
    }

    Tensor geluBackward(const Tensor &output_gradient) const
    {
        Tensor result(execution_target);
        geluBackward(output_gradient, result);
        return result;
    }

    Tensor softmax() const
    {
        Tensor result(execution_target);
        softmax(result);
        return result;
    }

    Tensor softmaxBackward(const Tensor &output_gradient) const
    {
        Tensor result(execution_target);
        softmaxBackward(output_gradient, result);
        return result;
    }

    Tensor matmulAdd(const Tensor &other, const Tensor &biases) const
    {
        Tensor result(execution_target);
        matmulAdd(other, biases, result);
        return result;
    }

    void concatenateColumns(const Tensor &other, Tensor &output) const
    {
        implementation->concatenateColumns(*other.implementation, *output.implementation);
    }

    Tensor concatenateColumns(const Tensor &other) const
    {
        Tensor result(execution_target);
        concatenateColumns(other, result);
        return result;
    }

    void concatenateRows(const Tensor &other, Tensor &output) const
    {
        implementation->concatenateRows(*other.implementation, *output.implementation);
    }

    Tensor concatenateRows(const Tensor &other) const
    {
        Tensor result(execution_target);
        concatenateRows(other, result);
        return result;
    }

    void splitColumns(size_t split_index, Tensor &result_left, Tensor &result_right) const
    {
        implementation->splitColumns(split_index, *result_left.implementation, *result_right.implementation);
    }

    std::pair<Tensor, Tensor> splitColumns(size_t split_index) const
    {
        Tensor result_left(execution_target);
        Tensor result_right(execution_target);
        splitColumns(split_index, result_left, result_right);
        return {std::move(result_left), std::move(result_right)};
    }

    void splitRows(size_t split_index, Tensor &result_up, Tensor &result_down) const
    {
        implementation->splitRows(split_index, *result_up.implementation, *result_down.implementation);
    }

    std::pair<Tensor, Tensor> splitRows(size_t split_index) const
    {
        Tensor result_up(execution_target);
        Tensor result_down(execution_target);
        splitRows(split_index, result_up, result_down);
        return {std::move(result_up), std::move(result_down)};
    }

    float getScalar() const
    {
        const auto host_data = getData();
        float sum = 0.0F;
        for (float val : host_data)
        {
            sum += val;
        }
        return sum;
    }

    void print(size_t max_display_rows = 10, size_t max_display_columns = 10) const
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

    void saveTensor(std::ofstream &output_file_stream) const
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

    static Tensor loadTensor(std::ifstream &input_file_stream, Execution_Target target = Execution_Target::CPU)
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

    void saveMatrix(std::ofstream &output_file_stream) const
    {
        saveTensor(output_file_stream);
    }

    static Tensor loadMatrix(std::ifstream &input_file_stream, Execution_Target target = Execution_Target::CPU)
    {
        return loadTensor(input_file_stream, target);
    }

    Tensor clone() const
    {
        return Tensor(getShape(), getData(), getDataType(), execution_target);
    }

    void fill(float value)
    {
        implementation->fill(value);
    }

    void zero()
    {
        implementation->zero();
    }

    void invalidateFp16Cache() noexcept
    {
        if (implementation)
        {
            implementation->invalidateFp16Cache();
        }
    }

    void prewarmFp16Cache()
    {
        if (implementation)
        {
            implementation->prewarmFp16Cache();
        }
    }

    const Shape &getShape() const noexcept { return implementation->getShape(); }
    const Stride &getStrides() const noexcept { return implementation->getStrides(); }
    Storage_Handle getStorage() const { const Tensor_Impl &const_implementation = *implementation; return const_implementation.getStorage(); }
    Mutable_Storage_Handle getStorage() { return implementation->getStorage(); }
    std::vector<float> getData() const { return implementation->getData(); }
    std::shared_ptr<Tensor_Impl> getImplementation() const noexcept { return implementation; }
    size_t getTotalElements() const noexcept { return implementation->getTotalElements(); }
    size_t getColumns() const noexcept { return implementation->getColumns(); }
    size_t getRank() const noexcept { return implementation->getRank(); }
    size_t getRows() const noexcept { return implementation->getRows(); }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    Data_Type getDataType() const noexcept { return implementation->getDataType(); }
    bool isEmpty() const noexcept { return implementation->isEmpty(); }

    void uploadData(const std::vector<float> &host_data) { implementation->uploadData(host_data); }
    void setImplementation(std::shared_ptr<Tensor_Impl> _impl) noexcept { implementation = std::move(_impl); }
    void setDataType(Data_Type _type) noexcept { implementation->setDataType(_type); }
    void logFp16Stats(std::string_view tensor_name, Log_Level level = Log_Level::LOG_DEBUG) const
    {
        std::vector<float> host_data = getData();
        Logger::logFp16TensorStats(tensor_name, host_data, getTotalElements(), getDataType() == Data_Type::FLOAT16 ? "FLOAT16" : "FLOAT32", level);
    }
    void setExecutionTarget(Execution_Target new_target)
    {
        if (execution_target == new_target)
        {
            return;
        }
        Shape current_shape = getShape();
        std::vector<float> current_data = getData();
        *this = Tensor(current_shape, current_data, getDataType(), new_target);
        if (new_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
    }
};