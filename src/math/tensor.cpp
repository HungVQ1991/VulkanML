#include "math/tensor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
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

namespace
{
    void validateImplMatchesTarget(const std::shared_ptr<Tensor_Impl> &impl, Execution_Target target, const char *caller)
    {
        if (!impl)
        {
            throw std::invalid_argument(std::format("Tensor::{}: Null tensor implementation", caller));
        }
        if (target == Execution_Target::CPU)
        {
            if (dynamic_cast<const Cpu_Tensor_Impl *>(impl.get()) == nullptr)
            {
                throw std::invalid_argument(std::format("Tensor::{}: Expected Cpu_Tensor_Impl for Execution_Target::CPU", caller));
            }
        }
        else
        {
            if (dynamic_cast<const Gpu_Tensor_Impl *>(impl.get()) == nullptr)
            {
                throw std::invalid_argument(std::format("Tensor::{}: Expected Gpu_Tensor_Impl for Execution_Target::VULKAN_GPU", caller));
            }
        }
    }

    void validateSameExecutionTarget(const Tensor &lhs, const Tensor &rhs, const char *op_name)
    {
        if (!lhs.getImplementation() || !rhs.getImplementation())
        {
            throw std::invalid_argument(std::format("Tensor::{}: Null tensor implementation", op_name));
        }
        if (lhs.getExecutionTarget() != rhs.getExecutionTarget())
        {
            throw std::invalid_argument(std::format("Tensor::{}: Execution_Target mismatch between operands", op_name));
        }
    }

    void readExact(std::ifstream &stream, void *dst, size_t bytes, const char *err_msg)
    {
        if (bytes == 0)
        {
            return;
        }
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
        {
            throw std::overflow_error(err_msg);
        }
        stream.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(bytes));
        if (!stream || static_cast<size_t>(stream.gcount()) != bytes)
        {
            throw std::runtime_error(err_msg);
        }
    }

    size_t safeMulSizeT(size_t a, size_t b, const char *err_msg)
    {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        {
            throw std::overflow_error(err_msg);
        }
        return a * b;
    }

    template <typename... Args>
    std::shared_ptr<Tensor_Impl> makeTensorImpl(Execution_Target target, Args &&...args)
    {
        if (target == Execution_Target::CPU)
        {
            return std::make_shared<Cpu_Tensor_Impl>(std::forward<Args>(args)...);
        }
        return std::make_shared<Gpu_Tensor_Impl>(std::forward<Args>(args)...);
    }

    template <typename... Tensors>
    void rejectTernary(const Tensors &...tensors)
    {
        if (((tensors.getDataType() == Data_Type::TERNARY_INT2) || ...))
        {
            throw std::invalid_argument("Operation not supported on TERNARY_INT2 tensor");
        }
    }

    void ensureTargetMatch(const Tensor &self, Tensor &output)
    {
        if (!output.getImplementation() || output.getExecutionTarget() != self.getExecutionTarget())
        {
            output = Tensor(self.getExecutionTarget());
        }
    }

    void validateOrInitTarget(const Tensor &self, Tensor &target, const char *op_name)
    {
        if (!target.isEmpty())
        {
            validateSameExecutionTarget(self, target, op_name);
        }
        else
        {
            ensureTargetMatch(self, target);
        }
    }

    void validateAccumulateTarget(const Tensor &self, Tensor &target, bool accumulate, const char *op_name)
    {
        if (!accumulate || target.isEmpty())
        {
            ensureTargetMatch(self, target);
        }
        else
        {
            validateSameExecutionTarget(self, target, op_name);
        }
    }

    void prepareUnary(const Tensor &self, Tensor &output, const char *op_name)
    {
        if (!self.getImplementation())
        {
            throw std::invalid_argument(std::format("Tensor::{}: Null tensor implementation", op_name));
        }
        rejectTernary(self);
        ensureTargetMatch(self, output);
    }

    void prepareBinary(const Tensor &self, const Tensor &other, Tensor &output, const char *op_name)
    {
        validateSameExecutionTarget(self, other, op_name);
        rejectTernary(self, other);
        ensureTargetMatch(self, output);
    }

    Tensor buildIndexTensor(const Tensor &self, const std::vector<int32_t> &indices)
    {
        const Execution_Target target = self.getExecutionTarget();
        if (target == Execution_Target::VULKAN_GPU)
        {
            const auto &gpu_impl = Gpu_Tensor_Impl::castToGpu(*self.getImplementation());
            auto indices_buf = gpu_impl.getOrUploadIndicesBuffer(indices);
            Shape idx_shape{indices.size()};
            auto idx_impl = std::make_shared<Gpu_Tensor_Impl>(
                idx_shape, idx_shape.computeContiguousStrides(), indices_buf, 0, Data_Type::FLOAT32);
            return Tensor(std::move(idx_impl), Execution_Target::VULKAN_GPU);
        }
        std::vector<float> idx_float(indices.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            idx_float[i] = static_cast<float>(indices[i]);
        }
        return Tensor(Shape{indices.size()}, std::move(idx_float), target);
    }

    void syncGpuIfNeeded(Execution_Target target)
    {
        if (target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
            Execution_Engine::getInstance().waitIdle();
        }
    }
}

void Tensor::requireImplementation(const char *op_name) const
{
    if (!implementation)
    {
        throw std::invalid_argument(std::format("Tensor::{}: Null tensor implementation", op_name));
    }
}

Tensor::Tensor(Execution_Target target)
    : implementation(makeTensorImpl(target, 0, 0)), execution_target(target)
{
}

Tensor::Tensor(size_t rows, size_t columns, Execution_Target target)
    : implementation(makeTensorImpl(target, rows, columns)), execution_target(target)
{
}

Tensor::Tensor(size_t rows, size_t columns, const std::vector<float> &host_data, Execution_Target target)
    : implementation(makeTensorImpl(target, rows, columns, host_data)), execution_target(target)
{
}

Tensor::Tensor(size_t rows, size_t columns, std::vector<float> &&host_data, Execution_Target target)
    : implementation(makeTensorImpl(target, rows, columns, std::move(host_data))), execution_target(target)
{
}

Tensor::Tensor(Shape shape, Execution_Target target)
    : implementation(makeTensorImpl(target, shape)), execution_target(target)
{
}

Tensor::Tensor(Shape shape, const std::vector<float> &host_data, Execution_Target target)
    : implementation(makeTensorImpl(target, shape, host_data)), execution_target(target)
{
}

Tensor::Tensor(Shape shape, std::vector<float> &&host_data, Execution_Target target)
    : implementation(makeTensorImpl(target, shape, std::move(host_data))), execution_target(target)
{
}

Tensor::Tensor(Shape shape, Data_Type type, Execution_Target target)
    : implementation(makeTensorImpl(target, shape, type)), execution_target(target)
{
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
        requireImplementation("to");
        auto src_impl = implementation;
        if (!output.implementation ||
            output.implementation == src_impl ||
            output.getExecutionTarget() != execution_target ||
            output.getShape() != src_impl->getShape() ||
            output.getDataType() != target_type ||
            !output.implementation->isContiguous() ||
            output.implementation->getByteOffset() != 0)
        {
            output = Tensor(src_impl->getShape(), target_type, execution_target);
        }
        src_impl->to(target_type, *output.implementation);
    }

Tensor Tensor::to(Data_Type target_type) const
{
        requireImplementation("to");
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
    : execution_target(target)
{
        validateImplMatchesTarget(impl, target, "Tensor");
        implementation = std::move(impl);
    }

void Tensor::initializeShape(size_t rows, size_t columns)
{
        reshape(Shape{rows, columns});
    }

void Tensor::reshape(size_t rows, size_t columns)
{
        reshape(Shape{rows, columns});
    }

void Tensor::reshape(Shape new_shape)
{
        if (!implementation)
        {
            Tensor fresh(new_shape, Data_Type::FLOAT32, execution_target);
            implementation = std::move(fresh.implementation);
            return;
        }
        if (implementation->getShape() == new_shape &&
            implementation->isContiguous() &&
            implementation->getByteOffset() == 0)
        {
            return;
        }
        if (implementation.use_count() > 1)
        {
            if (!implementation->isContiguous() || implementation->getByteOffset() != 0)
            {
                throw std::runtime_error("Tensor::reshape: Cannot reshape non-contiguous tensor view");
            }
            if (new_shape.getTotalElements() == implementation->getTotalElements() &&
                implementation->getDataType() != Data_Type::TERNARY_INT2)
            {
                implementation = view(new_shape).getImplementation();
                return;
            }
            Tensor fresh(new_shape, implementation->getDataType(), execution_target);
            implementation = std::move(fresh.implementation);
            return;
        }
        implementation->reshape(new_shape);
    }

Tensor Tensor::view(Shape new_shape) const
{
    requireImplementation("view");
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Tensor::view: Cannot create view of TERNARY_INT2 tensor");
    }
    if (new_shape.getTotalElements() != getTotalElements())
    {
        throw std::invalid_argument("Tensor::view: Total elements mismatch between current shape and view shape");
    }
    Tensor base = contiguous();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        const auto &gpu_base = Gpu_Tensor_Impl::castToGpu(*base.getImplementation());
        auto view_impl = std::make_shared<Gpu_Tensor_Impl>(
            new_shape,
            new_shape.computeContiguousStrides(),
            gpu_base.getStorageBuffer(),
            0,
            base.getDataType());
        return Tensor(std::move(view_impl), Execution_Target::VULKAN_GPU);
    }
    const auto &cpu_base = Cpu_Tensor_Impl::castToCpu(*base.getImplementation());
    if (base.getDataType() == Data_Type::FLOAT16)
    {
        auto view_impl = std::make_shared<Cpu_Tensor_Impl>(0, 0);
        view_impl->setDataType(Data_Type::FLOAT16);
        view_impl->setShape(new_shape);
        view_impl->setByteOffset(0);
        view_impl->getStorageBuffer().reset();
        view_impl->setStorageBufferFp16(cpu_base.getStorageBufferFp16());
        return Tensor(std::move(view_impl), Execution_Target::CPU);
    }
    auto view_impl = std::make_shared<Cpu_Tensor_Impl>(
        new_shape,
        new_shape.computeContiguousStrides(),
        cpu_base.getStorageBuffer(),
        0);
    return Tensor(std::move(view_impl), Execution_Target::CPU);
}

Tensor Tensor::permute(const std::vector<size_t> &axes_permutation) const
{
        requireImplementation("permute");
        if (getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Tensor::permute: TERNARY_INT2 is not supported");
        }
        size_t rank = getRank();
        if (axes_permutation.size() != rank)
        {
            throw std::invalid_argument("Tensor::permute: Permutation size must match tensor rank");
        }
        std::array<bool, MAX_TENSOR_RANK> seen{};
        for (size_t ax : axes_permutation)
        {
            if (ax >= rank || seen[ax])
            {
                throw std::invalid_argument("Tensor::permute: Invalid or duplicate axis in permutation");
            }
            seen[ax] = true;
        }
        Tensor result(execution_target);
        implementation->permute(axes_permutation, *result.implementation);
        return result;
    }

Tensor Tensor::slice(size_t axis, size_t start, size_t length) const
{
        requireImplementation("slice");
        if (getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Tensor::slice: TERNARY_INT2 is not supported");
        }
        size_t rank = getRank();
        if (axis >= rank)
        {
            throw std::out_of_range("Tensor::slice: Axis out of range");
        }
        size_t dim = getShape()[axis];
        if (start > dim || length > dim - start)
        {
            throw std::out_of_range("Tensor::slice: Slice range exceeds dimension bounds");
        }
        Tensor result(execution_target);
        implementation->slice(axis, start, length, *result.implementation);
        return result;
    }

void Tensor::updateSlice(size_t axis, size_t start, const Tensor &source)
{
        validateSameExecutionTarget(*this, source, "updateSlice");
        if (getDataType() == Data_Type::TERNARY_INT2 || source.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Tensor::updateSlice: TERNARY_INT2 is not supported");
        }
        if (getDataType() != source.getDataType())
        {
            throw std::invalid_argument("Tensor::updateSlice: Data_Type mismatch");
        }
        size_t rank = getRank();
        if (rank != source.getRank() || axis >= rank)
        {
            throw std::invalid_argument("Tensor::updateSlice: Rank or axis mismatch");
        }
        const Shape &dst_shape = getShape();
        const Shape &src_shape = source.getShape();
        for (size_t d = 0; d < rank; ++d)
        {
            if (d != axis && dst_shape[d] != src_shape[d])
            {
                throw std::invalid_argument("Tensor::updateSlice: Non-axis dimensions must match");
            }
        }
        size_t dim = dst_shape[axis];
        size_t src_len = src_shape[axis];
        if (start > dim || src_len > dim - start)
        {
            throw std::out_of_range("Tensor::updateSlice: Slice range exceeds destination dimension");
        }
        implementation->updateSlice(axis, start, *source.implementation);
    }

Tensor Tensor::gatherRows(const std::vector<int32_t> &indices) const
{
        requireImplementation("gatherRows");
        if (getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Tensor::gatherRows: TERNARY_INT2 is not supported");
        }
        if (getRank() < 1)
        {
            throw std::invalid_argument("Tensor::gatherRows: Tensor rank must be at least 1");
        }
        size_t S = indices.size();
        size_t D = (getRank() == 1) ? 1 : getColumns();
        Tensor result(Shape{ S, D }, getDataType(), execution_target);
        implementation->gatherRows(indices, *result.implementation);
        return result;
    }

Tensor Tensor::contiguous() const
{
        requireImplementation("contiguous");
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
        if (!output.implementation || output.getExecutionTarget() != execution_target)
        {
            output = Tensor(execution_target);
        }
    }

void Tensor::contiguous(Tensor &output) const
{
        requireImplementation("contiguous");
        auto src_impl = implementation;
        if (!output.implementation ||
            output.implementation == src_impl ||
            output.implementation.use_count() > 1 ||
            output.getExecutionTarget() != execution_target)
        {
            output = Tensor(execution_target);
        }
        src_impl->contiguous(*output.implementation);
    }

void Tensor::matmul(const Tensor &other, Tensor &output) const
{
    prepareBinary(*this, other, output, "matmul");
    implementation->matmul(*other.implementation, *output.implementation);
}

void Tensor::matdiv(const Tensor &other, Tensor &output) const
{
    prepareBinary(*this, other, output, "matdiv");
    implementation->matdiv(*other.implementation, *output.implementation);
}

void Tensor::add(const Tensor &other, Tensor &output) const
{
    prepareBinary(*this, other, output, "add");
    implementation->add(*other.implementation, *output.implementation);
}

void Tensor::sub(const Tensor &other, Tensor &output) const
{
    prepareBinary(*this, other, output, "sub");
    implementation->sub(*other.implementation, *output.implementation);
}

void Tensor::mulScalar(float scalar, Tensor &output) const
{
    prepareUnary(*this, output, "mulScalar");
    implementation->mulScalar(scalar, *output.implementation);
}

void Tensor::divScalar(float scalar, Tensor &output) const
{
    prepareUnary(*this, output, "divScalar");
    implementation->divScalar(scalar, *output.implementation);
}

void Tensor::hadamardMul(const Tensor &other, Tensor &output) const
{
    prepareBinary(*this, other, output, "hadamardMul");
    implementation->hadamardMul(*other.implementation, *output.implementation);
}

void Tensor::hadamardDiv(const Tensor &other, Tensor &output) const
{
    prepareBinary(*this, other, output, "hadamardDiv");
    implementation->hadamardDiv(*other.implementation, *output.implementation);
}

void Tensor::transpose(Tensor &output) const
{
    prepareUnary(*this, output, "transpose");
    implementation->transpose(*output.implementation);
}

void Tensor::inverse(Tensor &output) const
{
    prepareUnary(*this, output, "inverse");
    implementation->inverse(*output.implementation);
}

void Tensor::normalize(Tensor &output) const
{
    prepareUnary(*this, output, "normalize");
    implementation->normalize(*output.implementation);
}

void Tensor::relu(Tensor &output) const
{
    prepareUnary(*this, output, "relu");
    ops::activation::relu(*this, output);
}

void Tensor::reluBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
    prepareBinary(*this, output_gradient, input_gradient, "reluBackward");
    ops::activation::reluBackward(*this, output_gradient, input_gradient);
}

void Tensor::gelu(Tensor &output) const
{
    prepareUnary(*this, output, "gelu");
    ops::activation::gelu(*this, output);
}

void Tensor::geluBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
    prepareBinary(*this, output_gradient, input_gradient, "geluBackward");
    ops::activation::geluBackward(*this, output_gradient, input_gradient);
}

void Tensor::sigmoid(Tensor &output) const
{
    prepareUnary(*this, output, "sigmoid");
    ops::activation::sigmoid(*this, output);
}

void Tensor::sigmoidBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
    prepareBinary(*this, output_gradient, input_gradient, "sigmoidBackward");
    ops::activation::sigmoidBackward(*this, output_gradient, input_gradient);
}

void Tensor::softmax(Tensor &output) const
{
    prepareUnary(*this, output, "softmax");
    ops::activation::softmax(*this, output);
}

void Tensor::softmaxBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
    prepareBinary(*this, output_gradient, input_gradient, "softmaxBackward");
    ops::activation::softmaxBackward(*this, output_gradient, input_gradient);
}

void Tensor::matmulAdd(const Tensor &other, const Tensor &biases, Tensor &output) const
{
    validateSameExecutionTarget(*this, other, "matmulAdd");
    validateSameExecutionTarget(*this, biases, "matmulAdd");
    rejectTernary(*this, other, biases);
    ensureOutputTarget(output);
    implementation->matmulAdd(*other.implementation, *biases.implementation, *output.implementation);
}

void Tensor::sgdUpdate(const Tensor &gradient, float learning_rate, float max_gradient, float inv_scale)
{
    validateSameExecutionTarget(*this, gradient, "sgdUpdate");
    rejectTernary(*this, gradient);
    ops::optimizer::sgdStep(*this, gradient, learning_rate, max_gradient, inv_scale);
}

void Tensor::adamUpdate(const Tensor &gradient, Tensor &first_moment, Tensor &second_moment, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay)
{
    validateSameExecutionTarget(*this, gradient, "adamUpdate");
    validateSameExecutionTarget(*this, first_moment, "adamUpdate");
    validateSameExecutionTarget(*this, second_moment, "adamUpdate");
    rejectTernary(*this, gradient, first_moment, second_moment);
    ops::optimizer::adamStep(*this, gradient, first_moment, second_moment, learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
}

void Tensor::conv2d(const Tensor &weights, const Tensor &biases, Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor *scratch) const
{
    validateSameExecutionTarget(*this, weights, "conv2d");
    validateSameExecutionTarget(*this, biases, "conv2d");
    ensureOutputTarget(output);
    if (scratch != nullptr)
    {
        ensureOutputTarget(*scratch);
    }
    ops::vision::conv2d(*this, weights, biases, output,
                        input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, scratch);
}

void Tensor::conv2dBackwardInput(const Tensor &weights, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
    validateSameExecutionTarget(*this, weights, "conv2dBackwardInput");
    ensureOutputTarget(input_gradient);
    ops::vision::conv2dBackwardInput(*this, weights, input_gradient,
                                     input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
}

void Tensor::conv2dBackwardWeight(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor *im2col_scratch) const
{
    validateSameExecutionTarget(*this, output_gradient, "conv2dBackwardWeight");
    ensureOutputTarget(weight_gradient);
    ensureOutputTarget(bias_gradient);
    if (im2col_scratch != nullptr)
    {
        ensureOutputTarget(*im2col_scratch);
    }
    ops::vision::conv2dBackwardWeight(*this, output_gradient, weight_gradient, bias_gradient,
                                      input_height, input_width, input_channels, output_height, output_width, output_channels,
                                      kernel_size, stride, padding, im2col_scratch);
}

void Tensor::maxpool2d(Tensor &output, Tensor &output_mask, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
    requireImplementation("maxpool2d");
    ensureOutputTarget(output);
    ensureOutputTarget(output_mask);
    ops::vision::maxpool2d(*this, output, output_mask,
                           input_height, input_width, channels, kernel_size, stride, padding);
}

void Tensor::maxpool2dBackward(const Tensor &mask, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t output_height, uint32_t output_width, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
{
    validateSameExecutionTarget(*this, mask, "maxpool2dBackward");
    ensureOutputTarget(input_gradient);
    ops::vision::maxpool2dBackward(*this, mask, input_gradient,
                                   input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
}

void Tensor::globalAvgPool2d(Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const
{
    requireImplementation("globalAvgPool2d");
    ensureOutputTarget(output);
    ops::vision::globalAvgPool2d(*this, output, input_height, input_width, channels);
}

void Tensor::globalAvgPool2dBackward(Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const
{
    requireImplementation("globalAvgPool2dBackward");
    ensureOutputTarget(input_gradient);
    ops::vision::globalAvgPool2dBackward(*this, input_gradient, input_height, input_width, channels);
}

void Tensor::batchNormForward(const Tensor &gamma, const Tensor &beta, Tensor &running_mean, Tensor &running_variance, Tensor &batch_mean, Tensor &batch_variance, Tensor &normalized_input, Tensor &output, float epsilon, float momentum, bool is_training) const
{
    validateSameExecutionTarget(*this, gamma, "batchNormForward");
    validateSameExecutionTarget(*this, beta, "batchNormForward");
    validateOrInitTarget(*this, running_mean, "batchNormForward");
    validateOrInitTarget(*this, running_variance, "batchNormForward");
    ensureOutputTarget(batch_mean);
    ensureOutputTarget(batch_variance);
    ensureOutputTarget(normalized_input);
    ensureOutputTarget(output);
    ops::normalization::batchNormForward(*this, gamma, beta, running_mean, running_variance, batch_mean, batch_variance, normalized_input, output, epsilon, momentum, is_training);
}

void Tensor::batchNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input, Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, float epsilon) const
{
    validateSameExecutionTarget(*this, output_gradient, "batchNormBackward");
    validateSameExecutionTarget(*this, gamma, "batchNormBackward");
    validateSameExecutionTarget(*this, batch_variance, "batchNormBackward");
    validateSameExecutionTarget(*this, normalized_input, "batchNormBackward");
    ensureOutputTarget(gamma_gradient);
    ensureOutputTarget(beta_gradient);
    ensureOutputTarget(input_gradient);
    ops::normalization::batchNormBackward(*this, output_gradient, gamma, batch_variance, normalized_input, gamma_gradient, beta_gradient, input_gradient, epsilon);
}

void Tensor::rmsNormForward(const Tensor &gamma, Tensor &inv_rms, Tensor &output, float epsilon) const
{
    validateSameExecutionTarget(*this, gamma, "rmsNormForward");
    ensureOutputTarget(inv_rms);
    ensureOutputTarget(output);
    ops::normalization::rmsNormForward(*this, gamma, inv_rms, output, epsilon);
}

void Tensor::rmsNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &inv_rms, Tensor &gamma_gradient, Tensor &input_gradient, bool accumulate_gamma) const
{
    validateSameExecutionTarget(*this, output_gradient, "rmsNormBackward");
    validateSameExecutionTarget(*this, gamma, "rmsNormBackward");
    validateSameExecutionTarget(*this, inv_rms, "rmsNormBackward");
    validateAccumulateTarget(*this, gamma_gradient, accumulate_gamma, "rmsNormBackward");
    ensureOutputTarget(input_gradient);
    ops::normalization::rmsNormBackward(*this, output_gradient, gamma, inv_rms, gamma_gradient, input_gradient, accumulate_gamma);
}

void Tensor::applyRoPE(Tensor &output, uint32_t seq_len, uint32_t head_dim, int direction, float base, uint32_t num_heads, uint32_t mode, uint32_t position_offset) const
{
    requireImplementation("applyRoPE");
    ensureOutputTarget(output);
    ops::llm::applyRoPE(*this, output, seq_len, head_dim, direction, base, num_heads, mode, position_offset);
}

void Tensor::swigluForward(const Tensor &b, Tensor &output) const
{
    validateSameExecutionTarget(*this, b, "swigluForward");
    ensureOutputTarget(output);
    ops::llm::swigluForward(*this, b, output);
}

void Tensor::swigluBackward(const Tensor &output_gradient, const Tensor &b, Tensor &grad_a, Tensor &grad_b) const
{
    validateSameExecutionTarget(*this, output_gradient, "swigluBackward");
    validateSameExecutionTarget(*this, b, "swigluBackward");
    ensureOutputTarget(grad_a);
    ensureOutputTarget(grad_b);
    ops::llm::swigluBackward(*this, b, output_gradient, grad_a, grad_b, ops::llm::SwiGLU_Config{});
}

void Tensor::fusedSwiGLUForward(Tensor &output) const
{
    requireImplementation("fusedSwiGLUForward");
    ensureOutputTarget(output);
    ops::llm::fusedSwiGLUForward(*this, output);
}

void Tensor::fusedSwiGLUBackward(const Tensor &output_gradient, Tensor &input_gradient) const
{
    validateSameExecutionTarget(*this, output_gradient, "fusedSwiGLUBackward");
    ensureOutputTarget(input_gradient);
    ops::llm::fusedSwiGLUBackward(*this, output_gradient, input_gradient, ops::llm::SwiGLU_Config{});
}

void Tensor::flashAttentionForward(const Tensor &k, const Tensor &v, Tensor &output, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, Tensor *l_stats, uint32_t kv_seq_len, uint32_t causal_offset, uint32_t num_kv_heads) const
{
    validateSameExecutionTarget(*this, k, "flashAttentionForward");
    validateSameExecutionTarget(*this, v, "flashAttentionForward");
    ensureOutputTarget(output);
    if (l_stats != nullptr)
    {
        ensureOutputTarget(*l_stats);
    }
    ops::llm::flashAttentionForward(*this, k, v, output, num_heads, seq_len, head_dim, is_causal, scale, l_stats, kv_seq_len, causal_offset, num_kv_heads);
}

void Tensor::flashAttentionBackward(const Tensor &k, const Tensor &v, const Tensor &o, const Tensor &do_grad, Tensor &dq, Tensor &dk, Tensor &dv, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, const Tensor *l_stats, uint32_t num_kv_heads) const
{
    validateSameExecutionTarget(*this, k, "flashAttentionBackward");
    validateSameExecutionTarget(*this, v, "flashAttentionBackward");
    validateSameExecutionTarget(*this, o, "flashAttentionBackward");
    validateSameExecutionTarget(*this, do_grad, "flashAttentionBackward");
    if (l_stats != nullptr)
    {
        validateSameExecutionTarget(*this, *l_stats, "flashAttentionBackward");
    }
    ensureOutputTarget(dq);
    ensureOutputTarget(dk);
    ensureOutputTarget(dv);
    ops::llm::flashAttentionBackward(*this, k, v, o, do_grad, dq, dk, dv, num_heads, seq_len, head_dim, is_causal, scale, l_stats, num_kv_heads);
}

void Tensor::embeddingForward(const Tensor &indices, Tensor &output) const
{
    validateSameExecutionTarget(*this, indices, "embeddingForward");
    ensureOutputTarget(output);
    ops::llm::embeddingForward(*this, indices, output);
}

Tensor Tensor::embeddingForward(const Tensor &indices) const
{
    Tensor result(execution_target);
    embeddingForward(indices, result);
    return result;
}

void Tensor::embeddingForward(const std::vector<int32_t> &indices, Tensor &output) const
{
    requireImplementation("embeddingForward");
    ensureOutputTarget(output);
    ops::llm::embeddingForward(*this, buildIndexTensor(*this, indices), output);
}

Tensor Tensor::embeddingForward(const std::vector<int32_t> &indices) const
{
    Tensor result(execution_target);
    embeddingForward(indices, result);
    return result;
}

void Tensor::embeddingBackward(const Tensor &indices, const Tensor &output_gradient, Tensor &weight_gradient, bool accumulate) const
{
    validateSameExecutionTarget(*this, indices, "embeddingBackward");
    validateSameExecutionTarget(*this, output_gradient, "embeddingBackward");
    validateAccumulateTarget(*this, weight_gradient, accumulate, "embeddingBackward");
    ops::llm::embeddingBackward(*this, indices, output_gradient, weight_gradient, accumulate);
}

void Tensor::embeddingBackward(const std::vector<int32_t> &indices, const Tensor &output_gradient, Tensor &weight_gradient, bool accumulate) const
{
    validateSameExecutionTarget(*this, output_gradient, "embeddingBackward");
    validateAccumulateTarget(*this, weight_gradient, accumulate, "embeddingBackward");
    ops::llm::embeddingBackward(*this, buildIndexTensor(*this, indices), output_gradient, weight_gradient, accumulate);
}

void Tensor::singleTokenAttentionForward(const Tensor &k, const Tensor &v, Tensor &output, size_t num_heads, size_t head_dim, size_t total_seq_len, size_t num_kv_heads) const
{
    validateSameExecutionTarget(*this, k, "singleTokenAttentionForward");
    validateSameExecutionTarget(*this, v, "singleTokenAttentionForward");
    ensureOutputTarget(output);
    ops::llm::singleTokenAttentionForward(*this, k, v, output, num_heads, head_dim, total_seq_len, num_kv_heads);
}

Tensor Tensor::singleTokenAttentionForward(const Tensor &k, const Tensor &v, size_t num_heads, size_t head_dim, size_t total_seq_len, size_t num_kv_heads) const
{
    Tensor result(execution_target);
    singleTokenAttentionForward(k, v, result, num_heads, head_dim, total_seq_len, num_kv_heads);
    return result;
}

float Tensor::fusedCrossEntropyLoss(const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens) const
{
    validateSameExecutionTarget(*this, targets, "fusedCrossEntropyLoss");
    ensureOutputTarget(d_logits);
    return ops::loss::fusedCrossEntropy(*this, targets, d_logits, valid_tokens);
}

float Tensor::fusedCrossEntropyLoss(const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens) const
{
    requireImplementation("fusedCrossEntropyLoss");
    ensureOutputTarget(d_logits);
    return ops::loss::fusedCrossEntropy(*this, targets, d_logits, valid_tokens);
}

void Tensor::linearForward(const Tensor &weights, const Tensor &biases, Tensor &output) const
{
    validateSameExecutionTarget(*this, weights, "linearForward");
    validateSameExecutionTarget(*this, biases, "linearForward");
    ensureOutputTarget(output);
    ops::linear::linearForward(*this, weights, biases, output);
}

void Tensor::linearBackwardInput(const Tensor &weights, Tensor &input_gradient) const
{
    validateSameExecutionTarget(*this, weights, "linearBackwardInput");
    ensureOutputTarget(input_gradient);
    ops::linear::linearBackwardInput(*this, weights, input_gradient);
}

void Tensor::linearBackwardWeightBias(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate) const
{
    validateSameExecutionTarget(*this, output_gradient, "linearBackwardWeightBias");
    validateAccumulateTarget(*this, weight_gradient, accumulate, "linearBackwardWeightBias");
    validateAccumulateTarget(*this, bias_gradient, accumulate, "linearBackwardWeightBias");
    ops::linear::linearBackwardWeightBias(*this, output_gradient, weight_gradient, bias_gradient, accumulate);
}

void Tensor::linearBackwardWeightAdam(const Tensor &output_gradient, Tensor &weights, Tensor &first_moment, Tensor &second_moment, Tensor &bias_gradient, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay) const
{
    validateSameExecutionTarget(*this, output_gradient, "linearBackwardWeightAdam");
    validateSameExecutionTarget(*this, weights, "linearBackwardWeightAdam");
    validateOrInitTarget(*this, first_moment, "linearBackwardWeightAdam");
    validateOrInitTarget(*this, second_moment, "linearBackwardWeightAdam");
    ensureOutputTarget(bias_gradient);
    Tensor temp_w_grad(weights.getShape(), weights.getDataType(), weights.getExecutionTarget());
    linearBackwardWeightBias(output_gradient, temp_w_grad, bias_gradient, false);
    ops::optimizer::adamStep(weights, temp_w_grad, first_moment, second_moment, learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
}

void Tensor::batchNorm2dForward(const Tensor &gamma, const Tensor &beta, Tensor &running_mean, Tensor &running_variance, Tensor &batch_mean, Tensor &batch_variance, Tensor &normalized_input, Tensor &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon, float momentum, bool is_training) const
{
    validateSameExecutionTarget(*this, gamma, "batchNorm2dForward");
    validateSameExecutionTarget(*this, beta, "batchNorm2dForward");
    validateOrInitTarget(*this, running_mean, "batchNorm2dForward");
    validateOrInitTarget(*this, running_variance, "batchNorm2dForward");
    ensureOutputTarget(batch_mean);
    ensureOutputTarget(batch_variance);
    ensureOutputTarget(normalized_input);
    ensureOutputTarget(output);
    ops::normalization::batchNorm2dForward(*this, gamma, beta, running_mean, running_variance, batch_mean, batch_variance, normalized_input, output, input_height, input_width, input_channels, epsilon, momentum, is_training);
}

void Tensor::batchNorm2dBackward(const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input, Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const
{
    validateSameExecutionTarget(*this, gamma, "batchNorm2dBackward");
    validateSameExecutionTarget(*this, batch_variance, "batchNorm2dBackward");
    validateSameExecutionTarget(*this, normalized_input, "batchNorm2dBackward");
    ensureOutputTarget(gamma_gradient);
    ensureOutputTarget(beta_gradient);
    ensureOutputTarget(input_gradient);
    ops::normalization::batchNorm2dBackward(*this, gamma, batch_variance, normalized_input, gamma_gradient, beta_gradient, input_gradient, input_height, input_width, input_channels, epsilon);
}

void Tensor::cceLoss(const Tensor &target, Tensor &output, float epsilon) const
{
    validateSameExecutionTarget(*this, target, "cceLoss");
    ensureOutputTarget(output);
    ops::loss::cce(*this, target, output, epsilon);
}

void Tensor::mseLoss(const Tensor &target, Tensor &output) const
{
    validateSameExecutionTarget(*this, target, "mseLoss");
    ensureOutputTarget(output);
    ops::loss::mse(*this, target, output);
}

void Tensor::maeLoss(const Tensor &target, Tensor &output) const
{
    validateSameExecutionTarget(*this, target, "maeLoss");
    ensureOutputTarget(output);
    ops::loss::mae(*this, target, output);
}

void Tensor::bceLoss(const Tensor &target, Tensor &output, float epsilon) const
{
    validateSameExecutionTarget(*this, target, "bceLoss");
    ensureOutputTarget(output);
    ops::loss::bce(*this, target, output, epsilon);
}

void Tensor::huberLoss(const Tensor &target, Tensor &output, float delta) const
{
    validateSameExecutionTarget(*this, target, "huberLoss");
    ensureOutputTarget(output);
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
    prepareBinary(*this, other, output, "concatenateColumns");
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
    prepareBinary(*this, other, output, "concatenateRows");
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
    prepareUnary(*this, result_left, "splitColumns");
    ensureOutputTarget(result_right);
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
    prepareUnary(*this, result_up, "splitRows");
    ensureOutputTarget(result_down);
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

std::string Tensor::toString(size_t max_display_rows, size_t max_display_columns) const
{
    std::ostringstream oss;
    print(oss, max_display_rows, max_display_columns);
    return oss.str();
}

void Tensor::print(std::ostream &os, size_t max_display_rows, size_t max_display_columns) const
{
    const auto data_vector = getData();
    size_t print_rows = std::min(getRows(), max_display_rows);
    size_t print_cols = std::min(getColumns(), max_display_columns);
    os << "Tensor " << getShape().toString() << ":\n";
    for (size_t r = 0; r < print_rows; ++r)
    {
        os << "  [ ";
        for (size_t c = 0; c < print_cols; ++c)
        {
            os << std::format("{:8.4f} ", data_vector[r * getColumns() + c]);
        }
        if (getColumns() > print_cols)
        {
            os << "... ";
        }
        os << "]\n";
    }
    if (getRows() > print_rows)
    {
        os << "  ...\n";
    }
}

void Tensor::print(size_t max_display_rows, size_t max_display_columns) const
{
    const std::string formatted = toString(max_display_rows, max_display_columns);
    std::cout << formatted;
    Logger::logMessage(formatted, Log_Level::LOG_DEBUG, false, 0, Log_Feature::TENSOR_INSPECTION);
}

void Tensor::saveTensor(std::ofstream &output_file_stream) const
{
        requireImplementation("saveTensor");
        if (!output_file_stream.is_open() || !output_file_stream)
        {
            throw std::runtime_error("Tensor::saveTensor: Output stream is not open");
        }
        if (getRank() > MAX_TENSOR_RANK)
        {
            throw std::runtime_error("Tensor::saveTensor: Tensor rank exceeds MAX_TENSOR_RANK");
        }

        const auto &dims = getShape();
        bool dims_fit_u32 = true;
        for (size_t i = 0; i < dims.getRank(); ++i)
        {
            if (dims[i] > std::numeric_limits<uint32_t>::max())
            {
                dims_fit_u32 = false;
                break;
            }
        }

        if (getDataType() == Data_Type::FLOAT32 && dims_fit_u32)
        {
            output_file_stream.write(reinterpret_cast<const char *>(&TENSOR_MAGIC_HEADER), sizeof(TENSOR_MAGIC_HEADER));

            uint32_t rank = static_cast<uint32_t>(getRank());
            output_file_stream.write(reinterpret_cast<const char *>(&rank), sizeof(rank));

            for (size_t i = 0; i < rank; ++i)
            {
                uint32_t dim_val = static_cast<uint32_t>(dims[i]);
                output_file_stream.write(reinterpret_cast<const char *>(&dim_val), sizeof(dim_val));
            }

            if (getTotalElements() > 0)
            {
                std::vector<float> host_data = getData();
                output_file_stream.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
            }
        }
        else
        {
            output_file_stream.write(reinterpret_cast<const char *>(&TENSOR_MAGIC_HEADER_V2), sizeof(TENSOR_MAGIC_HEADER_V2));

            uint32_t dtype_u32 = static_cast<uint32_t>(getDataType());
            output_file_stream.write(reinterpret_cast<const char *>(&dtype_u32), sizeof(dtype_u32));

            uint32_t rank = static_cast<uint32_t>(getRank());
            output_file_stream.write(reinterpret_cast<const char *>(&rank), sizeof(rank));

            for (size_t i = 0; i < rank; ++i)
            {
                uint64_t dim_val = static_cast<uint64_t>(dims[i]);
                output_file_stream.write(reinterpret_cast<const char *>(&dim_val), sizeof(dim_val));
            }

            if (getDataType() == Data_Type::TERNARY_INT2)
            {
                syncGpuIfNeeded(execution_target);
                std::vector<uint32_t> words = getPackedData();
                uint64_t word_count = static_cast<uint64_t>(words.size());
                output_file_stream.write(reinterpret_cast<const char *>(&word_count), sizeof(word_count));
                if (!words.empty())
                {
                    output_file_stream.write(reinterpret_cast<const char *>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
                }
            }
            else if (getDataType() == Data_Type::FLOAT16)
            {
                if (getTotalElements() > 0)
                {
                    std::vector<float> host_data = getData();
                    std::vector<float16_t> fp16_data(host_data.size());
                    convertFp32ToFp16(host_data.data(), fp16_data.data(), host_data.size());
                    output_file_stream.write(reinterpret_cast<const char *>(fp16_data.data()), static_cast<std::streamsize>(fp16_data.size() * sizeof(float16_t)));
                }
            }
            else if (getTotalElements() > 0)
            {
                std::vector<float> host_data = getData();
                output_file_stream.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
            }
        }

        if (!output_file_stream)
        {
            throw std::runtime_error("Tensor::saveTensor: Failed to write to output stream");
        }
    }

Tensor Tensor::loadTensor(std::ifstream &input_file_stream, Execution_Target target)
{
        if (!input_file_stream.is_open() || !input_file_stream)
        {
            throw std::runtime_error("Tensor::loadTensor: Input stream is not open");
        }

        uint32_t first_header_field = 0;
        readExact(input_file_stream, &first_header_field, sizeof(first_header_field), "Tensor::loadTensor: Failed to read header");

        if (first_header_field == TENSOR_MAGIC_HEADER_V2)
        {
            uint32_t dtype_u32 = 0;
            uint32_t rank = 0;
            readExact(input_file_stream, &dtype_u32, sizeof(dtype_u32), "Tensor::loadTensor: Failed to read data type");
            readExact(input_file_stream, &rank, sizeof(rank), "Tensor::loadTensor: Failed to read rank");
            if (rank > MAX_TENSOR_RANK)
            {
                throw std::runtime_error("Tensor::loadTensor: Tensor rank exceeds MAX_TENSOR_RANK");
            }
            if (dtype_u32 != static_cast<uint32_t>(Data_Type::FLOAT32) &&
                dtype_u32 != static_cast<uint32_t>(Data_Type::FLOAT16) &&
                dtype_u32 != static_cast<uint32_t>(Data_Type::TERNARY_INT2))
            {
                throw std::runtime_error("Tensor::loadTensor: Invalid tensor data type");
            }
            Data_Type dtype = static_cast<Data_Type>(dtype_u32);

            std::vector<size_t> dims(rank);
            size_t total = (rank == 0) ? 0 : 1;
            for (size_t i = 0; i < rank; ++i)
            {
                uint64_t dim_val = 0;
                readExact(input_file_stream, &dim_val, sizeof(dim_val), "Tensor::loadTensor: Failed to read dimension");
                if (dim_val > std::numeric_limits<size_t>::max())
                {
                    throw std::overflow_error("Tensor::loadTensor: Dimension exceeds size_t");
                }
                dims[i] = static_cast<size_t>(dim_val);
                total = safeMulSizeT(total, dims[i], "Tensor::loadTensor: Element count overflow");
            }

            Shape loaded_shape(dims);
            if (dtype == Data_Type::TERNARY_INT2)
            {
                uint64_t word_count = 0;
                readExact(input_file_stream, &word_count, sizeof(word_count), "Tensor::loadTensor: Failed to read packed word count");
                size_t expected_bytes = getTensorStorageBytes(Data_Type::TERNARY_INT2, loaded_shape);
                size_t expected_words = expected_bytes / sizeof(uint32_t);
                if (word_count != expected_words)
                {
                    throw std::runtime_error("Tensor::loadTensor: TERNARY_INT2 word count mismatch");
                }
                std::vector<uint32_t> words(expected_words);
                readExact(input_file_stream, words.data(), expected_bytes, "Tensor::loadTensor: Failed to read TERNARY_INT2 payload");
                Tensor result(loaded_shape, Data_Type::TERNARY_INT2, target);
                if (!words.empty() && total > 0)
                {
                    result.uploadPackedData(words);
                }
                return result;
            }
            if (dtype == Data_Type::FLOAT16)
            {
                size_t byte_count = safeMulSizeT(total, sizeof(float16_t), "Tensor::loadTensor: Byte size overflow");
                std::vector<float16_t> fp16_data(total);
                readExact(input_file_stream, fp16_data.data(), byte_count, "Tensor::loadTensor: Failed to read FLOAT16 payload");
                std::vector<float> host_data(total);
                convertFp16ToFp32(fp16_data.data(), host_data.data(), total);
                return Tensor(loaded_shape, host_data, Data_Type::FLOAT16, target);
            }

            size_t byte_count = safeMulSizeT(total, sizeof(float), "Tensor::loadTensor: Byte size overflow");
            std::vector<float> host_data(total);
            readExact(input_file_stream, host_data.data(), byte_count, "Tensor::loadTensor: Failed to read FLOAT32 payload");
            return Tensor(loaded_shape, std::move(host_data), target);
        }

        if (first_header_field == TENSOR_MAGIC_HEADER || first_header_field == 0xFFFFFFFF)
        {
            uint32_t rank = 0;
            readExact(input_file_stream, &rank, sizeof(rank), "Tensor::loadTensor: Failed to read rank");
            if (rank > MAX_TENSOR_RANK)
            {
                throw std::runtime_error("Tensor::loadTensor: Tensor rank exceeds MAX_TENSOR_RANK");
            }

            std::vector<size_t> dims(rank);
            size_t total = (rank == 0) ? 0 : 1;
            for (size_t i = 0; i < rank; ++i)
            {
                uint32_t dim_val = 0;
                readExact(input_file_stream, &dim_val, sizeof(dim_val), "Tensor::loadTensor: Failed to read dimension");
                dims[i] = static_cast<size_t>(dim_val);
                total = safeMulSizeT(total, dims[i], "Tensor::loadTensor: Element count overflow");
            }

            size_t byte_count = safeMulSizeT(total, sizeof(float), "Tensor::loadTensor: Byte size overflow");
            std::vector<float> host_data(total);
            readExact(input_file_stream, host_data.data(), byte_count, "Tensor::loadTensor: Failed to read tensor payload");
            return Tensor(Shape(dims), std::move(host_data), target);
        }

        size_t rows_count = static_cast<size_t>(first_header_field);
        uint32_t columns_count = 0;
        readExact(input_file_stream, &columns_count, sizeof(columns_count), "Tensor::loadTensor: Failed to read column count");
        size_t cols_count = static_cast<size_t>(columns_count);
        size_t total = safeMulSizeT(rows_count, cols_count, "Tensor::loadTensor: Matrix element count overflow");
        size_t byte_count = safeMulSizeT(total, sizeof(float), "Tensor::loadTensor: Matrix byte size overflow");
        std::vector<float> host_data(total);
        readExact(input_file_stream, host_data.data(), byte_count, "Tensor::loadTensor: Failed to read matrix payload");
        return Tensor(rows_count, cols_count, std::move(host_data), target);
    }

Tensor Tensor::clone() const
{
    requireImplementation("clone");
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        syncGpuIfNeeded(execution_target);
        std::vector<uint32_t> words = getPackedData();
        Tensor cloned(getShape(), Data_Type::TERNARY_INT2, execution_target);
        if (!words.empty() && getTotalElements() > 0)
        {
            cloned.uploadPackedData(words);
        }
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
        return cloned;
    }
    return Tensor(getShape(), getData(), getDataType(), execution_target);
}

void Tensor::fill(float value)
{
        requireImplementation("fill");
        implementation->fill(value);
    }

void Tensor::zero()
{
        requireImplementation("zero");
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
{
    requireImplementation("getData");
    return implementation->getData();
}

void Tensor::uploadData(const std::vector<float> &host_data)
{
    requireImplementation("uploadData");
    implementation->uploadData(host_data);
}

void Tensor::setImplementation(std::shared_ptr<Tensor_Impl> _impl)
{
    validateImplMatchesTarget(_impl, execution_target, "setImplementation");
    implementation = std::move(_impl);
}

void Tensor::setDataType(Data_Type _type)
{
    if (!implementation)
    {
        Tensor fresh(Shape{}, _type, execution_target);
        implementation = std::move(fresh.implementation);
        return;
    }
    if (implementation->getDataType() == _type)
    {
        return;
    }
    if (implementation->isEmpty() && implementation->getTotalElements() == 0)
    {
        if (implementation.use_count() > 1)
        {
            Tensor fresh(implementation->getShape(), _type, execution_target);
            implementation = std::move(fresh.implementation);
        }
        else
        {
            implementation->setDataType(_type);
        }
        return;
    }
    Tensor converted = to(_type);
    implementation = std::move(converted.implementation);
}

void Tensor::logFp16Stats(std::string_view tensor_name, Log_Level level) const
{
#if !VULKANML_LOGGING_ENABLED
    (void)tensor_name;
    (void)level;
    return;
#else
    if (!implementation || getTotalElements() == 0)
    {
        return;
    }
    if (!Logger::shouldLog(level, Log_Feature::FP16_METRICS))
    {
        return;
    }
    std::vector<float> host_data = getData();
    Logger::logFp16TensorStats(tensor_name, host_data, getTotalElements(), getDataTypeName(getDataType()), level);
#endif
}

void Tensor::logTernaryStats(std::string_view tensor_name, float gamma, Log_Level level) const
{
#if !VULKANML_LOGGING_ENABLED
    (void)tensor_name;
    (void)gamma;
    (void)level;
    return;
#else
    if (!implementation || getTotalElements() == 0)
    {
        return;
    }
    if (!Logger::shouldLog(level, Log_Feature::TERNARY_METRICS))
    {
        return;
    }
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        const auto &packed = getPackedData();
        size_t rank = getRank();
        size_t cols = (rank == 0) ? 1 : getShape()[rank - 1];
        size_t rows = (rank <= 1 || cols == 0) ? 1 : (getTotalElements() / cols);
        std::vector<float> unpacked(getTotalElements(), 0.0f);
        if (!packed.empty())
        {
            Cpu_Tensor_Impl::unpackTernary(packed.data(), unpacked.data(), rows, cols, gamma);
        }
        Logger::logTernaryTensorStats(tensor_name, unpacked, getTotalElements(), gamma, level);
        return;
    }
    std::vector<float> host_data = getData();
    Logger::logTernaryTensorStats(tensor_name, host_data, getTotalElements(), gamma, level);
#endif
}

const Shape &Tensor::getShape() const noexcept
{
    if (!implementation)
    {
        static const Shape empty_shape;
        return empty_shape;
    }
    return implementation->getShape();
}

const Stride &Tensor::getStrides() const noexcept
{
    if (!implementation)
    {
        static const Stride empty_strides;
        return empty_strides;
    }
    return implementation->getStrides();
}

const std::vector<uint32_t> &Tensor::getPackedData() const
{
    requireImplementation("getPackedData");
    return implementation->getPackedData();
}

void Tensor::uploadPackedData(const std::vector<uint32_t> &packed_data)
{
    requireImplementation("uploadPackedData");
    implementation->uploadPackedData(packed_data);
}

size_t Tensor::getTotalElements() const noexcept
{
    return implementation ? implementation->getTotalElements() : 0;
}

size_t Tensor::getColumns() const
{
    return implementation ? implementation->getColumns() : 0;
}

size_t Tensor::getRank() const noexcept
{
    return implementation ? implementation->getRank() : 0;
}

size_t Tensor::getRows() const noexcept
{
    return implementation ? implementation->getRows() : 0;
}

Data_Type Tensor::getDataType() const noexcept
{
    return implementation ? implementation->getDataType() : Data_Type::FLOAT32;
}

bool Tensor::isEmpty() const noexcept
{
    return !implementation || implementation->isEmpty();
}

void Tensor::setExecutionTarget(Execution_Target new_target)
{
    requireImplementation("setExecutionTarget");
    if (execution_target == new_target)
    {
        return;
    }
    Shape current_shape = getShape();
    if (getDataType() == Data_Type::TERNARY_INT2)
    {
        syncGpuIfNeeded(execution_target);
        std::vector<uint32_t> words = getPackedData();
        Tensor moved(current_shape, Data_Type::TERNARY_INT2, new_target);
        if (!words.empty() && current_shape.getTotalElements() > 0)
        {
            moved.uploadPackedData(words);
        }
        implementation = std::move(moved.implementation);
        execution_target = new_target;
    }
    else
    {
        std::vector<float> current_data = getData();
        Tensor moved(current_shape, current_data, getDataType(), new_target);
        implementation = std::move(moved.implementation);
        execution_target = new_target;
    }
    if (new_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
}
