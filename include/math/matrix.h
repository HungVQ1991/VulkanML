#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <utility>
#include <iostream>
#include <iomanip>
#include <string>

#include "impl.h"
#include "cpu_matrix_impl.h"
#include "gpu_matrix_impl.h"
#include "logger.h"

enum class Execution_Target
{
    CPU,
    VULKAN_GPU
};

class Matrix
{
private:
    std::shared_ptr<Impl> pimpl;
    Execution_Target current_target;

public:
    explicit Matrix(Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(0, 0);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(0, 0);
        }
    }

    Matrix(std::size_t rows, std::size_t cols, Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        }
    }

    Matrix(std::size_t rows, std::size_t cols, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols, host_data);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols, host_data);
        }
    }

    Matrix(std::size_t rows, std::size_t cols, std::vector<float> &&host_data, Execution_Target target = Execution_Target::CPU)
        : current_target(target)
    {
        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(host_data));
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols, std::move(host_data));
        }
    }

    explicit Matrix(std::shared_ptr<Impl> implementation, Execution_Target target = Execution_Target::CPU)
        : pimpl(std::move(implementation)), current_target(target) {}

    ~Matrix() = default;
    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;
    Matrix(Matrix &&) noexcept = default;
    Matrix &operator=(Matrix &&) noexcept = default;

    void initShape(std::size_t rows, std::size_t cols)
    {
        if (pimpl->getRows() == rows && pimpl->getCols() == cols)
        {
            return;
        }

        if (current_target == Execution_Target::CPU)
        {
            pimpl = std::make_shared<Cpu_Matrix_Impl>(rows, cols);
        }
        else
        {
            pimpl = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        }
    }

    void print(const std::string &name = "") const
    {
        if (!name.empty())
            std::cout << name << " (" << getRows() << "x" << getCols() << ")\n";

        std::vector<float> data = getData();
        for (std::size_t i = 0; i < getRows(); ++i)
        {
            for (std::size_t j = 0; j < getCols(); ++j)
                std::cout << std::setprecision(6) << std::defaultfloat << data[i * getCols() + j] << ' ';
            std::cout << '\n';
        }
        std::cout << std::endl;
    }

    std::size_t getRows() const { return pimpl->getRows(); }
    std::size_t getCols() const { return pimpl->getCols(); }
    std::vector<float> getData() const { return pimpl->getData(); }
    Matrix operator*(const Matrix &other) const { return Matrix(pimpl->matmul(other.pimpl), current_target); }
    Matrix operator/(const Matrix &other) const { return Matrix(pimpl->matdiv(other.pimpl), current_target); }
    Matrix operator*(float scalar) const { return Matrix(pimpl->mulScalar(scalar), current_target); }
    Matrix operator/(float scalar) const { return Matrix(pimpl->divScalar(scalar), current_target); }
    Matrix operator+(const Matrix &other) const { return Matrix(pimpl->add(other.pimpl), current_target); }
    Matrix operator-(const Matrix &other) const { return Matrix(pimpl->sub(other.pimpl), current_target); }
    Matrix hadamardMul(const Matrix &other) const { return Matrix(pimpl->hadamardMul(other.pimpl), current_target); }
    Matrix hadamardDiv(const Matrix &other) const { return Matrix(pimpl->hadamardDiv(other.pimpl), current_target); }
    Matrix transpose() const { return Matrix(pimpl->transpose(), current_target); }
    Matrix inverse() const { return Matrix(pimpl->inverse(), current_target); }
    Matrix normalize() const { return Matrix(pimpl->normalize(), current_target); }
    Matrix relu() const { return Matrix(pimpl->relu(), current_target); }
    Matrix reluBackward(const Matrix &gradient) const { return Matrix(pimpl->reluBackward(gradient.pimpl), current_target); }
    Matrix gelu() const { return Matrix(pimpl->gelu(), current_target); }
    Matrix geluBackward(const Matrix &gradient) const { return Matrix(pimpl->geluBackward(gradient.pimpl), current_target); }
    Matrix softmax() const { return Matrix(pimpl->softmax(), current_target); }
    Matrix softmaxBackward(const Matrix &gradient) const { return Matrix(pimpl->softmaxBackward(gradient.pimpl), current_target); }
    void sgdUpdate(const Matrix &gradient, float learning_rate, float max_gradient = 0.0f) { pimpl->sgdUpdate(gradient.pimpl, learning_rate, max_gradient); }
    Matrix matmulAdd(const Matrix &other, const Matrix &biases) { return Matrix(pimpl->matmulAdd(other.pimpl, biases.pimpl), current_target); }
    Matrix matmulTransA(const Matrix &other) const { return Matrix(pimpl->matmulTransA(other.pimpl), current_target); }
    Matrix matmulTransB(const Matrix &other) const { return Matrix(pimpl->matmulTransB(other.pimpl), current_target); }

    Matrix conv2d(const Matrix &weights, const Matrix &bias,
                  std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
                  std::uint32_t out_c, std::uint32_t kernel_size,
                  std::uint32_t stride, std::uint32_t padding) const
    { return Matrix(pimpl->conv2d(weights.pimpl, bias.pimpl, in_h, in_w, in_c, out_c, kernel_size, stride, padding), current_target); }

    Matrix conv2dBackwardInput(const Matrix &weights,
                               std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
                               std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
                               std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    { return Matrix(pimpl->conv2dBackwardInput(weights.pimpl, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding), current_target); }

    void conv2dBackwardWeight(const Matrix &grad_output, Matrix &grad_weights, Matrix &grad_biases,
                              std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
                              std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
                              std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    { pimpl->conv2dBackwardWeight(grad_output.pimpl, grad_weights.pimpl, grad_biases.pimpl, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding); }

    std::pair<Matrix, Matrix> maxpool2d(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
                                        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        auto [out_impl, mask_impl] = pimpl->maxpool2d(in_h, in_w, channels, kernel_size, stride, padding);
        return {Matrix(out_impl, current_target), Matrix(mask_impl, current_target)};
    }

    Matrix maxpool2dBackward(const Matrix &mask,
                             std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
                             std::uint32_t out_h, std::uint32_t out_w,
                             std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    { return Matrix(pimpl->maxpool2dBackward(mask.pimpl, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding), current_target); }

    Matrix globalAvgPool2d(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const
    { return Matrix(pimpl->globalAvgPool2d(in_h, in_w, channels), current_target); }

    Matrix globalAvgPool2dBackward(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const
    { return Matrix(pimpl->globalAvgPool2dBackward(in_h, in_w, channels), current_target); }

    Execution_Target getTarget() const { return current_target; }
    void uploadData(const std::vector<float> &host_data) { pimpl->uploadData(host_data); }
};