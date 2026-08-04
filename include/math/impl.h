#pragma once

#include "helper/logger.h"

#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <string>

class Impl
{
public:
    virtual ~Impl() noexcept = default;

    virtual std::size_t getRows() const = 0;
    virtual std::size_t getCols() const = 0;
    virtual const std::vector<float> &getData() const = 0;

    void validateSameDimensions(const Impl &other_impl) const
    {
        if (getRows() != other_impl.getRows() || getCols() != other_impl.getCols())
        {
            std::string err_msg = "Impl::validateSameDimensions: Matrix dimension mismatch: (" + std::to_string(getRows()) + "x" +
                                  std::to_string(getCols()) + ") vs (" + std::to_string(other_impl.getRows()) + "x" +
                                  std::to_string(other_impl.getCols()) + ")";
            Logger::logMessage(err_msg, LOG_ERROR);
            throw std::invalid_argument(err_msg);
        }
    }

    void validateMatmulDimensions(const Impl &other_impl) const
    {
        if (getCols() != other_impl.getRows())
        {
            std::string err_msg = "Impl::validateMatmulDimensions: Matrix matmul dimension mismatch: cols_a (" + std::to_string(getCols()) +
                                  ") != rows_b (" + std::to_string(other_impl.getRows()) + ")";
            Logger::logMessage(err_msg, LOG_ERROR);
            throw std::invalid_argument(err_msg);
        }
    }

    void validateSquare() const
    {
        if (getRows() != getCols())
        {
            std::string err_msg = "Impl::validateSquare: Matrix is not square: (" + std::to_string(getRows()) + "x" +
                                  std::to_string(getCols()) + ")";
            Logger::logMessage(err_msg, LOG_ERROR);
            throw std::invalid_argument(err_msg);
        }
    }

    virtual std::shared_ptr<Impl> matmul(const std::shared_ptr<Impl> &other_impl) const = 0;
    virtual std::shared_ptr<Impl> matdiv(const std::shared_ptr<Impl> &other_impl) const = 0;
    virtual std::shared_ptr<Impl> add(const std::shared_ptr<Impl> &other_impl) const = 0;
    virtual std::shared_ptr<Impl> sub(const std::shared_ptr<Impl> &other_impl) const = 0;
    virtual std::shared_ptr<Impl> mulScalar(float scalar) const = 0;
    virtual std::shared_ptr<Impl> divScalar(float scalar) const = 0;
    virtual std::shared_ptr<Impl> hadamardMul(const std::shared_ptr<Impl> &other_impl) const = 0;
    virtual std::shared_ptr<Impl> hadamardDiv(const std::shared_ptr<Impl> &other_impl) const = 0;
    virtual std::shared_ptr<Impl> transpose() const = 0;
    virtual std::shared_ptr<Impl> inverse() const = 0;
    virtual std::shared_ptr<Impl> normalize() const = 0;
    virtual std::shared_ptr<Impl> relu() const = 0;
    virtual std::shared_ptr<Impl> reluBackward(const std::shared_ptr<Impl> &gradient) const = 0;
    virtual std::shared_ptr<Impl> gelu() const = 0;
    virtual std::shared_ptr<Impl> geluBackward(const std::shared_ptr<Impl> &gradient) const = 0;
    virtual std::shared_ptr<Impl> softmax() const = 0;
    virtual std::shared_ptr<Impl> softmaxBackward(const std::shared_ptr<Impl> &gradient) const = 0;
    virtual void sgdUpdate(const std::shared_ptr<Impl> &grad_impl, float learning_rate, float max_gradient = 0.0f) = 0;
    virtual void adamUpdate(
        const std::shared_ptr<Impl> &grad_impl,
        const std::shared_ptr<Impl> &m_impl,
        const std::shared_ptr<Impl> &v_impl,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        std::size_t timestep,
        float max_gradient = 1.0f) = 0;
    virtual std::shared_ptr<Impl> matmulAdd(const std::shared_ptr<Impl> &other, const std::shared_ptr<Impl> &bias) const = 0;
    virtual std::shared_ptr<Impl> matmulTransA(const std::shared_ptr<Impl> &other) const = 0;
    virtual std::shared_ptr<Impl> matmulTransB(const std::shared_ptr<Impl> &other) const = 0;
    virtual void uploadData(const std::vector<float> &host_data) = 0;
    virtual std::shared_ptr<Impl> conv2d(
        const std::shared_ptr<Impl> &weights,
        const std::shared_ptr<Impl> &bias,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_c, std::uint32_t kernel_size,
        std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual std::shared_ptr<Impl> conv2dBackwardInput(
        const std::shared_ptr<Impl> &weights,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void conv2dBackwardWeight(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &grad_weights,
        const std::shared_ptr<Impl> &grad_biases,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual std::pair<std::shared_ptr<Impl>, std::shared_ptr<Impl>> maxpool2d(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual std::shared_ptr<Impl> maxpool2dBackward(
        const std::shared_ptr<Impl> &mask,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t out_h, std::uint32_t out_w,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual std::shared_ptr<Impl> globalAvgPool2d(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const = 0;
    virtual std::shared_ptr<Impl> globalAvgPool2dBackward(std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const = 0;
    virtual std::shared_ptr<Impl> batchNormForward(
        const std::shared_ptr<Impl> &gamma,
        const std::shared_ptr<Impl> &beta,
        std::shared_ptr<Impl> &running_mean,
        std::shared_ptr<Impl> &running_var,
        std::shared_ptr<Impl> &batch_mean,
        std::shared_ptr<Impl> &batch_var,
        std::shared_ptr<Impl> &x_hat,
        float epsilon,
        float momentum,
        bool is_training) const = 0;

    virtual std::shared_ptr<Impl> batchNormBackward(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &gamma,
        const std::shared_ptr<Impl> &batch_mean,
        const std::shared_ptr<Impl> &batch_var,
        std::shared_ptr<Impl> &grad_gamma,
        std::shared_ptr<Impl> &grad_beta,
        float epsilon) const = 0;
};