#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"

#ifndef ENABLE_MATRIX_DEBUG_LOGS
#define ENABLE_MATRIX_DEBUG_LOGS 0
#endif

#if ENABLE_MATRIX_DEBUG_LOGS
#define MATRIX_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define MATRIX_LOG_DEBUG(msg) ((void)0)
#endif

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
            Logger::logMessage(err_msg, LOG_ERROR, true);
            throw std::invalid_argument(err_msg);
        }
    }

    void validateMatmulDimensions(const Impl &other_impl) const
    {
        if (getCols() != other_impl.getRows())
        {
            std::string err_msg = "Impl::validateMatmulDimensions: Matrix matmul dimension mismatch: cols_a (" + std::to_string(getCols()) +
                                  ") != rows_b (" + std::to_string(other_impl.getRows()) + ")";
            Logger::logMessage(err_msg, LOG_ERROR, true);
            throw std::invalid_argument(err_msg);
        }
    }

    void validateSquare() const
    {
        if (getRows() != getCols())
        {
            std::string err_msg = "Impl::validateSquare: Matrix is not square: (" + std::to_string(getRows()) + "x" +
                                  std::to_string(getCols()) + ")";
            Logger::logMessage(err_msg, LOG_ERROR, true);
            throw std::invalid_argument(err_msg);
        }
    }

    virtual std::shared_ptr<GVector> getGVector() { return {}; }

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

    virtual void maxpool2d(
        Impl &out_result,
        Impl &out_mask,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void maxpool2dBackward(
        const Impl &mask,
        Impl &grad_input,
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
        const std::shared_ptr<Impl> &batch_var,
        const std::shared_ptr<Impl> &x_hat,
        std::shared_ptr<Impl> &grad_gamma,
        std::shared_ptr<Impl> &grad_beta,
        float epsilon) const = 0;

    virtual void linearForward(
        const Impl &weights_w,
        const Impl &biases_b,
        Impl &output_y) const = 0;

    virtual void linearBackwardInput(
        const Impl &weights_w,
        Impl &grad_x) const = 0;

    virtual void linearBackwardWeightBias(
        const Impl &grad_y,
        Impl &grad_w,
        Impl &grad_b) const = 0;

    virtual void batchNorm2dForward(
        const Impl &gamma,
        const Impl &beta,
        Impl &running_mean,
        Impl &running_var,
        Impl &batch_mean,
        Impl &batch_var,
        Impl &x_hat,
        Impl &output_y,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        float epsilon, float momentum, bool is_training) const = 0;

    virtual void batchNorm2dBackward(
        const Impl &gamma,
        const Impl &batch_var,
        const Impl &x_hat,
        Impl &grad_gamma,
        Impl &grad_beta,
        Impl &grad_input,
        std::uint32_t in_h,
        std::uint32_t in_w,
        std::uint32_t in_c,
        float epsilon) const = 0;

    virtual std::shared_ptr<Impl> cceLoss(
        const std::shared_ptr<Impl> &target_impl,
        float epsilon_val) const = 0;
    virtual std::shared_ptr<Impl> mseLoss(
        const std::shared_ptr<Impl> &target_impl) const = 0;

    virtual std::shared_ptr<Impl> maeLoss(
        const std::shared_ptr<Impl> &target_impl) const = 0;

    virtual std::shared_ptr<Impl> bceLoss(
        const std::shared_ptr<Impl> &target_impl,
        float epsilon_val) const = 0;
};