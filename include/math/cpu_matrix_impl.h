#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "impl.h"
#include "helper/logger.h"

class Cpu_Matrix_Impl : public Impl
{
private:
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;

public:
    Cpu_Matrix_Impl(std::size_t r, std::size_t c)
        : rows(r), cols(c), data(r * c, 0.0f) {}

    Cpu_Matrix_Impl(std::size_t r, std::size_t c, const std::vector<float> &host_data)
        : rows(r), cols(c), data(host_data)
    {
        if (data.size() != rows * cols)
        {
            Logger::logMessage("Cpu_Matrix_Impl::Cpu_Matrix_Impl: Host data size mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Host data size mismatch");
        }
    }

    Cpu_Matrix_Impl(std::size_t r, std::size_t c, std::vector<float> &&host_data)
        : rows(r), cols(c), data(std::move(host_data))
    {
        if (data.size() != rows * cols)
        {
            Logger::logMessage("Cpu_Matrix_Impl::Cpu_Matrix_Impl (move): Host data size mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Host data size mismatch");
        }
    }

    ~Cpu_Matrix_Impl() override = default;

    std::size_t getRows() const override
    {
        return rows;
    }

    std::size_t getCols() const override
    {
        return cols;
    }

    const std::vector<float> &getData() const override { return data; }

    std::shared_ptr<Impl> matmul(const std::shared_ptr<Impl> &other_impl) const override
    {
        validateMatmulDimensions(*other_impl);

        std::size_t other_cols = other_impl->getCols();
        std::vector<float> result_data(rows * other_cols, 0.0f);
        std::vector<float> other_data = other_impl->getData();

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t k = 0; k < cols; ++k)
            {
                float a_val = data[i * cols + k];
                for (std::size_t j = 0; j < other_cols; ++j)
                {
                    result_data[i * other_cols + j] += a_val * other_data[k * other_cols + j];
                }
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, other_cols, std::move(result_data));
    }

    std::shared_ptr<Impl> matdiv(const std::shared_ptr<Impl> &other_impl) const override
    {
        return matmul(other_impl->inverse());
    }

    std::shared_ptr<Impl> add(const std::shared_ptr<Impl> &other_impl) const override
    {
        std::size_t other_rows = other_impl->getRows();
        std::size_t other_cols = other_impl->getCols();
        std::vector<float> other_data = other_impl->getData();
        std::vector<float> result_data(rows * cols);

        if (rows == other_rows && cols == other_cols)
        {
            for (std::size_t i = 0; i < data.size(); ++i)
            {
                result_data[i] = data[i] + other_data[i];
            }
        }
        else if (other_rows == 1 && cols == other_cols)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < cols; ++j)
                {
                    result_data[i * cols + j] = data[i * cols + j] + other_data[j];
                }
            }
        }
        else
        {
            validateSameDimensions(*other_impl);
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> sub(const std::shared_ptr<Impl> &other_impl) const override
    {
        std::size_t other_rows = other_impl->getRows();
        std::size_t other_cols = other_impl->getCols();
        std::vector<float> other_data = other_impl->getData();
        std::vector<float> result_data(rows * cols);

        if (rows == other_rows && cols == other_cols)
        {
            for (std::size_t i = 0; i < data.size(); ++i)
            {
                result_data[i] = data[i] - other_data[i];
            }
        }
        else if (other_rows == 1 && cols == other_cols)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < cols; ++j)
                {
                    result_data[i * cols + j] = data[i * cols + j] - other_data[j];
                }
            }
        }
        else
        {
            validateSameDimensions(*other_impl);
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> mulScalar(float scalar) const override
    {
        std::vector<float> result_data(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result_data[i] = data[i] * scalar;
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> divScalar(float scalar) const override
    {
        if (std::abs(scalar) < 1e-8f)
        {
            Logger::logMessage("Cpu_Matrix_Impl::divScalar: Division by zero", LOG_ERROR, true);
            throw std::runtime_error("Division by zero in divScalar");
        }

        std::vector<float> result_data(data.size());
        float inv_scalar = 1.0f / scalar;
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result_data[i] = data[i] * inv_scalar;
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> hadamardMul(const std::shared_ptr<Impl> &other_impl) const override
    {
        validateSameDimensions(*other_impl);

        std::vector<float> other_data = other_impl->getData();
        std::vector<float> result_data(data.size());

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result_data[i] = data[i] * other_data[i];
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> hadamardDiv(const std::shared_ptr<Impl> &other_impl) const override
    {
        validateSameDimensions(*other_impl);

        std::vector<float> other_data = other_impl->getData();
        std::vector<float> result_data(data.size());

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            if (std::abs(other_data[i]) < 1e-8f)
            {
                Logger::logMessage("Cpu_Matrix_Impl::hadamardDiv: Division by zero", LOG_ERROR, true);
                throw std::runtime_error("Division by zero in hadamardDiv");
            }
            result_data[i] = data[i] / other_data[i];
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> transpose() const override
    {
        std::vector<float> result_data(rows * cols);
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                result_data[j * rows + i] = data[i * cols + j];
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(cols, rows, std::move(result_data));
    }

    std::shared_ptr<Impl> inverse() const override
    {
        validateSquare();

        std::size_t n = rows;
        std::vector<float> aug(n * 2 * n, 0.0f);

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                aug[i * (2 * n) + j] = data[i * n + j];
            }
            aug[i * (2 * n) + (n + i)] = 1.0f;
        }

        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot_row = i;
            float max_val = std::abs(aug[i * (2 * n) + i]);

            for (std::size_t k = i + 1; k < n; ++k)
            {
                float val = std::abs(aug[k * (2 * n) + i]);
                if (val > max_val)
                {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if (max_val < 1e-7f)
            {
                Logger::logMessage("Cpu_Matrix_Impl::inverse: Matrix is singular and cannot be inverted", LOG_ERROR, true);
                throw std::runtime_error("Matrix is singular");
            }

            if (pivot_row != i)
            {
                for (std::size_t j = 0; j < 2 * n; ++j)
                {
                    std::swap(aug[i * (2 * n) + j], aug[pivot_row * (2 * n) + j]);
                }
            }

            float pivot = aug[i * (2 * n) + i];
            for (std::size_t j = 0; j < 2 * n; ++j)
            {
                aug[i * (2 * n) + j] /= pivot;
            }

            for (std::size_t k = 0; k < n; ++k)
            {
                if (k != i)
                {
                    float factor = aug[k * (2 * n) + i];
                    for (std::size_t j = 0; j < 2 * n; ++j)
                    {
                        aug[k * (2 * n) + j] -= factor * aug[i * (2 * n) + j];
                    }
                }
            }
        }

        std::vector<float> inv_data(n * n);
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                inv_data[i * n + j] = aug[i * (2 * n) + (n + j)];
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(n, n, std::move(inv_data));
    }

    std::shared_ptr<Impl> normalize() const override
    {
        float sum_sq = 0.0f;
        for (float val : data)
        {
            sum_sq += val * val;
        }

        float norm = std::sqrt(sum_sq);
        if (norm < 1e-8f)
        {
            Logger::logMessage("Cpu_Matrix_Impl::normalize: Matrix norm near zero during normalization", LOG_WARNING, true);
            return std::make_shared<Cpu_Matrix_Impl>(rows, cols, data);
        }

        std::vector<float> result_data(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result_data[i] = data[i] / norm;
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> relu() const override
    {
        std::vector<float> result_data(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result_data[i] = std::max(0.0f, data[i]);
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> reluBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        validateSameDimensions(*grad_impl);
        auto grad = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_impl);
        std::vector<float> result_data(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result_data[i] = (data[i] > 0.0f) ? grad->data[i] : 0.0f;
        }
        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> gelu() const override
    {
        constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
        constexpr float kBeta = 0.044715f;

        std::vector<float> result_data(data.size());

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            float x = data[i];
            float x3 = x * x * x;
            float t = std::tanh(kAlpha * (x + kBeta * x3));

            result_data[i] = 0.5f * x * (1.0f + t);
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> geluBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        validateSameDimensions(*grad_impl);

        auto grad = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_impl);

        constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
        constexpr float kBeta = 0.044715f;

        std::vector<float> result_data(data.size());

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            float x = data[i];

            float x2 = x * x;
            float x3 = x2 * x;

            float u = kAlpha * (x + kBeta * x3);
            float t = std::tanh(u);
            float sech2 = 1.0f - t * t;

            float derivative =
                0.5f * (1.0f + t) +
                0.5f * x * sech2 * kAlpha * (1.0f + 3.0f * kBeta * x2);

            result_data[i] = grad->getData()[i] * derivative;
        }

        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, std::move(result_data));
    }

    std::shared_ptr<Impl> softmax() const override
    {
        std::vector<float> out(rows * cols);
        for (std::size_t r = 0; r < rows; ++r)
        {
            float max_val = data[r * cols];

            for (std::size_t c = 1; c < cols; ++c)
                max_val = std::max(max_val, data[r * cols + c]);
            float sum = 0.0f;
            for (std::size_t c = 0; c < cols; ++c)
            {
                float e = std::exp(data[r * cols + c] - max_val);
                out[r * cols + c] = e;
                sum += e;
            }
            for (std::size_t c = 0; c < cols; ++c)
                out[r * cols + c] /= sum;
        }
        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, out);
    }

    std::shared_ptr<Impl> softmaxBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        auto grad = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_impl);
        std::vector<float> result(rows * cols);
        for (std::size_t r = 0; r < rows; ++r)
        {
            for (std::size_t i = 0; i < cols; ++i)
            {
                float value = 0.0f;
                for (std::size_t j = 0; j < cols; ++j)
                {
                    float delta = (i == j) ? 1.0f : 0.0f;
                    float jac = data[r * cols + i] * (delta - data[r * cols + j]);
                    value += jac * grad->data[r * cols + j];
                }
                result[r * cols + i] = value;
            }
        }
        return std::make_shared<Cpu_Matrix_Impl>(rows, cols, result);
    }

    void sgdUpdate(const std::shared_ptr<Impl> &grad_impl, float learning_rate, float max_gradient = 0.0f) override
    {
        validateSameDimensions(*grad_impl);
        const std::vector<float> &grad_data = grad_impl->getData();
        std::size_t total_elements = data.size();

        if (max_gradient > 0.0f)
        {
            for (std::size_t i = 0; i < total_elements; ++i)
            {
                float g = std::clamp(grad_data[i], -max_gradient, max_gradient);
                data[i] -= learning_rate * g;
            }
        }
        else
        {
            for (std::size_t i = 0; i < total_elements; ++i)
            {
                data[i] -= learning_rate * grad_data[i];
            }
        }
    }

    void adamUpdate(
        const std::shared_ptr<Impl> &grad_impl,
        const std::shared_ptr<Impl> &m_impl,
        const std::shared_ptr<Impl> &v_impl,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        std::size_t timestep,
        float max_gradient = 1.0f) override
    {
        validateSameDimensions(*grad_impl);
        validateSameDimensions(*m_impl);
        validateSameDimensions(*v_impl);

        auto grad_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_impl);
        auto m_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(m_impl);
        auto v_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(v_impl);

        const std::vector<float> &grad_data = grad_cpu->getData();
        std::vector<float> &m_data = m_cpu->data;
        std::vector<float> &v_data = v_cpu->data;

        std::size_t total_elements = data.size();

        float correction1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
        float correction2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));

        for (std::size_t i = 0; i < total_elements; ++i)
        {
            float g = (max_gradient > 0.0f) ? std::clamp(grad_data[i], -max_gradient, max_gradient) : grad_data[i];

            m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * g;
            v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * (g * g);

            float m_hat = m_data[i] / correction1;
            float v_hat = v_data[i] / correction2;

            data[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
        }
    }

    std::shared_ptr<Impl> matmulAdd(const std::shared_ptr<Impl> &other, const std::shared_ptr<Impl> &bias) const override
    {
        validateMatmulDimensions(*other);

        const auto &w_cpu = static_cast<const Cpu_Matrix_Impl &>(*other);
        const auto &b_cpu = static_cast<const Cpu_Matrix_Impl &>(*bias);

        std::size_t out_rows = rows;
        std::size_t out_cols = w_cpu.cols;

        std::vector<float> result_data(out_rows * out_cols);

        for (std::size_t i = 0; i < out_rows; ++i)
        {
            for (std::size_t j = 0; j < out_cols; ++j)
            {
                float bias_val = (b_cpu.rows == 1) ? b_cpu.data[j] : b_cpu.data[i * out_cols + j];
                result_data[i * out_cols + j] = bias_val;
            }

            for (std::size_t k = 0; k < cols; ++k)
            {
                float x_val = data[i * cols + k];
                for (std::size_t j = 0; j < out_cols; ++j)
                {
                    result_data[i * out_cols + j] += x_val * w_cpu.data[k * out_cols + j];
                }
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(out_rows, out_cols, std::move(result_data));
    }

    std::shared_ptr<Impl> conv2d(
        const std::shared_ptr<Impl> &weights,
        const std::shared_ptr<Impl> &bias,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_c, std::uint32_t kernel_size,
        std::uint32_t stride, std::uint32_t padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        const auto &w_data = weights->getData();
        const auto &b_data = bias->getData();

        std::vector<float> result_data(batch_size * out_h * out_w * out_c, 0.0f);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (std::uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (std::uint32_t oc = 0; oc < out_c; ++oc)
                    {
                        float sum = b_data[oc];
                        for (std::uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w))
                                {
                                    for (std::uint32_t ic = 0; ic < in_c; ++ic)
                                    {
                                        std::size_t in_idx = n * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                                        std::size_t w_idx = ky * kernel_size * in_c * out_c + kx * in_c * out_c + ic * out_c + oc;
                                        sum += data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        std::size_t out_idx = n * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
                        result_data[out_idx] = sum;
                    }
                }
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(batch_size, out_h * out_w * out_c, std::move(result_data));
    }

    std::shared_ptr<Impl> conv2dBackwardInput(
        const std::shared_ptr<Impl> &weights,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        const auto &w_data = weights->getData();

        std::vector<float> result_data(batch_size * in_h * in_w * in_c, 0.0f);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t ih = 0; ih < in_h; ++ih)
            {
                for (std::uint32_t iw = 0; iw < in_w; ++iw)
                {
                    for (std::uint32_t ic = 0; ic < in_c; ++ic)
                    {
                        float sum = 0.0f;
                        for (std::uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + padding - ky);
                                int ow_calc = static_cast<int>(iw + padding - kx);

                                if (oh_calc >= 0 && oh_calc % static_cast<int>(stride) == 0 &&
                                    ow_calc >= 0 && ow_calc % static_cast<int>(stride) == 0)
                                {
                                    std::uint32_t oh = static_cast<std::uint32_t>(oh_calc) / stride;
                                    std::uint32_t ow = static_cast<std::uint32_t>(ow_calc) / stride;

                                    if (oh < out_h && ow < out_w)
                                    {
                                        for (std::uint32_t oc = 0; oc < out_c; ++oc)
                                        {
                                            std::size_t grad_out_idx = n * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
                                            std::size_t w_idx = ky * kernel_size * in_c * out_c + kx * in_c * out_c + ic * out_c + oc;
                                            sum += data[grad_out_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        std::size_t in_idx = n * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                        result_data[in_idx] = sum;
                    }
                }
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(batch_size, in_h * in_w * in_c, std::move(result_data));
    }

    void conv2dBackwardWeight(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &grad_weights,
        const std::shared_ptr<Impl> &grad_biases,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        const auto &go_data = grad_output->getData();

        std::vector<float> gw_data(kernel_size * kernel_size * in_c * out_c, 0.0f);
        std::vector<float> gb_data(out_c, 0.0f);

        for (std::uint32_t oc = 0; oc < out_c; ++oc)
        {
            for (std::uint32_t ic = 0; ic < in_c; ++ic)
            {
                for (std::uint32_t ky = 0; ky < kernel_size; ++ky)
                {
                    for (std::uint32_t kx = 0; kx < kernel_size; ++kx)
                    {
                        float weight_sum = 0.0f;
                        for (std::uint32_t n = 0; n < batch_size; ++n)
                        {
                            for (std::uint32_t oh = 0; oh < out_h; ++oh)
                            {
                                for (std::uint32_t ow = 0; ow < out_w; ++ow)
                                {
                                    std::size_t grad_out_idx = n * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
                                    float grad_out_val = go_data[grad_out_idx];

                                    if (ic == 0 && ky == 0 && kx == 0)
                                    {
                                        gb_data[oc] += grad_out_val;
                                    }

                                    int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                    int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);

                                    if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w))
                                    {
                                        std::size_t in_idx = n * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                                        weight_sum += data[in_idx] * grad_out_val;
                                    }
                                }
                            }
                        }
                        std::size_t w_idx = ky * kernel_size * in_c * out_c + kx * in_c * out_c + ic * out_c + oc;
                        gw_data[w_idx] = weight_sum;
                    }
                }
            }
        }

        grad_weights->uploadData(gw_data);
        grad_biases->uploadData(gb_data);
    }

    std::pair<std::shared_ptr<Impl>, std::shared_ptr<Impl>> maxpool2d(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        std::vector<float> result_data(batch_size * out_h * out_w * channels);
        std::vector<float> mask_data(batch_size * out_h * out_w * channels);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (std::uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (std::uint32_t c = 0; c < channels; ++c)
                    {
                        float max_val = -3.402823466e+38f;
                        float max_idx = 0.0f;

                        for (std::uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);

                                if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w))
                                {
                                    std::size_t in_idx = n * in_h * in_w * channels + ih * in_w * channels + iw * channels + c;
                                    float val = data[in_idx];
                                    if (val > max_val)
                                    {
                                        max_val = val;
                                        max_idx = static_cast<float>(in_idx);
                                    }
                                }
                            }
                        }

                        std::size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                        result_data[out_idx] = max_val;
                        mask_data[out_idx] = max_idx;
                    }
                }
            }
        }

        auto result = std::make_shared<Cpu_Matrix_Impl>(batch_size, out_h * out_w * channels, std::move(result_data));
        auto mask = std::make_shared<Cpu_Matrix_Impl>(batch_size, out_h * out_w * channels, std::move(mask_data));
        return {result, mask};
    }

    std::shared_ptr<Impl> maxpool2dBackward(
        const std::shared_ptr<Impl> &mask,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t out_h, std::uint32_t out_w,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        const auto &mask_data = mask->getData();
        std::vector<float> result_data(batch_size * in_h * in_w * channels, 0.0f);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t ih = 0; ih < in_h; ++ih)
            {
                for (std::uint32_t iw = 0; iw < in_w; ++iw)
                {
                    for (std::uint32_t c = 0; c < channels; ++c)
                    {
                        std::size_t in_idx = n * in_h * in_w * channels + ih * in_w * channels + iw * channels + c;
                        float grad = 0.0f;

                        for (std::uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + padding - ky);
                                int ow_calc = static_cast<int>(iw + padding - kx);

                                if (oh_calc >= 0 && oh_calc % static_cast<int>(stride) == 0 &&
                                    ow_calc >= 0 && ow_calc % static_cast<int>(stride) == 0)
                                {
                                    std::uint32_t oh = static_cast<std::uint32_t>(oh_calc) / stride;
                                    std::uint32_t ow = static_cast<std::uint32_t>(ow_calc) / stride;

                                    if (oh < out_h && ow < out_w)
                                    {
                                        std::size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                                        if (static_cast<std::size_t>(mask_data[out_idx]) == in_idx)
                                        {
                                            grad += data[out_idx];
                                        }
                                    }
                                }
                            }
                        }
                        result_data[in_idx] = grad;
                    }
                }
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(batch_size, in_h * in_w * channels, std::move(result_data));
    }

    std::shared_ptr<Impl> globalAvgPool2d(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::vector<float> result_data(batch_size * channels, 0.0f);
        float pool_area = static_cast<float>(in_h * in_w);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t c = 0; c < channels; ++c)
            {
                float sum = 0.0f;
                for (std::uint32_t h = 0; h < in_h; ++h)
                {
                    for (std::uint32_t w = 0; w < in_w; ++w)
                    {
                        std::size_t in_idx = n * in_h * in_w * channels + h * in_w * channels + w * channels + c;
                        sum += data[in_idx];
                    }
                }
                result_data[n * channels + c] = sum / pool_area;
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(batch_size, channels, std::move(result_data));
    }

    std::shared_ptr<Impl> globalAvgPool2dBackward(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::vector<float> result_data(batch_size * in_h * in_w * channels);
        float pool_area = static_cast<float>(in_h * in_w);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t c = 0; c < channels; ++c)
            {
                float grad = data[n * channels + c] / pool_area;
                for (std::uint32_t h = 0; h < in_h; ++h)
                {
                    for (std::uint32_t w = 0; w < in_w; ++w)
                    {
                        std::size_t in_idx = n * in_h * in_w * channels + h * in_w * channels + w * channels + c;
                        result_data[in_idx] = grad;
                    }
                }
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(batch_size, in_h * in_w * channels, std::move(result_data));
    }

    std::shared_ptr<Impl> batchNormForward(
        const std::shared_ptr<Impl> &gamma,
        const std::shared_ptr<Impl> &beta,
        std::shared_ptr<Impl> &running_mean,
        std::shared_ptr<Impl> &running_var,
        std::shared_ptr<Impl> &batch_mean,
        std::shared_ptr<Impl> &batch_var,
        std::shared_ptr<Impl> &x_hat, 
        float epsilon,
        float momentum,
        bool is_training) const override
    {
        std::size_t n = rows;
        std::size_t d = cols;

        auto gamma_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(gamma);
        auto beta_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(beta);
        auto r_mean_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(running_mean);
        auto r_var_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(running_var);

        std::vector<float> out_data(n * d);
        std::vector<float> x_hat_data(n * d); // Tạo bộ đệm lưu trữ x_hat

        if (is_training)
        {
            std::vector<float> b_mean(d, 0.0f);
            std::vector<float> b_var(d, 0.0f);

            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < d; ++j)
                {
                    b_mean[j] += data[i * d + j];
                }
            }

            float inv_n = 1.0f / static_cast<float>(n);
            for (std::size_t j = 0; j < d; ++j)
            {
                b_mean[j] *= inv_n;
            }

            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < d; ++j)
                {
                    float diff = data[i * d + j] - b_mean[j];
                    b_var[j] += diff * diff;
                }
            }

            for (std::size_t j = 0; j < d; ++j)
            {
                b_var[j] *= inv_n;
            }

            std::vector<float> r_mean_data = r_mean_cpu->getData();
            std::vector<float> r_var_data = r_var_cpu->getData();

            for (std::size_t j = 0; j < d; ++j)
            {
                r_mean_data[j] = (1.0f - momentum) * r_mean_data[j] + momentum * b_mean[j];
                r_var_data[j] = (1.0f - momentum) * r_var_data[j] + momentum * b_var[j];
            }

            r_mean_cpu->uploadData(r_mean_data);
            r_var_cpu->uploadData(r_var_data);

            batch_mean = std::make_shared<Cpu_Matrix_Impl>(1, d, b_mean);
            batch_var = std::make_shared<Cpu_Matrix_Impl>(1, d, b_var);

            const std::vector<float> &gamma_data = gamma_cpu->getData();
            const std::vector<float> &beta_data = beta_cpu->getData();

            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < d; ++j)
                {
                    float val_x_hat = (data[i * d + j] - b_mean[j]) / std::sqrt(b_var[j] + epsilon);
                    x_hat_data[i * d + j] = val_x_hat; // Ghi nhận giá trị vào bộ đệm
                    out_data[i * d + j] = gamma_data[j] * val_x_hat + beta_data[j];
                }
            }
        }
        else
        {
            const std::vector<float> &r_mean_data = r_mean_cpu->getData();
            const std::vector<float> &r_var_data = r_var_cpu->getData();
            const std::vector<float> &gamma_data = gamma_cpu->getData();
            const std::vector<float> &beta_data = beta_cpu->getData();

            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t j = 0; j < d; ++j)
                {
                    float val_x_hat = (data[i * d + j] - r_mean_data[j]) / std::sqrt(r_var_data[j] + epsilon);
                    x_hat_data[i * d + j] = val_x_hat; // Ghi nhận giá trị vào bộ đệm
                    out_data[i * d + j] = gamma_data[j] * val_x_hat + beta_data[j];
                }
            }
        }

        x_hat = std::make_shared<Cpu_Matrix_Impl>(n, d, std::move(x_hat_data)); // Khởi tạo và gán đối tượng x_hat
        return std::make_shared<Cpu_Matrix_Impl>(n, d, std::move(out_data));
    }

    std::shared_ptr<Impl> batchNormBackward(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &gamma,
        const std::shared_ptr<Impl> &batch_var,
        const std::shared_ptr<Impl> &x_hat,
        std::shared_ptr<Impl> &grad_gamma,
        std::shared_ptr<Impl> &grad_beta,
        float epsilon) const override
    {
        std::size_t n = rows;
        std::size_t d = cols;

        auto dout_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_output);
        auto gamma_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(gamma);
        auto b_var_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(batch_var);
        auto x_hat_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(x_hat);
        
        auto g_gamma_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_gamma);
        auto g_beta_cpu = std::dynamic_pointer_cast<Cpu_Matrix_Impl>(grad_beta);

        const std::vector<float> &dout_data = dout_cpu->getData();
        const std::vector<float> &gamma_data = gamma_cpu->getData();
        const std::vector<float> &var_data = b_var_cpu->getData();
        const std::vector<float> &x_hat_data = x_hat_cpu->getData();

        std::vector<float> dgamma_data(d, 0.0f);
        std::vector<float> dbeta_data(d, 0.0f);
        std::vector<float> dx_data(n * d);

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < d; ++j)
            {
                std::size_t idx = i * d + j;
                float dout = dout_data[idx];
                
                dgamma_data[j] += dout * x_hat_data[idx];
                dbeta_data[j] += dout;
            }
        }

        g_gamma_cpu->uploadData(dgamma_data);
        g_beta_cpu->uploadData(dbeta_data);

        float inv_n = 1.0f / static_cast<float>(n);

        for (std::size_t j = 0; j < d; ++j)
        {
            float std_inv = 1.0f / std::sqrt(var_data[j] + epsilon);
            float g_val = gamma_data[j];
            float dg_val = dgamma_data[j];
            float db_val = dbeta_data[j];

            float coeff = g_val * std_inv * inv_n;

            for (std::size_t i = 0; i < n; ++i)
            {
                std::size_t idx = i * d + j;
                dx_data[idx] = coeff * (static_cast<float>(n) * dout_data[idx] - db_val - x_hat_data[idx] * dg_val);
            }
        }

        return std::make_shared<Cpu_Matrix_Impl>(n, d, std::move(dx_data));
    }
    void linearForward(
        const Impl &weights_w,
        const Impl &biases_b,
        Impl &output_y) const override
    {
        const auto &w_cpu = static_cast<const Cpu_Matrix_Impl &>(weights_w);
        const auto &b_cpu = static_cast<const Cpu_Matrix_Impl &>(biases_b);
        auto &y_cpu = static_cast<Cpu_Matrix_Impl &>(output_y);

        std::size_t batch_size = rows;
        std::size_t in_dim = cols;
        std::size_t out_dim = w_cpu.cols;

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < out_dim; ++j)
            {
                float bias_val = (b_cpu.rows == 1) ? b_cpu.data[j] : b_cpu.data[i * out_dim + j];
                y_cpu.data[i * out_dim + j] = bias_val;
            }

            for (std::size_t k = 0; k < in_dim; ++k)
            {
                float x_val = data[i * in_dim + k];
                for (std::size_t j = 0; j < out_dim; ++j)
                {
                    y_cpu.data[i * out_dim + j] += x_val * w_cpu.data[k * out_dim + j];
                }
            }
        }
    }

    void linearBackwardInput(
        const Impl &weights_w,
        Impl &grad_x) const override
    {
        const auto &w_cpu = static_cast<const Cpu_Matrix_Impl &>(weights_w);
        auto &dx_cpu = static_cast<Cpu_Matrix_Impl &>(grad_x);

        std::size_t batch_size = rows;
        std::size_t out_dim = cols;
        std::size_t in_dim = w_cpu.rows;

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < in_dim; ++j)
            {
                float sum = 0.0f;
                for (std::size_t k = 0; k < out_dim; ++k)
                {
                    sum += data[i * out_dim + k] * w_cpu.data[j * out_dim + k];
                }
                dx_cpu.data[i * in_dim + j] = sum;
            }
        }
    }

    void linearBackwardWeightBias(
        const Impl &grad_y,
        Impl &grad_w,
        Impl &grad_b) const override
    {
        const auto &dy_cpu = static_cast<const Cpu_Matrix_Impl &>(grad_y);
        auto &dw_cpu = static_cast<Cpu_Matrix_Impl &>(grad_w);
        auto &db_cpu = static_cast<Cpu_Matrix_Impl &>(grad_b);

        std::size_t batch_size = rows;
        std::size_t in_dim = cols;
        std::size_t out_dim = dy_cpu.cols;

        std::fill(dw_cpu.data.begin(), dw_cpu.data.end(), 0.0f);
        std::fill(db_cpu.data.begin(), db_cpu.data.end(), 0.0f);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < out_dim; ++j)
            {
                float dy_val = dy_cpu.data[i * out_dim + j];
                db_cpu.data[j] += dy_val;
                for (std::size_t k = 0; k < in_dim; ++k)
                {
                    dw_cpu.data[k * out_dim + j] += data[i * in_dim + k] * dy_val;
                }
            }
        }
    }
    void uploadData(const std::vector<float> &host_data) override
    {
        if (host_data.size() != data.size())
        {
            Logger::logMessage("Cpu_Matrix_Impl::uploadData: Size mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Size mismatch");
        }
        data = host_data;
    }
};