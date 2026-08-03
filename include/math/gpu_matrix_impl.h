#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <string>

#include "impl.h"
#include "logger.h"
#include "gpu_vector.h"
#include "execution_engine.h"

struct Matrix_Dimensions
{
    std::uint32_t rows_a;
    std::uint32_t cols_a;
    std::uint32_t cols_b;
};

struct Elementwise_Dimensions
{
    std::uint32_t total_elements;
    std::uint32_t cols;
    std::uint32_t is_broadcast;
};

struct Transpose_Dimensions
{
    std::uint32_t rows;
    std::uint32_t cols;
};

class Gpu_Matrix_Impl : public Impl
{
private:
    std::size_t rows;
    std::size_t cols;
    std::shared_ptr<GVector> storage;
    mutable std::vector<float> host_cache;

    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum pipeline_id, const std::vector<std::shared_ptr<GVector>> &buffers, const Push_Constants_Type &push_constants, std::uint32_t group_x, std::uint32_t group_y = 1, std::uint32_t group_z = 1) const
    {
        if (buffers.size() > 8)
        {
            Logger::logMessage("Gpu_Matrix_Impl::pushToGraph: Exceeded maximum supported buffer count", LOG_ERROR);
            throw std::runtime_error("Exceeded maximum supported buffer count");
        }

        Compute_Node node;
        node.pipeline_id = pipeline_id;
        node.buffers = buffers;

        const std::uint8_t *byte_ptr = reinterpret_cast<const std::uint8_t *>(&push_constants);
        node.push_constants_data.assign(byte_ptr, byte_ptr + sizeof(Push_Constants_Type));

        node.group_x = group_x;
        node.group_y = group_y;
        node.group_z = group_z;

        Execution_Engine::getInstance().getCurrentGraph().addNode(node);
    }

    std::shared_ptr<Gpu_Matrix_Impl> castToGpuMatrix(const std::shared_ptr<Impl> &other_impl, const std::string &error_msg) const
    {
        auto other_gpu = std::dynamic_pointer_cast<Gpu_Matrix_Impl>(other_impl);
        if (!other_gpu)
        {
            Logger::logMessage(error_msg, LOG_ERROR);
            throw std::invalid_argument(error_msg);
        }
        return other_gpu;
    }

    template <typename Pipeline_Enum>
    std::shared_ptr<Impl> executeElementwise(const std::shared_ptr<Impl> &other_impl, Pipeline_Enum pipeline_id, const std::string &error_msg, bool allow_broadcast) const
    {
        auto other_gpu = castToGpuMatrix(other_impl, error_msg);

        bool is_broadcast = false;
        if (allow_broadcast)
        {
            bool same_shape = (rows == other_gpu->getRows() && cols == other_gpu->getCols());
            is_broadcast = (other_gpu->getRows() == 1 && cols == other_gpu->getCols());

            if (!same_shape && !is_broadcast)
                validateSameDimensions(*other_impl);
        }
        else
            validateSameDimensions(*other_impl);

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);

        Elementwise_Dimensions dims{
            .total_elements = static_cast<std::uint32_t>(rows * cols),
            .cols = static_cast<std::uint32_t>(cols),
            .is_broadcast = static_cast<std::uint32_t>(is_broadcast ? 1 : 0)};

        pushToGraph(pipeline_id, {storage, other_gpu->storage, result->storage}, dims, (dims.total_elements + 255) / 256);

        return result;
    }

public:
    Gpu_Matrix_Impl(std::size_t r, std::size_t c)
        : rows(r), cols(c)
    {
        const auto &engine = Execution_Engine::getInstance();
        storage = std::make_shared<GVector>(engine.getContext(), rows * cols);
    }

    Gpu_Matrix_Impl(std::size_t r, std::size_t c, const std::vector<float> &host_data)
        : rows(r), cols(c)
    {
        if (host_data.size() != rows * cols)
        {
            Logger::logMessage("Gpu_Matrix_Impl::Gpu_Matrix_Impl: Host data size mismatch", LOG_ERROR);
            throw std::invalid_argument("Host data size mismatch");
        }
        const auto &engine = Execution_Engine::getInstance();
        storage = std::make_shared<GVector>(engine.getContext(), host_data);
    }

    Gpu_Matrix_Impl(std::size_t r, std::size_t c, std::shared_ptr<GVector> vec)
        : rows(r), cols(c), storage(std::move(vec)) {}

    ~Gpu_Matrix_Impl() override = default;

    std::size_t getRows() const override { return rows; }

    std::size_t getCols() const override { return cols; }

    const std::vector<float> &getData() const override
    {
        host_cache.resize(rows * cols);
        storage->downloadData(host_cache);
        return host_cache;
    }

    VkBuffer getBuffer() const { return storage->getBuffer(); }

    std::shared_ptr<Impl> matmul(const std::shared_ptr<Impl> &other_impl) const override
    {
        validateMatmulDimensions(*other_impl);
        auto other_gpu = castToGpuMatrix(other_impl, "Gpu_Matrix_Impl::matmul: Target matrix is not a Gpu_Matrix_Impl");

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, other_gpu->getCols());

        Matrix_Dimensions dims{
            .rows_a = static_cast<std::uint32_t>(rows),
            .cols_a = static_cast<std::uint32_t>(cols),
            .cols_b = static_cast<std::uint32_t>(result->getCols())};

        pushToGraph(MATMUL, {storage, other_gpu->storage, result->storage}, dims, (dims.cols_b + 15) / 16, (dims.rows_a + 15) / 16);
        return result;
    }

    std::shared_ptr<Impl> add(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, ADD, "Gpu_Matrix_Impl::add: Target matrix is not a Gpu_Matrix_Impl", true);
    }

    std::shared_ptr<Impl> sub(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, SUB, "Gpu_Matrix_Impl::sub: Target matrix is not a Gpu_Matrix_Impl", true);
    }

    std::shared_ptr<Impl> mulScalar(float scalar) const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Scalar_Push_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{total_elements, scalar};

        pushToGraph(MUL_SCALAR, {storage, result->storage}, constants, (total_elements + 255) / 256);

        return result;
    }

    std::shared_ptr<Impl> divScalar(float scalar) const override
    {
        if (std::abs(scalar) < 1e-8f)
        {
            Logger::logMessage("Gpu_Matrix_Impl::divScalar: Division by zero", LOG_ERROR, true);
            throw std::runtime_error("Division by zero in divScalar");
        }

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Scalar_Push_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{total_elements, 1.0f / scalar};

        pushToGraph(MUL_SCALAR, {storage, result->storage}, constants, (total_elements + 255) / 256);

        return result;
    }

    std::shared_ptr<Impl> hadamardMul(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, HADAMARD_MUL, "Gpu_Matrix_Impl::hadamardMul: Target matrix is not a Gpu_Matrix_Impl", false);
    }

    std::shared_ptr<Impl> hadamardDiv(const std::shared_ptr<Impl> &other_impl) const override
    {
        return executeElementwise(other_impl, HADAMARD_DIV, "Gpu_Matrix_Impl::hadamardDiv: Target matrix is not a Gpu_Matrix_Impl", false);
    }

    std::shared_ptr<Impl> transpose() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(cols, rows);

        Transpose_Dimensions dims{
            .rows = static_cast<std::uint32_t>(rows),
            .cols = static_cast<std::uint32_t>(cols)};

        pushToGraph(TRANSPOSE, {storage, result->storage}, dims, (dims.cols + 15) / 16, (dims.rows + 15) / 16);

        return result;
    }

    std::shared_ptr<Impl> relu() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        pushToGraph(RELU, {storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> reluBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        validateSameDimensions(*grad_impl);
        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::reluBackward: Target matrix is not a Gpu_Matrix_Impl");
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);
        pushToGraph(RELU_BACKWARD, {storage, grad_gpu->storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> gelu() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        pushToGraph(GELU, {storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> geluBackward(const std::shared_ptr<Impl> &grad_impl) const override
    {
        validateSameDimensions(*grad_impl);
        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::reluBackward: Target matrix is not a Gpu_Matrix_Impl");
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);
        pushToGraph(GELU_BACKWARD, {storage, grad_gpu->storage, result->storage}, total_elements, (total_elements + 255) / 256);
        return result;
    }

    std::shared_ptr<Impl> inverse() const override
    {
        validateSquare();

        std::size_t n = rows;
        std::vector<float> host_data = getData();
        std::vector<float> aug(n * 2 * n, 0.0f);

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
                aug[i * (2 * n) + j] = host_data[i * n + j];
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
                Logger::logMessage("Gpu_Matrix_Impl::inverse: Matrix is singular and cannot be inverted", LOG_ERROR);
                throw std::runtime_error("Matrix is singular");
            }

            if (pivot_row != i)
                for (std::size_t j = 0; j < 2 * n; ++j)
                    std::swap(aug[i * (2 * n) + j], aug[pivot_row * (2 * n) + j]);

            float pivot = aug[i * (2 * n) + i];
            for (std::size_t j = 0; j < 2 * n; ++j)
                aug[i * (2 * n) + j] /= pivot;

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
            for (std::size_t j = 0; j < n; ++j)
                inv_data[i * n + j] = aug[i * (2 * n) + (n + j)];

        return std::make_shared<Gpu_Matrix_Impl>(n, n, inv_data);
    }

    std::shared_ptr<Impl> normalize() const override
    {
        std::vector<float> host_data = getData();
        float sum_sq = 0.0f;
        for (float val : host_data)
            sum_sq += val * val;

        float norm = std::sqrt(sum_sq);
        if (norm < 1e-8f)
        {
            Logger::logMessage("Gpu_Matrix_Impl::normalize: Matrix norm near zero during normalization", LOG_WARNING, true);
            return std::make_shared<Gpu_Matrix_Impl>(rows, cols, host_data);
        }

        std::vector<float> result_data(host_data.size());
        for (std::size_t i = 0; i < host_data.size(); ++i)
            result_data[i] = host_data[i] / norm;

        return std::make_shared<Gpu_Matrix_Impl>(rows, cols, result_data);
    }

    std::shared_ptr<Impl> softmax() const override
    {
        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);

        struct Softmax_Dimensions
        {
            std::uint32_t rows;
            std::uint32_t cols;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols)};

        pushToGraph(SOFTMAX, {storage, result->storage}, constants, constants.rows, 1, 1);

        return result;
    }

    std::shared_ptr<Impl> softmaxBackward(const std::shared_ptr<Impl> &gradient_output) const override
    {
        validateSameDimensions(*gradient_output);
        auto grad_gpu = castToGpuMatrix(gradient_output, "Gpu_Matrix_Impl::softmaxBackward: Invalid matrix type");

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, cols);

        struct Softmax_Dimensions
        {
            std::uint32_t rows;
            std::uint32_t cols;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols)};

        pushToGraph(SOFTMAX_BACKWARD, {storage, grad_gpu->storage, result->storage}, constants, constants.rows, 1, 1);

        return result;
    }

    void sgdUpdate(const std::shared_ptr<Impl> &grad_impl, float learning_rate, float max_gradient = 0.0f) override
    {
        validateSameDimensions(*grad_impl);
        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::sgdUpdate: Target matrix is not a Gpu_Matrix_Impl");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        struct Sgd_Push_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float max_gradient;
        } constants{total_elements, learning_rate, max_gradient};

        pushToGraph(SGD_UPDATE, {storage, grad_gpu->storage}, constants, (total_elements + 255) / 256);
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

        auto grad_gpu = castToGpuMatrix(grad_impl, "Gpu_Matrix_Impl::adamUpdate: Invalid grad_impl matrix");
        auto m_gpu = castToGpuMatrix(m_impl, "Gpu_Matrix_Impl::adamUpdate: Invalid m_impl matrix");
        auto v_gpu = castToGpuMatrix(v_impl, "Gpu_Matrix_Impl::adamUpdate: Invalid v_impl matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * cols);

        float correction1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
        float correction2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));

        struct Adam_Push_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float correction1;
            float correction2;
        } constants{
            total_elements,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            max_gradient,
            correction1,
            correction2};

        pushToGraph(
            ADAM_UPDATE,
            {storage, grad_gpu->storage, m_gpu->storage, v_gpu->storage},
            constants,
            (total_elements + 255) / 256);
    }

    std::shared_ptr<Impl> matdiv(const std::shared_ptr<Impl> &other_impl) const override { return matmul(other_impl->inverse()); }

    std::shared_ptr<Impl> matmulAdd(const std::shared_ptr<Impl> &other, const std::shared_ptr<Impl> &bias) const override
    {
        validateMatmulDimensions(*other);
        auto w_gpu = castToGpuMatrix(other, "Invalid weight matrix type");
        auto b_gpu = castToGpuMatrix(bias, "Invalid bias matrix type");

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, w_gpu->cols);

        struct Push_Constants
        {
            std::uint32_t rows_x;
            std::uint32_t cols_x;
            std::uint32_t cols_w;
            std::uint32_t pad;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(w_gpu->cols),
            0};

        std::uint32_t group_x = (constants.cols_w + 15) / 16;
        std::uint32_t group_y = (constants.rows_x + 15) / 16;

        pushToGraph(MATMUL_ADD, {storage, w_gpu->storage, b_gpu->storage, result->storage}, constants, group_x, group_y, 1);

        return result;
    }

    std::shared_ptr<Impl> matmulTransA(const std::shared_ptr<Impl> &other) const override
    {
        auto other_gpu = castToGpuMatrix(other, "Invalid matrix type for matmulTransA");
        if (rows != other_gpu->rows)
        {
            Logger::logMessage("Gpu_Matrix_Impl::matmulTransA: Rows mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Rows mismatch in matmulTransA");
        }

        auto result = std::make_shared<Gpu_Matrix_Impl>(cols, other_gpu->cols);

        struct Push_Constants
        {
            std::uint32_t rows_a;
            std::uint32_t cols_a;
            std::uint32_t cols_b;
            std::uint32_t pad;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(other_gpu->cols),
            0};

        std::uint32_t group_x = (constants.cols_b + 15) / 16;
        std::uint32_t group_y = (constants.cols_a + 15) / 16;

        pushToGraph(MATMUL_TRANS_A, {storage, other_gpu->storage, result->storage}, constants, group_x, group_y, 1);

        return result;
    }

    std::shared_ptr<Impl> matmulTransB(const std::shared_ptr<Impl> &other) const override
    {
        auto other_gpu = castToGpuMatrix(other, "Invalid matrix type for matmulTransB");
        if (cols != other_gpu->cols)
        {
            Logger::logMessage("Gpu_Matrix_Impl::matmulTransB: Cols mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Cols mismatch in matmulTransB");
        }

        auto result = std::make_shared<Gpu_Matrix_Impl>(rows, other_gpu->rows);

        struct Push_Constants
        {
            std::uint32_t rows_a;
            std::uint32_t cols_a;
            std::uint32_t rows_b;
            std::uint32_t pad;
        } constants{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(other_gpu->rows),
            0};

        std::uint32_t group_x = (constants.rows_b + 15) / 16;
        std::uint32_t group_y = (constants.rows_a + 15) / 16;

        pushToGraph(MATMUL_TRANS_B, {storage, other_gpu->storage, result->storage}, constants, group_x, group_y, 1);

        return result;
    }

    std::shared_ptr<Impl> conv2d(
        const std::shared_ptr<Impl> &weights,
        const std::shared_ptr<Impl> &bias,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_c, std::uint32_t kernel_size,
        std::uint32_t stride, std::uint32_t padding) const override
    {
        auto w_gpu = castToGpuMatrix(weights, "Gpu_Matrix_Impl::conv2d: Invalid weights matrix");
        auto b_gpu = castToGpuMatrix(bias, "Gpu_Matrix_Impl::conv2d: Invalid bias matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, out_h * out_w * out_c);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t in_c;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t out_c;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding};

        std::uint32_t group_x = (out_c + 15) / 16;
        std::uint32_t group_y = (out_w + 15) / 16;
        std::uint32_t group_z = batch_size * out_h;

        pushToGraph(CONV2D_FORWARD_PASS, {storage, w_gpu->storage, b_gpu->storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
    }

    std::shared_ptr<Impl> conv2dBackwardInput(
        const std::shared_ptr<Impl> &weights,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto w_gpu = castToGpuMatrix(weights, "Gpu_Matrix_Impl::conv2dBackwardInput: Invalid weights matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, in_h * in_w * in_c);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t in_c;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t out_c;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding};

        std::uint32_t group_x = (in_c + 15) / 16;
        std::uint32_t group_y = (in_w + 15) / 16;
        std::uint32_t group_z = batch_size * in_h;

        pushToGraph(CONV2D_BACKWARD_PASS_INPUT_GRADIENT, {storage, w_gpu->storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
    }

    void conv2dBackwardWeight(
        const std::shared_ptr<Impl> &grad_output,
        const std::shared_ptr<Impl> &grad_weights,
        const std::shared_ptr<Impl> &grad_biases,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t in_c,
        std::uint32_t out_h, std::uint32_t out_w, std::uint32_t out_c,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto go_gpu = castToGpuMatrix(grad_output, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid grad_output matrix");
        auto gw_gpu = castToGpuMatrix(grad_weights, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid grad_weights matrix");
        auto gb_gpu = castToGpuMatrix(grad_biases, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid grad_biases matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t in_c;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t out_c;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding};

        std::uint32_t group_x = (out_c + 15) / 16;
        std::uint32_t group_y = (in_c + 15) / 16;
        std::uint32_t group_z = kernel_size * kernel_size;

        pushToGraph(CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT, {storage, go_gpu->storage, gw_gpu->storage, gb_gpu->storage}, constants, group_x, group_y, group_z);
    }

    std::pair<std::shared_ptr<Impl>, std::shared_ptr<Impl>> maxpool2d(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, out_h * out_w * channels);
        auto mask = std::make_shared<Gpu_Matrix_Impl>(batch_size, out_h * out_w * channels);

        struct Max_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding};

        std::uint32_t group_x = (channels + 15) / 16;
        std::uint32_t group_y = (out_w + 15) / 16;
        std::uint32_t group_z = batch_size * out_h;

        pushToGraph(MAXPOOL2D_FORWARD, {storage, result->storage, mask->storage}, constants, group_x, group_y, group_z);

        return {result, mask};
    }

    std::shared_ptr<Impl> maxpool2dBackward(
        const std::shared_ptr<Impl> &mask,
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
        std::uint32_t out_h, std::uint32_t out_w,
        std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto mask_gpu = castToGpuMatrix(mask, "Gpu_Matrix_Impl::maxpool2dBackward: Invalid mask matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, in_h * in_w * channels);

        struct Max_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
            std::uint32_t out_h;
            std::uint32_t out_w;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, in_h, in_w, channels, out_h, out_w, kernel_size, stride, padding};

        std::uint32_t group_x = (channels + 15) / 16;
        std::uint32_t group_y = (in_w + 15) / 16;
        std::uint32_t group_z = batch_size * in_h;

        pushToGraph(MAXPOOL2D_BACKWARD, {storage, mask_gpu->storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
    }

    std::shared_ptr<Impl> globalAvgPool2d(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, channels);

        struct Global_Avg_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
        } constants{batch_size, in_h, in_w, channels};

        std::uint32_t group_x = (channels + 255) / 256;
        std::uint32_t group_y = batch_size;

        pushToGraph(GLOBAL_AVGPOOL_FORWARD, {storage, result->storage}, constants, group_x, group_y, 1);

        return result;
    }

    std::shared_ptr<Impl> globalAvgPool2dBackward(
        std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto result = std::make_shared<Gpu_Matrix_Impl>(batch_size, in_h * in_w * channels);

        struct Global_Avg_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t in_h;
            std::uint32_t in_w;
            std::uint32_t channels;
        } constants{batch_size, in_h, in_w, channels};

        std::uint32_t group_x = (channels + 15) / 16;
        std::uint32_t group_y = (in_w + 15) / 16;
        std::uint32_t group_z = batch_size * in_h;

        pushToGraph(GLOBAL_AVGPOOL_BACKWARD, {storage, result->storage}, constants, group_x, group_y, group_z);

        return result;
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
        auto gamma_gpu = castToGpuMatrix(gamma, "Invalid gamma matrix");
        auto beta_gpu = castToGpuMatrix(beta, "Invalid beta matrix");
        auto r_mean_gpu = castToGpuMatrix(running_mean, "Invalid running_mean matrix");
        auto r_var_gpu = castToGpuMatrix(running_var, "Invalid running_var matrix");

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        std::uint32_t d = static_cast<std::uint32_t>(cols);

        auto result = std::make_shared<Gpu_Matrix_Impl>(n, d);
        x_hat = std::make_shared<Gpu_Matrix_Impl>(n, d);
        auto x_hat_gpu = castToGpuMatrix(x_hat, "Invalid x_hat matrix");

        if (is_training)
        {
            batch_mean = std::make_shared<Gpu_Matrix_Impl>(1, d);
            batch_var = std::make_shared<Gpu_Matrix_Impl>(1, d);
            auto b_mean_gpu = castToGpuMatrix(batch_mean, "Invalid batch_mean matrix");
            auto b_var_gpu = castToGpuMatrix(batch_var, "Invalid batch_var matrix");

            struct Stats_Push_Constants
            {
                std::uint32_t batch_size;
                std::uint32_t dim;
                float momentum;
            } stats_consts{n, d, momentum};

            pushToGraph(BATCH_NORM_STATS_FORWARD, {storage, b_mean_gpu->storage, b_var_gpu->storage, r_mean_gpu->storage, r_var_gpu->storage}, stats_consts, (d + 255) / 256, 1, 1);

            struct Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t dim;
                float epsilon;
            } trans_consts{n * d, d, epsilon};

            pushToGraph(BATCH_NORM_TRANSFORM_FORWARD, {storage, b_mean_gpu->storage, b_var_gpu->storage, gamma_gpu->storage, beta_gpu->storage, result->storage, x_hat_gpu->storage}, trans_consts, (n * d + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t dim;
                float epsilon;
            } trans_consts{n * d, d, epsilon};

            pushToGraph(BATCH_NORM_TRANSFORM_FORWARD, {storage, r_mean_gpu->storage, r_var_gpu->storage, gamma_gpu->storage, beta_gpu->storage, result->storage, x_hat_gpu->storage}, trans_consts, (n * d + 255) / 256, 1, 1);
        }

        return result;
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
        auto dout_gpu = castToGpuMatrix(grad_output, "Invalid grad_output matrix");
        auto gamma_gpu = castToGpuMatrix(gamma, "Invalid gamma matrix");
        auto b_var_gpu = castToGpuMatrix(batch_var, "Invalid batch_var matrix");
        auto x_hat_gpu = castToGpuMatrix(x_hat, "Invalid x_hat matrix");

        std::uint32_t n = static_cast<std::uint32_t>(rows);
        std::uint32_t d = static_cast<std::uint32_t>(cols);

        grad_gamma = std::make_shared<Gpu_Matrix_Impl>(1, d);
        grad_beta = std::make_shared<Gpu_Matrix_Impl>(1, d);
        auto g_gamma_gpu = castToGpuMatrix(grad_gamma, "Invalid grad_gamma matrix");
        auto g_beta_gpu = castToGpuMatrix(grad_beta, "Invalid grad_beta matrix");

        auto dx = std::make_shared<Gpu_Matrix_Impl>(n, d);

        struct Backward_Stats_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t dim;
        } b_stats_consts{n, d};

        pushToGraph(BATCH_NORM_STATS_BACKWARD, {dout_gpu->storage, x_hat_gpu->storage, g_gamma_gpu->storage, g_beta_gpu->storage}, b_stats_consts, (d + 255) / 256, 1, 1);

        struct Backward_Transform_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t batch_size;
            std::uint32_t dim;
            float epsilon;
        } b_trans_consts{n * d, n, d, epsilon};

        pushToGraph(BATCH_NORM_TRANSFORM_BACKWARD, {dout_gpu->storage, x_hat_gpu->storage, gamma_gpu->storage, g_gamma_gpu->storage, g_beta_gpu->storage, b_var_gpu->storage, dx->storage}, b_trans_consts, (n * d + 255) / 256, 1, 1);

        return dx;
    }

    void uploadData(const std::vector<float> &host_data) override { storage->uploadData(host_data); }
};