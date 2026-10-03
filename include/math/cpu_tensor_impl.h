#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "tensor_impl.h"

class Cpu_Tensor_Impl : public Tensor_Impl
{
private:
    std::shared_ptr<std::vector<float>> storage_buffer;
    std::shared_ptr<std::vector<float16_t>> storage_buffer_fp16;
    mutable std::vector<float> materialized_cache;
    mutable std::mutex cache_mutex;    static std::string formatDataSample(const std::vector<float> &data, size_t sample_limit = 5);
    size_t resolveFlatIndex(std::span<const size_t> indices) const noexcept;
    template <typename Op>
    void iterateCoordinates(Op &&op) const;


public:    Cpu_Tensor_Impl(size_t rows, size_t columns);
    Cpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data);
    Cpu_Tensor_Impl(size_t rows, size_t columns, std::vector<float> &&host_data);
    explicit Cpu_Tensor_Impl(Shape tensor_shape);
    Cpu_Tensor_Impl(Shape tensor_shape, Data_Type type);
    Cpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data);
    Cpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<std::vector<float>> buffer, size_t offset_elements);


    ~Cpu_Tensor_Impl() noexcept override = default;    void reshape(size_t rows, size_t columns) override;
    void reshape(Shape new_shape) override;
    void permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const override;
    void slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const override;
    void updateSlice(size_t axis, size_t start, const Tensor_Impl &source) override;
    void gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const override;
    void contiguous(Tensor_Impl &output) const override;
    void to(Data_Type target_type, Tensor_Impl &output) const override;
    void matmul(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void add(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void sub(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void mulScalar(float scalar, Tensor_Impl &output) const override;
    void divScalar(float scalar, Tensor_Impl &output) const override;
    void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void transpose(Tensor_Impl &output) const override;
    void inverse(Tensor_Impl &output) const override;
    void normalize(Tensor_Impl &output) const override;
    void relu(Tensor_Impl &output) const override;
    void reluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void gelu(Tensor_Impl &output) const override;
    void geluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void softmax(Tensor_Impl &output) const override;
    void softmaxBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient = 0.0F, float inv_scale = 1.0F) override;
    void adamUpdate(const Tensor_Impl &gradient,
                    const Tensor_Impl &first_moment,
                    const Tensor_Impl &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    size_t timestep,
                    float max_gradient = 1.0F,
                    float inv_scale = 1.0F,
                    float weight_decay = 0.0F) override;
    void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override;
    void uploadData(const std::vector<float> &host_data) override;
    void zero() override;
    void fill(float value) override;
    void conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output,
                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                uint32_t output_channels, uint32_t kernel_size,
                uint32_t stride, uint32_t padding,
                Tensor_Impl *scratch = nullptr) const override;
    void conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                             uint32_t kernel_size, uint32_t stride, uint32_t padding) const override;
    void conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient,
                              uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                              uint32_t kernel_size, uint32_t stride, uint32_t padding,
                              Tensor_Impl * /*im2col_scratch*/ = nullptr) const override;
    void maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask,
                   uint32_t input_height, uint32_t input_width, uint32_t channels,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding) const override;
    void maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient,
                           uint32_t input_height, uint32_t input_width, uint32_t channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) const override;
    void globalAvgPool2d(Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const override;
    void globalAvgPool2dBackward(Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const override;
    void batchNormForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                          Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                          Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                          Tensor_Impl &normalized_input, Tensor_Impl &output,
                          float epsilon, float momentum, bool is_training) const override;
    void batchNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                           Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, float epsilon) const override;
    void linearForward(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override;
    void linearBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient) const override;
    void linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient, bool accumulate = false) const override;
    void linearBackwardWeightAdam(const Tensor_Impl &output_gradient, Tensor_Impl &weights, Tensor_Impl &first_moment, Tensor_Impl &second_moment, Tensor_Impl &bias_gradient,
                                  float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient = 1.0F, float inv_scale = 1.0F, float weight_decay = 0.0F) override;
    void batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                            Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                            Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                            Tensor_Impl &normalized_input, Tensor_Impl &output,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const override;
    void batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                             Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const override;
    void cceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override;
    void mseLoss(const Tensor_Impl &target, Tensor_Impl &output) const override;
    void maeLoss(const Tensor_Impl &target, Tensor_Impl &output) const override;
    void bceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override;
    void huberLoss(const Tensor_Impl &target, Tensor_Impl &output, float delta) const override;
    void concatenateColumns(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override;
    void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override;
    const std::vector<float> &getData() const noexcept override;
    Mutable_Storage_Handle getStorage() override;
    void rmsNormForward(const Tensor_Impl &gamma, Tensor_Impl &inv_rms, Tensor_Impl &output, float epsilon) const override;
    void rmsNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &inv_rms,
                         Tensor_Impl &gamma_gradient, Tensor_Impl &input_gradient, bool accumulate_gamma = false) const override;
    void applyRoPE(Tensor_Impl &output, uint32_t seq_len, uint32_t head_dim, int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0) const override;
    void swigluForward(const Tensor_Impl &b, Tensor_Impl &output) const override;
    void swigluBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &b, Tensor_Impl &grad_a, Tensor_Impl &grad_b) const override;
    void fusedSwiGLUForward(Tensor_Impl &output) const override;
    void fusedSwiGLUBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override;
    void flashAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output,
                               uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                               bool is_causal = false, float scale = 0.0f,
                               Tensor_Impl *l_stats = nullptr) const override;
    void flashAttentionBackward(const Tensor_Impl &k, const Tensor_Impl &v,
                                const Tensor_Impl &o, const Tensor_Impl &do_grad,
                                Tensor_Impl &dq, Tensor_Impl &dk, Tensor_Impl &dv,
                                uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                                bool is_causal = false, float scale = 0.0f,
                                const Tensor_Impl *l_stats = nullptr) const override;
    void embeddingForward(const Tensor_Impl &indices, Tensor_Impl &output) const override;
    void embeddingBackward(const Tensor_Impl &indices, const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient) const override;
    void singleTokenAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output,
                                     size_t num_heads, size_t head_dim, size_t total_seq_len) const override;
    float fusedCrossEntropyLoss(const Tensor_Impl &targets, Tensor_Impl &d_logits, uint32_t valid_tokens = 0) const override;


    Storage_Handle getStorage() const override { return std::cref(getData()); }
    const std::shared_ptr<std::vector<float16_t>> &getStorageBufferFp16() const noexcept { return storage_buffer_fp16; }
    std::shared_ptr<std::vector<float16_t>> &getStorageBufferFp16() noexcept { return storage_buffer_fp16; }
    const std::shared_ptr<std::vector<float>> &getStorageBuffer() const noexcept { return storage_buffer; }
    std::shared_ptr<std::vector<float>> &getStorageBuffer() noexcept { return storage_buffer; }
    bool isEmpty() const noexcept override { return (data_type == Data_Type::FLOAT16) ? (!storage_buffer_fp16 || storage_buffer_fp16->empty()) : (!storage_buffer || storage_buffer->empty()); }

    void setStorageBufferFp16(std::shared_ptr<std::vector<float16_t>> _buf) noexcept { storage_buffer_fp16 = std::move(_buf); }
};;