#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <span>
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
    std::shared_ptr<std::vector<uint32_t>> storage_buffer_u32;
    mutable std::vector<float> materialized_cache;
    mutable std::mutex cache_mutex;
    static std::string formatDataSample(const std::vector<float> &data, size_t sample_limit = 5);
    size_t resolveFlatIndex(std::span<const size_t> indices) const noexcept;
    template <typename Op>
    void iterateCoordinates(Op &&op) const;

public:
    static const Cpu_Tensor_Impl &castToCpu(const Tensor_Impl &other)
    {
        const auto *ptr = dynamic_cast<const Cpu_Tensor_Impl *>(&other);
        if (!ptr)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::castToCpu: Expected Cpu_Tensor_Impl (Execution_Target mismatch)");
        }
        return *ptr;
    }

    static Cpu_Tensor_Impl &castToCpu(Tensor_Impl &other)
    {
        auto *ptr = dynamic_cast<Cpu_Tensor_Impl *>(&other);
        if (!ptr)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::castToCpu: Expected Cpu_Tensor_Impl (Execution_Target mismatch)");
        }
        return *ptr;
    }

    Cpu_Tensor_Impl(size_t rows, size_t columns);
    Cpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data);
    Cpu_Tensor_Impl(size_t rows, size_t columns, std::vector<float> &&host_data);
    explicit Cpu_Tensor_Impl(Shape tensor_shape);
    Cpu_Tensor_Impl(Shape tensor_shape, Data_Type type);
    Cpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data);
    Cpu_Tensor_Impl(Shape tensor_shape, std::vector<float> &&host_data);
    Cpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<std::vector<float>> buffer, size_t offset_elements);

    Cpu_Tensor_Impl(const Cpu_Tensor_Impl &) = delete;
    Cpu_Tensor_Impl &operator=(const Cpu_Tensor_Impl &) = delete;
    Cpu_Tensor_Impl(Cpu_Tensor_Impl &&) = delete;
    Cpu_Tensor_Impl &operator=(Cpu_Tensor_Impl &&) = delete;

    ~Cpu_Tensor_Impl() noexcept override = default;
    void reshape(size_t rows, size_t columns) override;
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
    void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override;
    void uploadData(const std::vector<float> &host_data) override;
    void zero() override;
    void fill(float value) override;
    void concatenateColumns(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override;
    void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override;
    const std::vector<float> &getData() const override;
    Mutable_Storage_Handle getStorage() override;

    Storage_Handle getStorage() const override { return std::cref(getData()); }
    const std::shared_ptr<std::vector<float16_t>> &getStorageBufferFp16() const noexcept { return storage_buffer_fp16; }
    std::shared_ptr<std::vector<float16_t>> &getStorageBufferFp16() noexcept { return storage_buffer_fp16; }
    const std::shared_ptr<std::vector<float>> &getStorageBuffer() const noexcept { return storage_buffer; }
    std::shared_ptr<std::vector<float>> &getStorageBuffer() noexcept { return storage_buffer; }
    const std::shared_ptr<std::vector<uint32_t>> &getStorageBufferU32() const noexcept { return storage_buffer_u32; }
    std::shared_ptr<std::vector<uint32_t>> &getStorageBufferU32() noexcept { return storage_buffer_u32; }

    const std::vector<uint32_t> &getPackedData() const override;
    void uploadPackedData(const std::vector<uint32_t> &packed_data) override;
    void setDataType(Data_Type _type) override;

    static void packTernary(const float *src, uint32_t *dst, size_t rows, size_t cols, float inv_gamma = 1.0f);
    static void unpackTernary(const uint32_t *src, float *dst, size_t rows, size_t cols, float gamma);
    static void ternaryMatmul(const float *A, const uint32_t *B_packed, float *C,
                              size_t M, size_t K, size_t N, float gamma, const float *biases = nullptr);

    bool isEmpty() const noexcept override
    {
        if (data_type == Data_Type::FLOAT16)
            return !storage_buffer_fp16 || storage_buffer_fp16->empty();
        if (data_type == Data_Type::TERNARY_INT2)
            return !storage_buffer_u32 || storage_buffer_u32->empty();
        return !storage_buffer || storage_buffer->empty();
    }

    void setStorageBufferFp16(std::shared_ptr<std::vector<float16_t>> _buf) noexcept { storage_buffer_fp16 = std::move(_buf); }
    void setStorageBufferU32(std::shared_ptr<std::vector<uint32_t>> _buf) noexcept { storage_buffer_u32 = std::move(_buf); }
};
