#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "engine/data_type.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "shape.h"

using Storage_Handle = std::variant<std::reference_wrapper<const std::vector<float>>, std::shared_ptr<gpu::vector>>;
using Mutable_Storage_Handle = std::variant<std::reference_wrapper<std::vector<float>>, std::shared_ptr<gpu::vector>>;

class Tensor_Impl
{
protected:
    Shape shape;
    Stride strides;
    size_t byte_offset = 0;
    size_t total_elements = 0;
    Data_Type data_type = Data_Type::FLOAT32;

    void updateShapeAndStrides(Shape target_shape)
    {
        shape = target_shape;
        strides = shape.computeContiguousStrides();
        total_elements = shape.getTotalElements();
    }

public:
    virtual ~Tensor_Impl() noexcept = default;

    virtual Storage_Handle getStorage() const = 0;
    virtual Mutable_Storage_Handle getStorage() = 0;
    virtual const std::vector<float> &getData() const = 0;
    virtual void uploadData(const std::vector<float> &host_data) = 0;
    virtual const std::vector<uint32_t> &getPackedData() const
    {
        static const std::vector<uint32_t> empty_vec;
        return empty_vec;
    }
    virtual void uploadPackedData(const std::vector<uint32_t> &)
    {
        throw std::invalid_argument("uploadPackedData not supported on this tensor type");
    }
    virtual bool isEmpty() const noexcept = 0;
    virtual void zero() = 0;
    virtual void fill(float value) = 0;
    virtual void invalidateFp16Cache() noexcept {}
    virtual void prewarmFp16Cache() {}

    bool isContiguous() const noexcept
    {
        size_t acc_stride = 1;
        for (size_t i = shape.getRank(); i > 0; --i)
        {
            if (shape[i - 1] == 1)
                continue;
            if (strides[i - 1] != acc_stride)
                return false;
            acc_stride *= shape[i - 1];
        }
        return true;
    }

    void validateSameDimensions(const Tensor_Impl &other) const
    {
        if (shape != other.shape)
        {
            Logger::logMessage(Input_Format{"Tensor_Impl::validateSameDimensions: Shape mismatch: {} vs {}",
                                            shape.toString(), other.shape.toString()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Tensor shape mismatch");
        }
    }

    void validateMatmulDimensions(const Tensor_Impl &other) const
    {
        if (getMatrixCols() != other.getMatrixRows())
        {
            Logger::logMessage(Input_Format{"Tensor_Impl::validateMatmulDimensions: Mismatch: cols_a ({}) != rows_b ({})",
                                            getMatrixCols(), other.getMatrixRows()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Tensor matmul dimension mismatch");
        }
    }

    void validateSquare() const
    {
        if (getRows() != getColumns())
        {
            Logger::logMessage(Input_Format{"Tensor_Impl::validateSquare: Non-square dimensions: ({}x{})",
                                            getRows(), getColumns()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Tensor is not square");
        }
    }

    virtual void reshape(size_t rows, size_t columns) = 0;
    virtual void reshape(Shape new_shape) = 0;
    virtual void permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const = 0;
    virtual void slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const = 0;
    virtual void updateSlice(size_t axis, size_t start, const Tensor_Impl &source) = 0;
    virtual void contiguous(Tensor_Impl &output) const = 0;
    virtual void to(Data_Type target_type, Tensor_Impl &output) const = 0;
    virtual void gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const = 0;

    virtual void matmul(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void add(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void sub(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void mulScalar(float scalar, Tensor_Impl &output) const = 0;
    virtual void divScalar(float scalar, Tensor_Impl &output) const = 0;
    virtual void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void transpose(Tensor_Impl &output) const = 0;
    virtual void inverse(Tensor_Impl &output) const = 0;
    virtual void normalize(Tensor_Impl &output) const = 0;

    virtual void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const = 0;

    virtual void concatenateColumns(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const = 0;
    virtual void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const = 0;

    const Shape &getShape() const noexcept { return shape; }
    const Stride &getStrides() const noexcept { return strides; }
    size_t getColumns() const
    {
        if (shape.getRank() == 0)
            return 0;
        if (shape.getRank() == 1)
            return shape[0];
        size_t cols = 1;
        for (size_t i = 1; i < shape.getRank(); ++i)
        {
            if (shape[i] == 0)
            {
                return 0;
            }
            if (cols > std::numeric_limits<size_t>::max() / shape[i])
            {
                throw std::overflow_error("Tensor_Impl::getColumns: size_t overflow");
            }
            cols *= shape[i];
        }
        return cols;
    }
    size_t getRows() const noexcept
    {
        if (shape.getRank() == 0)
            return 0;
        if (shape.getRank() == 1)
            return 1;
        return shape[0];
    }
    size_t getTotalElements() const noexcept { return total_elements; }
    size_t getByteOffset() const noexcept { return byte_offset; }
    size_t getRank() const noexcept { return shape.getRank(); }
    size_t getMatrixRows() const noexcept
    {
        size_t r = shape.getRank();
        return (r == 0) ? 0 : ((r == 1) ? 1 : shape[r - 2]);
    }
    size_t getMatrixCols() const noexcept
    {
        size_t r = shape.getRank();
        return (r == 0) ? 0 : shape[r - 1];
    }

    void setShape(const Shape &_shape) { updateShapeAndStrides(_shape); }
    void setByteOffset(size_t _byte_offset) noexcept { byte_offset = _byte_offset; }

protected:
    void setStrides(const Stride &_strides) noexcept { strides = _strides; }
    void setTotalElements(size_t _total_elements) noexcept { total_elements = _total_elements; }

public:
    Data_Type getDataType() const noexcept { return data_type; }
    virtual void setDataType(Data_Type _type)
    {
        data_type = _type;
        invalidateFp16Cache();
    }
};
