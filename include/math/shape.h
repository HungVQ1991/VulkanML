#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>

constexpr size_t MAX_TENSOR_RANK = 6;

class Shape
{
private:
    std::array<size_t, MAX_TENSOR_RANK> dimensions{};
    std::uint8_t rank_size = 0;

public:
    Shape() = default;
    Shape(std::initializer_list<size_t> dimension_list);
    Shape(std::span<const size_t> dimension_span);
    Shape(size_t dim_0, size_t dim_1)
    {
        rank_size = 2;
        dimensions[0] = dim_0;
        dimensions[1] = dim_1;
    }

    size_t operator[](size_t index) const
    {
        assert(index < MAX_TENSOR_RANK);
        return dimensions[index];
    }

    size_t &operator[](size_t index)
    {
        assert(index < MAX_TENSOR_RANK);
        return dimensions[index];
    }

    Shape computeContiguousStrides() const noexcept;
    std::string toString() const;
    bool operator==(const Shape &other) const noexcept;

    std::span<const size_t> getDimensions() const noexcept { return {dimensions.data(), rank_size}; }

    size_t getTotalElements() const noexcept
    {
        if (rank_size == 0)
        {
            return 0;
        }
        size_t total = 1;
        for (size_t i = 0; i < rank_size; ++i)
        {
            total *= dimensions[i];
        }
        return total;
    }

    size_t getRank() const noexcept { return rank_size; }
    void setDimensions(std::span<const size_t> _dimension_span);
    void setRank(size_t _rank);
};

using Stride = Shape;