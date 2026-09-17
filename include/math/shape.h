#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

constexpr std::size_t MAX_TENSOR_RANK = 6;

class Shape
{
private:
    std::array<std::size_t, MAX_TENSOR_RANK> dimensions{};
    std::uint8_t rank_size = 0;

public:
    Shape() = default;

    Shape(std::initializer_list<std::size_t> dimension_list)
    {
        if (dimension_list.size() > MAX_TENSOR_RANK)
        {
            throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
        }
        rank_size = static_cast<std::uint8_t>(dimension_list.size());
        std::copy(dimension_list.begin(), dimension_list.end(), dimensions.begin());
    }

    Shape(std::span<const std::size_t> dimension_span)
    {
        if (dimension_span.size() > MAX_TENSOR_RANK)
        {
            throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
        }
        rank_size = static_cast<std::uint8_t>(dimension_span.size());
        std::copy(dimension_span.begin(), dimension_span.end(), dimensions.begin());
    }

    Shape(std::size_t dim_0, std::size_t dim_1)
    {
        rank_size = 2;
        dimensions[0] = dim_0;
        dimensions[1] = dim_1;
    }

    std::size_t operator[](std::size_t index) const
    {
        assert(index < MAX_TENSOR_RANK);
        return dimensions[index];
    }

    std::size_t &operator[](std::size_t index)
    {
        assert(index < MAX_TENSOR_RANK);
        return dimensions[index];
    }

    Shape computeContiguousStrides() const noexcept
    {
        Shape strides_result;
        strides_result.rank_size = rank_size;
        if (rank_size == 0)
        {
            return strides_result;
        }

        std::size_t current_stride = 1;
        for (std::size_t i = rank_size; i > 0; --i)
        {
            strides_result.dimensions[i - 1] = current_stride;
            current_stride *= dimensions[i - 1];
        }
        return strides_result;
    }

    std::string toString() const
    {
        if (rank_size == 0)
        {
            return "()";
        }
        std::string result = "(";
        for (std::size_t i = 0; i < rank_size; ++i)
        {
            result += std::format("{}{}", dimensions[i], (i + 1 < rank_size) ? ", " : "");
        }
        result += ")";
        return result;
    }

    bool operator==(const Shape &other) const noexcept
    {
        if (rank_size != other.rank_size)
        {
            return false;
        }
        for (std::size_t i = 0; i < rank_size; ++i)
        {
            if (dimensions[i] != other.dimensions[i])
            {
                return false;
            }
        }
        return true;
    }

    std::span<const std::size_t> getDimensions() const noexcept { return {dimensions.data(), rank_size}; }

    std::size_t getTotalElements() const noexcept
    {
        if (rank_size == 0)
        {
            return 0;
        }
        std::size_t total = 1;
        for (std::size_t i = 0; i < rank_size; ++i)
        {
            total *= dimensions[i];
        }
        return total;
    }

    std::size_t getRank() const noexcept { return rank_size; }

    void setDimensions(std::span<const std::size_t> _dimension_span)
    {
        if (_dimension_span.size() > MAX_TENSOR_RANK)
        {
            throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
        }
        rank_size = static_cast<std::uint8_t>(_dimension_span.size());
        std::copy(_dimension_span.begin(), _dimension_span.end(), dimensions.begin());
    }

    void setRank(std::size_t _rank)
    {
        if (_rank > MAX_TENSOR_RANK)
        {
            throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
        }
        rank_size = static_cast<std::uint8_t>(_rank);
    }
};

using Stride = Shape;