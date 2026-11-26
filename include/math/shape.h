#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

constexpr size_t MAX_TENSOR_RANK = 6;

class Shape
{
private:
    std::array<size_t, MAX_TENSOR_RANK> dimensions{};
    std::uint8_t rank_size = 0;

    static void validateTotalElementsNoOverflow(std::span<const size_t> dims)
    {
        size_t acc = 1;
        for (size_t d : dims)
        {
            if (d == 0)
            {
                return;
            }
            if (acc > std::numeric_limits<size_t>::max() / d)
            {
                throw std::overflow_error("Shape: Total elements overflow size_t");
            }
            acc *= d;
        }
    }

public:
    Shape() = default;
    Shape(std::initializer_list<size_t> dimension_list);
    Shape(std::span<const size_t> dimension_span);
    Shape(size_t dim_0, size_t dim_1)
    {
        std::array<size_t, 2> dims{dim_0, dim_1};
        validateTotalElementsNoOverflow(dims);
        rank_size = 2;
        dimensions[0] = dim_0;
        dimensions[1] = dim_1;
    }

    size_t operator[](size_t index) const
    {
        if (index >= rank_size)
        {
            throw std::out_of_range("Shape::operator[]: Index out of rank bounds");
        }
        return dimensions[index];
    }

    size_t &operator[](size_t index)
    {
        if (index >= rank_size)
        {
            throw std::out_of_range("Shape::operator[]: Index out of rank bounds");
        }
        return dimensions[index];
    }

    size_t at(size_t index) const
    {
        if (index >= rank_size)
        {
            throw std::out_of_range("Shape::at: Index out of rank bounds");
        }
        return dimensions[index];
    }

    size_t &at(size_t index)
    {
        if (index >= rank_size)
        {
            throw std::out_of_range("Shape::at: Index out of rank bounds");
        }
        return dimensions[index];
    }

    Shape computeContiguousStrides() const;
    std::string toString() const;
    bool operator==(const Shape &other) const noexcept;

    std::span<const size_t> getDimensions() const noexcept { return {dimensions.data(), rank_size}; }

    size_t getTotalElements() const
    {
        if (rank_size == 0)
        {
            return 0;
        }
        size_t total = 1;
        for (size_t i = 0; i < rank_size; ++i)
        {
            if (dimensions[i] == 0)
            {
                return 0;
            }
            if (total > std::numeric_limits<size_t>::max() / dimensions[i])
            {
                throw std::overflow_error("Shape::getTotalElements: size_t overflow");
            }
            total *= dimensions[i];
        }
        return total;
    }

    size_t getRank() const noexcept { return rank_size; }
    void setDimensions(std::span<const size_t> _dimension_span);
    void setRank(size_t _rank);
};

using Stride = Shape;
