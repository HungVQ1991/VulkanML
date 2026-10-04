#include "math/shape.h"

#include <algorithm>
#include <format>
#include <stdexcept>

Shape::Shape(std::initializer_list<size_t> dimension_list)
{
    if (dimension_list.size() > MAX_TENSOR_RANK)
    {
        throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
    }
    rank_size = static_cast<std::uint8_t>(dimension_list.size());
    std::copy(dimension_list.begin(), dimension_list.end(), dimensions.begin());
}

Shape::Shape(std::span<const size_t> dimension_span)
{
    if (dimension_span.size() > MAX_TENSOR_RANK)
    {
        throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
    }
    rank_size = static_cast<std::uint8_t>(dimension_span.size());
    std::copy(dimension_span.begin(), dimension_span.end(), dimensions.begin());
}

Shape Shape::computeContiguousStrides() const noexcept
{
    Shape strides_result;
    strides_result.rank_size = rank_size;
    if (rank_size == 0)
    {
        return strides_result;
    }

    size_t current_stride = 1;
    for (size_t i = rank_size; i > 0; --i)
    {
        strides_result.dimensions[i - 1] = current_stride;
        current_stride *= dimensions[i - 1];
    }
    return strides_result;
}

std::string Shape::toString() const
{
    if (rank_size == 0)
    {
        return "()";
    }
    std::string result = "(";
    for (size_t i = 0; i < rank_size; ++i)
    {
        result += std::format("{}{}", dimensions[i], (i + 1 < rank_size) ? ", " : "");
    }
    result += ")";
    return result;
}

bool Shape::operator==(const Shape &other) const noexcept
{
    if (rank_size != other.rank_size)
    {
        return false;
    }
    for (size_t i = 0; i < rank_size; ++i)
    {
        if (dimensions[i] != other.dimensions[i])
        {
            return false;
        }
    }
    return true;
}

void Shape::setDimensions(std::span<const size_t> _dimension_span)
{
    if (_dimension_span.size() > MAX_TENSOR_RANK)
    {
        throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
    }
    rank_size = static_cast<std::uint8_t>(_dimension_span.size());
    std::copy(_dimension_span.begin(), _dimension_span.end(), dimensions.begin());
}

void Shape::setRank(size_t _rank)
{
    if (_rank > MAX_TENSOR_RANK)
    {
        throw std::invalid_argument("Shape: Rank exceeds maximum supported rank");
    }
    rank_size = static_cast<std::uint8_t>(_rank);
}
