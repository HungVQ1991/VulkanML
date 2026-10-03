#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "icost_function.h"
#include "math/tensor.h"

class Huber_Cost : public ICost_Function
{
private:
    float delta = 1.0f;
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;

public:    explicit Huber_Cost(float _delta = 1.0f, Execution_Target _execution_target = Execution_Target::CPU);


    ~Huber_Cost() noexcept override = default;    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;


    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&delta), sizeof(delta));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&delta), sizeof(delta));
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::HUBER; }
    float getDelta() const noexcept { return delta; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix) { loss_matrix = _matrix; }
    void setDelta(float _delta) noexcept { delta = _delta; }
};;