#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class SwiGLU_Layer : public ILayer
{
private:
    Tensor input_tensor;
    Tensor tensor_a;
    Tensor tensor_b;

    Tensor output_tensor;

    Tensor grad_a;
    Tensor grad_b;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;    explicit SwiGLU_Layer(Execution_Target _execution_target = Execution_Target::CPU);


    ~SwiGLU_Layer() noexcept override = default;    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &a, const Tensor &b);
    Tensor backward(const Tensor &_output_gradient) override;
    std::unique_ptr<ILayer> clone() const override;
    void resetGradient() override;
    void saveConfiguration(std::ofstream &) const override;
    void saveInference(std::ofstream &) const override;
    void loadInference(std::ifstream &) override;
    void saveCheckpoint(std::ofstream &) const override;
    void loadCheckpoint(std::ifstream &) override;


    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }    const Tensor &getGradA() const noexcept;
    const Tensor &getGradB() const noexcept;

    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::SWIGLU; }    void setExecutionTarget(Execution_Target _new_target) override;

};;
