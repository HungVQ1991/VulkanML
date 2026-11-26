#pragma once

#include <memory>
#include "layer/linear_layer.h"

class Ternary_Linear_Layer : public Linear_Layer
{
public:
    Ternary_Linear_Layer()
        : Linear_Layer(0, 0, Execution_Target::CPU, 2.0f, Data_Type::TERNARY_INT2)
    {
    }

    Ternary_Linear_Layer(size_t _input_dimension,
                         size_t _output_dimension,
                         Execution_Target _execution_target = Execution_Target::CPU,
                         float _initialization_gain = 2.0f)
        : Linear_Layer(_input_dimension, _output_dimension, _execution_target, _initialization_gain, Data_Type::TERNARY_INT2)
    {
    }

    ~Ternary_Linear_Layer() noexcept override = default;

    std::unique_ptr<ILayer> clone() const override
    {
        auto cloned = std::make_unique<Ternary_Linear_Layer>(getInputDimension(), getOutputDimension(), getExecutionTarget(), getInitializationGain());
        copyStateTo(*cloned);
        return cloned;
    }
};
