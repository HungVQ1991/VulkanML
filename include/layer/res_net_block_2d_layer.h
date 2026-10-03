#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class Res_Net_Block_2d_Layer : public ILayer
{
private:
    std::vector<std::unique_ptr<ILayer>> main_branch;
    std::vector<std::unique_ptr<ILayer>> shortcut_branch;
    std::unique_ptr<ILayer> post_activation;

    Tensor input_tensor;
    Tensor main_branch_output;
    Tensor shortcut_branch_output;
    Tensor sum_tensor;
    Tensor final_output;

    Tensor main_gradient;
    Tensor shortcut_gradient;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;    explicit Res_Net_Block_2d_Layer(Execution_Target _execution_target = Execution_Target::CPU);
    Res_Net_Block_2d_Layer(
        uint32_t _height,
        uint32_t _width,
        uint32_t _in_channels,
        uint32_t _out_channels,
        uint32_t _stride = 1,
        Execution_Target _execution_target = Execution_Target::CPU);


    ~Res_Net_Block_2d_Layer() noexcept override = default;

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addMainLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        main_branch.push_back(std::move(new_layer));
        return layer_reference;
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addShortcutLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        shortcut_branch.push_back(std::move(new_layer));
        return layer_reference;
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &setPostActivation(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        post_activation = std::move(new_layer);
        return layer_reference;
    }    void addMainLayer(std::unique_ptr<ILayer> _layer);
    void addShortcutLayer(std::unique_ptr<ILayer> _layer);
    void setPostActivation(std::unique_ptr<ILayer> _layer);
    std::unique_ptr<ILayer> clone() const override;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;

    std::vector<Shape> getPopulationParameterDims() const override
    {
        std::vector<Shape> total_dims;
        for (const auto& layer : main_branch)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        for (const auto& layer : shortcut_branch)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        if (post_activation)
        {
            auto dims = post_activation->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        return total_dims;
    }
    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        std::vector<bool> total_evolvable;
        for (const auto& layer : main_branch)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        for (const auto& layer : shortcut_branch)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        if (post_activation)
        {
            auto evolvable = post_activation->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        return total_evolvable;
    }    const std::vector<std::unique_ptr<ILayer>> &getShortcutBranch() const noexcept;
    const std::vector<std::unique_ptr<ILayer>> &getMainBranch() const noexcept;
    const Tensor &getShortcutOutput() const noexcept;

    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }    const Tensor &getShortcutGradient() const noexcept;
    const Tensor &getMainOutput() const noexcept;
    const Tensor &getMainGradient() const noexcept;
    const Tensor &getFinalOutput() const noexcept;
    const Tensor &getSumTensor() const noexcept;

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return final_output; }    const ILayer *getPostActivation() const noexcept;

    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::RES_NET_BLOCK_2D; }
    bool supportsPopulationBatch() const noexcept override
    {
        for (const auto& layer : main_branch)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        for (const auto& layer : shortcut_branch)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        if (post_activation && !post_activation->supportsPopulationBatch())
        {
            return false;
        }
        return true;
    }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;
    void setShortcutOutput(const Tensor &_tensor);

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }    void setShortcutGradient(const Tensor &_tensor);
    void setMainOutput(const Tensor &_tensor);
    void setMainGradient(const Tensor &_tensor);
    void setFinalOutput(const Tensor &_tensor);
    void setSumTensor(const Tensor &_tensor);

    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setIsForwardCompleted(bool _is_completed) noexcept;

};;