#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class PPO_Actor_Critic_Layer : public ILayer
{
private:
    std::vector<std::unique_ptr<ILayer>> actor;
    std::vector<std::unique_ptr<ILayer>> critic;

    uint64_t actor_output_dimension = 0;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor actor_output_tensor;
    Tensor critic_output_tensor;
    Tensor current_actor;
    Tensor current_critic;
    Tensor actor_gradient;
    Tensor critic_gradient;
    Tensor actor_input_gradient;
    Tensor critic_input_gradient;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;    PPO_Actor_Critic_Layer(uint64_t _actor_output_dimension, Execution_Target _execution_target = Execution_Target::CPU);


    ~PPO_Actor_Critic_Layer() noexcept override = default;

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addActorLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        actor.push_back(std::move(new_layer));
        return layer_reference;
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addCriticLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        critic.push_back(std::move(new_layer));
        return layer_reference;
    }    void addActorLayer(std::unique_ptr<ILayer> _layer);
    void addCriticLayer(std::unique_ptr<ILayer> _layer);
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override;
    std::unique_ptr<ILayer> clone() const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        std::vector<bool> total_evolvable;
        for (const auto& layer : actor)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        for (const auto& layer : critic)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        return total_evolvable;
    }
    std::vector<Shape> getPopulationParameterDims() const override
    {
        std::vector<Shape> total_dims;
        for (const auto& layer : actor)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        for (const auto& layer : critic)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        return total_dims;
    }    const std::vector<std::unique_ptr<ILayer>> &getActor() const noexcept;
    const std::vector<std::unique_ptr<ILayer>> &getCritic() const noexcept;
    const Tensor &getCriticInputGradient() const noexcept;
    const Tensor &getActorInputGradient() const noexcept;

    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }    const Tensor &getCriticOutput() const noexcept;
    const Tensor &getActorOutput() const noexcept;
    const Tensor &getCurrentCritic() const noexcept;
    const Tensor &getCurrentActor() const noexcept;
    const Tensor &getCriticGradient() const noexcept;
    const Tensor &getActorGradient() const noexcept;

    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getInput() const override { return input_tensor; }    uint64_t getActorOutputDimension() const noexcept;

    Layer_Type getLayerType() const noexcept override { return Layer_Type::PPO_ACTOR_CRITIC; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    bool supportsPopulationBatch() const noexcept override
    {
        for (const auto& layer : actor)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        for (const auto& layer : critic)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        return true;
    }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;
    void setCriticInputGradient(const Tensor &_tensor);
    void setActorInputGradient(const Tensor &_tensor);

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }    void setCriticGradient(const Tensor &_tensor);
    void setActorGradient(const Tensor &_tensor);
    void setCriticOutput(const Tensor &_tensor);
    void setActorOutput(const Tensor &_tensor);

    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }    void setActorOutputDimension(uint64_t _dimension) noexcept;
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setIsForwardCompleted(bool _is_completed) noexcept;

};;