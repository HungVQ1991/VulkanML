#include "layer/ppo_actor_critic_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "math/tensor_impl.h"

PPO_Actor_Critic_Layer::PPO_Actor_Critic_Layer(uint64_t _actor_output_dimension, Execution_Target _execution_target)
    : actor_output_dimension(_actor_output_dimension),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          actor_output_tensor(0, 0, _execution_target),
          critic_output_tensor(0, 0, _execution_target),
          current_actor(0, 0, _execution_target),
          current_critic(0, 0, _execution_target),
          actor_gradient(0, 0, _execution_target),
          critic_gradient(0, 0, _execution_target),
          actor_input_gradient(0, 0, _execution_target),
          critic_input_gradient(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{
    }

void PPO_Actor_Critic_Layer::addActorLayer(std::unique_ptr<ILayer> _layer)
{
        if (!_layer)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::addActorLayer: Null layer pointer");
        }
        _layer->setExecutionTarget(execution_target);
        _layer->setAccumulated(is_accumulated);
        _layer->setMixedPrecision(is_mixed_precision_enabled);
        actor.push_back(std::move(_layer));
    }

void PPO_Actor_Critic_Layer::addCriticLayer(std::unique_ptr<ILayer> _layer)
{
        if (!_layer)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::addCriticLayer: Null layer pointer");
        }
        _layer->setExecutionTarget(execution_target);
        _layer->setAccumulated(is_accumulated);
        _layer->setMixedPrecision(is_mixed_precision_enabled);
        critic.push_back(std::move(_layer));
    }

Tensor PPO_Actor_Critic_Layer::forward(const Tensor &_input_tensor)
{
        if (actor_output_dimension == 0)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: actor_output_dimension must be > 0");
        }
        if (!_input_tensor.getImplementation())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Input tensor has null implementation");
        }
        if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: TERNARY_INT2 input is not supported");
        }
        if (_input_tensor.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Execution_Target mismatch");
        }

        input_tensor = _input_tensor;
        const Shape saved_in_shape = _input_tensor.getShape();

        current_actor = _input_tensor.view(saved_in_shape);
        for (auto &layer : actor)
        {
            if (layer)
            {
                current_actor = layer->forward(current_actor);
            }
        }
        actor_output_tensor = current_actor;

        current_critic = _input_tensor.view(saved_in_shape);
        for (auto &layer : critic)
        {
            if (layer)
            {
                current_critic = layer->forward(current_critic);
            }
        }
        critic_output_tensor = current_critic;

        if (!actor_output_tensor.getImplementation() || !critic_output_tensor.getImplementation())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Branch output has null implementation");
        }
        if (actor_output_tensor.getExecutionTarget() != execution_target ||
            critic_output_tensor.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Branch output Execution_Target mismatch");
        }
        if (actor_output_tensor.getDataType() == Data_Type::TERNARY_INT2 ||
            critic_output_tensor.getDataType() == Data_Type::TERNARY_INT2 ||
            actor_output_tensor.getDataType() != critic_output_tensor.getDataType())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Actor and critic outputs must have matching non-ternary Data_Type");
        }

        const Shape &sa = actor_output_tensor.getShape();
        const Shape &sc = critic_output_tensor.getShape();
        const size_t rank = sa.getRank();
        if (rank < 2 || sc.getRank() != rank)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Actor and critic outputs must have matching rank >= 2");
        }
        size_t outer_rows = 1;
        std::vector<size_t> combined_dims(rank);
        for (size_t d = 0; d + 1 < rank; ++d)
        {
            if (sa[d] != sc[d])
            {
                throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Actor and critic leading dimensions mismatch");
            }
            combined_dims[d] = sa[d];
            outer_rows *= sa[d];
        }
        const size_t c_a = sa[rank - 1];
        const size_t c_c = sc[rank - 1];
        if (c_a != actor_output_dimension || c_c == 0)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: actor_output_dimension or critic output dimension mismatch");
        }
        combined_dims[rank - 1] = c_a + c_c;

        if (rank == 2)
        {
            actor_output_tensor.concatenateColumns(critic_output_tensor, output_tensor);
        }
        else
        {
            Tensor flat_actor = actor_output_tensor.view(Shape{ outer_rows, c_a });
            Tensor flat_critic = critic_output_tensor.view(Shape{ outer_rows, c_c });
            flat_actor.concatenateColumns(flat_critic, output_tensor);
            output_tensor.reshape(Shape(combined_dims));
        }

        is_forward_completed = true;
        return output_tensor;
    }

Tensor PPO_Actor_Critic_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (!_batched_input.getImplementation())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched input tensor has null implementation");
        }
        if (_batched_input.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: TERNARY_INT2 batched input is not supported");
        }
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "PPO_Actor_Critic_Layer::forward: expected {} batched parameter tensors, got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        const Shape saved_in_shape = _batched_input.getShape();
        size_t param_offset = 0;
        Tensor running_actor = _batched_input.view(saved_in_shape);
        for (const auto& layer : actor)
        {
            if (!layer) continue;
            size_t param_count = layer->getPopulationParameterDims().size();
            std::vector<Tensor> sub_params(
                _batched_params.begin() + param_offset,
                _batched_params.begin() + param_offset + param_count);
            running_actor = layer->forward(running_actor, sub_params);
            param_offset += param_count;
        }

        Tensor running_critic = _batched_input.view(saved_in_shape);
        for (const auto& layer : critic)
        {
            if (!layer) continue;
            size_t param_count = layer->getPopulationParameterDims().size();
            std::vector<Tensor> sub_params(
                _batched_params.begin() + param_offset,
                _batched_params.begin() + param_offset + param_count);
            running_critic = layer->forward(running_critic, sub_params);
            param_offset += param_count;
        }

        if (!running_actor.getImplementation() || !running_critic.getImplementation())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched branch output has null implementation");
        }
        const Execution_Target target = _batched_input.getExecutionTarget();
        if (running_actor.getExecutionTarget() != target || running_critic.getExecutionTarget() != target)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched branch Execution_Target mismatch");
        }
        if (running_actor.getDataType() == Data_Type::TERNARY_INT2 ||
            running_critic.getDataType() == Data_Type::TERNARY_INT2 ||
            running_actor.getDataType() != running_critic.getDataType())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched branch Data_Type mismatch");
        }

        const Shape &sa = running_actor.getShape();
        const Shape &sc = running_critic.getShape();
        const size_t rank = sa.getRank();
        if (rank < 2 || sc.getRank() != rank)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched branch outputs must have matching rank >= 2");
        }
        size_t outer_rows = 1;
        std::vector<size_t> combined_dims(rank);
        for (size_t d = 0; d + 1 < rank; ++d)
        {
            if (sa[d] != sc[d])
            {
                throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched branch leading dimensions mismatch");
            }
            combined_dims[d] = sa[d];
            outer_rows *= sa[d];
        }
        const size_t c_a = sa[rank - 1];
        const size_t c_c = sc[rank - 1];
        if ((actor_output_dimension > 0 && c_a != actor_output_dimension) || c_c == 0)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::forward: Batched actor_output_dimension or critic output dimension mismatch");
        }
        combined_dims[rank - 1] = c_a + c_c;

        Tensor combined_output(target);
        if (rank == 2)
        {
            running_actor.concatenateColumns(running_critic, combined_output);
        }
        else
        {
            Tensor flat_actor = running_actor.view(Shape{ outer_rows, c_a });
            Tensor flat_critic = running_critic.view(Shape{ outer_rows, c_c });
            flat_actor.concatenateColumns(flat_critic, combined_output);
            combined_output.reshape(Shape(combined_dims));
        }
        return combined_output;
    }

std::unique_ptr<ILayer> PPO_Actor_Critic_Layer::clone() const
{
        auto cloned_layer = std::make_unique<PPO_Actor_Critic_Layer>(actor_output_dimension, execution_target);
        cloned_layer->setAccumulated(is_accumulated);
        cloned_layer->setMixedPrecision(is_mixed_precision_enabled);
        for (const auto& layer : actor)
        {
            if (layer) cloned_layer->addActorLayer(layer->clone());
        }
        for (const auto& layer : critic)
        {
            if (layer) cloned_layer->addCriticLayer(layer->clone());
        }
        cloned_layer->setAccumulated(is_accumulated);
        cloned_layer->setMixedPrecision(is_mixed_precision_enabled);
        return cloned_layer;
    }

Tensor PPO_Actor_Critic_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"PPO_Actor_Critic_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }
        if (actor_output_dimension == 0)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::backward: actor_output_dimension must be > 0");
        }
        if (!_output_gradient.getImplementation())
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::backward: Output gradient has null implementation");
        }
        if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::backward: TERNARY_INT2 gradient is not supported");
        }
        if (_output_gradient.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("PPO_Actor_Critic_Layer::backward: Execution_Target mismatch");
        }
        if (_output_gradient.getShape() != output_tensor.getShape())
        {
            Logger::logMessage(Input_Format{"PPO_Actor_Critic_Layer::backward: Output gradient dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Output gradient dimension mismatch");
        }

        const Shape &out_shape = output_tensor.getShape();
        const size_t rank = out_shape.getRank();
        if (rank == 2)
        {
            _output_gradient.splitColumns(static_cast<size_t>(actor_output_dimension), actor_gradient, critic_gradient);
        }
        else
        {
            size_t outer_rows = 1;
            for (size_t d = 0; d + 1 < rank; ++d)
            {
                outer_rows *= out_shape[d];
            }
            const size_t c_total = out_shape[rank - 1];
            Tensor flat_grad = _output_gradient.view(Shape{ outer_rows, c_total });
            flat_grad.splitColumns(static_cast<size_t>(actor_output_dimension), actor_gradient, critic_gradient);
            actor_gradient.reshape(actor_output_tensor.getShape());
            critic_gradient.reshape(critic_output_tensor.getShape());
        }

        Tensor current_actor_gradient = actor_gradient;
        for (auto it = actor.rbegin(); it != actor.rend(); ++it)
        {
            if (*it)
            {
                current_actor_gradient = (*it)->backward(current_actor_gradient);
            }
        }
        actor_input_gradient = current_actor_gradient;

        Tensor current_critic_gradient = critic_gradient;
        for (auto it = critic.rbegin(); it != critic.rend(); ++it)
        {
            if (*it)
            {
                current_critic_gradient = (*it)->backward(current_critic_gradient);
            }
        }
        critic_input_gradient = current_critic_gradient;

        actor_input_gradient.add(critic_input_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

void PPO_Actor_Critic_Layer::resetGradient()
{
        is_forward_completed = false;
        for (auto &layer : actor)
        {
            if (layer) layer->resetGradient();
        }
        for (auto &layer : critic)
        {
            if (layer) layer->resetGradient();
        }
    }

void PPO_Actor_Critic_Layer::resetGradients()
{
        is_forward_completed = false;
        for (auto &layer : actor)
        {
            if (layer) layer->resetGradients();
        }
        for (auto &layer : critic)
        {
            if (layer) layer->resetGradients();
        }
    }

void PPO_Actor_Critic_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::saveConfiguration: Output stream is not open or writable");
        }
        _output_file_stream.write(reinterpret_cast<const char *>(&actor_output_dimension), sizeof(actor_output_dimension));
        uint64_t actor_count = 0;
        for (const auto &layer : actor) { if (layer) ++actor_count; }
        uint64_t critic_count = 0;
        for (const auto &layer : critic) { if (layer) ++critic_count; }
        _output_file_stream.write(reinterpret_cast<const char *>(&actor_count), sizeof(actor_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&critic_count), sizeof(critic_count));

        for (const auto &layer : actor)
        {
            if (!layer) continue;
            Layer_Type type = layer->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
            layer->saveConfiguration(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            if (!layer) continue;
            Layer_Type type = layer->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
            layer->saveConfiguration(_output_file_stream);
        }
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::saveConfiguration: Failed writing configuration");
        }
    }

void PPO_Actor_Critic_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::saveInference: Output stream is not open or writable");
        }
        for (const auto &layer : actor)
        {
            if (layer) layer->saveInference(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            if (layer) layer->saveInference(_output_file_stream);
        }
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::saveInference: Failed writing inference state");
        }
    }

void PPO_Actor_Critic_Layer::loadInference(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::loadInference: Input stream is not open or readable");
        }
        for (auto &layer : actor)
        {
            if (layer) layer->loadInference(_input_file_stream);
        }
        for (auto &layer : critic)
        {
            if (layer) layer->loadInference(_input_file_stream);
        }
        if (!_input_file_stream)
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::loadInference: Failed reading inference state");
        }
    }

void PPO_Actor_Critic_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::saveCheckpoint: Output stream is not open or writable");
        }
        for (const auto &layer : actor)
        {
            if (layer) layer->saveCheckpoint(_output_file_stream);
        }
        for (const auto &layer : critic)
        {
            if (layer) layer->saveCheckpoint(_output_file_stream);
        }
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::saveCheckpoint: Failed writing checkpoint state");
        }
    }

void PPO_Actor_Critic_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::loadCheckpoint: Input stream is not open or readable");
        }
        for (auto &layer : actor)
        {
            if (layer) layer->loadCheckpoint(_input_file_stream);
        }
        for (auto &layer : critic)
        {
            if (layer) layer->loadCheckpoint(_input_file_stream);
        }
        if (!_input_file_stream)
        {
            throw std::runtime_error("PPO_Actor_Critic_Layer::loadCheckpoint: Failed reading checkpoint state");
        }
    }

std::function<float(std::mt19937&)> PPO_Actor_Critic_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto& layer : actor)
        {
            if (!layer) continue;
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto& layer : critic)
        {
            if (!layer) continue;
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }

std::vector<float> PPO_Actor_Critic_Layer::getPopulationParameter(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto &layer : actor)
        {
            if (!layer) continue;
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto &layer : critic)
        {
            if (!layer) continue;
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::getPopulationParameter: Parameter index out of range");
    }

std::vector<std::pair<Tensor *, Tensor *>> PPO_Actor_Critic_Layer::getParametersAndGradients()
{
        std::vector<std::pair<Tensor *, Tensor *>> all_parameters;
        for (auto &layer : actor)
        {
            if (!layer) continue;
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        for (auto &layer : critic)
        {
            if (!layer) continue;
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        return all_parameters;
    }

std::vector<const Tensor *> PPO_Actor_Critic_Layer::getNoDecayParameters() const
{
        std::vector<const Tensor *> params;
        for (const auto &layer : actor)
        {
            if (!layer) continue;
            auto p = layer->getNoDecayParameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        for (const auto &layer : critic)
        {
            if (!layer) continue;
            auto p = layer->getNoDecayParameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

const std::vector<std::unique_ptr<ILayer>> & PPO_Actor_Critic_Layer::getActor() const noexcept
{ return actor; }

const std::vector<std::unique_ptr<ILayer>> & PPO_Actor_Critic_Layer::getCritic() const noexcept
{ return critic; }

const Tensor & PPO_Actor_Critic_Layer::getCriticInputGradient() const noexcept
{ return critic_input_gradient; }

const Tensor & PPO_Actor_Critic_Layer::getActorInputGradient() const noexcept
{ return actor_input_gradient; }

const Tensor & PPO_Actor_Critic_Layer::getCriticOutput() const noexcept
{ return critic_output_tensor; }

const Tensor & PPO_Actor_Critic_Layer::getActorOutput() const noexcept
{ return actor_output_tensor; }

const Tensor & PPO_Actor_Critic_Layer::getCurrentCritic() const noexcept
{ return current_critic; }

const Tensor & PPO_Actor_Critic_Layer::getCurrentActor() const noexcept
{ return current_actor; }

const Tensor & PPO_Actor_Critic_Layer::getCriticGradient() const noexcept
{ return critic_gradient; }

const Tensor & PPO_Actor_Critic_Layer::getActorGradient() const noexcept
{ return actor_gradient; }

uint64_t PPO_Actor_Critic_Layer::getActorOutputDimension() const noexcept
{ return actor_output_dimension; }

void PPO_Actor_Critic_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        size_t current_offset = 0;
        for (auto& layer : actor)
        {
            if (!layer) continue;
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        for (auto& layer : critic)
        {
            if (!layer) continue;
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("PPO_Actor_Critic_Layer::setPopulationParameter: Parameter index out of range");
    }

void PPO_Actor_Critic_Layer::setCriticInputGradient(const Tensor &_tensor)
{ critic_input_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setActorInputGradient(const Tensor &_tensor)
{ actor_input_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setCriticGradient(const Tensor &_tensor)
{ critic_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setActorGradient(const Tensor &_tensor)
{ actor_gradient = _tensor; }

void PPO_Actor_Critic_Layer::setCriticOutput(const Tensor &_tensor)
{ critic_output_tensor = _tensor; }

void PPO_Actor_Critic_Layer::setActorOutput(const Tensor &_tensor)
{ actor_output_tensor = _tensor; }

void PPO_Actor_Critic_Layer::setActorOutputDimension(uint64_t _dimension) noexcept
{ actor_output_dimension = _dimension; }

void PPO_Actor_Critic_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
        for (auto &layer : actor)
        {
            if (layer) layer->setExecutionTarget(_new_execution_target);
        }
        for (auto &layer : critic)
        {
            if (layer) layer->setExecutionTarget(_new_execution_target);
        }

        if (_new_execution_target == execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);
        execution_target = _new_execution_target;

        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        actor_output_tensor.setExecutionTarget(_new_execution_target);
        critic_output_tensor.setExecutionTarget(_new_execution_target);
        current_actor.setExecutionTarget(_new_execution_target);
        current_critic.setExecutionTarget(_new_execution_target);
        actor_gradient.setExecutionTarget(_new_execution_target);
        critic_gradient.setExecutionTarget(_new_execution_target);
        actor_input_gradient.setExecutionTarget(_new_execution_target);
        critic_input_gradient.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }

void PPO_Actor_Critic_Layer::setAccumulated(bool _is_accumulated) noexcept
{
        is_accumulated = _is_accumulated;
        for (auto &layer : actor)
        {
            if (layer) layer->setAccumulated(_is_accumulated);
        }
        for (auto &layer : critic)
        {
            if (layer) layer->setAccumulated(_is_accumulated);
        }
    }

void PPO_Actor_Critic_Layer::setMixedPrecision(bool _enable) noexcept
{
        ILayer::setMixedPrecision(_enable);
        for (auto &layer : actor)
        {
            if (layer) layer->setMixedPrecision(_enable);
        }
        for (auto &layer : critic)
        {
            if (layer) layer->setMixedPrecision(_enable);
        }
    }

void PPO_Actor_Critic_Layer::invalidateWeightCache() noexcept
{
        for (auto &layer : actor)
        {
            if (layer) layer->invalidateWeightCache();
        }
        for (auto &layer : critic)
        {
            if (layer) layer->invalidateWeightCache();
        }
    }

void PPO_Actor_Critic_Layer::setTrainingMode(bool _is_training)
{
        for (auto &layer : actor)
        {
            if (layer) layer->setTrainingMode(_is_training);
        }
        for (auto &layer : critic)
        {
            if (layer) layer->setTrainingMode(_is_training);
        }
    }

void PPO_Actor_Critic_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

void PPO_Actor_Critic_Layer::prepareInference()
{
    for (auto &layer : actor) { if (layer) layer->prepareInference(); }
    for (auto &layer : critic) { if (layer) layer->prepareInference(); }
}

void PPO_Actor_Critic_Layer::quantizeToTernary()
{
    for (auto &layer : actor)
    {
        if (layer) layer->quantizeToTernary();
    }
    for (auto &layer : critic)
    {
        if (layer) layer->quantizeToTernary();
    }
}

void PPO_Actor_Critic_Layer::dequantizeFromTernary()
{
    for (auto &layer : actor)
    {
        if (layer) layer->dequantizeFromTernary();
    }
    for (auto &layer : critic)
    {
        if (layer) layer->dequantizeFromTernary();
    }
}

void PPO_Actor_Critic_Layer::snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const
{
    for (const auto &layer : actor)
    {
        if (layer) layer->snapshotNonTrainableState(_out);
    }
    for (const auto &layer : critic)
    {
        if (layer) layer->snapshotNonTrainableState(_out);
    }
}

void PPO_Actor_Critic_Layer::restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor)
{
    for (auto &layer : actor)
    {
        if (layer) layer->restoreNonTrainableState(_in, _cursor);
    }
    for (auto &layer : critic)
    {
        if (layer) layer->restoreNonTrainableState(_in, _cursor);
    }
}

bool PPO_Actor_Critic_Layer::isTernary() const noexcept
{
    for (const auto &layer : actor)
    {
        if (layer && layer->isTernary()) return true;
    }
    for (const auto &layer : critic)
    {
        if (layer && layer->isTernary()) return true;
    }
    return false;
}
