#include "optimizer/sgd_optimizer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "engine/vulkan_context.h"
#include "helper/logger.h"
#include "math/gpu_tensor_impl.h"
#include "ops/optimizer/sgd_step.h"

struct Sgd_Optimizer::Backend_State
{
    std::array<std::shared_ptr<gpu::vector>, MAX_FRAMES_IN_FLIGHT> dynamic_params_buffers{};
    std::array<bool, MAX_FRAMES_IN_FLIGHT> frame_overflow_pending_check{false, false};
};

void Sgd_Optimizer::validateHyperparameters(float _lr, float _max_grad, float _wd, float _momentum)
{
    if (!std::isfinite(_lr) || _lr < 0.0f)
    {
        throw std::invalid_argument("Sgd_Optimizer: learning_rate must be finite and >= 0");
    }
    if (!std::isfinite(_max_grad) || _max_grad < 0.0f)
    {
        throw std::invalid_argument("Sgd_Optimizer: max_gradient must be finite and >= 0");
    }
    if (!std::isfinite(_wd) || _wd < 0.0f)
    {
        throw std::invalid_argument("Sgd_Optimizer: weight_decay must be finite and >= 0");
    }
    if (!std::isfinite(_momentum) || _momentum < 0.0f || _momentum >= 1.0f)
    {
        throw std::invalid_argument("Sgd_Optimizer: momentum must be finite and in [0, 1)");
    }
}

Tensor Sgd_Optimizer::createZeroVelocityTensor(const Shape &_shape, Execution_Target _target)
{
    Tensor v(_shape, Data_Type::FLOAT32, _target);
    size_t total = _shape.getTotalElements();
    if (_target == Execution_Target::VULKAN_GPU && total > 0)
    {
        auto &ctx = Execution_Engine::getInstance().getContext();
        auto &g = Gpu_Tensor_Impl::castToGpu(*v.getImplementation());
        if (g.getStorageBuffer() && g.getStorageBuffer()->getBuffer() != VK_NULL_HANDLE)
        {
            g.getStorageBuffer()->markAsUsedInFrame(ctx.getCurrentFrame());
            ctx.fillBuffer(g.getStorageBuffer()->getBuffer(), total * sizeof(float), 0, 0);
        }
        g.clearHostCache();
        g.invalidateFp16Cache();
    }
    return v;
}

Sgd_Optimizer::Sgd_Optimizer(float _learning_rate, float _max_gradient, float _weight_decay, float _momentum)
    : learning_rate(_learning_rate),
      max_gradient(_max_gradient),
      weight_decay(_weight_decay),
      momentum(_momentum),
      learning_rate_scheduler(nullptr),
      backend_state(std::make_unique<Backend_State>())
{
    validateHyperparameters(learning_rate, max_gradient, weight_decay, momentum);
    if (learning_rate <= 0.0f)
    {
        Logger::logMessage("Sgd_Optimizer::Sgd_Optimizer: Initial learning rate is non-positive",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }
}

Sgd_Optimizer::Sgd_Optimizer(ILearning_Rate &_learning_rate_scheduler, float _max_gradient, float _weight_decay, float _momentum)
    : learning_rate(_learning_rate_scheduler.getCurrentRate()),
      max_gradient(_max_gradient),
      weight_decay(_weight_decay),
      momentum(_momentum),
      learning_rate_scheduler(&_learning_rate_scheduler),
      scheduler_lifetime_token(_learning_rate_scheduler.getLifetimeToken()),
      backend_state(std::make_unique<Backend_State>())
{
    validateHyperparameters(learning_rate, max_gradient, weight_decay, momentum);
    if (learning_rate <= 0.0f)
    {
        Logger::logMessage("Sgd_Optimizer::Sgd_Optimizer: Initial learning rate is non-positive",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }
}

Sgd_Optimizer::Sgd_Optimizer(Sgd_Optimizer &&other) noexcept = default;
Sgd_Optimizer &Sgd_Optimizer::operator=(Sgd_Optimizer &&other) noexcept = default;
Sgd_Optimizer::~Sgd_Optimizer() noexcept = default;

void Sgd_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs)
{
    step(_parameter_gradient_pairs, 1.0f);
}

void Sgd_Optimizer::latchCompletedGpuOverflows() noexcept
{
    if (!backend_state)
    {
        return;
    }
    try
    {
        auto &ctx = Execution_Engine::getInstance().getContext();
        VkDevice device = ctx.getDevice();
        for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
        {
            if (!backend_state->frame_overflow_pending_check[f] ||
                !backend_state->dynamic_params_buffers[f] ||
                !backend_state->dynamic_params_buffers[f]->isHostMapped())
            {
                continue;
            }
            bool frame_done = ctx.isFrameReady(f);
            if (!frame_done && device != VK_NULL_HANDLE)
            {
                VkFence fence = ctx.getFrameFence(f);
                if (fence != VK_NULL_HANDLE && vkGetFenceStatus(device, fence) == VK_SUCCESS)
                {
                    frame_done = true;
                }
            }
            if (!frame_done)
            {
                continue;
            }
            Graph_Executor::Adam_Dynamic_Params snapshot{};
            std::memcpy(&snapshot, backend_state->dynamic_params_buffers[f]->getHostMappedPointer(), sizeof(snapshot));
            if (snapshot.overflow_detected != 0u)
            {
                pending_gpu_overflow = true;
                snapshot.overflow_detected = 0u;
                backend_state->dynamic_params_buffers[f]->writeHostDirect(&snapshot, sizeof(snapshot));
            }
            backend_state->frame_overflow_pending_check[f] = false;
        }
    }
    catch (...)
    {
    }
}

bool Sgd_Optimizer::consumeGpuOverflowDetected() noexcept
{
    latchCompletedGpuOverflows();
    bool detected = pending_gpu_overflow;
    pending_gpu_overflow = false;
    return detected;
}

void Sgd_Optimizer::stepDynamicParams(float _grad_scale)
{
    if (!std::isfinite(_grad_scale) || _grad_scale <= 0.0f)
    {
        throw std::invalid_argument("Sgd_Optimizer::stepDynamicParams: grad_scale must be finite and > 0");
    }

    if (learning_rate_scheduler != nullptr)
    {
        if (scheduler_lifetime_token.expired())
        {
            learning_rate_scheduler = nullptr;
        }
        else
        {
            learning_rate = learning_rate_scheduler->getCurrentRate();
        }
    }
    float inv_scale = 1.0f / _grad_scale;
    Execution_Engine &engine = Execution_Engine::getInstance();
    uint32_t current_frame = engine.getContext().getCurrentFrame();
    if (!engine.getContext().isFrameReady(current_frame))
    {
        engine.prepareCurrentFrame(Execution_Stage::OPTIMIZER);
        current_frame = engine.getContext().getCurrentFrame();
    }

    latchCompletedGpuOverflows();

    if (!backend_state)
    {
        backend_state = std::make_unique<Backend_State>();
    }
    if (current_frame < MAX_FRAMES_IN_FLIGHT)
    {
        if (!backend_state->dynamic_params_buffers[current_frame])
        {
            backend_state->dynamic_params_buffers[current_frame] = std::make_shared<gpu::vector>(engine.getContext());
            backend_state->dynamic_params_buffers[current_frame]->allocateHostVisible(256);
        }
        Graph_Executor::Adam_Dynamic_Params params{learning_rate, 1.0f, 1.0f, inv_scale, 0u};
        backend_state->dynamic_params_buffers[current_frame]->writeHostDirect(&params, sizeof(params));
        backend_state->frame_overflow_pending_check[current_frame] = true;
    }
}

void Sgd_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale)
{
    if (!std::isfinite(_grad_scale) || _grad_scale <= 0.0f)
    {
        throw std::invalid_argument("Sgd_Optimizer::step: grad_scale must be finite and > 0");
    }

    bool has_valid_pair = false;
    bool has_gpu_param = false;
    for (const auto &[parameter, gradient] : _parameter_gradient_pairs)
    {
        if (parameter && gradient)
        {
            has_valid_pair = true;
            if (parameter->getExecutionTarget() == Execution_Target::VULKAN_GPU)
            {
                has_gpu_param = true;
                break;
            }
        }
    }

    if (!has_valid_pair)
    {
        return;
    }

    std::shared_ptr<gpu::vector> active_dyn_buf = nullptr;
    if (has_gpu_param)
    {
        stepDynamicParams(_grad_scale);
        uint32_t current_frame = Execution_Engine::getInstance().getContext().getCurrentFrame();
        if (backend_state && current_frame < MAX_FRAMES_IN_FLIGHT)
        {
            active_dyn_buf = backend_state->dynamic_params_buffers[current_frame];
        }
    }
    else if (learning_rate_scheduler != nullptr)
    {
        if (scheduler_lifetime_token.expired())
        {
            learning_rate_scheduler = nullptr;
        }
        else
        {
            learning_rate = learning_rate_scheduler->getCurrentRate();
        }
    }

    Logger::logMessage(Input_Format{"Sgd_Optimizer::step: learning_rate={}, pairs_count={}",
                                    learning_rate,
                                    _parameter_gradient_pairs.size()},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::OPTIMIZER_STEP);

    float inv_scale = 1.0f / _grad_scale;

    for (const auto &[parameter, gradient] : _parameter_gradient_pairs)
    {
        if (!parameter || !gradient)
        {
            Logger::logMessage("Sgd_Optimizer::step: Null parameter or gradient pointer encountered",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
            continue;
        }

        Tensor *vel_ptr = nullptr;
        if (momentum > 0.0f)
        {
            auto it = velocity_states.find(parameter);
            if (it == velocity_states.end())
            {
                if (loaded_velocity_cursor < loaded_velocities.size())
                {
                    Tensor &loaded_v = loaded_velocities[loaded_velocity_cursor];
                    bool match = (loaded_v.getShape() == parameter->getShape()) ||
                                 (loaded_v.getTotalElements() == parameter->getTotalElements());
                    if (match)
                    {
                        if (loaded_v.getShape() != parameter->getShape())
                        {
                            loaded_v.reshape(parameter->getShape());
                        }
                        if (loaded_v.getExecutionTarget() != parameter->getExecutionTarget())
                        {
                            loaded_v.setExecutionTarget(parameter->getExecutionTarget());
                        }
                        auto [ins_it, ok] = velocity_states.emplace(parameter, std::move(loaded_v));
                        it = ins_it;
                    }
                    else
                    {
                        auto [ins_it, ok] = velocity_states.emplace(parameter, createZeroVelocityTensor(parameter->getShape(), parameter->getExecutionTarget()));
                        it = ins_it;
                    }

                    if (loaded_velocity_cursor < loaded_no_decay_flags.size() && loaded_no_decay_flags[loaded_velocity_cursor] != 0)
                    {
                        no_decay_parameters.insert(parameter);
                        no_decay_tokens[parameter] = parameter->getLifetimeToken();
                    }

                    ++loaded_velocity_cursor;
                    if (loaded_velocity_cursor >= loaded_velocities.size())
                    {
                        loaded_velocities.clear();
                        loaded_no_decay_flags.clear();
                        loaded_velocity_cursor = 0;
                    }
                }
                else
                {
                    auto [ins_it, ok] = velocity_states.emplace(parameter, createZeroVelocityTensor(parameter->getShape(), parameter->getExecutionTarget()));
                    it = ins_it;
                }
                velocity_tokens[parameter] = parameter->getLifetimeToken();
                velocity_order.push_back(parameter);
            }
            else
            {
                auto tok_it = velocity_tokens.find(parameter);
                bool token_valid = (tok_it == velocity_tokens.end()) ||
                                   (!tok_it->second.expired() && tok_it->second.lock() == parameter->getLifetimeToken().lock());
                if (!token_valid || it->second.getShape() != parameter->getShape())
                {
                    it->second = createZeroVelocityTensor(parameter->getShape(), parameter->getExecutionTarget());
                    velocity_tokens[parameter] = parameter->getLifetimeToken();
                }
                else if (it->second.getExecutionTarget() != parameter->getExecutionTarget())
                {
                    it->second.setExecutionTarget(parameter->getExecutionTarget());
                }
            }
            vel_ptr = &it->second;
        }

        float eff_weight_decay = weight_decay;
        if (!no_decay_parameters.empty() && no_decay_parameters.contains(parameter))
        {
            auto nd_it = no_decay_tokens.find(parameter);
            bool nd_valid = (nd_it == no_decay_tokens.end()) ||
                            (!nd_it->second.expired() && nd_it->second.lock() == parameter->getLifetimeToken().lock());
            if (nd_valid)
            {
                eff_weight_decay = 0.0f;
            }
            else
            {
                no_decay_parameters.erase(parameter);
                no_decay_tokens.erase(parameter);
            }
        }

        ops::optimizer::sgdStep(*parameter, *gradient, learning_rate, max_gradient, inv_scale, active_dyn_buf, eff_weight_decay, momentum, vel_ptr);
    }
}

void Sgd_Optimizer::setExecutionTarget(Execution_Target _target)
{
    for (auto &[param, v] : velocity_states)
    {
        if (v.getExecutionTarget() != _target)
        {
            v.setExecutionTarget(_target);
        }
    }
    for (auto &v : loaded_velocities)
    {
        if (v.getExecutionTarget() != _target)
        {
            v.setExecutionTarget(_target);
        }
    }
    ++config_generation;
}

void Sgd_Optimizer::reset()
{
    Logger::logMessage("Sgd_Optimizer::reset: Resetting SGD optimizer",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::OPTIMIZER_STEP);
    velocity_states.clear();
    velocity_tokens.clear();
    velocity_order.clear();
    loaded_velocities.clear();
    loaded_no_decay_flags.clear();
    loaded_velocity_cursor = 0;
    no_decay_parameters.clear();
    no_decay_tokens.clear();
    pending_gpu_overflow = false;
    if (backend_state)
    {
        backend_state->frame_overflow_pending_check.fill(false);
    }
    ++config_generation;
}

void Sgd_Optimizer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open())
    {
        Logger::logMessage("Sgd_Optimizer::saveCheckpoint: Output stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Sgd_Optimizer::saveCheckpoint: Output stream is not open");
    }

    Logger::logMessage("Sgd_Optimizer::saveCheckpoint: Saving SGD checkpoint",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    std::vector<Tensor *> valid_vels;
    valid_vels.reserve(velocity_order.size());
    for (Tensor *p : velocity_order)
    {
        if (p && velocity_states.find(p) != velocity_states.end())
        {
            valid_vels.push_back(p);
        }
    }

    uint32_t magic = SGD_CHECKPOINT_MAGIC_V2;
    uint32_t state_count = static_cast<uint32_t>(valid_vels.size());

    _output_file_stream.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
    _output_file_stream.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));
    _output_file_stream.write(reinterpret_cast<const char *>(&weight_decay), sizeof(weight_decay));
    _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
    _output_file_stream.write(reinterpret_cast<const char *>(&state_count), sizeof(state_count));

    if (state_count > 0)
    {
        std::vector<uint8_t> nd_flags(state_count, 0);
        for (uint32_t i = 0; i < state_count; ++i)
        {
            if (no_decay_parameters.contains(valid_vels[i]))
            {
                nd_flags[i] = 1;
            }
        }
        _output_file_stream.write(reinterpret_cast<const char *>(nd_flags.data()), static_cast<std::streamsize>(nd_flags.size()));
    }

    for (Tensor *p : valid_vels)
    {
        velocity_states.find(p)->second.saveTensor(_output_file_stream);
    }

    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Sgd_Optimizer::saveCheckpoint: Failed to write checkpoint");
    }
}

void Sgd_Optimizer::loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target)
{
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Sgd_Optimizer::loadCheckpoint: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Sgd_Optimizer::loadCheckpoint: Input stream is not open");
    }

    uint32_t first_word = 0;
    _input_file_stream.read(reinterpret_cast<char *>(&first_word), sizeof(first_word));
    if (!_input_file_stream.good())
    {
        throw std::runtime_error("Sgd_Optimizer::loadCheckpoint: Failed to read checkpoint header");
    }

    float lr = learning_rate;
    float mg = max_gradient;
    float wd = weight_decay;
    float mom = momentum;
    uint32_t state_count = 0;
    std::vector<uint8_t> tmp_nd_flags;
    std::vector<Tensor> tmp_vels;

    if (first_word == SGD_CHECKPOINT_MAGIC_V2)
    {
        _input_file_stream.read(reinterpret_cast<char *>(&lr), sizeof(lr));
        _input_file_stream.read(reinterpret_cast<char *>(&mg), sizeof(mg));
        _input_file_stream.read(reinterpret_cast<char *>(&wd), sizeof(wd));
        _input_file_stream.read(reinterpret_cast<char *>(&mom), sizeof(mom));
        _input_file_stream.read(reinterpret_cast<char *>(&state_count), sizeof(state_count));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Sgd_Optimizer::loadCheckpoint: Failed to read V2 header");
        }
        if (state_count > 1000000u)
        {
            throw std::runtime_error("Sgd_Optimizer::loadCheckpoint: state_count exceeds safety limit");
        }
        if (state_count > 0)
        {
            tmp_nd_flags.resize(state_count, 0);
            _input_file_stream.read(reinterpret_cast<char *>(tmp_nd_flags.data()), static_cast<std::streamsize>(tmp_nd_flags.size()));
            if (!_input_file_stream.good())
            {
                throw std::runtime_error("Sgd_Optimizer::loadCheckpoint: Failed to read V2 flags");
            }
            tmp_vels.reserve(state_count);
            for (uint32_t i = 0; i < state_count; ++i)
            {
                tmp_vels.push_back(Tensor::loadTensor(_input_file_stream, _execution_target));
            }
        }
    }
    else
    {
        std::memcpy(&lr, &first_word, sizeof(lr));
        _input_file_stream.read(reinterpret_cast<char *>(&mg), sizeof(mg));
    }

    if (!_input_file_stream.good())
    {
        throw std::runtime_error("Sgd_Optimizer::loadCheckpoint: Failed to read checkpoint body");
    }

    validateHyperparameters(lr, mg, wd, mom);

    reset();
    learning_rate = lr;
    max_gradient = mg;
    weight_decay = wd;
    momentum = mom;
    loaded_velocities = std::move(tmp_vels);
    loaded_no_decay_flags = std::move(tmp_nd_flags);
    loaded_velocity_cursor = 0;
    pending_gpu_overflow = false;
    if (backend_state)
    {
        backend_state->frame_overflow_pending_check.fill(false);
    }
    ++config_generation;

    Logger::logMessage(Input_Format{"Sgd_Optimizer::loadCheckpoint: Loaded learning_rate={}, max_gradient={}, weight_decay={}, momentum={}",
                                    learning_rate,
                                    max_gradient,
                                    weight_decay,
                                    momentum},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);
}

void Sgd_Optimizer::setVelocityStates(const std::unordered_map<Tensor *, Tensor> &_states)
{
    velocity_states = _states;
    velocity_tokens.clear();
    velocity_order.clear();
    velocity_order.reserve(velocity_states.size());
    for (const auto &[p, v] : velocity_states)
    {
        (void)v;
        if (p)
        {
            velocity_tokens[p] = p->getLifetimeToken();
            velocity_order.push_back(p);
        }
    }
}

void Sgd_Optimizer::setVelocityOrder(const std::vector<Tensor *> &_order)
{
    velocity_order.clear();
    velocity_order.reserve(_order.size());
    for (Tensor *p : _order)
    {
        if (p && velocity_states.contains(p))
        {
            velocity_order.push_back(p);
        }
    }
}

void Sgd_Optimizer::setLoadedVelocities(const std::vector<Tensor> &_loaded)
{
    loaded_velocities = _loaded;
    loaded_velocity_cursor = 0;
    pending_gpu_overflow = false;
    if (backend_state)
    {
        backend_state->frame_overflow_pending_check.fill(false);
    }
    ++config_generation;
}
