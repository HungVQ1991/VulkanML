#include "optimizer/adam_optimizer.h"

#include <algorithm>
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
#include "ops/optimizer/adam_step.h"

struct Adam_Optimizer::Backend_State
{
    std::array<std::shared_ptr<gpu::vector>, MAX_FRAMES_IN_FLIGHT> dynamic_params_buffers{};
    std::array<bool, MAX_FRAMES_IN_FLIGHT> frame_overflow_pending_check{false, false};
};

Adam_Optimizer::Parameter_State::Parameter_State(const Shape &_shape, Execution_Target _execution_target)
    : first_moment_matrix(_shape, Data_Type::FLOAT32, _execution_target),
      second_moment_matrix(_shape, Data_Type::FLOAT32, _execution_target)
{
    size_t total = _shape.getTotalElements();
    if (_execution_target == Execution_Target::VULKAN_GPU && total > 0)
    {
        auto &ctx = Execution_Engine::getInstance().getContext();
        size_t bytes = total * sizeof(float);
        for (Tensor *m : {&first_moment_matrix, &second_moment_matrix})
        {
            auto &g = Gpu_Tensor_Impl::castToGpu(*m->getImplementation());
            if (g.getStorageBuffer() && g.getStorageBuffer()->getBuffer() != VK_NULL_HANDLE)
            {
                g.getStorageBuffer()->markAsUsedInFrame(ctx.getCurrentFrame());
                ctx.fillBuffer(g.getStorageBuffer()->getBuffer(), bytes, 0, 0);
            }
            g.clearHostCache();
            g.invalidateFp16Cache();
        }
    }
}

void Adam_Optimizer::validateHyperparameters(float _lr, float _b1, float _b2, float _eps, float _max_grad, float _wd)
{
    if (!std::isfinite(_lr) || _lr < 0.0f)
    {
        throw std::invalid_argument("Adam_Optimizer: learning_rate must be finite and >= 0");
    }
    if (!std::isfinite(_b1) || _b1 < 0.0f || _b1 >= 1.0f)
    {
        throw std::invalid_argument("Adam_Optimizer: beta1 must be finite and in [0, 1)");
    }
    if (!std::isfinite(_b2) || _b2 < 0.0f || _b2 >= 1.0f)
    {
        throw std::invalid_argument("Adam_Optimizer: beta2 must be finite and in [0, 1)");
    }
    if (!std::isfinite(_eps) || _eps <= 0.0f)
    {
        throw std::invalid_argument("Adam_Optimizer: epsilon must be finite and > 0");
    }
    if (!std::isfinite(_max_grad) || _max_grad < 0.0f)
    {
        throw std::invalid_argument("Adam_Optimizer: max_gradient must be finite and >= 0");
    }
    if (!std::isfinite(_wd) || _wd < 0.0f)
    {
        throw std::invalid_argument("Adam_Optimizer: weight_decay must be finite and >= 0");
    }
}

Adam_Optimizer::Adam_Optimizer(float _learning_rate, float _beta1, float _beta2, float _epsilon, float _max_gradient, float _weight_decay)
    : learning_rate(_learning_rate),
      beta1(_beta1),
      beta2(_beta2),
      epsilon(_epsilon),
      max_gradient(_max_gradient),
      weight_decay(_weight_decay),
      learning_rate_scheduler(nullptr),
      backend_state(std::make_unique<Backend_State>())
{
    validateHyperparameters(learning_rate, beta1, beta2, epsilon, max_gradient, weight_decay);
    if (learning_rate <= 0.0f)
    {
        Logger::logMessage("Adam_Optimizer::Adam_Optimizer: Initial learning rate is non-positive",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }
    Logger::logMessage(Input_Format{"Adam_Optimizer::Adam_Optimizer: learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, weight_decay={}",
                                    learning_rate,
                                    beta1,
                                    beta2,
                                    epsilon,
                                    max_gradient,
                                    weight_decay},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::OPTIMIZER_STEP);
}

Adam_Optimizer::Adam_Optimizer(ILearning_Rate &_learning_rate_scheduler, float _beta1, float _beta2, float _epsilon, float _max_gradient, float _weight_decay)
    : learning_rate(_learning_rate_scheduler.getCurrentRate()),
      beta1(_beta1),
      beta2(_beta2),
      epsilon(_epsilon),
      max_gradient(_max_gradient),
      weight_decay(_weight_decay),
      learning_rate_scheduler(&_learning_rate_scheduler),
      scheduler_lifetime_token(_learning_rate_scheduler.getLifetimeToken()),
      backend_state(std::make_unique<Backend_State>())
{
    validateHyperparameters(learning_rate, beta1, beta2, epsilon, max_gradient, weight_decay);
    if (learning_rate <= 0.0f)
    {
        Logger::logMessage("Adam_Optimizer::Adam_Optimizer: Initial learning rate is non-positive",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }
    Logger::logMessage(Input_Format{"Adam_Optimizer::Adam_Optimizer (ILearning_Rate): learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, weight_decay={}",
                                    learning_rate,
                                    beta1,
                                    beta2,
                                    epsilon,
                                    max_gradient,
                                    weight_decay},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::OPTIMIZER_STEP);
}

Adam_Optimizer::Adam_Optimizer(Adam_Optimizer &&other) noexcept = default;
Adam_Optimizer &Adam_Optimizer::operator=(Adam_Optimizer &&other) noexcept = default;
Adam_Optimizer::~Adam_Optimizer() noexcept = default;

void Adam_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs)
{
    step(_parameter_gradient_pairs, 1.0f);
}

void Adam_Optimizer::latchCompletedGpuOverflows() noexcept
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

bool Adam_Optimizer::consumeGpuOverflowDetected() noexcept
{
    latchCompletedGpuOverflows();
    bool detected = pending_gpu_overflow;
    pending_gpu_overflow = false;
    if (detected)
    {
        Logger::logMessage(Input_Format{"Adam_Optimizer::consumeGpuOverflowDetected: GPU gradient overflow detected at timestep {}", timestep},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::GRADIENT_HEALTH | Log_Feature::OPTIMIZER_STEP);
    }
    return detected;
}

void Adam_Optimizer::stepDynamicParams(float _grad_scale)
{
    if (!std::isfinite(_grad_scale) || _grad_scale <= 0.0f)
    {
        throw std::invalid_argument("Adam_Optimizer::stepDynamicParams: grad_scale must be finite and > 0");
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

    ++timestep;
    float inv_scale = 1.0f / _grad_scale;
    size_t effective_t = std::max<size_t>(timestep, 1);
    float bc1 = std::max(1.0F - std::pow(beta1, static_cast<float>(effective_t)), 1e-8F);
    float bc2 = std::max(1.0F - std::pow(beta2, static_cast<float>(effective_t)), 1e-8F);

    Execution_Engine &engine = Execution_Engine::getInstance();
    uint32_t current_frame = engine.getContext().getCurrentFrame();
    if (!engine.getContext().isFrameReady(current_frame))
    {
        engine.prepareCurrentFrame(Execution_Stage::OPTIMIZER);
        current_frame = engine.getContext().getCurrentFrame();
    }

    latchCompletedGpuOverflows();

    float inv_bc1 = 1.0F / bc1;
    float inv_sqrt_bc2 = 1.0F / std::sqrt(bc2);

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
        Graph_Executor::Adam_Dynamic_Params params{learning_rate, inv_bc1, inv_sqrt_bc2, inv_scale, 0u};
        backend_state->dynamic_params_buffers[current_frame]->writeHostDirect(&params, sizeof(params));
        backend_state->frame_overflow_pending_check[current_frame] = true;
    }
}

void Adam_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale)
{
    if (!std::isfinite(_grad_scale) || _grad_scale <= 0.0f)
    {
        throw std::invalid_argument("Adam_Optimizer::step: grad_scale must be finite and > 0");
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
    else
    {
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
        ++timestep;
    }

    float inv_scale = 1.0f / _grad_scale;
    const bool use_multi4_fp16 =
        has_gpu_param &&
        active_dyn_buf != nullptr &&
        Execution_Engine::getInstance().getContext().isFloat16Enabled();

    struct Pending_Batch
    {
        std::array<ops::optimizer::Adam_Multi_Item, 4> items{};
        std::array<VkBuffer, 16> handles{};
        size_t count = 0;
        size_t handle_count = 0;

        bool conflictsWith(VkBuffer a, VkBuffer b, VkBuffer c, VkBuffer d) const noexcept
        {
            for (size_t i = 0; i < handle_count; ++i)
            {
                VkBuffer h = handles[i];
                if (h != VK_NULL_HANDLE && (h == a || h == b || h == c || h == d))
                {
                    return true;
                }
            }
            return false;
        }

        void push(const ops::optimizer::Adam_Multi_Item &item, VkBuffer a, VkBuffer b, VkBuffer c, VkBuffer d) noexcept
        {
            items[count++] = item;
            handles[handle_count++] = a;
            handles[handle_count++] = b;
            handles[handle_count++] = c;
            handles[handle_count++] = d;
        }
    };

    Pending_Batch decay_batch;
    Pending_Batch no_decay_batch;

    size_t multi4_batched_params = 0;
    auto flushBatch = [&](Pending_Batch &batch, float batch_wd)
    {
        if (batch.count == 0)
        {
            return;
        }
        if (batch.count == 1)
        {
            const auto &it = batch.items[0];
            ops::optimizer::adamStep(*it.parameter,
                                     *it.gradient,
                                     *it.first_moment,
                                     *it.second_moment,
                                     learning_rate,
                                     beta1,
                                     beta2,
                                     epsilon,
                                     timestep,
                                     max_gradient,
                                     inv_scale,
                                     batch_wd,
                                     active_dyn_buf);
        }
        else
        {
            multi4_batched_params += batch.count;
            ops::optimizer::adamStepMulti4Fp16(batch.items,
                                               batch.count,
                                               learning_rate,
                                               beta1,
                                               beta2,
                                               epsilon,
                                               timestep,
                                               max_gradient,
                                               inv_scale,
                                               batch_wd,
                                               active_dyn_buf);
        }
        batch.count = 0;
        batch.handle_count = 0;
    };

    for (const auto &[parameter, gradient] : _parameter_gradient_pairs)
    {
        if (!parameter || !gradient)
        {
            Logger::logMessage("Adam_Optimizer::step: Null parameter or gradient pointer encountered",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
            continue;
        }

        auto iterator = parameter_states.find(parameter);
        if (iterator == parameter_states.end())
        {
            if (loaded_state_cursor < loaded_states.size())
            {
                Parameter_State &loaded = loaded_states[loaded_state_cursor];
                bool matches_shape = (parameter->getShape() == loaded.first_moment_matrix.getShape()) ||
                                     (parameter->getTotalElements() == loaded.first_moment_matrix.getTotalElements() &&
                                      parameter->getTotalElements() == loaded.second_moment_matrix.getTotalElements());
                if (matches_shape)
                {
                    if (loaded.first_moment_matrix.getShape() != parameter->getShape())
                    {
                        loaded.first_moment_matrix.reshape(parameter->getShape());
                        loaded.second_moment_matrix.reshape(parameter->getShape());
                    }
                    if (loaded.first_moment_matrix.getExecutionTarget() != parameter->getExecutionTarget())
                    {
                        loaded.first_moment_matrix.setExecutionTarget(parameter->getExecutionTarget());
                    }
                    if (loaded.second_moment_matrix.getExecutionTarget() != parameter->getExecutionTarget())
                    {
                        loaded.second_moment_matrix.setExecutionTarget(parameter->getExecutionTarget());
                    }
                    auto [ins_it, ok] = parameter_states.emplace(parameter, std::move(loaded));
                    iterator = ins_it;
                }
                else
                {
                    Logger::logMessage("Adam_Optimizer::step: Parameter shape mismatch with loaded state, reinitializing state",
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::OPTIMIZER_STEP);
                    auto [ins_it, ok] = parameter_states.emplace(parameter, Parameter_State(parameter->getShape(), parameter->getExecutionTarget()));
                    iterator = ins_it;
                }

                if (loaded_state_cursor < loaded_no_decay_flags.size() && loaded_no_decay_flags[loaded_state_cursor] != 0)
                {
                    no_decay_parameters.insert(parameter);
                    no_decay_tokens[parameter] = parameter->getLifetimeToken();
                }

                ++loaded_state_cursor;
                if (loaded_state_cursor >= loaded_states.size())
                {
                    loaded_states.clear();
                    loaded_no_decay_flags.clear();
                    loaded_state_cursor = 0;
                }
            }
            else
            {
                auto [new_iterator, is_inserted] = parameter_states.emplace(parameter, Parameter_State(parameter->getShape(), parameter->getExecutionTarget()));
                iterator = new_iterator;
            }
            parameter_tokens[parameter] = parameter->getLifetimeToken();
            parameter_order.push_back(parameter);
        }
        else
        {
            auto tok_it = parameter_tokens.find(parameter);
            bool token_valid = (tok_it == parameter_tokens.end()) ||
                               (!tok_it->second.expired() && tok_it->second.lock() == parameter->getLifetimeToken().lock());
            if (!token_valid || iterator->second.first_moment_matrix.getShape() != parameter->getShape())
            {
                iterator->second = Parameter_State(parameter->getShape(), parameter->getExecutionTarget());
                parameter_tokens[parameter] = parameter->getLifetimeToken();
            }
            else
            {
                if (iterator->second.first_moment_matrix.getExecutionTarget() != parameter->getExecutionTarget())
                {
                    iterator->second.first_moment_matrix.setExecutionTarget(parameter->getExecutionTarget());
                }
                if (iterator->second.second_moment_matrix.getExecutionTarget() != parameter->getExecutionTarget())
                {
                    iterator->second.second_moment_matrix.setExecutionTarget(parameter->getExecutionTarget());
                }
            }
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

        bool queued_in_multi4 = false;
        if (use_multi4_fp16 &&
            parameter != gradient &&
            parameter->getExecutionTarget() == Execution_Target::VULKAN_GPU &&
            gradient->getExecutionTarget() == Execution_Target::VULKAN_GPU &&
            parameter->getDataType() == Data_Type::FLOAT32 &&
            gradient->getDataType() == Data_Type::FLOAT32 &&
            iterator->second.first_moment_matrix.getDataType() == Data_Type::FLOAT32 &&
            iterator->second.second_moment_matrix.getDataType() == Data_Type::FLOAT32)
        {
            const Shape &p_shape = parameter->getShape();
            const size_t total_elems = parameter->getTotalElements();
            const bool is_small_1d = (total_elems > 0 && total_elems <= 4096) &&
                                     (p_shape.getRank() <= 1 || p_shape[0] == 1);
            if (is_small_1d &&
                parameter->getImplementation() && gradient->getImplementation() &&
                iterator->second.first_moment_matrix.getImplementation() &&
                iterator->second.second_moment_matrix.getImplementation() &&
                p_shape == gradient->getShape())
            {
                auto &p_gpu = Gpu_Tensor_Impl::castToGpu(*parameter->getImplementation());
                const auto &g_gpu = Gpu_Tensor_Impl::castToGpu(*gradient->getImplementation());
                auto &m_gpu = Gpu_Tensor_Impl::castToGpu(*iterator->second.first_moment_matrix.getImplementation());
                auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*iterator->second.second_moment_matrix.getImplementation());
                bool use_fp16_cache = (Execution_Engine::getInstance().isCooperativeMatrixEnabled() ||
                                       p_gpu.getCachedFp16Storage() != nullptr);
                if (use_fp16_cache &&
                    p_gpu.isContiguous() && p_gpu.getByteOffset() == 0 &&
                    g_gpu.isContiguous() && g_gpu.getByteOffset() == 0 &&
                    m_gpu.isContiguous() && m_gpu.getByteOffset() == 0 &&
                    v_gpu.isContiguous() && v_gpu.getByteOffset() == 0 &&
                    p_gpu.getStorageBuffer() && g_gpu.getStorageBuffer() &&
                    m_gpu.getStorageBuffer() && v_gpu.getStorageBuffer())
                {
                    VkBuffer hp = p_gpu.getStorageBuffer()->getBuffer();
                    VkBuffer hg = g_gpu.getStorageBuffer()->getBuffer();
                    VkBuffer hm = m_gpu.getStorageBuffer()->getBuffer();
                    VkBuffer hv = v_gpu.getStorageBuffer()->getBuffer();
                    if (hp != VK_NULL_HANDLE && hp != hg && hm != VK_NULL_HANDLE && hv != VK_NULL_HANDLE)
                    {
                        Pending_Batch &target_batch = (eff_weight_decay == 0.0f) ? no_decay_batch : decay_batch;
                        if (target_batch.count == 4 || target_batch.conflictsWith(hp, hg, hm, hv))
                        {
                            flushBatch(target_batch, eff_weight_decay);
                        }
                        target_batch.push(
                            ops::optimizer::Adam_Multi_Item{
                                parameter,
                                gradient,
                                &iterator->second.first_moment_matrix,
                                &iterator->second.second_moment_matrix},
                            hp, hg, hm, hv);
                        queued_in_multi4 = true;
                    }
                }
            }
        }

        if (!queued_in_multi4)
        {
            ops::optimizer::adamStep(*parameter,
                                     *gradient,
                                     iterator->second.first_moment_matrix,
                                     iterator->second.second_moment_matrix,
                                     learning_rate,
                                     beta1,
                                     beta2,
                                     epsilon,
                                     timestep,
                                     max_gradient,
                                     inv_scale,
                                     eff_weight_decay,
                                     active_dyn_buf);
        }
    }

    flushBatch(decay_batch, weight_decay);
    flushBatch(no_decay_batch, 0.0f);
    if (multi4_batched_params > 0)
    {
        Logger::logMessage(Input_Format{"Adam_Optimizer::step: multi4_batched_params={} / {}",
                                        multi4_batched_params, _parameter_gradient_pairs.size()},
                           Log_Level::LOG_DEBUG,
                           false,
                           1,
                           Log_Feature::OPTIMIZER_STEP);
    }
}

void Adam_Optimizer::setExecutionTarget(Execution_Target _target)
{
    for (auto &[param, state] : parameter_states)
    {
        if (state.first_moment_matrix.getExecutionTarget() != _target)
        {
            state.first_moment_matrix.setExecutionTarget(_target);
        }
        if (state.second_moment_matrix.getExecutionTarget() != _target)
        {
            state.second_moment_matrix.setExecutionTarget(_target);
        }
    }
    for (auto &state : loaded_states)
    {
        if (state.first_moment_matrix.getExecutionTarget() != _target)
        {
            state.first_moment_matrix.setExecutionTarget(_target);
        }
        if (state.second_moment_matrix.getExecutionTarget() != _target)
        {
            state.second_moment_matrix.setExecutionTarget(_target);
        }
    }
    ++config_generation;
}

void Adam_Optimizer::reset()
{
    Logger::logMessage("Adam_Optimizer::reset: Resetting optimizer states and timestep",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::OPTIMIZER_STEP);
    parameter_states.clear();
    parameter_tokens.clear();
    parameter_order.clear();
    loaded_states.clear();
    loaded_no_decay_flags.clear();
    loaded_state_cursor = 0;
    no_decay_parameters.clear();
    no_decay_tokens.clear();
    timestep = 0;
    pending_gpu_overflow = false;
    if (backend_state)
    {
        backend_state->frame_overflow_pending_check.fill(false);
    }
    ++config_generation;
}

void Adam_Optimizer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open())
    {
        Logger::logMessage("Adam_Optimizer::saveCheckpoint: Output stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Adam_Optimizer::saveCheckpoint: Output stream is not open");
    }

    Logger::logMessage(Input_Format{"Adam_Optimizer::saveCheckpoint: Saving checkpoint at timestep={}, learning_rate={}",
                                    timestep,
                                    learning_rate},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    std::vector<Tensor *> valid_params;
    valid_params.reserve(parameter_order.size());
    for (Tensor *parameter : parameter_order)
    {
        if (parameter && parameter_states.find(parameter) != parameter_states.end())
        {
            valid_params.push_back(parameter);
        }
    }

    uint64_t magic = ADAM_CHECKPOINT_MAGIC_V2;
    uint64_t timestep_value = static_cast<uint64_t>(timestep);
    uint32_t state_count = static_cast<uint32_t>(valid_params.size());

    _output_file_stream.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    _output_file_stream.write(reinterpret_cast<const char *>(&timestep_value), sizeof(timestep_value));
    _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
    _output_file_stream.write(reinterpret_cast<const char *>(&beta1), sizeof(beta1));
    _output_file_stream.write(reinterpret_cast<const char *>(&beta2), sizeof(beta2));
    _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    _output_file_stream.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));
    _output_file_stream.write(reinterpret_cast<const char *>(&weight_decay), sizeof(weight_decay));
    _output_file_stream.write(reinterpret_cast<const char *>(&state_count), sizeof(state_count));

    if (state_count > 0)
    {
        std::vector<uint8_t> nd_flags(state_count, 0);
        for (uint32_t i = 0; i < state_count; ++i)
        {
            if (no_decay_parameters.contains(valid_params[i]))
            {
                nd_flags[i] = 1;
            }
        }
        _output_file_stream.write(reinterpret_cast<const char *>(nd_flags.data()), static_cast<std::streamsize>(nd_flags.size()));
    }

    for (Tensor *parameter : valid_params)
    {
        const auto &state = parameter_states.find(parameter)->second;
        state.first_moment_matrix.saveTensor(_output_file_stream);
        state.second_moment_matrix.saveTensor(_output_file_stream);
    }

    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Adam_Optimizer::saveCheckpoint: Failed to write checkpoint");
    }
}

void Adam_Optimizer::loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target)
{
    if (!_input_file_stream.is_open())
    {
        Logger::logMessage("Adam_Optimizer::loadCheckpoint: Input stream is not open",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Adam_Optimizer::loadCheckpoint: Input stream is not open");
    }

    uint64_t first_word = 0;
    _input_file_stream.read(reinterpret_cast<char *>(&first_word), sizeof(first_word));
    if (!_input_file_stream.good())
    {
        throw std::runtime_error("Adam_Optimizer::loadCheckpoint: Failed to read checkpoint header");
    }

    uint64_t timestep_value = 0;
    float lr = learning_rate;
    float b1 = beta1;
    float b2 = beta2;
    float eps = epsilon;
    float mg = max_gradient;
    float wd = weight_decay;
    uint32_t state_count = 0;
    std::vector<uint8_t> tmp_nd_flags;

    if (first_word == ADAM_CHECKPOINT_MAGIC_V2)
    {
        _input_file_stream.read(reinterpret_cast<char *>(&timestep_value), sizeof(timestep_value));
        _input_file_stream.read(reinterpret_cast<char *>(&lr), sizeof(lr));
        _input_file_stream.read(reinterpret_cast<char *>(&b1), sizeof(b1));
        _input_file_stream.read(reinterpret_cast<char *>(&b2), sizeof(b2));
        _input_file_stream.read(reinterpret_cast<char *>(&eps), sizeof(eps));
        _input_file_stream.read(reinterpret_cast<char *>(&mg), sizeof(mg));
        _input_file_stream.read(reinterpret_cast<char *>(&wd), sizeof(wd));
        _input_file_stream.read(reinterpret_cast<char *>(&state_count), sizeof(state_count));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Adam_Optimizer::loadCheckpoint: Failed to read V2 header");
        }
        if (state_count > 1000000u)
        {
            throw std::runtime_error("Adam_Optimizer::loadCheckpoint: state_count exceeds safety limit");
        }
        if (state_count > 0)
        {
            tmp_nd_flags.resize(state_count, 0);
            _input_file_stream.read(reinterpret_cast<char *>(tmp_nd_flags.data()), static_cast<std::streamsize>(tmp_nd_flags.size()));
        }
    }
    else
    {
        timestep_value = first_word;
        _input_file_stream.read(reinterpret_cast<char *>(&lr), sizeof(lr));
        _input_file_stream.read(reinterpret_cast<char *>(&b1), sizeof(b1));
        _input_file_stream.read(reinterpret_cast<char *>(&b2), sizeof(b2));
        _input_file_stream.read(reinterpret_cast<char *>(&eps), sizeof(eps));
        _input_file_stream.read(reinterpret_cast<char *>(&mg), sizeof(mg));
        _input_file_stream.read(reinterpret_cast<char *>(&state_count), sizeof(state_count));
    }

    if (!_input_file_stream.good())
    {
        throw std::runtime_error("Adam_Optimizer::loadCheckpoint: Failed to read optimizer parameters");
    }

    validateHyperparameters(lr, b1, b2, eps, mg, wd);
    if (state_count > 1000000u)
    {
        throw std::runtime_error("Adam_Optimizer::loadCheckpoint: state_count exceeds safety limit");
    }

    std::vector<Parameter_State> tmp_states;
    tmp_states.reserve(state_count);
    for (uint32_t i = 0; i < state_count; ++i)
    {
        Tensor first_moment = Tensor::loadTensor(_input_file_stream, _execution_target);
        Tensor second_moment = Tensor::loadTensor(_input_file_stream, _execution_target);
        tmp_states.emplace_back(std::move(first_moment), std::move(second_moment));
    }

    reset();
    timestep = static_cast<size_t>(timestep_value);
    learning_rate = lr;
    beta1 = b1;
    beta2 = b2;
    epsilon = eps;
    max_gradient = mg;
    weight_decay = wd;
    loaded_states = std::move(tmp_states);
    loaded_no_decay_flags = std::move(tmp_nd_flags);
    loaded_state_cursor = 0;
    pending_gpu_overflow = false;
    if (backend_state)
    {
        backend_state->frame_overflow_pending_check.fill(false);
    }
    ++config_generation;

    Logger::logMessage(Input_Format{"Adam_Optimizer::loadCheckpoint: Loaded timestep={}, learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, weight_decay={}, state_count={}",
                                    timestep,
                                    learning_rate,
                                    beta1,
                                    beta2,
                                    epsilon,
                                    max_gradient,
                                    weight_decay,
                                    state_count},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);
}

void Adam_Optimizer::setLoadedStates(const std::vector<Parameter_State> &_loaded_states)
{
    loaded_states = _loaded_states;
    loaded_state_cursor = 0;
}

void Adam_Optimizer::setParameterOrder(const std::vector<Tensor *> &_parameter_order)
{
    parameter_order.clear();
    parameter_order.reserve(_parameter_order.size());
    for (Tensor *p : _parameter_order)
    {
        if (p && parameter_states.contains(p))
        {
            parameter_order.push_back(p);
        }
    }
}
