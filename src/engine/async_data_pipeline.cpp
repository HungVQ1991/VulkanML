#include "engine/async_data_pipeline.h"

#include <chrono>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <vulkan/vulkan.h>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"
#include "math/tensor_impl.h"

namespace
{
size_t checkedMultiply(size_t a, size_t b, const char *context_label)
{
    if (a != 0 && b > (std::numeric_limits<size_t>::max)() / a)
    {
        throw std::overflow_error(std::string(context_label) + ": integer overflow in buffer size calculation");
    }
    return a * b;
}

Sync_Handle toSyncHandle(VkFence fence) noexcept
{
    return Sync_Handle{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fence))};
}
}

struct Async_Data_Pipeline::Backend_State
{
    VkDevice device = VK_NULL_HANDLE;
    std::array<VkFence, BUFFER_SLOTS_COUNT> fences{};
    std::array<bool, BUFFER_SLOTS_COUNT> fence_submitted{};

    Backend_State()
    {
        fences.fill(VK_NULL_HANDLE);
        fence_submitted.fill(false);
    }
};

void Async_Data_Pipeline::createFences()
{
    if (!backend_state || backend_state->device == VK_NULL_HANDLE)
    {
        return;
    }

    VkFenceCreateInfo fence_create_information{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (size_t i = 0; i < BUFFER_SLOTS_COUNT; ++i)
    {
        if (vkCreateFence(backend_state->device, &fence_create_information, nullptr, &backend_state->fences[i]) != VK_SUCCESS)
        {
            Logger::logMessage("Async_Data_Pipeline::createFences: Failed to create fence",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
            destroyFences();
            throw std::runtime_error("Failed to create fence");
        }
        backend_state->fence_submitted[i] = false;
    }

    Logger::logMessage("Async_Data_Pipeline::createFences: Successfully created fences",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
}

void Async_Data_Pipeline::destroyFences()
{
    if (!backend_state || backend_state->device == VK_NULL_HANDLE)
    {
        return;
    }

    Logger::logMessage("Async_Data_Pipeline::destroyFences: Destroying fences",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);

    for (size_t i = 0; i < BUFFER_SLOTS_COUNT; ++i)
    {
        VkFence &slot_fence = backend_state->fences[i];
        if (slot_fence != VK_NULL_HANDLE)
        {
            if (backend_state->fence_submitted[i])
            {
                auto start_time = std::chrono::high_resolution_clock::now();
                VkResult wait_res = vkWaitForFences(backend_state->device, 1, &slot_fence, VK_TRUE, UINT64_MAX);
                if (wait_res != VK_SUCCESS)
                {
                    Logger::logMessage(Input_Format{"Async_Data_Pipeline::destroyFences: vkWaitForFences failed ({})", static_cast<int>(wait_res)},
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                double time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                Step_Timings::getInstance().recordDuration(Timing_Stage::DATA_PREP, time);
            }
            vkDestroyFence(backend_state->device, slot_fence, nullptr);
            slot_fence = VK_NULL_HANDLE;
            backend_state->fence_submitted[i] = false;
        }
    }
}

void Async_Data_Pipeline::stopUnlocked()
{
    Logger::logMessage("Async_Data_Pipeline::stop: Stopping data pipeline worker thread",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);
    is_running.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
    }
    producer_condition_variable.notify_all();
    consumer_condition_variable.notify_all();

    if (worker_thread.joinable())
    {
        worker_thread.join();
    }
}

void Async_Data_Pipeline::workerLoop()
{
    Logger::logMessage("Async_Data_Pipeline::workerLoop: Worker thread loop started",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);
    size_t current_batch_step = 0;

    while (is_running.load(std::memory_order_acquire))
    {
        size_t slot_index = producer_index.load(std::memory_order_acquire) % BUFFER_SLOTS_COUNT;

        {
            std::unique_lock<std::mutex> lock(pipeline_mutex);
            producer_condition_variable.wait(lock, [this, slot_index]
                                             { return !buffer_slots[slot_index].is_ready.load() || !is_running.load(std::memory_order_acquire); });
        }

        if (!is_running.load(std::memory_order_acquire))
        {
            break;
        }

        try
        {
            prepareBatchHost(current_batch_step, buffer_slots[slot_index].host_inputs, buffer_slots[slot_index].host_targets);
            Logger::logMessage(Input_Format{"Async_Data_Pipeline::workerLoop: Step {}: Slot {} host batch prepared",
                                            current_batch_step, slot_index},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE);

            {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                buffer_slots[slot_index].is_ready.store(true);
            }

            consumer_condition_variable.notify_one();
            producer_index.fetch_add(1, std::memory_order_acq_rel);
            current_batch_step++;
        }
        catch (const std::exception &exception)
        {
            Logger::logMessage(Input_Format{"Async_Data_Pipeline::workerLoop: Exception in prepareBatchHost: {}", exception.what()},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE);
            {
                std::lock_guard<std::mutex> lock(pipeline_mutex);
                worker_exception = std::current_exception();
                is_running.store(false, std::memory_order_release);
            }
            consumer_condition_variable.notify_all();
            producer_condition_variable.notify_all();
            break;
        }
        catch (...)
        {
            Logger::logMessage("Async_Data_Pipeline::workerLoop: Unknown exception in prepareBatchHost",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE);
            {
                std::lock_guard<std::mutex> lock(pipeline_mutex);
                worker_exception = std::current_exception();
                is_running.store(false, std::memory_order_release);
            }
            consumer_condition_variable.notify_all();
            producer_condition_variable.notify_all();
            break;
        }
    }

    Logger::logMessage("Async_Data_Pipeline::workerLoop: Worker thread loop finished",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);
}

Async_Data_Pipeline::Async_Data_Pipeline(Execution_Target _execution_target)
    : backend_state(std::make_unique<Backend_State>()),
      execution_target(_execution_target)
{
    if (_execution_target == Execution_Target::VULKAN_GPU)
    {
        bindActiveDevice();
    }
    Execution_Target target = execution_target.load(std::memory_order_acquire);
    for (auto &slot : buffer_slots)
    {
        slot.input_matrix = Tensor(0, 0, target);
        slot.target_matrix = Tensor(0, 0, target);
    }
}

Async_Data_Pipeline::~Async_Data_Pipeline()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
    if (is_running.load(std::memory_order_acquire))
    {
        Logger::logMessage("Async_Data_Pipeline::~Async_Data_Pipeline: Stopping still-running pipeline in base destructor; subclasses should call stop() in their own destructors",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
    }
    stopUnlocked();
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    destroyFences();
}

void Async_Data_Pipeline::bindActiveDevice()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
    VkDevice active_dev = Execution_Engine::getInstance().getContext().getDevice();
    if (backend_state->device == active_dev)
    {
        return;
    }
    stopUnlocked();
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    Logger::logMessage("Async_Data_Pipeline::bindActiveDevice: Binding Vulkan device handle",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DATA_PIPELINE);
    destroyFences();
    backend_state->device = active_dev;
    if (backend_state->device != VK_NULL_HANDLE)
    {
        createFences();
    }
}

void Async_Data_Pipeline::releaseDevice()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
    if (!backend_state || backend_state->device == VK_NULL_HANDLE)
    {
        return;
    }
    stopUnlocked();
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    Logger::logMessage("Async_Data_Pipeline::releaseDevice: Releasing Vulkan device handle",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DATA_PIPELINE);
    destroyFences();
    backend_state->device = VK_NULL_HANDLE;
}

void Async_Data_Pipeline::initializeBuffers(size_t batch_size,
                                            size_t input_dimension,
                                            size_t output_dimension,
                                            Execution_Target _execution_target)
{
    if (_execution_target == Execution_Target::VULKAN_GPU)
    {
        bindActiveDevice();
    }
    else
    {
        releaseDevice();
    }

    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
    if (is_running.load(std::memory_order_acquire))
    {
        throw std::logic_error("Async_Data_Pipeline::initializeBuffers: Cannot reinitialize while running");
    }
    const size_t input_elems = checkedMultiply(batch_size, input_dimension, "Async_Data_Pipeline::initializeBuffers");
    const size_t output_elems = checkedMultiply(batch_size, output_dimension, "Async_Data_Pipeline::initializeBuffers");
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    execution_target.store(_execution_target, std::memory_order_release);
    for (size_t i = 0; i < BUFFER_SLOTS_COUNT; ++i)
    {
        buffer_slots[i].host_inputs.resize(input_elems, 0.0f);
        buffer_slots[i].host_targets.resize(output_elems, 0.0f);
        buffer_slots[i].input_matrix = Tensor(batch_size, input_dimension, _execution_target);
        buffer_slots[i].target_matrix = Tensor(batch_size, output_dimension, _execution_target);
    }
}

void Async_Data_Pipeline::start()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
    if (is_running.load(std::memory_order_acquire))
    {
        return;
    }

    if (worker_thread.joinable())
    {
        worker_thread.join();
    }

    Logger::logMessage("Async_Data_Pipeline::start: Starting data pipeline worker thread",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);

    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        for (size_t i = 0; i < BUFFER_SLOTS_COUNT; ++i)
        {
            if (backend_state->device != VK_NULL_HANDLE &&
                backend_state->fences[i] != VK_NULL_HANDLE &&
                backend_state->fence_submitted[i])
            {
                VkResult wait_res = vkWaitForFences(backend_state->device, 1, &backend_state->fences[i], VK_TRUE, UINT64_MAX);
                if (wait_res != VK_SUCCESS)
                {
                    Logger::logMessage(Input_Format{"Async_Data_Pipeline::start: vkWaitForFences failed ({})", static_cast<int>(wait_res)},
                                       Log_Level::LOG_ERROR,
                                       true,
                                       0,
                                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
                    throw std::runtime_error("Async_Data_Pipeline::start: vkWaitForFences failed");
                }
            }
            buffer_slots[i].is_ready.store(false);
            backend_state->fence_submitted[i] = false;
        }
        worker_exception = nullptr;
    }
    producer_index.store(0, std::memory_order_release);
    consumer_index.store(0, std::memory_order_release);
    is_running.store(true, std::memory_order_release);

    worker_thread = std::jthread([this]
                                 { workerLoop(); });
}

void Async_Data_Pipeline::stop()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
    stopUnlocked();
}

void Async_Data_Pipeline::setRunning(bool _running)
{
    if (_running)
    {
        start();
    }
    else
    {
        stop();
    }
}

Batch_Data Async_Data_Pipeline::nextBatch(size_t batch_size, size_t input_dimension, size_t output_dimension)
{
    const size_t expected_inputs = checkedMultiply(batch_size, input_dimension, "Async_Data_Pipeline::nextBatch");
    const size_t expected_targets = checkedMultiply(batch_size, output_dimension, "Async_Data_Pipeline::nextBatch");

    size_t cur_consumer = consumer_index.load(std::memory_order_acquire);
    size_t slot_index = cur_consumer % BUFFER_SLOTS_COUNT;
    Buffer_Slot &slot = buffer_slots[slot_index];

    VkDevice wait_device = VK_NULL_HANDLE;
    VkFence wait_fence = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        if (backend_state->device != VK_NULL_HANDLE &&
            backend_state->fences[slot_index] != VK_NULL_HANDLE &&
            backend_state->fence_submitted[slot_index])
        {
            wait_device = backend_state->device;
            wait_fence = backend_state->fences[slot_index];
        }
    }
    if (wait_device != VK_NULL_HANDLE && wait_fence != VK_NULL_HANDLE)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        VkResult wait_res = vkWaitForFences(wait_device, 1, &wait_fence, VK_TRUE, UINT64_MAX);
        if (wait_res != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Async_Data_Pipeline::nextBatch: vkWaitForFences failed ({})", static_cast<int>(wait_res)},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
            throw std::runtime_error("Async_Data_Pipeline::nextBatch: vkWaitForFences failed");
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(pipeline_mutex);
            if (backend_state->device == wait_device && backend_state->fences[slot_index] == wait_fence)
            {
                backend_state->fence_submitted[slot_index] = false;
            }
        }
        double time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        Step_Timings::getInstance().recordDuration(Timing_Stage::DATA_PREP, time);
    }

    {
        std::unique_lock<std::mutex> lock(pipeline_mutex);
        consumer_condition_variable.wait(lock, [this, slot_index]
                                         { return buffer_slots[slot_index].is_ready.load() || !is_running.load(std::memory_order_acquire); });

        if (!buffer_slots[slot_index].is_ready.load() && !is_running.load(std::memory_order_acquire))
        {
            if (worker_exception)
            {
                std::exception_ptr ex = worker_exception;
                worker_exception = nullptr;
                lock.unlock();
                std::rethrow_exception(ex);
            }
            return Batch_Data();
        }
    }

    Execution_Target target = execution_target.load(std::memory_order_acquire);
    try
    {
        if (slot.host_inputs.size() != expected_inputs)
        {
            throw std::runtime_error(std::format(
                "Async_Data_Pipeline::nextBatch: host_inputs size ({}) does not match batch_size * input_dimension ({}); check initializeBuffers() / prepareBatchHost()",
                slot.host_inputs.size(), expected_inputs));
        }
        if (slot.host_targets.size() != expected_targets)
        {
            throw std::runtime_error(std::format(
                "Async_Data_Pipeline::nextBatch: host_targets size ({}) does not match batch_size * output_dimension ({}); check initializeBuffers() / prepareBatchHost()",
                slot.host_targets.size(), expected_targets));
        }

        if (slot.input_matrix.getExecutionTarget() != target)
        {
            slot.input_matrix.setExecutionTarget(target);
        }
        if (slot.input_matrix.getRows() != batch_size || slot.input_matrix.getColumns() != input_dimension)
        {
            slot.input_matrix.initializeShape(batch_size, input_dimension);
        }
        slot.input_matrix.uploadData(slot.host_inputs);

        if (slot.target_matrix.getExecutionTarget() != target)
        {
            slot.target_matrix.setExecutionTarget(target);
        }
        if (slot.target_matrix.getRows() != batch_size || slot.target_matrix.getColumns() != output_dimension)
        {
            slot.target_matrix.initializeShape(batch_size, output_dimension);
        }
        slot.target_matrix.uploadData(slot.host_targets);
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        slot.is_ready.store(false);
        producer_condition_variable.notify_one();
        throw;
    }

    uint64_t input_buffer_handle = 0;
    uint64_t target_buffer_handle = 0;

    if (slot.input_matrix.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        auto storage_handle = slot.input_matrix.getImplementation()->getStorage();
        if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
        {
            const auto &gpu_vec = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
            if (gpu_vec)
            {
                input_buffer_handle = reinterpret_cast<uint64_t>(gpu_vec->getBuffer());
            }
        }
    }

    if (slot.target_matrix.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        auto storage_handle = slot.target_matrix.getImplementation()->getStorage();
        if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
        {
            const auto &gpu_vec = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
            if (gpu_vec)
            {
                target_buffer_handle = reinterpret_cast<uint64_t>(gpu_vec->getBuffer());
            }
        }
    }

    Logger::logMessage(Input_Format{"Async_Data_Pipeline::nextBatch: Step {}: Slot {} | input_buffer={} | target_buffer={}",
                                    cur_consumer, slot_index, input_buffer_handle, target_buffer_handle},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);

    Sync_Handle batch_sync{};
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        if (target == Execution_Target::VULKAN_GPU &&
            backend_state->device != VK_NULL_HANDLE &&
            backend_state->fences[slot_index] != VK_NULL_HANDLE)
        {
            backend_state->fence_submitted[slot_index] = true;
            batch_sync = toSyncHandle(backend_state->fences[slot_index]);
        }
        slot.is_ready.store(false);
    }

    Batch_Data batch_data{
        .input_matrix = &slot.input_matrix,
        .target_matrix = &slot.target_matrix,
        .sync_handle = batch_sync};

    producer_condition_variable.notify_one();
    consumer_index.fetch_add(1, std::memory_order_acq_rel);

    return batch_data;
}
