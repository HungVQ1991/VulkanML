#include "engine/async_data_pipeline.h"

#include <chrono>
#include <format>
#include <stdexcept>

#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"

void Async_Data_Pipeline::createFences()
{
    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    VkFenceCreateInfo fence_create_information{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (auto &slot : buffer_slots)
    {
        if (vkCreateFence(device, &fence_create_information, nullptr, &slot.fence) != VK_SUCCESS)
        {
            Logger::logMessage("Async_Data_Pipeline::createFences: Failed to create fence",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
            throw std::runtime_error("Failed to create fence");
        }
        slot.is_fence_submitted = false;
    }

    Logger::logMessage("Async_Data_Pipeline::createFences: Successfully created fences",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);
}

void Async_Data_Pipeline::destroyFences()
{
    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    Logger::logMessage("Async_Data_Pipeline::destroyFences: Destroying fences",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);

    for (auto &slot : buffer_slots)
    {
        if (slot.fence != VK_NULL_HANDLE)
        {
            if (slot.is_fence_submitted)
            {
                auto start_time = std::chrono::high_resolution_clock::now();
                vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
                auto end_time = std::chrono::high_resolution_clock::now();
                double time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                Step_Timings::getInstance()[Timing_Stage::DATA_PREP] += time;
            }
            vkDestroyFence(device, slot.fence, nullptr);
            slot.fence = VK_NULL_HANDLE;
            slot.is_fence_submitted = false;
        }
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

    while (is_running.load())
    {
        size_t slot_index = producer_index % BUFFER_SLOTS_COUNT;

        {
            std::unique_lock<std::mutex> lock(pipeline_mutex);
            producer_condition_variable.wait(lock, [this, slot_index]
                                             { return !buffer_slots[slot_index].is_ready.load() || !is_running.load(); });
        }

        if (!is_running.load())
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
            producer_index++;
            current_batch_step++;
        }
        catch (const std::exception &exception)
        {
            Logger::logMessage(Input_Format{"Async_Data_Pipeline::workerLoop: Exception in prepareBatchHost: {}", exception.what()},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DATA_PIPELINE);
            is_running.store(false);
            consumer_condition_variable.notify_all();
            break;
        }
    }

    Logger::logMessage("Async_Data_Pipeline::workerLoop: Worker thread loop finished",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);
}

Async_Data_Pipeline::Async_Data_Pipeline(VkDevice _device, Execution_Target _execution_target)
    : device(_device), execution_target(_execution_target)
{
    if (device != VK_NULL_HANDLE)
    {
        createFences();
    }
    for (auto &slot : buffer_slots)
    {
        slot.input_matrix = Tensor(0, 0, execution_target);
        slot.target_matrix = Tensor(0, 0, execution_target);
    }
}

Async_Data_Pipeline::~Async_Data_Pipeline()
{
    stop();
    destroyFences();
}

void Async_Data_Pipeline::initializeBuffers(size_t batch_size,
                                           size_t input_dimension,
                                           size_t output_dimension,
                                           Execution_Target _execution_target)
{
    execution_target = _execution_target;
    for (size_t i = 0; i < BUFFER_SLOTS_COUNT; ++i)
    {
        buffer_slots[i].host_inputs.resize(batch_size * input_dimension, 0.0f);
        buffer_slots[i].host_targets.resize(batch_size * output_dimension, 0.0f);
        buffer_slots[i].input_matrix = Tensor(batch_size, input_dimension, execution_target);
        buffer_slots[i].target_matrix = Tensor(batch_size, output_dimension, execution_target);
    }
}

void Async_Data_Pipeline::start()
{
    if (is_running.load())
    {
        return;
    }

    Logger::logMessage("Async_Data_Pipeline::start: Starting data pipeline worker thread",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);
    is_running.store(true);
    producer_index = 0;
    consumer_index = 0;

    for (auto &slot : buffer_slots)
    {
        slot.is_ready.store(false);
        slot.is_fence_submitted = false;
    }

    worker_thread = std::jthread([this]
                                 { workerLoop(); });
}

void Async_Data_Pipeline::stop()
{
    if (!is_running.load())
    {
        return;
    }

    Logger::logMessage("Async_Data_Pipeline::stop: Stopping data pipeline worker thread",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE);
    is_running.store(false);
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

Batch_Data Async_Data_Pipeline::nextBatch(size_t batch_size, size_t input_dimension, size_t output_dimension)
{
    size_t slot_index = consumer_index % BUFFER_SLOTS_COUNT;
    Buffer_Slot &slot = buffer_slots[slot_index];

    if (device != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE)
    {
        if (slot.is_fence_submitted)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);

            auto end_time = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            Step_Timings::getInstance()[Timing_Stage::DATA_PREP] += time;
        }
        vkResetFences(device, 1, &slot.fence);
        slot.is_fence_submitted = true;
    }

    {
        std::unique_lock<std::mutex> lock(pipeline_mutex);
        consumer_condition_variable.wait(lock, [this, slot_index]
                                         { return buffer_slots[slot_index].is_ready.load() || !is_running.load(); });

        if (!buffer_slots[slot_index].is_ready.load() && !is_running.load())
        {
            return Batch_Data();
        }
    }

    try
    {
        if (slot.input_matrix.getExecutionTarget() != execution_target)
        {
            slot.input_matrix.setExecutionTarget(execution_target);
        }
        if (slot.input_matrix.getRows() != batch_size || slot.input_matrix.getColumns() != input_dimension)
        {
            slot.input_matrix.initShape(batch_size, input_dimension);
        }
        slot.input_matrix.uploadData(slot.host_inputs);

        if (slot.target_matrix.getExecutionTarget() != execution_target)
        {
            slot.target_matrix.setExecutionTarget(execution_target);
        }
        if (slot.target_matrix.getRows() != batch_size || slot.target_matrix.getColumns() != output_dimension)
        {
            slot.target_matrix.initShape(batch_size, output_dimension);
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
        auto storage_handle = slot.input_matrix.getStorage();
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
        auto storage_handle = slot.target_matrix.getStorage();
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
                                   consumer_index, slot_index, input_buffer_handle, target_buffer_handle},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DATA_PIPELINE | Log_Feature::SYNCHRONIZATION);

    Batch_Data batch_data{
        .input_matrix = &slot.input_matrix,
        .target_matrix = &slot.target_matrix,
        .fence = slot.fence};

    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        slot.is_ready.store(false);
    }

    producer_condition_variable.notify_one();
    consumer_index++;

    return batch_data;
}

void Async_Data_Pipeline::setDevice(VkDevice _device)
{
    Logger::logMessage("Async_Data_Pipeline::setDevice: Updating Vulkan device handle",
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DATA_PIPELINE);
    destroyFences();
    device = _device;
    if (device != VK_NULL_HANDLE)
    {
        createFences();
    }
}
