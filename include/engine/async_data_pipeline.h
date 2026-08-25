#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
#include <vulkan/vulkan.h>

#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "math/matrix.h"

struct Batch_Data
{
    Matrix *input_matrix = nullptr;
    Matrix *target_matrix = nullptr;
    VkFence fence = VK_NULL_HANDLE;
};

class Async_Data_Pipeline
{
private:
    static constexpr std::size_t BUFFER_SLOTS_COUNT = 2;

    struct Buffer_Slot
    {
        std::vector<float> host_inputs;
        std::vector<float> host_targets;
        Matrix input_matrix;
        Matrix target_matrix;
        VkFence fence = VK_NULL_HANDLE;
        std::atomic<bool> is_ready{false};
        bool is_fence_submitted = false;
    };

    std::array<Buffer_Slot, BUFFER_SLOTS_COUNT> buffer_slots;
    std::size_t producer_index = 0;
    std::size_t consumer_index = 0;

    std::atomic<bool> is_running{false};
    std::jthread worker_thread;

    std::mutex pipeline_mutex;
    std::condition_variable producer_condition_variable;
    std::condition_variable consumer_condition_variable;

    VkDevice device = VK_NULL_HANDLE;
    Execution_Target execution_target = Execution_Target::VULKAN_GPU;

    void createFences()
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

    void destroyFences()
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
                    vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
                }
                vkDestroyFence(device, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
                slot.is_fence_submitted = false;
            }
        }
    }

    void workerLoop()
    {
        Logger::logMessage("Async_Data_Pipeline::workerLoop: Worker thread loop started",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
        std::size_t current_batch_step = 0;

        while (is_running.load())
        {
            std::size_t slot_index = producer_index % BUFFER_SLOTS_COUNT;

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
                auto start_preparation_time = std::chrono::high_resolution_clock::now();
                prepareBatchHost(current_batch_step, buffer_slots[slot_index].host_inputs, buffer_slots[slot_index].host_targets);
                auto end_preparation_time = std::chrono::high_resolution_clock::now();

                double preparation_time_in_milliseconds = std::chrono::duration<double, std::milli>(end_preparation_time - start_preparation_time).count();
                Logger::logMessage(std::format("Async_Data_Pipeline::workerLoop: Step {}: Slot {} host batch prepared in {:.3f} ms",
                                               current_batch_step, slot_index, preparation_time_in_milliseconds),
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
                Logger::logMessage(std::format("Async_Data_Pipeline::workerLoop: Exception in prepareBatchHost: {}", exception.what()),
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

protected:
    virtual void prepareBatchHost(std::size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets) = 0;

public:
    explicit Async_Data_Pipeline(VkDevice _device = VK_NULL_HANDLE, Execution_Target _execution_target = Execution_Target::VULKAN_GPU)
        : device(_device), execution_target(_execution_target)
    {
        if (device != VK_NULL_HANDLE)
        {
            createFences();
        }
        for (auto &slot : buffer_slots)
        {
            slot.input_matrix = Matrix(0, 0, execution_target);
            slot.target_matrix = Matrix(0, 0, execution_target);
        }
    }

    virtual ~Async_Data_Pipeline()
    {
        stop();
        destroyFences();
    }

    void initializeBuffers(std::size_t batch_size,
                           std::size_t input_dimension,
                           std::size_t output_dimension,
                           Execution_Target _execution_target = Execution_Target::VULKAN_GPU)
    {
        execution_target = _execution_target;
        for (std::size_t i = 0; i < BUFFER_SLOTS_COUNT; ++i)
        {
            buffer_slots[i].host_inputs.resize(batch_size * input_dimension, 0.0f);
            buffer_slots[i].host_targets.resize(batch_size * output_dimension, 0.0f);
            buffer_slots[i].input_matrix = Matrix(batch_size, input_dimension, execution_target);
            buffer_slots[i].target_matrix = Matrix(batch_size, output_dimension, execution_target);
        }
    }

    void setDevice(VkDevice _device)
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

    void setExecutionTarget(Execution_Target _execution_target) noexcept
    {
        execution_target = _execution_target;
    }

    void start()
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

    void stop()
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

    Batch_Data nextBatch(std::size_t batch_size, std::size_t input_dimension, std::size_t output_dimension)
    {
        std::size_t slot_index = consumer_index % BUFFER_SLOTS_COUNT;
        Buffer_Slot &slot = buffer_slots[slot_index];

        double fence_wait_time_in_milliseconds = 0.0;
        if (device != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE)
        {
            if (slot.is_fence_submitted)
            {
                auto start_fence_time = std::chrono::high_resolution_clock::now();
                vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
                auto end_fence_time = std::chrono::high_resolution_clock::now();
                fence_wait_time_in_milliseconds = std::chrono::duration<double, std::milli>(end_fence_time - start_fence_time).count();
            }
            vkResetFences(device, 1, &slot.fence);
            slot.is_fence_submitted = true;
        }

        double condition_variable_wait_time_in_milliseconds = 0.0;
        {
            auto start_condition_variable_time = std::chrono::high_resolution_clock::now();
            std::unique_lock<std::mutex> lock(pipeline_mutex);
            consumer_condition_variable.wait(lock, [this, slot_index]
                                             { return buffer_slots[slot_index].is_ready.load() || !is_running.load(); });

            if (!buffer_slots[slot_index].is_ready.load() && !is_running.load())
            {
                return Batch_Data();
            }
            auto end_condition_variable_time = std::chrono::high_resolution_clock::now();
            condition_variable_wait_time_in_milliseconds = std::chrono::duration<double, std::milli>(end_condition_variable_time - start_condition_variable_time).count();
        }

        try
        {
            if (slot.input_matrix.getTarget() != execution_target)
            {
                slot.input_matrix.setExecutionTarget(execution_target);
            }
            if (slot.input_matrix.getRows() != batch_size || slot.input_matrix.getCols() != input_dimension)
            {
                slot.input_matrix.initShape(batch_size, input_dimension);
            }
            slot.input_matrix.uploadData(slot.host_inputs);

            if (slot.target_matrix.getTarget() != execution_target)
            {
                slot.target_matrix.setExecutionTarget(execution_target);
            }
            if (slot.target_matrix.getRows() != batch_size || slot.target_matrix.getCols() != output_dimension)
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

        std::uint64_t input_buffer_handle = 0;
        std::uint64_t target_buffer_handle = 0;

        if (slot.input_matrix.getTarget() == Execution_Target::VULKAN_GPU)
        {
            auto storage_handle = slot.input_matrix.getStorage();
            if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
            {
                const auto &gpu_vec = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
                if (gpu_vec)
                {
                    input_buffer_handle = reinterpret_cast<std::uint64_t>(gpu_vec->getBuffer());
                }
            }
        }

        if (slot.target_matrix.getTarget() == Execution_Target::VULKAN_GPU)
        {
            auto storage_handle = slot.target_matrix.getStorage();
            if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
            {
                const auto &gpu_vec = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
                if (gpu_vec)
                {
                    target_buffer_handle = reinterpret_cast<std::uint64_t>(gpu_vec->getBuffer());
                }
            }
        }

        Logger::logMessage(std::format("Async_Data_Pipeline::nextBatch: Step {}: Slot {} | gpu_fence_wait={:.3f}ms | cpu_data_wait={:.3f}ms | input_buffer={} | target_buffer={}",
                                       consumer_index, slot_index, fence_wait_time_in_milliseconds, condition_variable_wait_time_in_milliseconds, input_buffer_handle, target_buffer_handle),
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

     VkDevice getDevice() const noexcept
    {
        return device;
    }

     Execution_Target getExecutionTarget() const noexcept
    {
        return execution_target;
    }

     bool isRunning() const noexcept
    {
        return is_running.load();
    }

     std::size_t getProducerIndex() const noexcept
    {
        return producer_index;
    }

     std::size_t getConsumerIndex() const noexcept
    {
        return consumer_index;
    }

     std::size_t getBufferSlotsCount() const noexcept
    {
        return BUFFER_SLOTS_COUNT;
    }

     virtual std::size_t getBatchSize() const = 0;
};