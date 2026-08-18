#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"
#include "math/matrix.h"

#ifndef ENABLE_PIPELINE_DEBUG_LOGS
#define ENABLE_PIPELINE_DEBUG_LOGS 0
#endif

#if ENABLE_PIPELINE_DEBUG_LOGS
#define PIPELINE_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define PIPELINE_LOG_DEBUG(msg) ((void)0)
#endif

struct Batch_Data
{
    Matrix *inputs = nullptr;
    Matrix *targets = nullptr;
    VkFence fence = VK_NULL_HANDLE;
};

class Async_Data_Pipeline
{
private:
    static constexpr std::size_t BUFFER_COUNT = 2;

    struct Buffer_Slot
    {
        std::vector<float> input_host;
        std::vector<float> target_host;
        Matrix inputs;
        Matrix targets;
        VkFence fence = VK_NULL_HANDLE;
        std::atomic<bool> is_ready{false};
        bool is_fence_submitted = false;
    };

    std::array<Buffer_Slot, BUFFER_COUNT> slots;
    std::size_t producer_index = 0;
    std::size_t consumer_index = 0;

    std::atomic<bool> is_running{false};
    std::jthread worker_thread;

    std::mutex pipeline_mutex;
    std::condition_variable cv_producer;
    std::condition_variable cv_consumer;

    VkDevice device = VK_NULL_HANDLE;

    void createFences()
    {
        if (device == VK_NULL_HANDLE)
        {
            return;
        }

        VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (auto &slot : slots)
        {
            if (vkCreateFence(device, &fence_info, nullptr, &slot.fence) != VK_SUCCESS)
            {
                Logger::logMessage("Async_Data_Pipeline::createFences: Failed to create fence", LOG_ERROR, true);
                throw std::runtime_error("Failed to create fence");
            }
            slot.is_fence_submitted = false;
        }
        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::createFences: Successfully created fences");
    }

    void destroyFences()
    {
        if (device == VK_NULL_HANDLE)
        {
            return;
        }

        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::destroyFences: Destroying fences");
        for (auto &slot : slots)
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
        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::workerLoop: Worker thread loop started");
        std::size_t current_batch_step = 0;
        while (is_running.load())
        {
            std::size_t slot_idx = producer_index % BUFFER_COUNT;

            {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                cv_producer.wait(lock, [this, slot_idx]
                                 { return !slots[slot_idx].is_ready.load() || !is_running.load(); });
            }

            if (!is_running.load())
            {
                break;
            }

            auto start_prep = std::chrono::high_resolution_clock::now();
            prepareBatchHost(current_batch_step, slots[slot_idx].input_host, slots[slot_idx].target_host);
            auto end_prep = std::chrono::high_resolution_clock::now();

            double prep_time_ms = std::chrono::duration<double, std::milli>(end_prep - start_prep).count();
            PIPELINE_LOG_DEBUG(std::format("Async_Data_Pipeline::workerLoop: Step {}: Slot {} host batch prepared in {:.3f} ms (inputs_size={}, targets_size={})",
                                           current_batch_step, slot_idx, prep_time_ms, slots[slot_idx].input_host.size(), slots[slot_idx].target_host.size()));

            {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                slots[slot_idx].is_ready.store(true);
            }

            cv_consumer.notify_one();
            producer_index++;
            current_batch_step++;
        }
        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::workerLoop: Worker thread loop finished");
    }

protected:
    virtual void prepareBatchHost(std::size_t batch_step, std::vector<float> &out_inputs, std::vector<float> &out_targets) = 0;

public:
    explicit Async_Data_Pipeline(VkDevice vk_device = VK_NULL_HANDLE)
        : device(vk_device)
    {
        if (device != VK_NULL_HANDLE)
        {
            createFences();
        }
    }

    virtual ~Async_Data_Pipeline()
    {
        stop();
        destroyFences();
    }

    void setDevice(VkDevice vk_device)
    {
        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::setDevice: Updating Vulkan device handle");
        destroyFences();
        device = vk_device;
        if (device != VK_NULL_HANDLE)
        {
            createFences();
        }
    }

    void start()
    {
        if (is_running.load())
        {
            return;
        }

        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::start: Starting data pipeline worker thread");
        is_running.store(true);
        producer_index = 0;
        consumer_index = 0;

        for (auto &slot : slots)
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

        PIPELINE_LOG_DEBUG("Async_Data_Pipeline::stop: Stopping data pipeline worker thread");
        is_running.store(false);
        {
            std::lock_guard<std::mutex> lock(pipeline_mutex);
        }
        cv_producer.notify_all();
        cv_consumer.notify_all();

        if (worker_thread.joinable())
        {
            worker_thread.join();
        }
    }

    Batch_Data nextBatch(std::size_t batch_size, std::size_t input_dim, std::size_t output_dim)
    {
        std::size_t slot_idx = consumer_index % BUFFER_COUNT;
        Buffer_Slot &slot = slots[slot_idx];

        double fence_wait_ms = 0.0;
        if (device != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE)
        {
            if (slot.is_fence_submitted)
            {
                auto start_fence = std::chrono::high_resolution_clock::now();
                vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
                auto end_fence = std::chrono::high_resolution_clock::now();
                fence_wait_ms = std::chrono::duration<double, std::milli>(end_fence - start_fence).count();
            }
            vkResetFences(device, 1, &slot.fence);
            slot.is_fence_submitted = true;
        }

        double cv_wait_ms = 0.0;
        {
            auto start_cv = std::chrono::high_resolution_clock::now();
            std::unique_lock<std::mutex> lock(pipeline_mutex);
            cv_consumer.wait(lock, [this, slot_idx]
                             { return slots[slot_idx].is_ready.load() || !is_running.load(); });

            if (!slots[slot_idx].is_ready.load() && !is_running.load())
            {
                return Batch_Data();
            }
            auto end_cv = std::chrono::high_resolution_clock::now();
            cv_wait_ms = std::chrono::duration<double, std::milli>(end_cv - start_cv).count();
        }

        try
        {
            if (slot.inputs.getRows() != batch_size || slot.inputs.getCols() != input_dim)
            {
                slot.inputs = Matrix(batch_size, input_dim, slot.input_host, Execution_Target::VULKAN_GPU);
            }
            else
            {
                slot.inputs.uploadData(slot.input_host);
            }

            if (slot.targets.getRows() != batch_size || slot.targets.getCols() != output_dim)
            {
                slot.targets = Matrix(batch_size, output_dim, slot.target_host, Execution_Target::VULKAN_GPU);
            }
            else
            {
                slot.targets.uploadData(slot.target_host);
            }
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(pipeline_mutex);
            slot.is_ready.store(false);
            cv_producer.notify_one();
            throw;
        }

        uint64_t input_buf_handle = reinterpret_cast<uint64_t>(slot.inputs.getGVector()->getBuffer());
        uint64_t target_buf_handle = reinterpret_cast<uint64_t>(slot.targets.getGVector()->getBuffer());

        PIPELINE_LOG_DEBUG(std::format("Async_Data_Pipeline::nextBatch: Step {}: Slot {} | gpu_fence_wait={:.3f}ms | cpu_data_wait={:.3f}ms | input_buf={} | target_buf={}",
                                       consumer_index, slot_idx, fence_wait_ms, cv_wait_ms, input_buf_handle, target_buf_handle));

        Batch_Data batch{.inputs = &slot.inputs,
                         .targets = &slot.targets,
                         .fence = slot.fence};

        {
            std::lock_guard<std::mutex> lock(pipeline_mutex);
            slot.is_ready.store(false);
        }

        cv_producer.notify_one();
        consumer_index++;

        return batch;
    }

    virtual std::size_t getBatchSize() const = 0;
};