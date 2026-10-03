#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>
#include <vulkan/vulkan.h>

#include "math/tensor.h"

struct Batch_Data
{
    Tensor *input_matrix = nullptr;
    Tensor *target_matrix = nullptr;
    VkFence fence = VK_NULL_HANDLE;

    Tensor *getInputMatrix() const noexcept { return input_matrix; }
    Tensor *getTargetMatrix() const noexcept { return target_matrix; }
    VkFence getFence() const noexcept { return fence; }

    void setInputMatrix(Tensor *_matrix) noexcept { input_matrix = _matrix; }
    void setTargetMatrix(Tensor *_matrix) noexcept { target_matrix = _matrix; }
    void setFence(VkFence _fence) noexcept { fence = _fence; }
};

class Async_Data_Pipeline
{
private:
    static constexpr size_t BUFFER_SLOTS_COUNT = 2;

    struct Buffer_Slot
    {
        std::vector<float> host_inputs;
        std::vector<float> host_targets;
        Tensor input_matrix;
        Tensor target_matrix;
        VkFence fence = VK_NULL_HANDLE;
        std::atomic<bool> is_ready{false};
        bool is_fence_submitted = false;
    };

    std::array<Buffer_Slot, BUFFER_SLOTS_COUNT> buffer_slots;
    size_t producer_index = 0;
    size_t consumer_index = 0;

    std::atomic<bool> is_running{false};
    std::jthread worker_thread;

    std::mutex pipeline_mutex;
    std::condition_variable producer_condition_variable;
    std::condition_variable consumer_condition_variable;

    VkDevice device = VK_NULL_HANDLE;
    Execution_Target execution_target = Execution_Target::VULKAN_GPU;

    void createFences();
    void destroyFences();
    void workerLoop();

protected:
    virtual void prepareBatchHost(size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets) = 0;

public:
    explicit Async_Data_Pipeline(VkDevice _device = VK_NULL_HANDLE, Execution_Target _execution_target = Execution_Target::VULKAN_GPU);
    virtual ~Async_Data_Pipeline();

    void initializeBuffers(size_t batch_size,
                           size_t input_dimension,
                           size_t output_dimension,
                           Execution_Target _execution_target = Execution_Target::VULKAN_GPU);

    void start();
    void stop();

    Batch_Data nextBatch(size_t batch_size, size_t input_dimension, size_t output_dimension);

    virtual size_t getBatchSize() const = 0;
    size_t getBufferSlotsCount() const noexcept { return BUFFER_SLOTS_COUNT; }
    size_t getProducerIndex() const noexcept { return producer_index; }
    size_t getConsumerIndex() const noexcept { return consumer_index; }
    VkDevice getDevice() const noexcept { return device; }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    bool isRunning() const noexcept { return is_running.load(); }

    void setDevice(VkDevice _device);
    void setProducerIndex(size_t _index) noexcept { producer_index = _index; }
    void setConsumerIndex(size_t _index) noexcept { consumer_index = _index; }
    void setExecutionTarget(Execution_Target _execution_target) noexcept { execution_target = _execution_target; }
    void setRunning(bool _running) noexcept { is_running.store(_running); }
};