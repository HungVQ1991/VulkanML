#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "math/tensor.h"

struct Sync_Handle
{
    uint64_t raw = 0;
    explicit operator bool() const noexcept { return raw != 0; }
    bool operator==(const Sync_Handle &) const = default;
};

struct Batch_Data
{
    Tensor *input_matrix = nullptr;
    Tensor *target_matrix = nullptr;
    Sync_Handle sync_handle{};

    Tensor *getInputMatrix() const noexcept { return input_matrix; }
    Tensor *getTargetMatrix() const noexcept { return target_matrix; }
    Sync_Handle getSyncHandle() const noexcept { return sync_handle; }

    void setInputMatrix(Tensor *_matrix) noexcept { input_matrix = _matrix; }
    void setTargetMatrix(Tensor *_matrix) noexcept { target_matrix = _matrix; }
    void setSyncHandle(Sync_Handle _handle) noexcept { sync_handle = _handle; }
};

class Async_Data_Pipeline
{
private:
    static constexpr size_t BUFFER_SLOTS_COUNT = 2;

    struct Backend_State;

    struct Buffer_Slot
    {
        std::vector<float> host_inputs;
        std::vector<float> host_targets;
        Tensor input_matrix;
        Tensor target_matrix;
        std::atomic<bool> is_ready{false};
    };

    std::array<Buffer_Slot, BUFFER_SLOTS_COUNT> buffer_slots;
    std::unique_ptr<Backend_State> backend_state;
    std::atomic<size_t> producer_index{0};
    std::atomic<size_t> consumer_index{0};

    std::atomic<bool> is_running{false};
    std::jthread worker_thread;
    std::exception_ptr worker_exception = nullptr;

    std::mutex pipeline_mutex;
    std::mutex lifecycle_mutex;
    std::condition_variable producer_condition_variable;
    std::condition_variable consumer_condition_variable;

    std::atomic<Execution_Target> execution_target{Execution_Target::VULKAN_GPU};

    void createFences();
    void destroyFences();
    void stopUnlocked();
    void workerLoop();

protected:
    virtual void prepareBatchHost(size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets) = 0;

public:
    explicit Async_Data_Pipeline(Execution_Target _execution_target = Execution_Target::VULKAN_GPU);
    Async_Data_Pipeline(const Async_Data_Pipeline &) = delete;
    Async_Data_Pipeline &operator=(const Async_Data_Pipeline &) = delete;
    virtual ~Async_Data_Pipeline();

    void initializeBuffers(size_t batch_size,
                           size_t input_dimension,
                           size_t output_dimension,
                           Execution_Target _execution_target = Execution_Target::VULKAN_GPU);

    void bindActiveDevice();
    void releaseDevice();

    void start();
    void stop();

    Batch_Data nextBatch(size_t batch_size, size_t input_dimension, size_t output_dimension);

    virtual size_t getBatchSize() const = 0;
    size_t getBufferSlotsCount() const noexcept { return BUFFER_SLOTS_COUNT; }
    size_t getProducerIndex() const noexcept { return producer_index.load(std::memory_order_acquire); }
    size_t getConsumerIndex() const noexcept { return consumer_index.load(std::memory_order_acquire); }
    Execution_Target getExecutionTarget() const noexcept { return execution_target.load(std::memory_order_acquire); }
    bool isRunning() const noexcept { return is_running.load(std::memory_order_acquire); }

    void setProducerIndex(size_t _index) noexcept { producer_index.store(_index, std::memory_order_release); }
    void setConsumerIndex(size_t _index) noexcept { consumer_index.store(_index, std::memory_order_release); }
    void setExecutionTarget(Execution_Target _execution_target) noexcept { execution_target.store(_execution_target, std::memory_order_release); }
    void setRunning(bool _running);
};
