#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "math/matrix.h"

struct Batch_Data
{
    Matrix inputs;
    Matrix targets;
};

class Async_Data_Pipeline
{
private:
    static constexpr std::size_t BUFFER_COUNT = 2;

    struct Buffer_Slot
    {
        Batch_Data batch;
        std::atomic<bool> is_ready{false};
    };

    std::array<Buffer_Slot, BUFFER_COUNT> slots;
    std::size_t producer_index = 0;
    std::size_t consumer_index = 0;

    std::atomic<bool> is_running{false};
    std::jthread worker_thread;

    std::mutex pipeline_mutex;
    std::condition_variable cv_producer;
    std::condition_variable cv_consumer;

    void workerLoop()
    {
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

            // Trong Async_Data_Pipeline::workerLoop
            auto start_cpu = std::chrono::high_resolution_clock::now();

            Batch_Data new_batch = prepareBatch(current_batch_step);

            auto end_cpu = std::chrono::high_resolution_clock::now();
            Logger::logMessage(std::format("[Thread CPU {}] Finished loading Batch {} in {:.2f} ms",
                                           std::this_thread::get_id(), current_batch_step,
                                           std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count()),
                               LOG_DEBUG, true);
            {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                slots[slot_idx].batch = std::move(new_batch);
                slots[slot_idx].is_ready.store(true);
            }

            cv_consumer.notify_one();
            producer_index++;
        }
    }

protected:
    virtual Batch_Data prepareBatch(std::size_t batch_step) = 0;

public:
    Async_Data_Pipeline() = default;

    virtual ~Async_Data_Pipeline()
    {
        stop();
    }

    void start()
    {
        is_running.store(true);
        worker_thread = std::jthread([this]
                                     { workerLoop(); });
    }

    void stop()
    {
        is_running.store(false);
        cv_producer.notify_all();
        cv_consumer.notify_all();
        if (worker_thread.joinable())
        {
            worker_thread.join();
        }
    }

    Batch_Data nextBatch()
    {
        std::size_t slot_idx = consumer_index % BUFFER_COUNT;

        {
            std::unique_lock<std::mutex> lock(pipeline_mutex);
            cv_consumer.wait(lock, [this, slot_idx]
                             { return slots[slot_idx].is_ready.load(); });
        }

        Batch_Data batch = std::move(slots[slot_idx].batch);
        slots[slot_idx].is_ready.store(false);

        cv_producer.notify_one();
        consumer_index++;

        return batch;
    }
};