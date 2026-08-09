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

            auto start_cpu = std::chrono::high_resolution_clock::now();
            Batch_Data new_batch = prepareBatch(current_batch_step);
            auto end_cpu = std::chrono::high_resolution_clock::now();

            PIPELINE_LOG_DEBUG(std::format("[Thread CPU {}] Finished loading Batch {} in {:.2f} ms",
                                           std::this_thread::get_id(), current_batch_step,
                                           std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count()));

            {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                slots[slot_idx].batch = std::move(new_batch);
                slots[slot_idx].is_ready.store(true);
            }

            cv_consumer.notify_one();
            producer_index++;
            current_batch_step++;
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
        if (is_running.load())
        {
            Logger::logMessage("Async_Data_Pipeline::start: Pipeline is already running", LOG_WARNING);
            return;
        }

        is_running.store(true);
        producer_index = 0;
        consumer_index = 0;

        for (auto &slot : slots)
        {
            slot.is_ready.store(false);
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

    Batch_Data nextBatch()
    {
        std::size_t slot_idx = consumer_index % BUFFER_COUNT;

        Batch_Data batch;
        {
            std::unique_lock<std::mutex> lock(pipeline_mutex);
            cv_consumer.wait(lock, [this, slot_idx]
                             { return slots[slot_idx].is_ready.load() || !is_running.load(); });

            if (!slots[slot_idx].is_ready.load() && !is_running.load())
            {
                Logger::logMessage("Async_Data_Pipeline::nextBatch: Pipeline stopped while waiting for batch", LOG_WARNING);
                return Batch_Data();
            }

            batch = std::move(slots[slot_idx].batch);
            slots[slot_idx].is_ready.store(false);
        }

        cv_producer.notify_one();
        consumer_index++;

        return batch;
    }
};