#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "helper/logger.h"

struct Gpu_Kernel_Stat
{
    std::string name;
    uint64_t dispatch_count = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
};

struct Gpu_Profile_Summary
{
    std::vector<Gpu_Kernel_Stat> kernels;
    double total_kernel_ms = 0.0;
    double total_span_ms = 0.0;
    uint64_t total_dispatches = 0;
    uint64_t total_barriers = 0;
    uint64_t profiled_submits = 0;
    float timestamp_period_ns = 0.0f;
    uint32_t timestamp_valid_bits = 0;
};

enum class Timing_Stage : size_t
{
    DATA_PREP = 0,
    RESET_GRAD,
    FWD_RECORD,
    FWD_LOSS_EXEC,
    FWD_LOSS_GPU_SUBMIT,
    FWD_LOSS_FENCE_WAIT,
    LOSS_READBACK,
    MUL_SCALAR,
    BWD_RECORD,
    BWD_EXEC,
    BWD_GPU_SUBMIT,
    BWD_FENCE_WAIT,
    OPT_RECORD,
    OPT_EXEC,
    OPT_GPU_SUBMIT,
    OPT_FENCE_WAIT,
    BWD_WAITED_IN_OPT,
    OPT_WAITED_IN_FWD,
    SCHED_STEP,
    TOTAL_BATCH
};

enum class Node_Stage : size_t
{
    FWD_LOSS_GPU_NODES = 0,
    BWD_GPU_NODES,
    OPT_GPU_NODES
};

class Step_Timings
{
public:
    static constexpr size_t NUM_TIMING_STAGES = static_cast<size_t>(Timing_Stage::TOTAL_BATCH) + 1;
    static constexpr size_t NUM_NODE_STAGES = static_cast<size_t>(Node_Stage::OPT_GPU_NODES) + 1;

    std::array<double, NUM_TIMING_STAGES> timings{};
    std::array<size_t, NUM_NODE_STAGES> node_counts{};
    bool is_chained_bwd_opt = false;

    static Step_Timings &getInstance();

    Step_Timings(const Step_Timings &) = delete;
    Step_Timings &operator=(const Step_Timings &) = delete;
    Step_Timings(Step_Timings &&) = delete;
    Step_Timings &operator=(Step_Timings &&) = delete;

    double &operator[](Timing_Stage stage)
    {
        return timings.at(static_cast<size_t>(stage));
    }

    double operator[](Timing_Stage stage) const
    {
        return timings.at(static_cast<size_t>(stage));
    }

    size_t &operator[](Node_Stage stage)
    {
        return node_counts.at(static_cast<size_t>(stage));
    }

    size_t operator[](Node_Stage stage) const
    {
        return node_counts.at(static_cast<size_t>(stage));
    }

    double &fwdGpuMs() { return (*this)[Timing_Stage::FWD_LOSS_EXEC]; }
    double &bwdGpuMs() { return (*this)[Timing_Stage::BWD_EXEC]; }
    double &optGpuMs() { return (*this)[Timing_Stage::OPT_EXEC]; }
    double &totalMs() { return (*this)[Timing_Stage::TOTAL_BATCH]; }
    double &lossReadMs() { return (*this)[Timing_Stage::LOSS_READBACK]; }

    void recordDuration(Timing_Stage stage, double ms);
    double getDuration(Timing_Stage stage) const;
    void setDuration(Timing_Stage stage, double ms);
    void setNodeCount(Node_Stage stage, size_t count);
    size_t getNodeCount(Node_Stage stage) const;
    void setChainedBwdOpt(bool chained);
    bool isChainedBwdOpt() const;
    void reset();
    void add(const Step_Timings &other);
    void scale(double factor);
    std::string formatTable(size_t batch_idx, size_t total_batches, double throughput, size_t count = 1, std::string_view unit = "tok/s") const;
    void logTable(size_t batch_idx, size_t total_batches, double throughput, size_t count = 1, std::string_view unit = "tok/s", Log_Level level = Log_Level::LOG_INFO, bool print_to_console = true) const;
    void printTable(size_t batch_idx, size_t total_batches, double throughput, size_t count = 1, std::string_view unit = "tok/s") const;
    void printTable(size_t batch_idx, size_t total_batches, double throughput, size_t count, std::string_view unit, std::ostream &os) const;

private:
    mutable std::mutex profiler_mutex;
    Step_Timings() = default;
    ~Step_Timings() = default;
};
