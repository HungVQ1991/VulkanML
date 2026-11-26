#include "helper/training_profiler.h"
#include "helper/magic_enum.hpp"

#include <algorithm>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

Step_Timings& Step_Timings::getInstance()
{
    static Step_Timings instance;
    return instance;
}

void Step_Timings::recordDuration(Timing_Stage stage, double ms)
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    timings.at(std::to_underlying(stage)) += ms;
}

double Step_Timings::getDuration(Timing_Stage stage) const
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    return timings.at(std::to_underlying(stage));
}

void Step_Timings::setDuration(Timing_Stage stage, double ms)
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    timings.at(std::to_underlying(stage)) = ms;
}

void Step_Timings::setNodeCount(Node_Stage stage, size_t count)
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    node_counts.at(std::to_underlying(stage)) = count;
}

size_t Step_Timings::getNodeCount(Node_Stage stage) const
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    return node_counts.at(std::to_underlying(stage));
}

void Step_Timings::setChainedBwdOpt(bool chained)
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    is_chained_bwd_opt = chained;
}

bool Step_Timings::isChainedBwdOpt() const
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    return is_chained_bwd_opt;
}

void Step_Timings::reset()
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    timings.fill(0.0);
    node_counts.fill(0);
    is_chained_bwd_opt = false;
}

void Step_Timings::add(const Step_Timings& other)
{
    if (this == &other)
    {
        std::lock_guard<std::mutex> lock(profiler_mutex);
        for (size_t i = 0; i < NUM_TIMING_STAGES; ++i)
        {
            timings[i] += timings[i];
        }
        return;
    }

    std::scoped_lock lock(profiler_mutex, other.profiler_mutex);
    for (size_t i = 0; i < NUM_TIMING_STAGES; ++i)
    {
        timings[i] += other.timings[i];
    }
    for (size_t i = 0; i < NUM_NODE_STAGES; ++i)
    {
        if (other.node_counts[i] > 0)
        {
            node_counts[i] = other.node_counts[i];
        }
    }
    is_chained_bwd_opt = is_chained_bwd_opt || other.is_chained_bwd_opt;
}

void Step_Timings::scale(double factor)
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    for (double& val : timings)
    {
        val *= factor;
    }
}

std::string Step_Timings::formatTable(size_t batch_idx, size_t total_batches, double throughput, size_t count, std::string_view unit) const
{
    std::ostringstream oss;
    printTable(batch_idx, total_batches, throughput, count, unit, oss);
    return oss.str();
}

void Step_Timings::logTable(size_t batch_idx, size_t total_batches, double throughput, size_t count, std::string_view unit, Log_Level level, bool print_to_console) const
{
    if (!Logger::shouldLog(level, Log_Feature::PROFILER))
    {
        return;
    }
    std::string table = formatTable(batch_idx, total_batches, throughput, count, unit);
    while (!table.empty() && table.back() == '\n')
    {
        table.pop_back();
    }
    Logger::logMessage(table, level, print_to_console, 0, Log_Feature::PROFILER);
}

void Step_Timings::printTable(size_t batch_idx, size_t total_batches, double throughput, size_t count, std::string_view unit) const
{
    logTable(batch_idx, total_batches, throughput, count, unit, Log_Level::LOG_INFO, true);
}

void Step_Timings::printTable(size_t batch_idx, size_t total_batches, double throughput, size_t count, std::string_view unit, std::ostream& os) const
{
    std::lock_guard<std::mutex> lock(profiler_mutex);
    const double divisor = (count > 0) ? static_cast<double>(count) : 1.0;
    double total_time = timings[std::to_underlying(Timing_Stage::TOTAL_BATCH)] / divisor;
    if (total_time <= 0.0) total_time = 1e-6;

    auto get = [&](Timing_Stage s) { return timings[std::to_underlying(s)] / divisor; };

    double v_data_prep    = get(Timing_Stage::DATA_PREP);
    double v_reset_grad   = get(Timing_Stage::RESET_GRAD);
    double v_fwd_record   = get(Timing_Stage::FWD_RECORD);
    double v_fwd_submit   = get(Timing_Stage::FWD_LOSS_GPU_SUBMIT);
    double v_fwd_fence    = get(Timing_Stage::FWD_LOSS_FENCE_WAIT);
    double v_fwd_exec     = get(Timing_Stage::FWD_LOSS_EXEC);
    if (v_fwd_exec <= 0.0 && (v_fwd_submit > 0.0 || v_fwd_fence > 0.0))
    {
        v_fwd_exec = v_fwd_submit + v_fwd_fence;
    }
    double v_opt_waited_in_fwd = get(Timing_Stage::OPT_WAITED_IN_FWD);
    double v_loss_read    = get(Timing_Stage::LOSS_READBACK);
    double v_mul_scalar   = get(Timing_Stage::MUL_SCALAR);

    double v_bwd_record   = get(Timing_Stage::BWD_RECORD);
    double v_bwd_submit   = get(Timing_Stage::BWD_GPU_SUBMIT);
    double v_bwd_fence    = get(Timing_Stage::BWD_FENCE_WAIT);
    double v_bwd_exec     = get(Timing_Stage::BWD_EXEC);
    if (v_bwd_exec <= 0.0 && (v_bwd_submit > 0.0 || v_bwd_fence > 0.0))
    {
        v_bwd_exec = v_bwd_submit + v_bwd_fence;
    }

    double v_opt_record   = get(Timing_Stage::OPT_RECORD);
    double v_opt_submit   = get(Timing_Stage::OPT_GPU_SUBMIT);
    double v_opt_fence    = get(Timing_Stage::OPT_FENCE_WAIT);
    double v_opt_exec     = get(Timing_Stage::OPT_EXEC);
    double v_bwd_waited_in_opt = get(Timing_Stage::BWD_WAITED_IN_OPT);
    double v_sched_step   = get(Timing_Stage::SCHED_STEP);

    auto print_major = [&](std::string_view name, double ms) {
        double pct = (total_time > 0.0) ? (ms / total_time) * 100.0 : 0.0;
        os << std::format("| {:<35} | {:>10.3f} | {:>7.1f}% |\n", name, ms, pct);
    };

    auto print_sub = [&](std::string_view name, double ms) {
        double pct = (total_time > 0.0) ? (ms / total_time) * 100.0 : 0.0;
        os << std::format("| {:<35} | {:>10.3f} | {:>8} |\n", name, ms, std::format("({:.1f}%)", pct));
    };

    const std::string line61(61, '-');
    const std::string line37(37, '-');
    const std::string line12(12, '-');
    const std::string line10(10, '-');

    os << std::format("\n+{}+\n", line61);
    std::string header_str = std::format("Batch [{}/{}] | Throughput: {:.2f} {}", batch_idx, total_batches, throughput, unit);
    os << std::format("| {:<59} |\n", header_str);
    os << std::format("+{}+{}+{}+\n", line37, line12, line10);
    os << std::format("| {:<35} | {:>10} | {:>8} |\n", "Stage / Sub-operation", "Time (ms)", "%");
    os << std::format("+{}+{}+{}+\n", line37, line12, line10);

    size_t stage_num = 1;
    double sum_major = 0.0;

    if (v_data_prep > 0.0001)
    {
        print_major(std::format("[{}] DATA_PREP", stage_num++), v_data_prep);
        sum_major += v_data_prep;
    }

    if (v_reset_grad > 0.0001)
    {
        print_major(std::format("[{}] RESET_GRAD", stage_num++), v_reset_grad);
        sum_major += v_reset_grad;
    }

    bool has_fwd_loss = (v_fwd_record > 0.0001 || v_fwd_exec > 0.0001 || v_loss_read > 0.0001 || v_mul_scalar > 0.0001 || v_opt_waited_in_fwd > 0.0001);
    if (has_fwd_loss)
    {
        double fwd_stage_ms = v_fwd_record + v_fwd_exec + v_loss_read + v_mul_scalar;
        print_major(std::format("[{}] FORWARD & LOSS", stage_num++), fwd_stage_ms);
        sum_major += fwd_stage_ms;

        if (v_fwd_record > 0.0001)
        {
            print_sub("  |- FWD_RECORD (CPU)", v_fwd_record);
        }
        if (v_fwd_exec > 0.0001)
        {
            bool has_sub = (v_fwd_submit > 0.0001 || v_fwd_fence > 0.0001);
            std::string_view prefix = (v_loss_read > 0.0001 || v_mul_scalar > 0.0001 || v_opt_waited_in_fwd > 0.0001) ? "  |- " : "  `- ";
            print_sub(std::format("{}FWD_LOSS_EXEC (GPU)", prefix), v_fwd_exec);
            if (has_sub)
            {
                std::string_view sub_p = (v_loss_read > 0.0001 || v_mul_scalar > 0.0001 || v_opt_waited_in_fwd > 0.0001) ? "  |   -> " : "      -> ";
                if (v_fwd_submit > 0.0001) print_sub(std::format("{}GPU Submit", sub_p), v_fwd_submit);
                if (v_fwd_fence > 0.0001)  print_sub(std::format("{}Fence Wait", sub_p), v_fwd_fence);
            }
        }
        if (v_opt_waited_in_fwd > 0.0001)
        {
            std::string_view prefix = (v_loss_read > 0.0001 || v_mul_scalar > 0.0001) ? "  |- " : "  `- ";
            print_sub(std::format("{}OPT_WAITED_IN_FWD", prefix), v_opt_waited_in_fwd);
        }
        if (v_loss_read > 0.0001)
        {
            std::string_view prefix = (v_mul_scalar > 0.0001) ? "  |- " : "  `- ";
            print_sub(std::format("{}LOSS_READBACK", prefix), v_loss_read);
        }
        if (v_mul_scalar > 0.0001)
        {
            print_sub("  `- MUL_SCALAR", v_mul_scalar);
        }
    }

    if (is_chained_bwd_opt)
    {
        double chained_gpu_ms = v_bwd_submit + v_bwd_fence;
        if (chained_gpu_ms <= 0.0) chained_gpu_ms = v_bwd_exec;

        double host_overhead = std::max(0.0, v_opt_exec - chained_gpu_ms - v_opt_record);
        double chained_stage_ms = v_bwd_record + v_opt_record + chained_gpu_ms + host_overhead;
        print_major(std::format("[{}] CHAINED BWD + OPT", stage_num++), chained_stage_ms);
        sum_major += chained_stage_ms;

        if (v_bwd_record > 0.0001)
        {
            print_sub("  |- BWD_RECORD (CPU)", v_bwd_record);
        }
        if (v_opt_record > 0.0001)
        {
            print_sub("  |- OPT_RECORD (CPU)", v_opt_record);
        }
        if (chained_gpu_ms > 0.0001)
        {
            bool has_sub = (v_bwd_submit > 0.0001 || v_bwd_fence > 0.0001);
            std::string_view prefix = (host_overhead > 0.0001 || v_bwd_waited_in_opt > 0.0001) ? "  |- " : "  `- ";
            print_sub(std::format("{}BWD+OPT GPU EXEC", prefix), chained_gpu_ms);
            if (has_sub)
            {
                std::string_view sub_p = (host_overhead > 0.0001 || v_bwd_waited_in_opt > 0.0001) ? "  |   -> " : "      -> ";
                if (v_bwd_submit > 0.0001) print_sub(std::format("{}GPU Submit", sub_p), v_bwd_submit);
                if (v_bwd_fence > 0.0001)  print_sub(std::format("{}Fence Wait", sub_p), v_bwd_fence);
            }
        }
        if (v_bwd_waited_in_opt > 0.0001)
        {
            std::string_view prefix = (host_overhead > 0.0001) ? "  |- " : "  `- ";
            print_sub(std::format("{}BWD_WAITED_IN_OPT", prefix), v_bwd_waited_in_opt);
        }
        if (host_overhead > 0.0001)
        {
            print_sub("  `- Host Overhead (Sync/Clip)", host_overhead);
        }
    }
    else
    {
        bool has_bwd = (v_bwd_record > 0.0001 || v_bwd_exec > 0.0001);
        if (has_bwd)
        {
            double bwd_stage_ms = v_bwd_record + v_bwd_exec;
            print_major(std::format("[{}] BACKWARD PASS", stage_num++), bwd_stage_ms);
            sum_major += bwd_stage_ms;

            if (v_bwd_record > 0.0001)
            {
                std::string_view prefix = (v_bwd_exec > 0.0001) ? "  |- " : "  `- ";
                print_sub(std::format("{}BWD_RECORD (CPU)", prefix), v_bwd_record);
            }
            if (v_bwd_exec > 0.0001)
            {
                print_sub("  `- BWD GPU EXEC", v_bwd_exec);
                if (v_bwd_submit > 0.0001) print_sub("      -> GPU Submit", v_bwd_submit);
                if (v_bwd_fence > 0.0001)  print_sub("      -> Fence Wait", v_bwd_fence);
            }
        }

        double opt_gpu = (v_opt_submit + v_opt_fence > 0.0) ? (v_opt_submit + v_opt_fence) : 0.0;
        double opt_host = std::max(0.0, v_opt_exec - opt_gpu - v_opt_record);
        double opt_stage_ms = v_opt_record + opt_gpu + opt_host;
        if (opt_stage_ms <= 0.0 && v_opt_exec > 0.0) opt_stage_ms = v_opt_exec;

        if (opt_stage_ms > 0.0001 || v_bwd_waited_in_opt > 0.0001)
        {
            print_major(std::format("[{}] OPTIMIZER STEP", stage_num++), opt_stage_ms);
            sum_major += opt_stage_ms;

            if (v_opt_record > 0.0001)
            {
                std::string_view prefix = (opt_gpu > 0.0001 || v_bwd_waited_in_opt > 0.0001 || opt_host > 0.0001) ? "  |- " : "  `- ";
                print_sub(std::format("{}OPT_RECORD (CPU)", prefix), v_opt_record);
            }
            if (opt_gpu > 0.0001)
            {
                std::string_view prefix = (v_bwd_waited_in_opt > 0.0001 || opt_host > 0.0001) ? "  |- " : "  `- ";
                print_sub(std::format("{}OPT GPU EXEC", prefix), opt_gpu);
                std::string_view sub_p = (v_bwd_waited_in_opt > 0.0001 || opt_host > 0.0001) ? "  |   -> " : "      -> ";
                if (v_opt_submit > 0.0001) print_sub(std::format("{}GPU Submit", sub_p), v_opt_submit);
                if (v_opt_fence > 0.0001)  print_sub(std::format("{}Fence Wait", sub_p), v_opt_fence);
            }
            if (v_bwd_waited_in_opt > 0.0001)
            {
                std::string_view prefix = (opt_host > 0.0001) ? "  |- " : "  `- ";
                print_sub(std::format("{}BWD_WAITED_IN_OPT", prefix), v_bwd_waited_in_opt);
            }
            if (opt_host > 0.0001)
            {
                print_sub("  `- Host Overhead (Sync/Clip)", opt_host);
            }
        }
    }

    if (!has_fwd_loss && !is_chained_bwd_opt && (total_time > sum_major + v_sched_step + 0.001))
    {
        double step_exec = total_time - (sum_major + v_sched_step);
        print_major(std::format("[{}] MODEL_TRAIN_STEP", stage_num++), step_exec);
        sum_major += step_exec;
    }

    if (v_sched_step > 0.0001)
    {
        print_major(std::format("[{}] SCHED_STEP", stage_num++), v_sched_step);
        sum_major += v_sched_step;
    }

    if (total_time > sum_major + 0.001)
    {
        double overhead = total_time - sum_major;
        print_major(std::format("[{}] LOOP_OVERHEAD", stage_num++), overhead);
    }

    os << std::format("+{}+{}+{}+\n", line37, line12, line10);
    print_major("TOTAL_BATCH", total_time);

    os << std::format("+{}+\n", line61);
    os << std::format("| {:<59} |\n", "GPU Dispatch Node Counts:");
    for (Node_Stage node_stage : magic_enum::enum_values<Node_Stage>())
    {
        std::string_view name = magic_enum::enum_name(node_stage);
        size_t nodes = node_counts[std::to_underlying(node_stage)];
        std::string node_line = std::format("  - {:<28} : {}", name, nodes);
        os << std::format("| {:<59} |\n", node_line);
    }
    os << std::format("+{}+\n", line61);
}
