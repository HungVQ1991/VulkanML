// #include "helper/training_profiler.h"

// #include <algorithm>
// #include <iomanip>
// #include <iostream>

// Step_Timings& Step_Timings::getInstance()
// {
//     static Step_Timings instance;
//     return instance;
// }

// void Step_Timings::syncAliases()
// {
//     if (fwd_loss_gpu_submit_ms > 0.0 || fwd_loss_fence_wait_ms > 0.0)
//     {
//         fwd_loss_exec_ms = fwd_loss_gpu_submit_ms + fwd_loss_fence_wait_ms;
//     }
//     else if (fwd_loss_exec_ms == 0.0 && fwd_gpu_ms != 0.0)
//     {
//         fwd_loss_exec_ms = fwd_gpu_ms;
//     }
//     fwd_gpu_ms = fwd_loss_exec_ms;

//     if (fwd_loss_gpu_nodes == 0 && fwd_gpu_nodes != 0) fwd_loss_gpu_nodes = fwd_gpu_nodes;
//     fwd_gpu_nodes = fwd_loss_gpu_nodes;

//     if (loss_readback_ms == 0.0 && loss_read_ms != 0.0) loss_readback_ms = loss_read_ms;
//     loss_read_ms = loss_readback_ms;

//     if (bwd_gpu_submit_ms > 0.0 || bwd_fence_wait_ms > 0.0)
//     {
//         bwd_exec_ms = bwd_gpu_submit_ms + bwd_fence_wait_ms;
//     }
//     else if (bwd_exec_ms == 0.0 && bwd_gpu_ms != 0.0)
//     {
//         bwd_exec_ms = bwd_gpu_ms;
//     }
//     bwd_gpu_ms = bwd_exec_ms;

//     if (opt_gpu_submit_ms > 0.0 || opt_fence_wait_ms > 0.0)
//     {
//         opt_exec_ms = opt_gpu_submit_ms + opt_fence_wait_ms;
//     }
//     else if (opt_exec_ms == 0.0 && opt_gpu_ms != 0.0)
//     {
//         opt_exec_ms = opt_gpu_ms;
//     }
//     opt_gpu_ms = opt_exec_ms;

//     // Apply measured inter-stage fence shifts caused by double-buffering
//     if (bwd_waited_in_opt_ms > 0.0)
//     {
//         fwd_loss_bwd_ms += bwd_waited_in_opt_ms;
//         opt_step_ms = std::max(0.0, opt_step_ms - bwd_waited_in_opt_ms);
//         bwd_waited_in_opt_ms = 0.0;
//     }

//     if (opt_waited_in_fwd_ms > 0.0)
//     {
//         opt_step_ms += opt_waited_in_fwd_ms;
//         fwd_loss_bwd_ms = std::max(0.0, fwd_loss_bwd_ms - opt_waited_in_fwd_ms);
//         opt_waited_in_fwd_ms = 0.0;
//     }

//     // Structural invariance guarantees:
//     // A major stage must be at least the sum of its measured sub-operations
//     double fwd_bwd_sub_sum = fwd_record_ms + fwd_loss_exec_ms + loss_readback_ms + mul_scalar_ms + bwd_record_ms + bwd_exec_ms;
//     if (fwd_loss_bwd_ms < fwd_bwd_sub_sum)
//     {
//         double phase_shift = fwd_bwd_sub_sum - fwd_loss_bwd_ms;
//         fwd_loss_bwd_ms = fwd_bwd_sub_sum;
//         double opt_sub_sum = opt_record_ms + opt_exec_ms;
//         if (opt_step_ms > opt_sub_sum)
//         {
//             double excess = opt_step_ms - opt_sub_sum;
//             opt_step_ms -= std::min(excess, phase_shift);
//         }
//     }

//     double opt_sub_sum = opt_record_ms + opt_exec_ms;
//     if (opt_step_ms < opt_sub_sum && opt_sub_sum > 0.0)
//     {
//         opt_step_ms = opt_sub_sum;
//     }

//     double major_stages_sum = data_prep_ms + reset_grad_ms + fwd_loss_bwd_ms + opt_step_ms + sched_step_ms;
//     if (total_batch_ms < major_stages_sum)
//     {
//         total_batch_ms = major_stages_sum;
//     }
//     total_ms = total_batch_ms;
// }

// void Step_Timings::reset()
// {
//     data_prep_ms = 0.0;
//     reset_grad_ms = 0.0;
//     fwd_loss_bwd_ms = 0.0;
//     opt_step_ms = 0.0;
//     sched_step_ms = 0.0;
//     total_batch_ms = 0.0;
//     total_ms = 0.0;

//     fwd_record_ms = 0.0;
//     fwd_loss_exec_ms = 0.0;
//     fwd_gpu_ms = 0.0;
//     fwd_loss_gpu_submit_ms = 0.0;
//     fwd_loss_fence_wait_ms = 0.0;
//     fwd_loss_gpu_nodes = 0;
//     fwd_gpu_nodes = 0;

//     loss_readback_ms = 0.0;
//     loss_read_ms = 0.0;
//     mul_scalar_ms = 0.0;

//     bwd_record_ms = 0.0;
//     bwd_exec_ms = 0.0;
//     bwd_gpu_ms = 0.0;
//     bwd_gpu_submit_ms = 0.0;
//     bwd_fence_wait_ms = 0.0;
//     bwd_gpu_nodes = 0;

//     opt_record_ms = 0.0;
//     opt_exec_ms = 0.0;
//     opt_gpu_ms = 0.0;
//     opt_gpu_submit_ms = 0.0;
//     opt_fence_wait_ms = 0.0;
//     opt_gpu_nodes = 0;

//     bwd_waited_in_opt_ms = 0.0;
//     opt_waited_in_fwd_ms = 0.0;
//     is_chained_bwd_opt = false;
// }

// void Step_Timings::add(const Step_Timings& other)
// {
//     data_prep_ms += other.data_prep_ms;
//     reset_grad_ms += other.reset_grad_ms;
//     fwd_loss_bwd_ms += other.fwd_loss_bwd_ms;
//     opt_step_ms += other.opt_step_ms;
//     sched_step_ms += other.sched_step_ms;
//     total_batch_ms += other.total_batch_ms;

//     fwd_record_ms += other.fwd_record_ms;
//     fwd_loss_exec_ms += other.fwd_loss_exec_ms;
//     fwd_loss_gpu_submit_ms += other.fwd_loss_gpu_submit_ms;
//     fwd_loss_fence_wait_ms += other.fwd_loss_fence_wait_ms;
//     fwd_loss_gpu_nodes = other.fwd_loss_gpu_nodes;

//     loss_readback_ms += other.loss_readback_ms;
//     mul_scalar_ms += other.mul_scalar_ms;

//     bwd_record_ms += other.bwd_record_ms;
//     bwd_exec_ms += other.bwd_exec_ms;
//     bwd_gpu_submit_ms += other.bwd_gpu_submit_ms;
//     bwd_fence_wait_ms += other.bwd_fence_wait_ms;
//     bwd_gpu_nodes = other.bwd_gpu_nodes;

//     opt_record_ms += other.opt_record_ms;
//     opt_exec_ms += other.opt_exec_ms;
//     opt_gpu_submit_ms += other.opt_gpu_submit_ms;
//     opt_fence_wait_ms += other.opt_fence_wait_ms;
//     opt_gpu_nodes = other.opt_gpu_nodes;

//     bwd_waited_in_opt_ms += other.bwd_waited_in_opt_ms;
//     opt_waited_in_fwd_ms += other.opt_waited_in_fwd_ms;
//     is_chained_bwd_opt = is_chained_bwd_opt || other.is_chained_bwd_opt;

//     syncAliases();
// }

// void Step_Timings::scale(double factor)
// {
//     data_prep_ms *= factor;
//     reset_grad_ms *= factor;
//     fwd_loss_bwd_ms *= factor;
//     opt_step_ms *= factor;
//     sched_step_ms *= factor;
//     total_batch_ms *= factor;

//     fwd_record_ms *= factor;
//     fwd_loss_exec_ms *= factor;
//     fwd_loss_gpu_submit_ms *= factor;
//     fwd_loss_fence_wait_ms *= factor;

//     loss_readback_ms *= factor;
//     mul_scalar_ms *= factor;

//     bwd_record_ms *= factor;
//     bwd_exec_ms *= factor;
//     bwd_gpu_submit_ms *= factor;
//     bwd_fence_wait_ms *= factor;

//     opt_record_ms *= factor;
//     opt_exec_ms *= factor;
//     opt_gpu_submit_ms *= factor;
//     opt_fence_wait_ms *= factor;

//     bwd_waited_in_opt_ms *= factor;
//     opt_waited_in_fwd_ms *= factor;

//     syncAliases();
// }

// void Step_Timings::printTable(size_t batch_idx, size_t total_batches, double tok_s, size_t count)
// {
//     syncAliases();

//     double factor = (count > 0) ? (1.0 / static_cast<double>(count)) : 1.0;
//     double total = total_batch_ms * factor;
//     if (total <= 0.0) total = 1e-6;

//     auto calc = [factor](double val) { return val * factor; };

//     double v_data_prep = calc(data_prep_ms);
//     double v_reset_grad = calc(reset_grad_ms);
//     double v_fwd_loss_bwd = calc(fwd_loss_bwd_ms);
//     double v_fwd_record = calc(fwd_record_ms);
//     double v_fwd_exec = calc(fwd_loss_exec_ms);
//     double v_fwd_submit = calc(fwd_loss_gpu_submit_ms);
//     double v_fwd_fence = calc(fwd_loss_fence_wait_ms);
//     double v_loss_read = calc(loss_readback_ms);
//     double v_mul_scalar = calc(mul_scalar_ms);
//     double v_bwd_record = calc(bwd_record_ms);
//     double v_bwd_exec = calc(bwd_exec_ms);
//     double v_bwd_submit = calc(bwd_gpu_submit_ms);
//     double v_bwd_fence = calc(bwd_fence_wait_ms);

//     double fwd_bwd_sub_sum = v_fwd_record + v_fwd_exec + v_loss_read + v_mul_scalar + v_bwd_record + v_bwd_exec;
//     double fwd_bwd_overhead = std::max(0.0, v_fwd_loss_bwd - fwd_bwd_sub_sum);

//     double v_opt_step = calc(opt_step_ms);
//     double v_opt_record = calc(opt_record_ms);
//     double v_opt_exec = calc(opt_exec_ms);
//     double v_opt_submit = calc(opt_gpu_submit_ms);
//     double v_opt_fence = calc(opt_fence_wait_ms);
//     double opt_sub_sum = v_opt_record + v_opt_exec;
//     double opt_overhead = std::max(0.0, v_opt_step - opt_sub_sum);

//     double v_sched_step = calc(sched_step_ms);

//     double major_stages_sum = v_data_prep + v_reset_grad + v_fwd_loss_bwd + v_opt_step + v_sched_step;
//     double batch_loop_overhead = std::max(0.0, total - major_stages_sum);

//     std::cout << "\n  +----------------- [Training Step Profiler: Batch " << batch_idx << "/" << total_batches << "] -----------------+\n";
//     std::cout << "  | Major Stage / Sub-operation   |   Time (ms)   |  Ratio (%) | Note / Breakdown          |\n";
//     std::cout << "  +-------------------------------+---------------+------------+---------------------------+\n";
//     std::cout << std::fixed << std::setprecision(2);

//     // [1] Data Prep
//     std::cout << "  | [1] Data Prep & Upload        | " << std::setw(10) << v_data_prep << " ms | " << std::setw(8) << (v_data_prep / total * 100.0) << " % | CPU tokens + setData()    |\n";

//     // [2] Reset Gradients
//     std::cout << "  | [2] Reset Gradients           | " << std::setw(10) << v_reset_grad << " ms | " << std::setw(8) << (v_reset_grad / total * 100.0) << " % | Gradient buffer reset     |\n";

//     // [3] Forward + Loss + Backward
//     std::cout << "  | [3] Forward + Loss + Backward | " << std::setw(10) << v_fwd_loss_bwd << " ms | " << std::setw(8) << (v_fwd_loss_bwd / total * 100.0) << " % | Complete FWD + BWD        |\n";
//     std::cout << "  |   - Fwd Graph Record (CPU)    | " << std::setw(10) << v_fwd_record << " ms | " << std::setw(8) << (v_fwd_record / total * 100.0) << " % | CPU layer graph build     |\n";
//     std::cout << "  |   - Fwd + Loss GPU Execute    | " << std::setw(10) << v_fwd_exec << " ms | " << std::setw(8) << (v_fwd_exec / total * 100.0) << " % | (" << std::setw(3) << fwd_loss_gpu_nodes << " nodes)               |\n";
//     if (v_fwd_submit > 0.001 || v_fwd_fence > 0.001)
//     {
//         std::cout << "  |       * CPU Submit to Queue   | " << std::setw(10) << v_fwd_submit << " ms | " << std::setw(8) << (v_fwd_submit / total * 100.0) << " % | Compile & vkQueueSubmit   |\n";
//         std::cout << "  |       * GPU Fence Wait        | " << std::setw(10) << v_fwd_fence << " ms | " << std::setw(8) << (v_fwd_fence / total * 100.0) << " % | vkWaitForFences (GPU work)|\n";
//     }
//     std::cout << "  |   - Loss Buffer Readback      | " << std::setw(10) << v_loss_read << " ms | " << std::setw(8) << (v_loss_read / total * 100.0) << " % | Read loss scalar from GPU |\n";
//     if (v_mul_scalar > 0.001)
//     {
//         std::cout << "  |   - mulScalar Gradient Scale  | " << std::setw(10) << v_mul_scalar << " ms | " << std::setw(8) << (v_mul_scalar / total * 100.0) << " % | d_logits mulScalar        |\n";
//     }
//     std::cout << "  |   - Bwd Graph Record (CPU)    | " << std::setw(10) << v_bwd_record << " ms | " << std::setw(8) << (v_bwd_record / total * 100.0) << " % | CPU backward graph build  |\n";
//     if (is_chained_bwd_opt)
//     {
//         std::cout << "  |   - Bwd + Adam GPU Execute    | " << std::setw(10) << v_bwd_exec << " ms | " << std::setw(8) << (v_bwd_exec / total * 100.0) << " % | (" << std::setw(3) << bwd_gpu_nodes << " nodes incl. Adam)    |\n";
//     }
//     else
//     {
//         std::cout << "  |   - Bwd GPU Execute           | " << std::setw(10) << v_bwd_exec << " ms | " << std::setw(8) << (v_bwd_exec / total * 100.0) << " % | (" << std::setw(3) << bwd_gpu_nodes << " nodes)               |\n";
//     }
//     if (v_bwd_submit > 0.001 || v_bwd_fence > 0.001)
//     {
//         std::cout << "  |       * CPU Submit to Queue   | " << std::setw(10) << v_bwd_submit << " ms | " << std::setw(8) << (v_bwd_submit / total * 100.0) << " % | Compile & vkQueueSubmit   |\n";
//         std::cout << "  |       * GPU Fence Wait        | " << std::setw(10) << v_bwd_fence << " ms | " << std::setw(8) << (v_bwd_fence / total * 100.0) << " % | vkWaitForFences (GPU work)|\n";
//     }
//     if (fwd_bwd_overhead > 0.01)
//     {
//         std::cout << "  |   - FWD/BWD Pipeline Overhead | " << std::setw(10) << fwd_bwd_overhead << " ms | " << std::setw(8) << (fwd_bwd_overhead / total * 100.0) << " % | Unmeasured dispatch/alloc |\n";
//     }

//     // [4] Optimizer Step
//     if (is_chained_bwd_opt)
//     {
//         std::cout << "  | [4] Optimizer Step (Chained)  | " << std::setw(10) << v_opt_step << " ms | " << std::setw(8) << (v_opt_step / total * 100.0) << " % | Chained with BWD (0 GPU wait) |\n";
//         std::cout << "  |   - Adam Graph Record (CPU)   | " << std::setw(10) << v_opt_record << " ms | " << std::setw(8) << (v_opt_record / total * 100.0) << " % | CPU Adam node build       |\n";
//         std::cout << "  |   - Adam GPU Execute          | " << std::setw(10) << 0.00 << " ms | " << std::setw(8) << 0.00 << " % | (Fused in BWD Chained)    |\n";
//         if (opt_overhead > 0.01)
//         {
//             std::cout << "  |   - Optimizer Step Overhead   | " << std::setw(10) << opt_overhead << " ms | " << std::setw(8) << (opt_overhead / total * 100.0) << " % | Clipping/scaling overhead |\n";
//         }
//     }
//     else
//     {
//         std::cout << "  | [4] Optimizer Step            | " << std::setw(10) << v_opt_step << " ms | " << std::setw(8) << (v_opt_step / total * 100.0) << " % | Adam update pipeline      |\n";
//         std::cout << "  |   - Adam Graph Record (CPU)   | " << std::setw(10) << v_opt_record << " ms | " << std::setw(8) << (v_opt_record / total * 100.0) << " % | CPU Adam node build       |\n";
//         std::cout << "  |   - Adam GPU Execute          | " << std::setw(10) << v_opt_exec << " ms | " << std::setw(8) << (v_opt_exec / total * 100.0) << " % | (" << std::setw(3) << opt_gpu_nodes << " nodes)               |\n";
//         if (v_opt_submit > 0.001 || v_opt_fence > 0.001)
//         {
//             std::cout << "  |       * CPU Submit to Queue   | " << std::setw(10) << v_opt_submit << " ms | " << std::setw(8) << (v_opt_submit / total * 100.0) << " % | Compile & vkQueueSubmit   |\n";
//             std::cout << "  |       * GPU Fence Wait        | " << std::setw(10) << v_opt_fence << " ms | " << std::setw(8) << (v_opt_fence / total * 100.0) << " % | vkWaitForFences (GPU work)|\n";
//         }
//         if (opt_overhead > 0.01)
//         {
//             std::cout << "  |   - Optimizer Step Overhead   | " << std::setw(10) << opt_overhead << " ms | " << std::setw(8) << (opt_overhead / total * 100.0) << " % | Clipping/scaling overhead |\n";
//         }
//     }

//     // [5] LR Scheduler Step
//     if (v_sched_step > 0.001)
//     {
//         std::cout << "  | [5] LR Scheduler Step         | " << std::setw(10) << v_sched_step << " ms | " << std::setw(8) << (v_sched_step / total * 100.0) << " % | Learning rate update      |\n";
//     }

//     // [6] Batch Loop Level Overhead
//     if (batch_loop_overhead > 0.01)
//     {
//         std::cout << "  | [6] Loop Level Overhead / Sync| " << std::setw(10) << batch_loop_overhead << " ms | " << std::setw(8) << (batch_loop_overhead / total * 100.0) << " % | CPU loop/idle/sync gap    |\n";
//     }

//     std::cout << "  +-------------------------------+---------------+------------+---------------------------+\n";
//     std::cout << "  | Total Measured Batch Latency  | " << std::setw(10) << total << " ms |   100.00 % | Speed: " << std::setprecision(0) << tok_s << " tok/s       |\n";
//     std::cout << "  +-----------------------------------------------------------------------------------+\n\n";
// }
