#include "test_common.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace
{

bool runLinearAndLoRATests(Execution_Target target)
{
    Scoped_Coop_Mat_Guard coop_guard;
    if (target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().setCooperativeMatrixEnabled(false);
    }
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "LUP-LoRA-GPU" : "LUP-LoRA-CPU";

    {
        Linear_Layer no_bias(6, 5, target, 1.0f, Data_Type::FLOAT32, false);
        if (no_bias.hasBias())
        {
            return failTestStep(tag, "Linear_Layer use_bias=false should report hasBias() == false");
        }
        if (no_bias.getParametersAndGradients().size() != 1)
        {
            return failTestStep(tag, "Linear_Layer use_bias=false should expose only weights parameter");
        }

        Tensor x(Shape{ 3, 6 }, generateRandomVector(18, -0.8f, 0.8f, 11), target);
        Tensor target_grad(Shape{ 3, 5 }, generateRandomVector(15, -0.6f, 0.6f, 12), target);
        Tensor y = no_bias.forward(x);
        Tensor dx = no_bias.backward(target_grad);
        flushIfGpu(target, true);

        const std::vector<float> tg_vec = target_grad.getData();
        const std::vector<float> dx_ana = dx.getData();
        const double dx_err = computeNumericalGradientMaxRelErr(
            x.getData(),
            1e-3f,
            dx_ana,
            [&](const std::vector<float> &vals) -> float {
                Tensor tx(Shape{ 3, 6 }, vals, target);
                Tensor ty = no_bias.forward(tx);
                flushIfGpu(target, true);
                const auto yv = ty.getData();
                double loss = 0.0;
                for (size_t i = 0; i < yv.size(); ++i)
                {
                    loss += static_cast<double>(yv[i]) * static_cast<double>(tg_vec[i]);
                }
                return static_cast<float>(loss);
            });
        if (dx_err > 1e-2)
        {
            return failTestStep(tag, std::format("no-bias Linear_Layer dx gradcheck failed (err={:.6f})", dx_err));
        }

        no_bias.resetGradients();
        no_bias.setTrainable(false);
        if (!no_bias.getParametersAndGradients().empty())
        {
            return failTestStep(tag, "setTrainable(false) should return empty params");
        }
        (void)no_bias.forward(x);
        Tensor dx_frozen = no_bias.backward(target_grad);
        flushIfGpu(target, true);
        if (!checkVectorsClose(dx_frozen.getData(), dx_ana, 1e-4f, tag, "frozen Linear_Layer dx mismatch"))
        {
            return false;
        }
        for (float g : no_bias.getWeightsGradient().getData())
        {
            if (std::abs(g) > 1e-6f)
            {
                return failTestStep(tag, "frozen Linear_Layer modified weight gradient");
            }
        }
    }

    {
        LoRA_Linear_Layer lora(8, 6, 3, 6.0f, 0.0f, target, 1.0f, Data_Type::FLOAT32, false);
        lora.setTrainingMode(true);
        const auto a_init = generateRandomVector(8 * 3, -0.5f, 0.5f, 21);
        const auto b_init = generateRandomVector(3 * 6, -0.5f, 0.5f, 22);
        lora.getLoRAA().setWeights(Tensor(Shape{ 8, 3 }, a_init, target));
        lora.getLoRAB().setWeights(Tensor(Shape{ 3, 6 }, b_init, target));

        if (!lora.isLoRAEnabled() || lora.isBaseTrainable())
        {
            return failTestStep(tag, "LoRA initial state mismatch");
        }
        const auto params = lora.getParametersAndGradients();
        if (params.size() != 2)
        {
            return failTestStep(tag, "LoRA with frozen base should expose 2 trainable tensors (A, B)");
        }

        Tensor x(Shape{ 4, 8 }, generateRandomVector(32, -0.9f, 0.9f, 23), target);
        Tensor dy(Shape{ 4, 6 }, generateRandomVector(24, -0.7f, 0.7f, 24), target);
        Tensor y_unmerged = lora.forward(x);
        Tensor dx = lora.backward(dy);
        flushIfGpu(target, true);
        const std::vector<float> y_unmerged_vec = y_unmerged.getData();

        const auto dy_vec = dy.getData();
        const auto da_ana = lora.getLoRAA().getWeightsGradient().getData();
        const double da_err = computeNumericalGradientMaxRelErr(
            a_init,
            1e-3f,
            da_ana,
            [&](const std::vector<float> &vals) -> float {
                lora.getLoRAA().setWeights(Tensor(Shape{ 8, 3 }, vals, target));
                Tensor ty = lora.forward(x);
                flushIfGpu(target, true);
                const auto yv = ty.getData();
                double loss = 0.0;
                for (size_t i = 0; i < yv.size(); ++i)
                {
                    loss += static_cast<double>(yv[i]) * static_cast<double>(dy_vec[i]);
                }
                return static_cast<float>(loss);
            });
        lora.getLoRAA().setWeights(Tensor(Shape{ 8, 3 }, a_init, target));
        if (da_err > 1e-2)
        {
            return failTestStep(tag, std::format("LoRA A gradcheck failed (err={:.6f})", da_err));
        }

        const auto db_ana = lora.getLoRAB().getWeightsGradient().getData();
        const double db_err = computeNumericalGradientMaxRelErr(
            b_init,
            1e-3f,
            db_ana,
            [&](const std::vector<float> &vals) -> float {
                lora.getLoRAB().setWeights(Tensor(Shape{ 3, 6 }, vals, target));
                Tensor ty = lora.forward(x);
                flushIfGpu(target, true);
                const auto yv = ty.getData();
                double loss = 0.0;
                for (size_t i = 0; i < yv.size(); ++i)
                {
                    loss += static_cast<double>(yv[i]) * static_cast<double>(dy_vec[i]);
                }
                return static_cast<float>(loss);
            });
        lora.getLoRAB().setWeights(Tensor(Shape{ 3, 6 }, b_init, target));
        if (db_err > 1e-2)
        {
            return failTestStep(tag, std::format("LoRA B gradcheck failed (err={:.6f})", db_err));
        }

        const auto dx_err = computeNumericalGradientMaxRelErr(
            x.getData(),
            1e-3f,
            dx.getData(),
            [&](const std::vector<float> &vals) -> float {
                Tensor tx(Shape{ 4, 8 }, vals, target);
                Tensor ty = lora.forward(tx);
                flushIfGpu(target, true);
                const auto yv = ty.getData();
                double loss = 0.0;
                for (size_t i = 0; i < yv.size(); ++i)
                {
                    loss += static_cast<double>(yv[i]) * static_cast<double>(dy_vec[i]);
                }
                return static_cast<float>(loss);
            });
        if (dx_err > 1e-2)
        {
            return failTestStep(tag, std::format("LoRA dx gradcheck failed (err={:.6f})", dx_err));
        }

        const float lora_tol = (target == Execution_Target::VULKAN_GPU) ? 2e-3f : 1e-4f;
        lora.mergeWeights();
        Tensor y_merged = lora.forward(x);
        flushIfGpu(target, true);
        if (!checkVectorsClose(y_unmerged_vec, y_merged.getData(), lora_tol, tag, "LoRA mergeWeights forward parity"))
        {
            return false;
        }
        lora.unmergeWeights();
        Tensor y_restored = lora.forward(x);
        flushIfGpu(target, true);
        if (!checkVectorsClose(y_unmerged_vec, y_restored.getData(), lora_tol, tag, "LoRA unmergeWeights forward parity"))
        {
            return false;
        }

        auto cloned = lora.clone();
        Tensor y_cloned = cloned->forward(x);
        flushIfGpu(target, true);
        if (!checkVectorsClose(y_unmerged_vec, y_cloned.getData(), lora_tol, tag, "LoRA clone forward parity"))
        {
            return false;
        }

        const std::filesystem::path tmp_path = makeUniqueTestTempPath("lora_ckpt");
        Scoped_Temp_File_Guard cleanup({ tmp_path.string() });
        {
            std::ofstream out(tmp_path, std::ios::binary);
            lora.saveCheckpoint(out);
        }
        LoRA_Linear_Layer loaded(8, 6, 3, 6.0f, 0.0f, target, 1.0f, Data_Type::FLOAT32, false);
        {
            std::ifstream in(tmp_path, std::ios::binary);
            loaded.loadCheckpoint(in);
        }
        Tensor y_loaded = loaded.forward(x);
        flushIfGpu(target, true);
        if (!checkVectorsClose(y_unmerged_vec, y_loaded.getData(), lora_tol, tag, "LoRA checkpoint round-trip"))
        {
            return false;
        }
    }

    return true;
}

bool testPhase1LinearAndLoRA()
{
    if (!runLinearAndLoRATests(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice())
    {
        if (!runLinearAndLoRATests(Execution_Target::VULKAN_GPU))
        {
            return false;
        }
        LoRA_Linear_Layer cpu_l(8, 6, 2, 4.0f, 0.0f, Execution_Target::CPU, 1.0f, Data_Type::FLOAT32, false);
        LoRA_Linear_Layer gpu_l(8, 6, 2, 4.0f, 0.0f, Execution_Target::VULKAN_GPU, 1.0f, Data_Type::FLOAT32, false);
        const auto w0 = generateRandomVector(48, -0.5f, 0.5f, 31);
        const auto a0 = generateRandomVector(16, -0.5f, 0.5f, 32);
        const auto b0 = generateRandomVector(12, -0.5f, 0.5f, 33);
        const auto xv = generateRandomVector(24, -1.0f, 1.0f, 34);
        const auto dyv = generateRandomVector(18, -0.5f, 0.5f, 35);

        cpu_l.getBaseLayer().setWeights(Tensor(Shape{ 8, 6 }, w0, Execution_Target::CPU));
        gpu_l.getBaseLayer().setWeights(Tensor(Shape{ 8, 6 }, w0, Execution_Target::VULKAN_GPU));
        cpu_l.getLoRAA().setWeights(Tensor(Shape{ 8, 2 }, a0, Execution_Target::CPU));
        gpu_l.getLoRAA().setWeights(Tensor(Shape{ 8, 2 }, a0, Execution_Target::VULKAN_GPU));
        cpu_l.getLoRAB().setWeights(Tensor(Shape{ 2, 6 }, b0, Execution_Target::CPU));
        gpu_l.getLoRAB().setWeights(Tensor(Shape{ 2, 6 }, b0, Execution_Target::VULKAN_GPU));

        Tensor yc = cpu_l.forward(Tensor(Shape{ 3, 8 }, xv, Execution_Target::CPU));
        Tensor yg = gpu_l.forward(Tensor(Shape{ 3, 8 }, xv, Execution_Target::VULKAN_GPU));
        Tensor dxc = cpu_l.backward(Tensor(Shape{ 3, 6 }, dyv, Execution_Target::CPU));
        Tensor dxg = gpu_l.backward(Tensor(Shape{ 3, 6 }, dyv, Execution_Target::VULKAN_GPU));
        flushGpuEngine(true);

        if (!checkVectorsClose(yc.getData(), yg.getData(), 1e-3f, "LUP-LoRA-Parity", "forward"))
        {
            return false;
        }
        if (!checkVectorsClose(dxc.getData(), dxg.getData(), 1e-3f, "LUP-LoRA-Parity", "backward dx"))
        {
            return false;
        }
        if (!checkVectorsClose(cpu_l.getLoRAA().getWeightsGradient().getData(),
                        gpu_l.getLoRAA().getWeightsGradient().getData(), 1e-3f, "LUP-LoRA-Parity", "backward dA"))
        {
            return false;
        }
    }
    return true;
}

bool runTiedAndIndexedTests(Execution_Target target)
{
    Scoped_Coop_Mat_Guard coop_guard;
    if (target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().setCooperativeMatrixEnabled(false);
    }
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "LUP-Tied-GPU" : "LUP-Tied-CPU";

    {
        const size_t V = 13;
        const size_t D = 8;
        Embedding_Layer emb(V, D, target, Data_Type::FLOAT32);
        Linear_Layer lm_head(D, V, target, 1.0f, Data_Type::FLOAT32, false);
        const auto w_init = generateRandomVector(V * D, -0.4f, 0.4f, 41);
        emb.getWeight().uploadData(w_init);
        lm_head.tieWeights(&emb.getWeight(), &emb.getWeightGradient(), true);
        emb.setAccumulateForTiedWeights(true);

        std::vector<int32_t> tok_ids{ 1, 5, 1, 9 };
        Tensor tok_tensor(Shape{ 1, 4 }, { 1.0f, 5.0f, 1.0f, 9.0f }, target);
        Tensor hidden = emb.forward(tok_tensor);
        Tensor logits = lm_head.forward(hidden.view(Shape{ 4, D }));

        const auto d_logits_vec = generateRandomVector(4 * V, -0.5f, 0.5f, 42);
        Tensor d_logits(Shape{ 4, V }, d_logits_vec, target);
        Tensor d_hidden = lm_head.backward(d_logits);
        (void)emb.backward(d_hidden.view(Shape{ 1, 4, D }));
        flushIfGpu(target, true);

        std::vector<float> expected_grad(V * D, 0.0f);
        const auto h_vec = hidden.getData();
        const auto dh_vec = d_hidden.getData();
        for (size_t r = 0; r < 4; ++r)
        {
            for (size_t v = 0; v < V; ++v)
            {
                for (size_t d = 0; d < D; ++d)
                {
                    expected_grad[v * D + d] += h_vec[r * D + d] * d_logits_vec[r * V + v];
                }
            }
            const size_t tid = static_cast<size_t>(tok_ids[r]);
            for (size_t d = 0; d < D; ++d)
            {
                expected_grad[tid * D + d] += dh_vec[r * D + d];
            }
        }
        const float tied_tol = (target == Execution_Target::VULKAN_GPU) ? 2e-3f : 1e-4f;
        if (!checkVectorsClose(emb.getWeightGradient().getData(), expected_grad, tied_tol, tag, "tied weight grad accumulation"))
        {
            return false;
        }
        const auto emb_row = emb.getWeight().getData();
        const auto logits_vec = logits.getData();
        double dot_check = 0.0;
        for (size_t d = 0; d < D; ++d)
        {
            dot_check += static_cast<double>(h_vec[d]) * static_cast<double>(emb_row[3 * D + d]);
        }
        if (std::abs(dot_check - static_cast<double>(logits_vec[3])) > 1e-4)
        {
            return failTestStep(tag, "tied forward dot product mismatch");
        }
    }

    {
        const size_t N = 6;
        const size_t D = 10;
        const size_t V = 11;
        Linear_Layer full_lm(D, V, target, 1.0f, Data_Type::FLOAT32, true);
        Linear_Layer idx_lm(D, V, target, 1.0f, Data_Type::FLOAT32, true);
        const auto w_data = generateRandomVector(D * V, -0.5f, 0.5f, 43);
        const auto b_data = generateRandomVector(V, -0.2f, 0.2f, 44);
        full_lm.setWeights(Tensor(Shape{ D, V }, w_data, target));
        full_lm.setBiases(Tensor(Shape{ 1, V }, b_data, target));
        idx_lm.setWeights(Tensor(Shape{ D, V }, w_data, target));
        idx_lm.setBiases(Tensor(Shape{ 1, V }, b_data, target));

        const auto h_data = generateRandomVector(N * D, -0.7f, 0.7f, 45);
        Tensor h_full(Shape{ N, D }, h_data, target);
        Tensor h_idx(Shape{ N, D }, h_data, target);
        Tensor target_indices(Shape{ 4 }, { 1.0f, 4.0f, 1.0f, 5.0f }, target);

        Tensor full_logits = full_lm.forward(h_full);
        Tensor sel_logits = idx_lm.forwardIndexed(h_idx, target_indices);

        const auto sel_dy = generateRandomVector(4 * V, -0.4f, 0.4f, 46);
        Tensor sel_dy_t(Shape{ 4, V }, sel_dy, target);
        std::vector<float> full_dy(N * V, 0.0f);
        const std::vector<size_t> idx_list{ 1, 4, 1, 5 };
        for (size_t k = 0; k < idx_list.size(); ++k)
        {
            const size_t row = idx_list[k];
            for (size_t v = 0; v < V; ++v)
            {
                full_dy[row * V + v] += sel_dy[k * V + v];
            }
        }

        Tensor dx_full = full_lm.backward(Tensor(Shape{ N, V }, full_dy, target));
        Tensor dx_idx = idx_lm.backwardIndexed(sel_dy_t, target_indices, Shape{ N, D });
        flushIfGpu(target, true);

        const auto fl_vec = full_logits.getData();
        const auto sl_vec = sel_logits.getData();
        for (size_t k = 0; k < idx_list.size(); ++k)
        {
            std::vector<float> exp_row(fl_vec.begin() + idx_list[k] * V, fl_vec.begin() + (idx_list[k] + 1) * V);
            std::vector<float> got_row(sl_vec.begin() + k * V, sl_vec.begin() + (k + 1) * V);
            if (!checkVectorsClose(exp_row, got_row, 1e-4f, tag, "forwardIndexed row parity"))
            {
                return false;
            }
        }
        if (!checkVectorsClose(dx_full.getData(), dx_idx.getData(), 1e-4f, tag, "backwardIndexed dx parity") ||
            !checkVectorsClose(full_lm.getWeightsGradient().getData(), idx_lm.getWeightsGradient().getData(), 1e-4f, tag, "backwardIndexed dW parity") ||
            !checkVectorsClose(full_lm.getBiasesGradient().getData(), idx_lm.getBiasesGradient().getData(), 1e-4f, tag, "backwardIndexed db parity"))
        {
            return false;
        }
    }

    return true;
}

bool testPhase2TiedAndIndexedLM()
{
    if (!runTiedAndIndexedTests(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice() && !runTiedAndIndexedTests(Execution_Target::VULKAN_GPU))
    {
        return false;
    }
    return true;
}

bool runRoPEAnd4DNrmTests(Execution_Target target)
{
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "LUP-RoPE-GPU" : "LUP-RoPE-CPU";

    {
        const size_t B = 2;
        const size_t S = 4;
        const size_t H = 3;
        const size_t D = 8;
        Tensor in_t(Shape{ B, S, H, D }, generateRandomVector(B * S * H * D, -1.0f, 1.0f, 51), target);
        Tensor pos_ids(Shape{ B, S }, { 0.0f, 4.0f, 4.0f, 9.0f, 7.0f, 2.0f, 2.0f, 11.0f }, target);

        ops::llm::RoPE_Config fwd_cfg{};
        fwd_cfg.seq_len = static_cast<uint32_t>(S);
        fwd_cfg.head_dim = static_cast<uint32_t>(D);
        fwd_cfg.direction = ops::llm::RoPE_Direction::FORWARD;
        fwd_cfg.base = 10000.0f;
        fwd_cfg.num_heads = static_cast<uint32_t>(H);
        fwd_cfg.mode = ops::llm::RoPE_Mode::BSHD_INPLACE;
        fwd_cfg.position_ids = &pos_ids;
        fwd_cfg.yarn.enabled = true;
        fwd_cfg.yarn.factor = 4.0f;
        fwd_cfg.yarn.original_max_position_embeddings = 32u;
        fwd_cfg.yarn.beta_fast = 32.0f;
        fwd_cfg.yarn.beta_slow = 1.0f;
        fwd_cfg.yarn.attention_factor = 1.0f;

        Tensor out_fwd(target);
        ops::llm::applyRoPE(in_t, out_fwd, fwd_cfg);

        ops::llm::RoPE_Config bwd_cfg = fwd_cfg;
        bwd_cfg.direction = ops::llm::RoPE_Direction::BACKWARD;
        Tensor out_inv(target);
        ops::llm::applyRoPE(out_fwd, out_inv, bwd_cfg);
        flushIfGpu(target, true);

        const float rope_tol = (target == Execution_Target::VULKAN_GPU) ? 2e-3f : 1e-4f;
        if (!checkVectorsClose(in_t.getData(), out_inv.getData(), rope_tol, tag, "RoPE position_ids+YaRN round-trip"))
        {
            return false;
        }
    }

    {
        Scoped_Coop_Mat_Guard coop_guard;
        if (target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().setCooperativeMatrixEnabled(false);
        }
        RMSNorm_Layer qk_norm(8, 1e-6f, target, Data_Type::FLOAT32);
        qk_norm.getParametersAndGradients()[0].first->uploadData(generateRandomVector(8, 0.5f, 1.5f, 52));
        const Shape s4{ 2, 3, 2, 8 };
        const auto xv = generateRandomVector(s4.getTotalElements(), -0.8f, 0.8f, 53);
        const auto dyv = generateRandomVector(s4.getTotalElements(), -0.5f, 0.5f, 54);
        Tensor x(s4, xv, target);
        Tensor dy(s4, dyv, target);

        Tensor y = qk_norm.forward(x);
        Tensor dx = qk_norm.backward(dy);
        flushIfGpu(target, true);

        if (y.getShape() != s4 || dx.getShape() != s4)
        {
            return failTestStep(tag, "4D RMSNorm shape mismatch");
        }

        const double dx_err = computeNumericalGradientMaxRelErr(
            xv,
            1e-3f,
            dx.getData(),
            [&](const std::vector<float> &vals) -> float {
                Tensor tx(s4, vals, target);
                Tensor ty = qk_norm.forward(tx);
                flushIfGpu(target, true);
                const auto yv = ty.getData();
                double loss = 0.0;
                for (size_t i = 0; i < yv.size(); ++i)
                {
                    loss += static_cast<double>(yv[i]) * static_cast<double>(dyv[i]);
                }
                return static_cast<float>(loss);
            });
        if (dx_err > 1e-2)
        {
            return failTestStep(tag, std::format("4D RMSNorm dx gradcheck failed (err={:.6f})", dx_err));
        }
    }

    return true;
}

bool testPhase3RoPEAnd4DQKNorm()
{
    if (!runRoPEAnd4DNrmTests(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice())
    {
        if (!runRoPEAnd4DNrmTests(Execution_Target::VULKAN_GPU))
        {
            return false;
        }
        const Shape qshape{ 2, 4, 2, 8 };
        const auto qv = generateRandomVector(qshape.getTotalElements(), -1.0f, 1.0f, 55);
        const std::vector<float> pv{ 0.0f, 5.0f, 2.0f, 9.0f, 3.0f, 3.0f, 7.0f, 1.0f };
        Tensor qc(qshape, qv, Execution_Target::CPU), qg(qshape, qv, Execution_Target::VULKAN_GPU);
        Tensor pc(Shape{ 2, 4 }, pv, Execution_Target::CPU), pg(Shape{ 2, 4 }, pv, Execution_Target::VULKAN_GPU);

        ops::llm::RoPE_Config cc{}, cg{};
        cc.seq_len = 4; cc.head_dim = 8; cc.num_heads = 2; cc.mode = ops::llm::RoPE_Mode::BSHD_INPLACE;
        cc.yarn.enabled = true; cc.yarn.factor = 4.0f; cc.yarn.original_max_position_embeddings = 16;
        cg = cc;
        cc.position_ids = &pc;
        cg.position_ids = &pg;

        Tensor oc(Execution_Target::CPU), og(Execution_Target::VULKAN_GPU);
        ops::llm::applyRoPE(qc, oc, cc);
        ops::llm::applyRoPE(qg, og, cg);
        flushGpuEngine(true);
        if (!checkVectorsClose(oc.getData(), og.getData(), 2e-3f, "LUP-RoPE-Parity", "position_ids+YaRN CPU vs GPU"))
        {
            return false;
        }
    }
    return true;
}

bool runAttentionMaskAndKvIndexTests(Execution_Target target)
{
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "LUP-Attn-GPU" : "LUP-Attn-CPU";

    {
        const uint32_t B = 1, Q = 4, K = 4, H = 2, KVH = 1, D = 8;
        const auto qv = generateRandomVector(B * Q * H * D, -0.7f, 0.7f, 61);
        const auto kv = generateRandomVector(B * Q * KVH * D, -0.7f, 0.7f, 62);
        const auto vv = generateRandomVector(B * Q * KVH * D, -0.7f, 0.7f, 63);
        const auto dov = generateRandomVector(B * Q * H * D, -0.5f, 0.5f, 64);

        auto kv_mut = kv;
        auto vv_mut = vv;
        for (size_t d = 0; d < KVH * D; ++d)
        {
            kv_mut[1 * KVH * D + d] += 13.0f;
            vv_mut[1 * KVH * D + d] -= 9.0f;
        }

        std::vector<float> mask_vals{
            1, 0, 0, 0,
            1, 1, 0, 0,
            1, 0, 1, 0,
            0, 0, 0, 0
        };
        Tensor q_t(Shape{ B, Q, H, D }, qv, target);
        Tensor k1(Shape{ B, K, KVH, D }, kv, target), v1(Shape{ B, K, KVH, D }, vv, target);
        Tensor k2(Shape{ B, K, KVH, D }, kv_mut, target), v2(Shape{ B, K, KVH, D }, vv_mut, target);
        Tensor mask_t(Shape{ B, Q, K }, mask_vals, target);
        Tensor do_t(Shape{ B, Q, H, D }, dov, target);

        ops::llm::Attention_Config cfg{};
        cfg.num_heads = H;
        cfg.num_kv_heads = KVH;
        cfg.seq_len = Q;
        cfg.kv_seq_len = K;
        cfg.head_dim = D;
        cfg.is_bshd = true;
        cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
        cfg.attn_mask = &mask_t;

        Tensor o1(target), l1(target), dq1(target), dk1(target), dv1(target);
        Tensor o2(target), l2(target), dq2(target), dk2(target), dv2(target);
        ops::llm::flashAttentionForward(q_t, k1, v1, o1, cfg, &l1);
        ops::llm::flashAttentionBackward(q_t, k1, v1, o1, do_t, dq1, dk1, dv1, cfg, &l1);
        ops::llm::flashAttentionForward(q_t, k2, v2, o2, cfg, &l2);
        ops::llm::flashAttentionBackward(q_t, k2, v2, o2, do_t, dq2, dk2, dv2, cfg, &l2);
        flushIfGpu(target, true);

        const auto o1_d = o1.getData();
        const auto o2_d = o2.getData();
        const auto dq1_d = dq1.getData();
        const auto dq2_d = dq2.getData();
        for (size_t d = 0; d < H * D; ++d)
        {
            if (std::abs(o1_d[3 * H * D + d]) > 1e-6f || std::abs(dq1_d[3 * H * D + d]) > 1e-6f)
            {
                return failTestStep(tag, "all-masked query row must output 0 with 0 dq");
            }
            if (std::abs(o1_d[2 * H * D + d] - o2_d[2 * H * D + d]) > 1e-5f ||
                std::abs(dq1_d[2 * H * D + d] - dq2_d[2 * H * D + d]) > 1e-5f)
            {
                return failTestStep(tag, "SealedRead Property 1 invariance violated on GEN row");
            }
        }
    }

    {
        const uint32_t B = 1, Q = 2, K_full = 4, H = 2, KVH = 1, D = 8;
        const auto qv = generateRandomVector(B * Q * H * D, -0.6f, 0.6f, 65);
        const auto kv = generateRandomVector(B * K_full * KVH * D, -0.6f, 0.6f, 66);
        const auto vv = generateRandomVector(B * K_full * KVH * D, -0.6f, 0.6f, 67);
        const auto dov = generateRandomVector(B * Q * H * D, -0.5f, 0.5f, 68);

        Tensor q_t(Shape{ B, Q, H, D }, qv, target);
        Tensor k_full(Shape{ B, K_full, KVH, D }, kv, target);
        Tensor v_full(Shape{ B, K_full, KVH, D }, vv, target);
        Tensor do_t(Shape{ B, Q, H, D }, dov, target);
        Tensor kv_idx(Shape{ 2 }, { 0.0f, 2.0f }, target);
        Tensor mask_sub(Shape{ B, Q, 2 }, { 1.0f, 1.0f, 1.0f, 1.0f }, target);

        std::vector<float> k_sub_v(2 * KVH * D), v_sub_v(2 * KVH * D);
        std::copy_n(kv.data() + 0 * KVH * D, KVH * D, k_sub_v.data());
        std::copy_n(kv.data() + 2 * KVH * D, KVH * D, k_sub_v.data() + KVH * D);
        std::copy_n(vv.data() + 0 * KVH * D, KVH * D, v_sub_v.data());
        std::copy_n(vv.data() + 2 * KVH * D, KVH * D, v_sub_v.data() + KVH * D);
        Tensor k_sub(Shape{ B, 2, KVH, D }, k_sub_v, target);
        Tensor v_sub(Shape{ B, 2, KVH, D }, v_sub_v, target);

        ops::llm::Attention_Config idx_cfg{};
        idx_cfg.num_heads = H;
        idx_cfg.num_kv_heads = KVH;
        idx_cfg.seq_len = Q;
        idx_cfg.head_dim = D;
        idx_cfg.is_bshd = true;
        idx_cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
        idx_cfg.attn_mask = &mask_sub;
        idx_cfg.kv_index = &kv_idx;

        ops::llm::Attention_Config sub_cfg = idx_cfg;
        sub_cfg.kv_seq_len = 2;
        sub_cfg.kv_index = nullptr;

        Tensor o_idx(target), l_idx(target), dq_idx(target), dk_idx(target), dv_idx(target);
        Tensor o_sub(target), l_sub(target), dq_sub(target), dk_sub(target), dv_sub(target);
        ops::llm::flashAttentionForward(q_t, k_full, v_full, o_idx, idx_cfg, &l_idx);
        ops::llm::flashAttentionBackward(q_t, k_full, v_full, o_idx, do_t, dq_idx, dk_idx, dv_idx, idx_cfg, &l_idx);
        ops::llm::flashAttentionForward(q_t, k_sub, v_sub, o_sub, sub_cfg, &l_sub);
        ops::llm::flashAttentionBackward(q_t, k_sub, v_sub, o_sub, do_t, dq_sub, dk_sub, dv_sub, sub_cfg, &l_sub);
        flushIfGpu(target, true);

        const float kv_idx_tol = (target == Execution_Target::VULKAN_GPU) ? 2e-3f : 1e-4f;
        if (!checkVectorsClose(o_idx.getData(), o_sub.getData(), kv_idx_tol, tag, "kv_index forward parity") ||
            !checkVectorsClose(dq_idx.getData(), dq_sub.getData(), kv_idx_tol, tag, "kv_index dq parity"))
        {
            return false;
        }
        const auto dk_idx_d = dk_idx.getData();
        const auto dk_sub_d = dk_sub.getData();
        for (size_t d = 0; d < KVH * D; ++d)
        {
            if (std::abs(dk_idx_d[1 * KVH * D + d]) > 1e-6f || std::abs(dk_idx_d[3 * KVH * D + d]) > 1e-6f)
            {
                return failTestStep(tag, "kv_index unselected KV slots must receive 0 gradient");
            }
            if (std::abs(dk_idx_d[0 * KVH * D + d] - dk_sub_d[0 * KVH * D + d]) > kv_idx_tol ||
                std::abs(dk_idx_d[2 * KVH * D + d] - dk_sub_d[1 * KVH * D + d]) > kv_idx_tol)
            {
                return failTestStep(tag, "kv_index dk scatter mismatch");
            }
        }
    }

    return true;
}

bool testPhase4FlashAttentionMasked()
{
    if (!runAttentionMaskAndKvIndexTests(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice())
    {
        if (!runAttentionMaskAndKvIndexTests(Execution_Target::VULKAN_GPU))
        {
            return false;
        }
        const uint32_t B = 2, Q = 4, K = 4, H = 4, KVH = 2, D = 8;
        const auto qv = generateRandomVector(B * Q * H * D, -0.7f, 0.7f, 69);
        const auto kv = generateRandomVector(B * K * KVH * D, -0.7f, 0.7f, 70);
        const auto vv = generateRandomVector(B * K * KVH * D, -0.7f, 0.7f, 71);
        const auto dov = generateRandomVector(B * Q * H * D, -0.5f, 0.5f, 72);
        const std::vector<float> mask_v{
            1, 0, 0, 0,
            1, 1, 0, 0,
            1, 0, 1, 0,
            1, 0, 1, 1,
            1, 1, 0, 0,
            0, 1, 0, 0,
            1, 1, 1, 0,
            0, 0, 0, 0
        };
        Tensor qc(Shape{ B, Q, H, D }, qv, Execution_Target::CPU), kc(Shape{ B, K, KVH, D }, kv, Execution_Target::CPU), vc(Shape{ B, K, KVH, D }, vv, Execution_Target::CPU);
        Tensor doc(Shape{ B, Q, H, D }, dov, Execution_Target::CPU), mc(Shape{ B, Q, K }, mask_v, Execution_Target::CPU);
        Tensor qg(Shape{ B, Q, H, D }, qv, Execution_Target::VULKAN_GPU), kg(Shape{ B, K, KVH, D }, kv, Execution_Target::VULKAN_GPU), vg(Shape{ B, K, KVH, D }, vv, Execution_Target::VULKAN_GPU);
        Tensor dog(Shape{ B, Q, H, D }, dov, Execution_Target::VULKAN_GPU), mg(Shape{ B, Q, K }, mask_v, Execution_Target::VULKAN_GPU);

        ops::llm::Attention_Config cc{};
        cc.num_heads = H; cc.num_kv_heads = KVH; cc.seq_len = Q; cc.kv_seq_len = K; cc.head_dim = D;
        cc.is_bshd = true; cc.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
        auto cg = cc;
        cc.attn_mask = &mc;
        cg.attn_mask = &mg;

        Tensor oc(Execution_Target::CPU), lc(Execution_Target::CPU), dqc(Execution_Target::CPU), dkc(Execution_Target::CPU), dvc(Execution_Target::CPU);
        Tensor og(Execution_Target::VULKAN_GPU), lg(Execution_Target::VULKAN_GPU), dqg(Execution_Target::VULKAN_GPU), dkg(Execution_Target::VULKAN_GPU), dvg(Execution_Target::VULKAN_GPU);
        ops::llm::flashAttentionForward(qc, kc, vc, oc, cc, &lc);
        ops::llm::flashAttentionBackward(qc, kc, vc, oc, doc, dqc, dkc, dvc, cc, &lc);
        ops::llm::flashAttentionForward(qg, kg, vg, og, cg, &lg);
        ops::llm::flashAttentionBackward(qg, kg, vg, og, dog, dqg, dkg, dvg, cg, &lg);
        flushGpuEngine(true);

        if (!checkVectorsClose(oc.getData(), og.getData(), 1e-3f, "LUP-Attn-Parity", "forward CPU vs GPU") ||
            !checkVectorsClose(dqc.getData(), dqg.getData(), 1e-3f, "LUP-Attn-Parity", "dq CPU vs GPU") ||
            !checkVectorsClose(dkc.getData(), dkg.getData(), 1e-3f, "LUP-Attn-Parity", "dk CPU vs GPU") ||
            !checkVectorsClose(dvc.getData(), dvg.getData(), 1e-3f, "LUP-Attn-Parity", "dv CPU vs GPU"))
        {
            return false;
        }
    }
    return true;
}

bool runQwen3BlockTests(Execution_Target target)
{
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "LUP-Block-GPU" : "LUP-Block-CPU";

    Transformer_Block_Config cfg;
    cfg.hidden_dim = 32;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 2;
    cfg.intermediate_dim = 48;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_base = 10000.0f;
    cfg.execution_target = target;
    cfg.data_type = Data_Type::FLOAT32;
    cfg.num_layers = 2;
    cfg.residual_scale = 1.0f;
    cfg.use_qk_norm = true;
    cfg.use_attention_bias = false;
    cfg.use_mlp_bias = false;
    cfg.gradient_checkpointing = false;

    Transformer_Block ref_blk(cfg);
    ref_blk.setTrainingMode(true);
    auto ckpt_blk_base = ref_blk.clone();
    auto *ckpt_blk = dynamic_cast<Transformer_Block *>(ckpt_blk_base.get());
    ckpt_blk->setGradientCheckpointing(true);
    ckpt_blk->setTrainingMode(true);

    if (ref_blk.getQProj().getOutputDimension() != 64 ||
        ref_blk.getKProj().getOutputDimension() != 32 ||
        ref_blk.getOProj().getInputDimension() != 64 ||
        ref_blk.getQNorm() == nullptr || ref_blk.getKNorm() == nullptr ||
        ref_blk.getQProj().hasBias() || ref_blk.getGateProj().hasBias())
    {
        return failTestStep(tag, "Qwen3 Transformer_Block geometry or bias mismatch");
    }

    const size_t B = 2, S = 4;
    Tensor x(Shape{ B, S, 32 }, generateRandomVector(B * S * 32, -0.5f, 0.5f, 81), target);
    Tensor pos_ids(Shape{ B, S }, { 0.0f, 1.0f, 3.0f, 4.0f, 0.0f, 2.0f, 5.0f, 6.0f }, target);
    Tensor mask_t(Shape{ B, S, S }, {
        1, 0, 0, 0,  1, 1, 0, 0,  1, 0, 1, 0,  1, 0, 1, 1,
        1, 0, 0, 0,  1, 1, 0, 0,  1, 1, 1, 0,  1, 1, 1, 1
    }, target);
    ops::llm::Attention_Config attn_cfg{};
    attn_cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
    attn_cfg.attn_mask = &mask_t;
    Tensor dy(Shape{ B, S, 32 }, generateRandomVector(B * S * 32, -0.3f, 0.3f, 82), target);

    Tensor y_ref = ref_blk.forward(x, nullptr, &pos_ids, &attn_cfg);
    Tensor dx_ref = ref_blk.backward(dy);
    Tensor y_ckpt = ckpt_blk->forward(x, nullptr, &pos_ids, &attn_cfg);
    Tensor dx_ckpt = ckpt_blk->backward(dy);
    flushIfGpu(target, true);

    if (!checkVectorsClose(y_ref.getData(), y_ckpt.getData(), 1e-4f, tag, "ckpt forward parity") ||
        !checkVectorsClose(dx_ref.getData(), dx_ckpt.getData(), 1e-4f, tag, "ckpt backward dx parity"))
    {
        return false;
    }
    auto p_ref = ref_blk.getParametersAndGradients();
    auto p_ckpt = ckpt_blk->getParametersAndGradients();
    if (p_ref.size() != p_ckpt.size())
    {
        return failTestStep(tag, "parameter count mismatch between ref and ckpt block");
    }
    for (size_t i = 0; i < p_ref.size(); ++i)
    {
        if (!checkVectorsClose(p_ref[i].second->getData(), p_ckpt[i].second->getData(), 1e-4f, tag, "ckpt param grad parity"))
        {
            return false;
        }
    }

    ref_blk.resetGradients();
    ref_blk.enableLoRA(4, 8.0f, 0.0f, { "q_proj", "o_proj", "down_proj" }, true);
    if (!ref_blk.hasActiveLoRA() || ref_blk.getParametersAndGradients().size() != 6)
    {
        return failTestStep(tag, "LoRA on Transformer_Block should expose 6 trainable tensors");
    }
    ref_blk.getQLoRA().getLoRAB().setWeights(Tensor(Shape{ 4, 64 }, generateRandomVector(4 * 64, -0.2f, 0.2f, 83), target));
    Tensor y_lora = ref_blk.forward(x, nullptr, &pos_ids, &attn_cfg);
    (void)ref_blk.backward(dy);
    flushIfGpu(target, true);

    const std::filesystem::path ckpt_file = makeUniqueTestTempPath("tb_qwen3");
    const std::filesystem::path cfg_file = makeUniqueTestTempPath("tb_cfg");
    Scoped_Temp_File_Guard guard({ ckpt_file.string(), cfg_file.string() });
    {
        std::ofstream oc(cfg_file, std::ios::binary);
        ref_blk.saveConfiguration(oc);
        std::ofstream ow(ckpt_file, std::ios::binary);
        ref_blk.saveCheckpoint(ow);
    }
    std::unique_ptr<ILayer> rebuilt;
    {
        std::ifstream ic(cfg_file, std::ios::binary);
        rebuilt = Training_Context::constructLayerFromConfig(ic, Layer_Type::TRANSFORMER_BLOCK, target);
    }
    auto *loaded_blk = dynamic_cast<Transformer_Block *>(rebuilt.get());
    if (!loaded_blk || !loaded_blk->usesQKNorm() || loaded_blk->getHeadDim() != 16 ||
        loaded_blk->usesAttentionBias() || loaded_blk->usesMlpBias())
    {
        return failTestStep(tag, "Training_Context::constructLayerFromConfig EXT metadata mismatch");
    }
    {
        std::ifstream iw(ckpt_file, std::ios::binary);
        loaded_blk->loadCheckpoint(iw);
    }
    Tensor y_loaded = loaded_blk->forward(x, nullptr, &pos_ids, &attn_cfg);
    flushIfGpu(target, true);
    if (!checkVectorsClose(y_lora.getData(), y_loaded.getData(), 1e-4f, tag, "Qwen3+LoRA block checkpoint round-trip"))
    {
        return false;
    }

    return true;
}

bool testPhase5Qwen3Block()
{
    if (!runQwen3BlockTests(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice() && !runQwen3BlockTests(Execution_Target::VULKAN_GPU))
    {
        return false;
    }
    return true;
}

bool runKVCacheAndCausalLMTests(Execution_Target target)
{
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "LUP-CLM-GPU" : "LUP-CLM-CPU";

    {
        KV_Cache_Manager kv(1, 2, 4, 8, target, Data_Type::FLOAT32, KV_Cache_Mode::STATIC, true);
        Tensor k1(Shape{ 1, 2, 3, 8 }, generateRandomVector(48, -1.0f, 1.0f, 91), target);
        Tensor v1(Shape{ 1, 2, 3, 8 }, generateRandomVector(48, -1.0f, 1.0f, 92), target);
        kv.append(k1, v1);
        Tensor k2(Shape{ 1, 2, 3, 8 }, generateRandomVector(48, -1.0f, 1.0f, 93), target);
        Tensor v2(Shape{ 1, 2, 3, 8 }, generateRandomVector(48, -1.0f, 1.0f, 94), target);
        kv.append(k2, v2);
        flushIfGpu(target, true);
        if (kv.getCurrentSeqLen() != 6 || kv.getMaxSeqLen() < 6)
        {
            return failTestStep(tag, "KV_Cache_Manager auto_grow failed");
        }
        auto kv_clone = kv.clone(4);
        flushIfGpu(target, true);
        if (kv_clone->getCurrentSeqLen() != 4 || kv_clone->getTotalTokensSeen() != 4)
        {
            return failTestStep(tag, "KV_Cache_Manager clone(4) len mismatch");
        }
        kv.truncate(2);
        if (kv.getCurrentSeqLen() != 2 || kv.getTotalTokensSeen() != 2)
        {
            return failTestStep(tag, "KV_Cache_Manager truncate(2) len mismatch");
        }
    }

    Causal_LM_Config cfg;
    cfg.vocab_size = 29;
    cfg.hidden_dim = 32;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 2;
    cfg.intermediate_dim = 48;
    cfg.num_layers = 2;
    cfg.max_seq_len = 16;
    cfg.rms_norm_eps = 1e-6f;
    cfg.execution_target = target;
    cfg.data_type = Data_Type::FLOAT32;
    cfg.use_qk_norm = true;
    cfg.attention_bias = false;
    cfg.mlp_bias = false;
    cfg.tie_word_embeddings = true;
    cfg.residual_scale = 1.0f;
    cfg.gradient_checkpointing = false;
    cfg.tokenizer_path = "";

    Causal_LM ref_lm(cfg);
    Causal_LM tgt_lm(cfg);
    if (!ref_lm.getLMHead().isWeightTied() || ref_lm.getLMHead().hasBias() ||
        !ref_lm.getTokenEmbedding().isAccumulatingForTiedWeights())
    {
        return failTestStep(tag, "Causal_LM tied embeddings setup failed");
    }

    const std::filesystem::path sync_file = makeUniqueTestTempPath("clm_qwen3");
    Scoped_Temp_File_Guard guard({ sync_file.string() });
    ref_lm.saveCheckpoint(sync_file.string());
    tgt_lm.loadCheckpoint(sync_file.string());
    if (!tgt_lm.getLMHead().isWeightTied())
    {
        return failTestStep(tag, "Causal_LM weight tying lost after loadCheckpoint");
    }

    Tensor in_t(Shape{ 1, 5 }, { 2.0f, 5.0f, 9.0f, 11.0f, 14.0f }, target);
    Tensor pos_t(Shape{ 1, 5 }, { 0.0f, 1.0f, 1.0f, 2.0f, 3.0f }, target);
    Tensor mask_t(Shape{ 1, 5, 5 }, {
        1, 0, 0, 0, 0,
        1, 1, 0, 0, 0,
        1, 0, 1, 0, 0,
        1, 0, 1, 1, 0,
        1, 0, 1, 1, 1
    }, target);
    ops::llm::Attention_Config attn_cfg{};
    attn_cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
    attn_cfg.attn_mask = &mask_t;

    Tensor full_logits = ref_lm.forward(in_t, false, &pos_t, &attn_cfg);
    Tensor full_dlogits(target);
    const std::vector<int32_t> full_targets{ -1, -1, 7, 13, -1 };
    const float ref_loss = ref_lm.computeLossAndGradient(full_logits.view(Shape{ 5, 29 }), full_targets, full_dlogits);
    (void)ref_lm.backward(full_dlogits);

    Tensor target_indices(Shape{ 2 }, { 2.0f, 3.0f }, target);
    Tensor target_labels(Shape{ 2 }, { 7.0f, 13.0f }, target);
    const float tgt_loss = tgt_lm.computeTargetLossAndBackward(in_t, target_indices, target_labels, &pos_t, &attn_cfg, 1.0f);
    flushIfGpu(target, true);

    if (std::abs(ref_loss - tgt_loss) > 1e-4f)
    {
        return failTestStep(tag, std::format("target loss mismatch (ref={:.6f} vs tgt={:.6f})", ref_loss, tgt_loss));
    }
    auto pg_ref = ref_lm.getParametersAndGradients();
    auto pg_tgt = tgt_lm.getParametersAndGradients();
    if (pg_ref.size() != pg_tgt.size())
    {
        return failTestStep(tag, "Causal_LM param count mismatch");
    }
    for (size_t i = 0; i < pg_ref.size(); ++i)
    {
        if (!checkVectorsClose(pg_ref[i].second->getData(), pg_tgt[i].second->getData(), 1e-4f, tag, "target-only vs full masked param grad"))
        {
            return false;
        }
    }

    tgt_lm.resetGradients();
    tgt_lm.enableLoRA(4, 8.0f, 0.0f, { "q_proj", "v_proj", "o_proj" }, true);
    if (!tgt_lm.hasActiveLoRA() || tgt_lm.getTokenEmbedding().isTrainable())
    {
        return failTestStep(tag, "Causal_LM::enableLoRA did not freeze base embedding");
    }
    tgt_lm.mergeLoRA();
    tgt_lm.unmergeLoRA();
    tgt_lm.disableLoRA();
    if (tgt_lm.hasActiveLoRA() || !tgt_lm.getTokenEmbedding().isTrainable())
    {
        return failTestStep(tag, "Causal_LM::disableLoRA did not restore base trainability");
    }

    if (target == Execution_Target::CPU)
    {
        Causal_LM_Config gen_cfg_m;
        gen_cfg_m.vocab_size = 0;
        gen_cfg_m.hidden_dim = 32;
        gen_cfg_m.num_heads = 4;
        gen_cfg_m.head_dim = 16;
        gen_cfg_m.num_kv_heads = 2;
        gen_cfg_m.intermediate_dim = 48;
        gen_cfg_m.num_layers = 1;
        gen_cfg_m.max_seq_len = 32;
        gen_cfg_m.use_qk_norm = true;
        gen_cfg_m.attention_bias = false;
        gen_cfg_m.mlp_bias = false;
        gen_cfg_m.tie_word_embeddings = true;
        gen_cfg_m.execution_target = Execution_Target::CPU;
        gen_cfg_m.tokenizer_path = "tokenizer/tokenizer.json";
        Causal_LM gen_lm(gen_cfg_m);

        const int32_t forced_id = 1095;
        Generate_Config gcfg;
        gcfg.max_new_tokens = 2;
        gcfg.temperature = 0.0f;
        gcfg.repetition_penalty = 1.0f;
        gcfg.logits_processor = [forced_id](std::vector<float> &logits, const std::vector<int32_t> &) {
            std::fill(logits.begin(), logits.end(), -1e9f);
            logits[forced_id] = 10.0f;
        };
        const std::string out_txt = gen_lm.generate("Hello", gcfg);
        const std::string expected_twice = gen_lm.getTokenizer().decode({ forced_id, forced_id }, true, true);
        if (out_txt != expected_twice)
        {
            return failTestStep(tag, "Generate_Config::logits_processor constrained decoding failed");
        }
    }

    return true;
}

bool testPhase6KVCacheAndCausalLM()
{
    if (!runKVCacheAndCausalLMTests(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice() && !runKVCacheAndCausalLMTests(Execution_Target::VULKAN_GPU))
    {
        return false;
    }
    return true;
}

}

void runLayerUpgradesSuite(Test_Result_Counter &counter)
{
    counter.record("  Phase 1 Linear/Bias/LoRA:         ", testPhase1LinearAndLoRA());
    counter.record("  Phase 2 Tied & Indexed LM Head:   ", testPhase2TiedAndIndexedLM());
    counter.record("  Phase 3 RoPE PosID/YaRN & QKNorm: ", testPhase3RoPEAnd4DQKNorm());
    counter.record("  Phase 4 Attn DENSE_BOOL/kv_index: ", testPhase4FlashAttentionMasked());
    counter.record("  Phase 5 Qwen3 Block/Ckpt/LoRA:    ", testPhase5Qwen3Block());
    counter.record("  Phase 6 KVCache & CausalLM Qwen3: ", testPhase6KVCacheAndCausalLM());
}
