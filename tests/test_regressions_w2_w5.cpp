#include "test_common.h"

bool testWave2Regressions()
{
    {
        Conv2d_Layer conv(6, 6, 2, 4, 3, 1, 1, Execution_Target::CPU, Data_Type::FLOAT32);
        conv.quantizeToTernary();
        conv.prepareInference();
        conv.setTrainingMode(true);

        Tensor in_c(Shape{1, 6 * 6 * 2}, Execution_Target::CPU);
        in_c.uploadData(std::vector<float>(72, 0.25f));
        Tensor out_c = conv.forward(in_c);
        Tensor grad_c(out_c.getShape(), std::vector<float>(out_c.getTotalElements(), 0.1f), Execution_Target::CPU);
        Tensor in_grad_c = conv.backward(grad_c);
        if (in_grad_c.getTotalElements() != 72) return false;

        auto conv_clone = conv.clone();
        if (!conv_clone) return false;
        Tensor out_c2 = conv_clone->forward(in_c);
        if (out_c.getData() != out_c2.getData()) return false;
    }

    {
        Res_Net_Block_2d_Layer res(7, 7, 4, 8, 2, Execution_Target::CPU, true);
        Tensor in_r(Shape{1, 7 * 7 * 4}, Execution_Target::CPU);
        in_r.uploadData(std::vector<float>(7 * 7 * 4, 0.2f));
        Tensor out_r = res.forward(in_r);
        if (out_r.getTotalElements() != 4 * 4 * 8) return false;

        Tensor grad_r(out_r.getShape(), std::vector<float>(out_r.getTotalElements(), 0.05f), Execution_Target::CPU);
        Tensor in_grad_r = res.backward(grad_r);
        if (in_grad_r.getTotalElements() != 7 * 7 * 4) return false;

        res.setTrainingMode(false);
        auto res_clone = res.clone();
        if (!res_clone) return false;
        out_r = res.forward(in_r);
        Tensor out_r2 = res_clone->forward(in_r);
        if (out_r.getData().size() != out_r2.getData().size()) return false;
        for (size_t i = 0; i < out_r.getData().size(); ++i) {
            if (!nearlyEqual(out_r.getData()[i], out_r2.getData()[i], 1e-4f)) return false;
        }
    }

    {
        Multi_Scale_ResNet18_Balanced_Layer ms_net(24, 16, 1, Execution_Target::CPU, true);
        Tensor in_ms(Shape{1, 24 * 16}, Execution_Target::CPU);
        in_ms.uploadData(std::vector<float>(24 * 16, 0.15f));
        Tensor out_ms = ms_net.forward(in_ms);
        if (out_ms.getTotalElements() == 0) return false;

        Tensor grad_ms(out_ms.getShape(), std::vector<float>(out_ms.getTotalElements(), 0.01f), Execution_Target::CPU);
        Tensor in_grad_ms = ms_net.backward(grad_ms);
        if (in_grad_ms.getTotalElements() != 24 * 16) return false;

        ms_net.setTrainingMode(false);
        auto ms_clone = ms_net.clone();
        if (!ms_clone) return false;
        out_ms = ms_net.forward(in_ms);
        Tensor out_ms2 = ms_clone->forward(in_ms);
        if (out_ms.getData().size() != out_ms2.getData().size()) return false;
        for (size_t i = 0; i < out_ms.getData().size(); ++i) {
            if (!nearlyEqual(out_ms.getData()[i], out_ms2.getData()[i], 1e-4f)) return false;
        }

        Multi_Scale_ResNet18_Balanced_Layer ms_r4(32, 32, 2, Execution_Target::VULKAN_GPU, true);
        ms_r4.setLambdaGmp(0.2f);
        Tensor in_r4(Shape{1, 32, 32, 2}, std::vector<float>(1 * 32 * 32 * 2, 0.1f), Execution_Target::VULKAN_GPU);
        Tensor out_r4 = ms_r4.forward(in_r4);
        if (out_r4.getShape() != Shape{1, 560}) return false;
        Tensor grad_r4(Shape{1, 560}, std::vector<float>(560, 0.01f), Execution_Target::VULKAN_GPU);
        Tensor in_grad_r4 = ms_r4.backward(grad_r4);
        flushGpuEngine(true);
        if (in_grad_r4.getShape() != Shape{1, 32, 32, 2}) return false;
        for (float v : in_grad_r4.getData()) {
            if (!std::isfinite(v)) return false;
        }
    }

    {
        std::vector<float> seg_in_data(1 * 2 * 16 * 16);
        for (size_t i = 0; i < seg_in_data.size(); ++i) {
            seg_in_data[i] = 0.05f * static_cast<float>((i % 7) + 1);
        }
        Tensor seg_in(Shape{1, 2, 16, 16}, Execution_Target::CPU);
        seg_in.uploadData(seg_in_data);

        U_Net_Layer unet(16, 16, 2, 2, 4, Execution_Target::CPU);
        unet.setTrainingMode(false);
        Tensor u_out = unet.forward(seg_in);
        if (u_out.getShape() != Shape{1, 2, 16, 16}) return false;
        auto unet_clone = unet.clone();
        if (!unet_clone) return false;
        Tensor u_out2 = unet_clone->forward(seg_in);
        if (u_out.getData().size() != u_out2.getData().size()) return false;
        for (size_t i = 0; i < u_out.getData().size(); ++i) {
            if (!nearlyEqual(u_out.getData()[i], u_out2.getData()[i], 1e-4f)) return false;
        }

        NN_UNet_Layer nn_unet(16, 16, 2, 2, 4, Execution_Target::CPU);
        nn_unet.setTrainingMode(false);
        Tensor nn_out = nn_unet.forward(seg_in);
        if (nn_out.getShape() != Shape{1, 2, 16, 16}) return false;
        auto nn_clone = nn_unet.clone();
        if (!nn_clone) return false;
        Tensor nn_out2 = nn_clone->forward(seg_in);
        if (nn_out.getData().size() != nn_out2.getData().size()) return false;
        for (size_t i = 0; i < nn_out.getData().size(); ++i) {
            if (!nearlyEqual(nn_out.getData()[i], nn_out2.getData()[i], 1e-4f)) return false;
        }
    }

    {
        std::vector<float> q_all_data(16);
        for (size_t i = 0; i < 16; ++i) q_all_data[i] = static_cast<float>(i + 1) * 0.25f;
        Tensor q_all(Shape{1, 1, 4, 4}, Execution_Target::CPU);
        q_all.uploadData(q_all_data);
        Tensor rope_all(Shape{1, 1, 4, 4}, Execution_Target::CPU);
        ops::llm::applyRoPE(q_all, rope_all, 4, 4, 1, 10000.0f, 1, 0, 0);

        std::vector<float> q_tok3_data(q_all_data.begin() + 12, q_all_data.end());
        Tensor q_tok3(Shape{1, 1, 1, 4}, Execution_Target::CPU);
        q_tok3.uploadData(q_tok3_data);
        Tensor rope_tok3(Shape{1, 1, 1, 4}, Execution_Target::CPU);
        ops::llm::applyRoPE(q_tok3, rope_tok3, 1, 4, 1, 10000.0f, 1, 0, 3);

        const auto& all_vec = rope_all.getData();
        const auto& tok3_vec = rope_tok3.getData();
        if (tok3_vec.size() != 4 || all_vec.size() != 16) return false;
        for (size_t i = 0; i < 4; ++i) {
            if (!nearlyEqual(all_vec[12 + i], tok3_vec[i], 1e-5f)) return false;
        }
    }

    {
        KV_Cache_Manager kv(1, 1, 4, 4, Execution_Target::CPU, Data_Type::FLOAT32, KV_Cache_Mode::RING_BUFFER);
        std::vector<float> k_data(24), v_data(24);
        for (size_t i = 0; i < 24; ++i) {
            k_data[i] = static_cast<float>(i);
            v_data[i] = static_cast<float>(i + 100);
        }
        Tensor k_new(Shape{1, 1, 6, 4}, Execution_Target::CPU);
        Tensor v_new(Shape{1, 1, 6, 4}, Execution_Target::CPU);
        k_new.uploadData(k_data);
        v_new.uploadData(v_data);
        kv.append(k_new, v_new);

        if (kv.getCurrentSeqLen() != 4) return false;
        Tensor k_out = kv.getK();
        const auto& ko = k_out.getData();
        if (ko.size() != 16) return false;
        for (size_t i = 0; i < 16; ++i) {
            if (!nearlyEqual(ko[i], static_cast<float>(8 + i), 1e-5f)) return false;
        }
    }

    {
        const std::string tmp_tok = "temp_wave2_tok.json";
        {
            std::ofstream out(tmp_tok);
            out << R"({"model":{"vocab":{"<unk>":0,"<s>":1,"</s>":2,"a":3,"b":4,"ab":5},"merges":[["a","b"]]}})";
        }
        Bpe_Tokenizer tok;
        bool loaded = tok.load(tmp_tok);
        std::remove(tmp_tok.c_str());
        if (!loaded || !tok.isLoaded()) return false;
    }

    {
        Adam_Optimizer adam;
        {
            auto sched = std::make_unique<No_Decay>(0.01f);
            adam.setLearningRateScheduler(sched.get());
        }
        Tensor w4(Shape{2, 2, 3, 3}, std::vector<float>(36, 1.0f), Execution_Target::CPU);
        Tensor g4(Shape{2, 2, 3, 3}, std::vector<float>(36, 0.1f), Execution_Target::CPU);
        adam.step({{&w4, &g4}});
        if (adam.getTimestep() != 1) return false;

        Neural_Network tmpl(Execution_Target::CPU);
        tmpl.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
        bool caught_zero = false;
        try {
            Population bad_pop(0, tmpl, 4, 2, Execution_Target::CPU);
        } catch (const std::invalid_argument&) {
            caught_zero = true;
        }
        if (!caught_zero) return false;

        Population good_pop(2, tmpl, 4, 2, Execution_Target::CPU);
        bool caught_null = false;
        try {
            good_pop.selectBatchActions(nullptr, nullptr, 1, nullptr);
        } catch (const std::invalid_argument&) {
            caught_null = true;
        }
        if (!caught_null) return false;
    }

    return true;
}

bool testWave3Regressions()
{
    {
        const std::string tmp_tensor = makeUniqueTestTempPath("temp_wave3_tensor").string();
        Scoped_Temp_File_Guard tmp_guard({tmp_tensor});
        std::vector<float> orig_vals = { 1.5f, -2.25f, 3.0f, 0.5f, -1.0f, 4.5f };
        Tensor t_orig(Shape{2, 3}, Data_Type::FLOAT16, Execution_Target::CPU);
        t_orig.uploadData(orig_vals);
        {
            std::ofstream ofs(tmp_tensor, std::ios::binary);
            t_orig.saveTensor(ofs);
        }
        Tensor t_loaded(Execution_Target::CPU);
        {
            std::ifstream ifs(tmp_tensor, std::ios::binary);
            t_loaded = Tensor::loadTensor(ifs, Execution_Target::CPU);
        }
        if (t_loaded.getDataType() != Data_Type::FLOAT16 || t_loaded.getShape() != Shape{2, 3}) return false;
        if (!verifyMatrix(t_loaded, orig_vals, 1e-2f)) return false;

        bool caught_permute = false;
        try {
            Tensor bad_perm = t_orig.permute({0, 0});
            (void)bad_perm;
        } catch (const std::invalid_argument&) {
            caught_permute = true;
        }
        if (!caught_permute) return false;

        Tensor cpu_a(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, Execution_Target::CPU);
        Tensor gpu_b(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, Execution_Target::VULKAN_GPU);
        Tensor out_cross(Shape{2, 2}, Execution_Target::CPU);
        bool caught_cross = false;
        try {
            cpu_a.add(gpu_b, out_cross);
        } catch (const std::invalid_argument&) {
            caught_cross = true;
        }
        if (!caught_cross) return false;
    }

    {
        Tensor base(Shape{2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, Execution_Target::CPU);
        Tensor s = base.slice(1, 1, 2);
        s.uploadData({10.0f, 20.0f, 30.0f, 40.0f});
        if (!verifyMatrix(base, {1, 10, 20, 4, 5, 30, 40, 8})) return false;
        s.zero();
        if (!verifyMatrix(base, {1, 0, 0, 4, 5, 0, 0, 8})) return false;

        Tensor f16_alias(Shape{1, 4}, Data_Type::FLOAT16, Execution_Target::CPU);
        f16_alias.uploadData({1.0f, -2.0f, 3.5f, 4.0f});
        f16_alias.add(f16_alias, f16_alias);
        if (f16_alias.getDataType() != Data_Type::FLOAT16) return false;
        if (!verifyMatrix(f16_alias, {2.0f, -4.0f, 7.0f, 8.0f}, 1e-2f)) return false;
    }

    {
        Tensor odd_fp16(Shape{1, 5}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        odd_fp16.uploadData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        odd_fp16.zero();
        Execution_Engine::getInstance().executeGraph();
        if (odd_fp16.getDataType() != Data_Type::FLOAT16) return false;
        if (!verifyMatrix(odd_fp16, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 1e-3f)) return false;

        Tensor a(Shape{1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f}, Execution_Target::VULKAN_GPU);
        Tensor out(Shape{1, 4}, Execution_Target::VULKAN_GPU);
        a.relu(out);
        if (!verifyMatrix(out, {0.0f, 2.0f, 0.0f, 4.0f})) return false;
        Tensor ones(Shape{1, 4}, {1.0f, 1.0f, 1.0f, 1.0f}, Execution_Target::VULKAN_GPU);
        out.add(ones, out);
        if (!verifyMatrix(out, {1.0f, 3.0f, 1.0f, 5.0f})) return false;
    }

    {
        Tensor sig_in(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::CPU);
        sig_in.uploadData({0.0f, 1.0f});
        Tensor sig_out(Execution_Target::CPU);
        ops::activation::sigmoid(sig_in, sig_out);
        if (sig_out.getDataType() != Data_Type::FLOAT16) return false;
        if (!verifyMatrix(sig_out, {0.5f, 1.0f / (1.0f + std::exp(-1.0f))}, 1e-2f)) return false;

        Tensor dy(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::CPU);
        dy.uploadData({1.0f, 2.0f});
        Tensor w(Shape{3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}, Execution_Target::CPU);
        Tensor dx(Execution_Target::CPU);
        ops::linear::linearBackwardInput(dy, w, dx);
        if (dx.getDataType() != Data_Type::FLOAT16) return false;
        if (!verifyMatrix(dx, {1.0f, 2.0f, 1.5f}, 1e-2f)) return false;

        Tensor p_cpu(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::CPU);
        Tensor g_cpu(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::CPU);
        Tensor m_cpu(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::CPU);
        Tensor v_cpu(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::CPU);
        p_cpu.uploadData({1.0f, 2.0f});
        g_cpu.uploadData({0.1f, -0.2f});
        m_cpu.zero();
        v_cpu.zero();
        ops::optimizer::adamStep(p_cpu, g_cpu, m_cpu, v_cpu, 0.1f, 0.9f, 0.999f, 1e-8f, 0);
        auto p_after = p_cpu.getData();
        if (p_after.size() != 2 || std::isnan(p_after[0]) || std::isinf(p_after[0]) || !(p_after[0] < 1.0f)) return false;

        Tensor p_gpu(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor g_gpu(Shape{1, 2}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        p_gpu.uploadData({2.0f, -1.0f});
        g_gpu.uploadData({0.5f, -0.5f});
        ops::optimizer::sgdStep(p_gpu, g_gpu, 0.2f);
        if (p_gpu.getDataType() != Data_Type::FLOAT16) return false;
        if (!verifyMatrix(p_gpu, {1.9f, -0.9f}, 1e-2f)) return false;
    }

    {
        Tensor x_cpu16(Shape{2, 64}, Data_Type::FLOAT16, Execution_Target::CPU);
        Tensor g_cpu16(Shape{1, 64}, Data_Type::FLOAT16, Execution_Target::CPU);
        std::vector<float> x_vals(128), g_vals(64, 1.0f);
        for (size_t i = 0; i < 128; ++i) x_vals[i] = static_cast<float>((i % 7) + 1) * 0.25f;
        x_cpu16.uploadData(x_vals);
        g_cpu16.uploadData(g_vals);
        Tensor inv_rms_cpu(Execution_Target::CPU), out_cpu16(Execution_Target::CPU);
        ops::normalization::rmsNormForward(x_cpu16, g_cpu16, inv_rms_cpu, out_cpu16, 1e-5f);
        if (out_cpu16.getDataType() != Data_Type::FLOAT16 || out_cpu16.getTotalElements() != 128) return false;

        Tensor x_gpu(Shape{2, 64}, x_vals, Execution_Target::VULKAN_GPU);
        Tensor g_gpu(Shape{1, 64}, g_vals, Execution_Target::VULKAN_GPU);
        Tensor inv_rms_gpu(Execution_Target::VULKAN_GPU), out_gpu(Execution_Target::VULKAN_GPU);
        ops::normalization::rmsNormForward(x_gpu, g_gpu, inv_rms_gpu, out_gpu, 1e-5f);
        if (!verifyMatrix(out_gpu, out_cpu16.getData(), 2e-2f)) return false;

        Tensor bn_x(Shape{2, 4}, std::vector<float>(8, 1.0f), Execution_Target::CPU);
        Tensor bn_dy(Shape{2, 4}, std::vector<float>(8, 1.0f), Execution_Target::CPU);
        Tensor bn_g(Shape{1, 4}, std::vector<float>(4, 1.0f), Execution_Target::CPU);
        Tensor bn_bv(Shape{1, 4}, std::vector<float>(4, 1.0f), Execution_Target::CPU);
        Tensor bn_norm(Shape{2, 4}, std::vector<float>(8, 0.0f), Execution_Target::CPU);
        Tensor bn_dg(Execution_Target::CPU), bn_db(Execution_Target::CPU), bn_dx(Execution_Target::CPU);
        bool caught_rv = false;
        try {
            ops::normalization::batchNormBackward(bn_x, bn_dy, bn_g, bn_bv, bn_norm, bn_dg, bn_db, bn_dx, 1e-5f, false);
        } catch (const std::invalid_argument&) {
            caught_rv = true;
        }
        if (!caught_rv) return false;

        std::vector<float> cin(16, 1.0f);
        std::vector<float> cw = {1.0f, 0.5f, -0.5f, 1.0f};
        std::vector<float> cb = {0.25f};
        Tensor c_in_cpu(Shape{1, 16}, cin, Execution_Target::CPU);
        Tensor c_w_cpu(Shape{4, 1}, cw, Execution_Target::CPU);
        Tensor c_b_cpu(Shape{1, 1}, cb, Execution_Target::CPU);
        Tensor c_out_cpu(Execution_Target::CPU);
        ops::vision::conv2d(c_in_cpu, c_w_cpu, c_b_cpu, c_out_cpu, 4, 4, 1, 1, 2, 1, 0);

        Tensor c_in_gpu(Shape{1, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        c_in_gpu.uploadData(cin);
        Tensor c_w_gpu(Shape{4, 1}, cw, Execution_Target::VULKAN_GPU);
        Tensor c_b_gpu(Shape{1, 1}, cb, Execution_Target::VULKAN_GPU);
        Tensor c_out_gpu(Execution_Target::VULKAN_GPU);
        ops::vision::conv2d(c_in_gpu, c_w_gpu, c_b_gpu, c_out_gpu, 4, 4, 1, 1, 2, 1, 0);
        if (!verifyMatrix(c_out_gpu, c_out_cpu.getData(), 2e-2f)) return false;
    }

    {
        bool caught_stride0 = false;
        try {
            Max_Pool_2d_Layer mp(4, 4, 1, 2, 0, 0, Execution_Target::CPU);
        } catch (const std::invalid_argument&) {
            caught_stride0 = true;
        }
        if (!caught_stride0) return false;

        bool caught_underflow = false;
        try {
            Avg_Pool_2d_Layer ap(2, 2, 1, 5, 1, 0, Execution_Target::CPU);
        } catch (const std::invalid_argument&) {
            caught_underflow = true;
        }
        if (!caught_underflow) return false;

        Dropout_Layer drop(0.5f, Execution_Target::CPU, 42);
        if (drop.supportsStaticGraph()) return false;
        drop.setTrainingMode(false);
        if (!drop.supportsStaticGraph()) return false;
        drop.setTrainingMode(true);

        Tensor ones(Shape{1, 128}, std::vector<float>(128, 1.0f), Execution_Target::CPU);
        Tensor y1 = drop.forward(ones);
        auto m1 = drop.getMask().getData();
        Tensor y2 = drop.forward(ones);
        auto m2 = drop.getMask().getData();
        if (m1.size() != 128 || m2.size() != 128) return false;
        bool masks_differ = false;
        for (size_t i = 0; i < 128; ++i) {
            if (!nearlyEqual(m1[i], m2[i], 1e-5f)) {
                masks_differ = true;
                break;
            }
        }
        if (!masks_differ) return false;

        drop.setSeed(42);
        drop.forward(ones);
        auto m1_replay = drop.getMask().getData();
        for (size_t i = 0; i < 128; ++i) {
            if (!nearlyEqual(m1[i], m1_replay[i], 1e-5f)) return false;
        }
    }

    {
        std::vector<float> logits_vec = {
            2.0f, 0.5f, -0.5f, 0.1f,
            0.3f, 1.8f,  0.2f, 0.1f,
            0.1f, 2.2f,  0.4f, -0.3f,
            1.2f, -0.2f, 0.9f, 0.5f
        };
        std::vector<int32_t> idx_targets = { 0, 2, 1, 2 };
        std::vector<float> f_targets = { 0.0f, 2.0f, 1.0f, 2.0f };
        Fused_Cross_Entropy fce(2);

        for (Execution_Target et : { Execution_Target::CPU, Execution_Target::VULKAN_GPU }) {
            Tensor logits(Shape{4, 4}, logits_vec, et);
            Tensor t_tensor(Shape{4}, f_targets, et);
            Tensor grad_vec, grad_ten;
            float l_vec = fce.computeLossAndGradient(logits, idx_targets, grad_vec);
            float l_ten = fce.computeLossAndGradient(logits, t_tensor, grad_ten);
            if (!nearlyEqual(l_vec, l_ten, 1e-3f)) return false;
            if (!verifyMatrix(grad_ten, grad_vec.getData(), 1e-3f)) return false;
            const auto gt = grad_ten.getData();
            for (size_t v = 0; v < 4; ++v) {
                if (!nearlyEqual(gt[1 * 4 + v], 0.0f, 1e-5f)) return false;
                if (!nearlyEqual(gt[3 * 4 + v], 0.0f, 1e-5f)) return false;
            }
        }

        Tensor mse_p(Shape{1, 4}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor mse_t(Shape{1, 4}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        mse_p.uploadData({1.0f, 2.0f, 3.0f, 4.0f});
        mse_t.uploadData({2.0f, 2.0f, 1.0f, 4.0f});
        Mse_Cost mse_gpu(Execution_Target::VULKAN_GPU);
        if (!nearlyEqual(mse_gpu.computeLoss(mse_p, mse_t), 1.25f, 1e-2f)) return false;

        std::vector<float> dp_vals = {0.8f, 0.2f, 0.9f, 0.1f};
        std::vector<float> dt_vals = {1.0f, 0.0f, 1.0f, 0.0f};
        Dice_Cost dice_cpu(1e-6f, Execution_Target::CPU);
        Tensor dp_cpu(Shape{1, 4}, dp_vals, Execution_Target::CPU);
        Tensor dt_cpu(Shape{1, 4}, dt_vals, Execution_Target::CPU);
        float expected_dice = dice_cpu.computeLoss(dp_cpu, dt_cpu);

        Dice_Cost dice_gpu(1e-6f, Execution_Target::VULKAN_GPU);
        Tensor dp_gpu(Shape{1, 4}, dp_vals, Execution_Target::VULKAN_GPU);
        Tensor dt_gpu16(Shape{1, 4}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        dt_gpu16.uploadData(dt_vals);
        if (!nearlyEqual(dice_gpu.computeLoss(dp_gpu, dt_gpu16), expected_dice, 1e-2f)) return false;

        bool caught_loss_cross = false;
        try {
            (void)mse_gpu.computeLoss(mse_p, dt_cpu);
        } catch (const std::invalid_argument&) {
            caught_loss_cross = true;
        }
        bool caught_grad_cross = false;
        try {
            (void)mse_gpu.computeGradient(mse_p, dt_cpu);
        } catch (const std::invalid_argument&) {
            caught_grad_cross = true;
        }
        if (!caught_loss_cross || !caught_grad_cross) return false;
    }

    {
        Neural_Network cnn(Execution_Target::CPU);
        cnn.addLayer<Conv2d_Layer>(8, 8, 1, 4, 3, 1, 1, Execution_Target::CPU);
        cnn.addLayer<Relu_Layer>(Execution_Target::CPU);
        cnn.addLayer<Global_Avg_Pool_2d_Layer>(8, 8, 4, Execution_Target::CPU);
        cnn.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);

        Tensor img(Shape{1, 64}, std::vector<float>(64, 1.0f), Execution_Target::CPU);
        chexlocalize::GradcamConfig cfg;
        cfg.target_height = 8;
        cfg.target_width = 8;
        chexlocalize::Gradcam_Engine cam(cnn, cfg);
        std::vector<float> heatmap = cam.generateHeatmap(img, 0, 8, 8);
        if (heatmap.size() != 64) return false;
        for (float v : heatmap) {
            if (std::isnan(v) || std::isinf(v) || v < -1e-5f || v > 1.0f + 1e-5f) return false;
        }
    }

    {
        auto &engine = Execution_Engine::getInstance();
        auto tasks_copy = engine.getContext().getTransferTasks();
        (void)tasks_copy;
        bool ok_t1 = true, ok_t2 = true;
        std::thread t1([&]() {
            ok_t1 = (engine.getNetwork().getPipeline(Compute_Pipeline::ADD) != VK_NULL_HANDLE);
            (void)engine.getPipelineCacheManager().hasPipeline("");
        });
        std::thread t2([&]() {
            ok_t2 = (engine.getNetwork().getPipeline(Compute_Pipeline::ADD) != VK_NULL_HANDLE);
            (void)engine.getPipelineCacheManager().hasPipeline("");
        });
        t1.join();
        t2.join();
        if (!ok_t1 || !ok_t2) return false;
    }

    return true;
}

bool testWave4Regressions()
{
    namespace fs = std::filesystem;

    auto failStep = [](const char* step) -> bool {
        return failTestStep("Wave4", step);
    };

    Scoped_Temp_File_Guard cleanup;

    {
        fs::path bin_path = fs::temp_directory_path() / "vulkanml_wave4_tokens.bin";
        cleanup.files.push_back(bin_path.string());
        std::vector<int32_t> saved = {10, 20, 30, 42};
        if (!Binary_Token_Dataset::save(bin_path.string(), saved)) return failStep("1a Binary_Token_Dataset::save");

        std::vector<int32_t> loaded;
        if (!Binary_Token_Dataset::load(bin_path.string(), loaded)) return failStep("1b Binary_Token_Dataset::load");
        if (loaded != saved) return failStep("1c Binary_Token_Dataset loaded!=saved");

        {
            std::ofstream out(bin_path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) return failStep("1d ofstream open");
            Binary_Token_Dataset::Header bad_hdr;
            bad_hdr.total_tokens = 1000;
            out.write(reinterpret_cast<const char*>(&bad_hdr), sizeof(bad_hdr));
            int32_t partial[2] = {7, 8};
            out.write(reinterpret_cast<const char*>(partial), sizeof(partial));
        }

        std::vector<int32_t> sentinel = {99, 100};
        if (Binary_Token_Dataset::load(bin_path.string(), sentinel)) return failStep("1e bad load succeeded");
        if (sentinel != std::vector<int32_t>{99, 100}) return failStep("1f sentinel mutated");
        if (Binary_Token_Dataset::existsAndValid(bin_path.string())) return failStep("1g existsAndValid true");
    }

    {
        fs::path tok_path = fs::temp_directory_path() / "vulkanml_wave4_tok.json";
        cleanup.files.push_back(tok_path.string());
        {
            std::ofstream out(tok_path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) return failStep("2a tok ofstream");
            out << R"({
  "model": {
    "vocab": {
      "<unk>": 0,
      "<s>": 1,
      "</s>": 2,
      "<0x": 3,
      "41>": 4,
      "\u2581hello": 5,
      "<0x41>": 6
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 7, "content": "<spec>", "special": false},
    {"id": 8, "content": "<spec>_long", "special": false}
  ]
})";
        }

        Bpe_Tokenizer tok;
        if (!tok.load(tok_path.string()) || !tok.isLoaded()) return failStep("2b tok.load");
        if (tok.decode({3, 4}, true, true) != "<0x41>") return failStep("2c decode {3,4}");
        if (tok.decode({6}, true, true) != "A") return failStep("2d decode {6}");
        if (tok.decode({5}, true, true) != "hello") return failStep("2e decode {5} strip=true");
        if (tok.decode({5}, true, false) != " hello") return failStep("2f decode {5} strip=false");
        auto enc = tok.encode("<spec>_long", false);
        if (enc.size() != 1 || enc[0] != 8) return failStep("2g encode <spec>_long");
    }

    {
        Replay_Buffer empty_rb(4, 42);
        auto zero_from_empty = empty_rb.sample(0);
        if (zero_from_empty.batch_size != 0 || !zero_from_empty.states.empty()) return failStep("3a empty sample(0)");

        Replay_Buffer rb(8, 42);
        rb.push(Transition{{1.0f, 2.0f}, 0, 1.0f, {3.0f, 4.0f}, false});
        if (rb.isReady(0)) return failStep("3b isReady(0)");
        if (!rb.isReady(1)) return failStep("3c isReady(1)");

        auto zero_batch = rb.sample(0);
        if (zero_batch.batch_size != 0 || !zero_batch.actions.empty()) return failStep("3d sample(0)");

        if (!expectThrows([&]() { (void)rb.sample(2); })) return failStep("3e oversize sample(2) not caught");
        if (!expectThrows([&]() { rb.push(Transition{{1.0f, 2.0f, 3.0f}, 0, 0.0f, {1.0f, 2.0f, 3.0f}, false}); })) return failStep("3f mismatch not caught");

        rb.clear();
        rb.push(Transition{{1.0f, 2.0f, 3.0f}, 1, 0.5f, {4.0f, 5.0f, 6.0f}, true});
        if (rb.getSize() != 1) return failStep("3g getSize!=1");
    }

    {
        Neural_Network proto(Execution_Target::CPU);
        proto.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
        proto.setCostFunction<Mse_Cost>(Execution_Target::CPU);
        proto.setOptimizer<Adam_Optimizer>(1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Dqn_Agent dqn(std::move(proto), 4, 2, 16, 0.99f, 1.0f, 0.05f, 0.9f, Execution_Target::CPU, 42);
        if (dqn.getTargetNetwork().getLayerCount() != 1) return failStep("4a dqn target layers!=1");

        dqn.setEpsilon(0.01f);
        dqn.decayEpsilon();
        if (!nearlyEqual(dqn.getEpsilon(), 0.01f, 1e-6f)) return failStep("4b decayEpsilon");

        if (!expectThrows([&]() { dqn.setTargetUpdateParameters(0, false, 0.005f); })) return failStep("4c setTargetUpdateParameters(0)");
        if (!expectThrows([&]() { Ppo_Agent bad_ppo(4, 2, 3e-4f, 0.99f, 0.95f, 0.2f, 0.5f, 0.01f, 4, 0, Execution_Target::CPU, 42); })) return failStep("4d bad_ppo");
    }

    {
        auto& st = Step_Timings::getInstance();
        st.reset();
        st.is_chained_bwd_opt = true;
        st[Timing_Stage::OPT_EXEC] = 5.0;
        st[Timing_Stage::OPT_RECORD] = 1.0;

        {
            std::ostringstream sink;
            std::streambuf* old_buf = std::cout.rdbuf(sink.rdbuf());
            struct CoutGuard {
                std::streambuf* old;
                ~CoutGuard() { std::cout.rdbuf(old); }
            } cout_guard{old_buf};

            st.printTable(0, 1, 10.0, 0, "tok/s");
            if (sink.str().empty()) return failStep("5a printTable empty");
        }

        st.add(st);
        if (!st.is_chained_bwd_opt || !nearlyEqual(static_cast<float>(st[Timing_Stage::OPT_EXEC]), 10.0f, 1e-4f))
        {
            st.reset();
            return failStep("5b st.add");
        }
        st.reset();
    }

    {
        auto expectParseThrow = [](std::string_view bad) -> bool {
            return expectThrows<std::invalid_argument>([&]() { (void)chexlocalize::Rle_Decoder::parseCounts(bad); });
        };
        if (!expectParseThrow(" ")) return failStep("6a parseCounts space");
        if (!expectParseThrow("P")) return failStep("6b parseCounts P");
        if (!expectParseThrow("PPPPPPPPPPPPPP")) return failStep("6c parseCounts long");

        auto cnts = chexlocalize::Rle_Decoder::parseCounts("33");
        if (cnts.size() != 2 || cnts[0] != 3 || cnts[1] != 3) return failStep("6d parseCounts 33");
        if (chexlocalize::Rle_Decoder::computeArea("33") != 3) return failStep("6e computeArea 33");
        if (!chexlocalize::Rle_Decoder::decodeToRowMajor("", 0, 0).empty()) return failStep("6f decode 0x0");
    }

    {
        chexlocalize::BlenderConfig edge_cfg;
        edge_cfg.active_threshold = 1.0f;
        edge_cfg.use_slerp_top2 = true;
        chexlocalize::Quaternion_Blender qb_edge(edge_cfg);
        if (qb_edge.getConfig().active_threshold >= 1.0f) return failStep("7a active_threshold");

        std::array<float, 10> high_act{};
        high_act[0] = 0.95f;
        auto edge_col = qb_edge.blendPixel(high_act, 100);
        if (edge_col.r != 100 || edge_col.g != 100 || edge_col.b != 100) return failStep("7b blendPixel threshold");

        chexlocalize::Quaternion_Blender qb;
        std::array<float, 10> red_dom{};
        red_dom[0] = 0.85f;
        red_dom[4] = 0.55f;
        std::array<float, 10> cyan_dom{};
        cyan_dom[0] = 0.55f;
        cyan_dom[4] = 0.85f;
        auto c_red = qb.blendPixel(red_dom, 128);
        auto c_cyan = qb.blendPixel(cyan_dom, 128);
        if (!(c_red.r > c_cyan.r && c_cyan.b > c_red.b)) return failStep("7c slerp polarity");

        std::vector<std::vector<float>> maps(10, std::vector<float>(16, 0.2f));
        maps[3].resize(4);
        std::vector<uint8_t> gray(16, 128);
        if (!expectThrows<std::invalid_argument>([&]() { (void)qb.blendCompositeImage(maps, gray, 4, 4); })) return failStep("7d blendCompositeImage size");

        std::vector<uint8_t> bad_rgb(10, 0);
        if (chexlocalize::Quaternion_Blender::saveToPpm("wave4_unused.ppm", bad_rgb, 4, 4)) return failStep("7e saveToPpm");
        if (chexlocalize::Quaternion_Blender::saveToJpg("wave4_unused.jpg", bad_rgb, 4, 4, 90)) return failStep("7f saveToJpg");
    }

    {
        Neural_Network cnn(Execution_Target::CPU);
        cnn.addLayer<Conv2d_Layer>(8, 8, 1, 4, 3, 1, 1, Execution_Target::CPU);
        cnn.addLayer<Relu_Layer>(Execution_Target::CPU);
        cnn.addLayer<Global_Avg_Pool_2d_Layer>(8, 8, 4, Execution_Target::CPU);
        cnn.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);

        cnn.setTrainingMode(true);
        cnn.enableGradientAccumulation(true);
        cnn.getLossScaler().setEnabled(true);
        cnn.getLossScaler().setScaleFactor(64.0f);

        chexlocalize::GradcamConfig cfg;
        if (cfg.thin_structure_classes != std::vector<size_t>{7, 9}) return failStep("8a thin_structure_classes");
        cfg.target_height = 8;
        cfg.target_width = 8;
        chexlocalize::Gradcam_Engine cam(cnn, cfg);

        Tensor bad_batch(Shape{2, 64}, std::vector<float>(128, 0.5f), Execution_Target::CPU);
        if (!expectThrows<std::invalid_argument>([&]() { (void)cam.generateHeatmap(bad_batch, 0, 8, 8); })) return failStep("8b bad_batch");

        Tensor good_batch(Shape{1, 64}, std::vector<float>(64, 0.5f), Execution_Target::CPU);
        auto hmaps = cam.generateMultiHeatmaps(good_batch, std::array<size_t, 1>{0}, 8, 8);
        if (hmaps.size() != 1 || hmaps[0].size() != 64) return failStep("8c generateMultiHeatmaps");

        if (!cnn.isTrainingMode() || !cnn.isGradientAccumulationEnabled() ||
            !cnn.getLossScaler().isEnabled() ||
            !nearlyEqual(cnn.getLossScaler().getScaleFactor(), 64.0f, 1e-5f))
        {
            return failStep("8d NetworkStateGuard restore");
        }
    }

    {
        Multi_Scale_ResNet18_Balanced_Layer ms(32, 32, 1, Execution_Target::CPU, true);
        ms.setTrainingMode(false);
        std::vector<float> img1(32 * 32, 0.0f);
        for (size_t i = 0; i < img1.size(); ++i)
        {
            img1[i] = static_cast<float>((i * 7 + 3) % 23) * 0.04f;
        }
        Tensor out1 = ms.forward(Tensor(Shape{1, 32, 32, 1}, img1, Execution_Target::CPU));
        auto out1_vec = out1.getData();

        if (!ms.expandStemFrom1To2Channels() || ms.getInputChannels() != 2) return failStep("9a expandStem");

        std::vector<float> img2(32 * 32 * 2, 0.0f);
        for (size_t p = 0; p < 32 * 32; ++p)
        {
            img2[p * 2 + 0] = img1[p];
        }
        Tensor out2 = ms.forward(Tensor(Shape{1, 32, 32, 2}, img2, Execution_Target::CPU));
        auto out2_vec = out2.getData();

        if (!verifyVectorsClose(out1_vec, out2_vec, 1e-4f) || out1_vec.empty()) return failStep("9b/9c out mismatch");
    }

    return true;
}

bool testWave5Phase5ACRegressions()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("Wave5 5A-5C", msg);
    };

    {
        Loss_Scaler scaler(4.0f, 2.0f, 0.5f, 2, true, 64.0f, 2.0f);
        if (!nearlyEqual(scaler.getMinScale(), 2.0f, 1e-6f) ||
            !nearlyEqual(scaler.getScaleFactor(), 4.0f, 1e-6f))
        {
            return failStep("5A Loss_Scaler ctor min_scale");
        }
        scaler.setBackoffFactor(1.5f);
        scaler.setGrowthFactor(0.5f);
        scaler.setScaleFactor(-1.0f);
        if (!nearlyEqual(scaler.getBackoffFactor(), 0.5f, 1e-6f) ||
            !nearlyEqual(scaler.getGrowthFactor(), 2.0f, 1e-6f) ||
            !nearlyEqual(scaler.getScaleFactor(), 4.0f, 1e-6f))
        {
            return failStep("5A Loss_Scaler invalid setter guard");
        }
        scaler.step(true);
        scaler.step(true);
        if (!nearlyEqual(scaler.getScaleFactor(), 2.0f, 1e-6f))
        {
            return failStep("5A Loss_Scaler min_scale clamp");
        }

        Tensor finite_fp16 = Tensor(Shape{2, 2}, std::vector<float>{1.0f, -2.0f, 3.5f, 0.25f}, Execution_Target::CPU).toFp16();
        Tensor inf_fp16 = Tensor(Shape{2, 2}, std::vector<float>{1.0f, std::numeric_limits<float>::infinity(), 0.0f, 0.0f}, Execution_Target::CPU).toFp16();
        Tensor nan_fp16 = Tensor(Shape{1, 2}, std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f}, Execution_Target::CPU).toFp16();
        if (scaler.hasOverflow(finite_fp16)) return failStep("5A Loss_Scaler FP16 finite false positive");
        if (!scaler.hasOverflow(inf_fp16)) return failStep("5A Loss_Scaler FP16 inf missed");
        if (!scaler.hasOverflow(nan_fp16)) return failStep("5A Loss_Scaler FP16 nan missed");

        auto &ctx = Execution_Engine::getInstance().getContext();
        VkMemoryRequirements req0{};
        req0.size = 0;
        req0.alignment = 16;
        req0.memoryTypeBits = 0xFFFFFFFFu;
        auto alloc0 = ctx.getAllocator().allocate(req0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (alloc0.memory != VK_NULL_HANDLE || alloc0.size != 0)
        {
            return failStep("5A Vulkan_Sub_Allocator::allocate(0)");
        }

        gpu::vector hv(ctx);
        hv.allocateHostVisible(16);
        uint32_t valid4[4] = {1u, 2u, 3u, 4u};
        hv.writeHostDirect(valid4, sizeof(valid4));
        bool threw_null = false;
        try { hv.writeHostDirect(nullptr, 4); } catch (...) { threw_null = true; }
        bool threw_oob = false;
        std::vector<uint8_t> big_buf(1024, 0);
        try { hv.writeHostDirect(big_buf.data(), big_buf.size()); } catch (...) { threw_oob = true; }
        if (!threw_null || !threw_oob) return failStep("5A gpu::vector::writeHostDirect validation");

        Specialization_Map_Builder spec_builder;
        spec_builder.addConstant<bool>(0, true);
        if (spec_builder.getEntries().size() != 1 || spec_builder.getEntries()[0].size != sizeof(VkBool32))
        {
            return failStep("5A Specialization_Map_Builder bool size");
        }
        bool threw_dup_spec = false;
        try { spec_builder.addConstant<uint32_t>(0, 1u); } catch (...) { threw_dup_spec = true; }
        if (!threw_dup_spec) return failStep("5A Specialization_Map_Builder duplicate ID");
        VkSpecializationInfo spec_info = spec_builder.build();
        if (spec_info.mapEntryCount != 1 || spec_info.dataSize != sizeof(VkBool32))
        {
            return failStep("5A Specialization_Map_Builder build");
        }

        bool threw_zero_wg = false;
        try { Shader_Generator bad_gen(0, 1, 1); } catch (...) { threw_zero_wg = true; }
        if (!threw_zero_wg) return failStep("5A Shader_Generator zero workgroup ctor");

        Shader_Generator gen(16, 1, 1);
        gen.setGroupX(0);
        if (gen.getGroupX() != 1) return failStep("5A Shader_Generator setGroupX clamp");
        gen.setPushConstants("uint n;");
        bool threw_dup_pc = false;
        try { gen.setPushConstants("uint m;"); } catch (...) { threw_dup_pc = true; }
        if (!threw_dup_pc) return failStep("5A Shader_Generator duplicate push constants");
        bool threw_zero_sm = false;
        try { (void)gen.addSharedMemory(0, "s", "float"); } catch (...) { threw_zero_sm = true; }
        if (!threw_zero_sm) return failStep("5A Shader_Generator zero shared memory");

        Shader_Dictionary dict("compute_shader/shader_dictionary.json");
        Operation_Class meta_before = dict.getMetadata(Compute_Pipeline::ADD).operation_class;
        bool threw_dict = false;
        try { dict.loadFromFile("nonexistent_wave5_file.json"); } catch (...) { threw_dict = true; }
        if (!threw_dict || dict.getMetadata(Compute_Pipeline::ADD).operation_class != meta_before)
        {
            return failStep("5A Shader_Dictionary atomic rollback");
        }
    }

    {
        auto &engine = Execution_Engine::getInstance();
        if (engine.isCooperativeMatrixEnabled() != engine.getContext().isCooperativeMatrixEnabled())
        {
            return failStep("5B Execution_Engine::isCooperativeMatrixEnabled parity");
        }
        Scoped_Engine_Flags_Guard flags_guard;
        bool orig_cache = engine.isGraphCacheEnabled();
        bool orig_fused = engine.isFusedGemmAdamEnabled();
        engine.enableGraphCaching(false);
        if (engine.isGraphCacheEnabled()) return failStep("5B enableGraphCaching(false)");
        engine.enableGraphCaching(true);
        if (!engine.isGraphCacheEnabled()) return failStep("5B enableGraphCaching(true)");
        engine.enableGraphCaching(orig_cache);

        engine.enableFusedGemmAdam(!orig_fused);
        if (engine.isFusedGemmAdamEnabled() == orig_fused) return failStep("5B enableFusedGemmAdam");
        engine.enableFusedGemmAdam(orig_fused);

        struct Throwing_Wave5_Pipeline final : public Async_Data_Pipeline
        {
            explicit Throwing_Wave5_Pipeline()
                : Async_Data_Pipeline(Execution_Target::CPU)
            {
            }
            ~Throwing_Wave5_Pipeline() override { stop(); }
            size_t getBatchSize() const override { return 2; }
        protected:
            void prepareBatchHost(size_t, std::vector<float> &, std::vector<float> &) override
            {
                throw std::runtime_error("wave5 loader boom");
            }
        };

        Throwing_Wave5_Pipeline pipeline;
        pipeline.initializeBuffers(2, 3, 1, Execution_Target::CPU);
        pipeline.start();
        bool threw_worker = false;
        try
        {
            (void)pipeline.nextBatch(2, 3, 1);
        }
        catch (const std::runtime_error &e)
        {
            if (std::string(e.what()).find("wave5 loader boom") != std::string::npos)
            {
                threw_worker = true;
            }
        }
        pipeline.stop();
        if (!threw_worker) return failStep("5B Async_Data_Pipeline worker exception propagation");
    }

    {
        std::vector<float> mv_vec = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        Tensor mv(Shape{2, 3}, std::move(mv_vec), Execution_Target::CPU);
        mv.reshape(3, 2);
        if (mv.getRank() != 2 || mv.getRows() != 3 || mv.getColumns() != 2)
        {
            return failStep("5C Tensor::reshape(rows, cols)");
        }
        mv.contiguous(mv);
        std::vector<float> mv_got = mv.getData();
        if (mv_got.size() != 6 || !nearlyEqual(mv_got[0], 1.f) || !nearlyEqual(mv_got[5], 6.f))
        {
            return failStep("5C Tensor::contiguous self-alias");
        }
        Tensor alias = mv;
        Tensor out = alias;
        mv.contiguous(out);
        out.fill(99.0f);
        if (!nearlyEqual(alias.getData()[0], 1.f) || !nearlyEqual(mv.getData()[0], 1.f))
        {
            return failStep("5C Tensor::contiguous COW isolation");
        }

        Tensor gpu_probe(Shape{1, 4}, std::vector<float>{1.f, -1.f, 0.f, 0.5f}, Execution_Target::VULKAN_GPU);
        Logger::setOnlyActiveFeatures(Log_Feature::NONE);
        gpu_probe.logFp16Stats("wave5_probe_off");
        gpu_probe.logTernaryStats("wave5_probe_off", 0.5f);
        if (!Gpu_Tensor_Impl::castToGpu(*gpu_probe.getImplementation()).getHostCache().empty())
        {
            return failStep("5C logFp16/TernaryStats readback stall while feature disabled");
        }
        Tensor tern_cpu(Shape{1, 4}, Data_Type::TERNARY_INT2, Execution_Target::CPU);
        tern_cpu.uploadData({1.f, 0.f, -1.f, 1.f});
        Logger::setOnlyActiveFeatures(Log_Feature::TERNARY_METRICS);
        tern_cpu.logTernaryStats("wave5_tern_on", 0.5f);
        Logger::setOnlyActiveFeatures(Log_Feature::NONE);

        auto cpu_impl = std::make_shared<Cpu_Tensor_Impl>(Shape{2, 2}, std::vector<float>{1.5f, -2.5f, 0.0f, 3.25f});
        cpu_impl->setDataType(Data_Type::FLOAT16);
        if (cpu_impl->isEmpty() || cpu_impl->getDataType() != Data_Type::FLOAT16)
        {
            return failStep("5C Cpu_Tensor_Impl::setDataType(FLOAT16) state");
        }
        auto impl_fp16_vals = cpu_impl->getData();
        if (!nearlyEqual(impl_fp16_vals[0], 1.5f, 1e-2f) || !nearlyEqual(impl_fp16_vals[1], -2.5f, 1e-2f))
        {
            return failStep("5C Cpu_Tensor_Impl::setDataType(FLOAT16) value preservation");
        }
        cpu_impl->setDataType(Data_Type::FLOAT32);
        if (cpu_impl->isEmpty() || cpu_impl->getDataType() != Data_Type::FLOAT32 ||
            !nearlyEqual(cpu_impl->getData()[3], 3.25f, 1e-2f))
        {
            return failStep("5C Cpu_Tensor_Impl::setDataType(FLOAT32) round-trip");
        }

        Tensor af16 = Tensor(Shape{2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, Execution_Target::CPU).toFp16();
        Tensor bf16 = Tensor(Shape{2, 2}, std::vector<float>{2.f, 1.f, 0.5f, 4.f}, Execution_Target::CPU).toFp16();
        Tensor mul_fp16(Execution_Target::CPU);
        af16.mulScalar(2.0f, mul_fp16);
        if ((af16 + bf16).getDataType() != Data_Type::FLOAT16 ||
            (af16 * bf16).getDataType() != Data_Type::FLOAT16 ||
            af16.transpose().getDataType() != Data_Type::FLOAT16 ||
            mul_fp16.getDataType() != Data_Type::FLOAT16)
        {
            return failStep("5C CPU FP16 output dtype parity");
        }

        Tensor c3d(Shape{2, 2, 3}, std::vector<float>{1,2,3, 4,5,6, 7,8,9, 10,11,12}, Execution_Target::CPU);
        Tensor c11d(Shape{1, 1, 3}, std::vector<float>{10.f, 20.f, 30.f}, Execution_Target::CPU);
        Tensor cadd = c3d + c11d;
        auto cadd_d = cadd.getData();
        if (cadd.getShape() != Shape{2, 2, 3} || !nearlyEqual(cadd_d[0], 11.f) || !nearlyEqual(cadd_d[11], 42.f))
        {
            return failStep("5C CPU 3D broadcast [2,2,3] + [1,1,3]");
        }
        Tensor r1(Shape{1, 3}, std::vector<float>{1.f, 2.f, 3.f}, Execution_Target::CPU);
        Tensor r2(Shape{2, 3}, std::vector<float>{4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, Execution_Target::CPU);
        Tensor sym = r1 + r2;
        if (sym.getShape() != Shape{2, 3} || !nearlyEqual(sym.getData()[0], 5.f) || !nearlyEqual(sym.getData()[5], 12.f))
        {
            return failStep("5C CPU symmetric broadcast [1,3] + [2,3]");
        }

        std::vector<float> ga_vals = {1,2,3, 4,5,6,   7,8,9, 10,11,12};
        std::vector<float> gb_vals = {1,0, 0,1, 1,1,   2,0, 0,2, 0,0,   1,-1, 0,1, 2,0};
        Tensor ga(Shape{2, 1, 2, 3}, ga_vals, Execution_Target::CPU);
        Tensor gb(Shape{1, 3, 3, 2}, gb_vals, Execution_Target::CPU);
        Tensor gc = ga * gb;
        auto gc_d = gc.getData();
        if (gc.getShape() != Shape{2, 3, 2, 2}) return failStep("5C CPU 4D GEMM shape");
        if (!nearlyEqual(gc_d[0], 4.f) || !nearlyEqual(gc_d[1], 5.f) ||
            !nearlyEqual(gc_d[2], 10.f) || !nearlyEqual(gc_d[3], 11.f) ||
            !nearlyEqual(gc_d[22], 34.f) || !nearlyEqual(gc_d[23], 1.f))
        {
            return failStep("5C CPU 4D GEMM values");
        }
        bool threw_4d_cpu = false;
        try
        {
            Tensor bad_a(Shape{2, 3, 2, 2}, Execution_Target::CPU);
            Tensor bad_b(Shape{3, 2, 2, 2}, Execution_Target::CPU);
            (void)(bad_a * bad_b);
        }
        catch (const std::invalid_argument &) { threw_4d_cpu = true; }
        if (!threw_4d_cpu) return failStep("5C CPU 4D mismatched batch rejection");

        bool threw_nan_inv = false;
        try
        {
            Tensor nan_sq(Shape{2, 2}, std::vector<float>{1.f, 0.f, 0.f, std::numeric_limits<float>::quiet_NaN()}, Execution_Target::CPU);
            (void)nan_sq.inverse();
        }
        catch (...) { threw_nan_inv = true; }
        if (!threw_nan_inv) return failStep("5C inverse NaN rejection");

        Tensor big_v(Shape{1, 2}, std::vector<float>{3e25f, 4e25f}, Execution_Target::CPU);
        Tensor norm_v = big_v.normalize();
        if (!nearlyEqual(norm_v.getData()[0], 0.6f, 1e-4f) || !nearlyEqual(norm_v.getData()[1], 0.8f, 1e-4f))
        {
            return failStep("5C normalize overflow resistance");
        }

        Tensor tiny(Shape{1, 1}, std::vector<float>{1e-9f}, Execution_Target::CPU);
        Tensor div_tiny(Execution_Target::CPU);
        tiny.divScalar(1e-9f, div_tiny);
        if (!nearlyEqual(div_tiny.getData()[0], 1.0f, 1e-3f)) return failStep("5C divScalar small valid scalar");
        bool threw_div0 = false;
        try { tiny.divScalar(0.0f, div_tiny); } catch (...) { threw_div0 = true; }
        if (!threw_div0) return failStep("5C divScalar zero rejection");

        Tensor num(Shape{1, 2}, std::vector<float>{4.f, 6.f}, Execution_Target::CPU);
        Tensor den(Shape{1, 2}, std::vector<float>{2.f, 0.f}, Execution_Target::CPU);
        Tensor div_out(Shape{1, 2}, std::vector<float>{7.f, 8.f}, Execution_Target::CPU);
        bool threw_hdiv = false;
        try { num.hadamardDiv(den, div_out); } catch (...) { threw_hdiv = true; }
        if (!threw_hdiv || !nearlyEqual(div_out.getData()[0], 7.f) || !nearlyEqual(div_out.getData()[1], 8.f))
        {
            return failStep("5C hadamardDiv no partial output mutation");
        }

        std::vector<float> pt_in = {0.5f, -0.5f, 0.2f};
        std::vector<uint32_t> pt_packed(1, 0u);
        std::vector<float> pt_out(3, 0.0f);
        Cpu_Tensor_Impl::packTernary(pt_in.data(), pt_packed.data(), 1, 3, 1.0f);
        Cpu_Tensor_Impl::unpackTernary(pt_packed.data(), pt_out.data(), 1, 3, 1.0f);
        if (!nearlyEqual(pt_out[0], 1.0f) || !nearlyEqual(pt_out[1], -1.0f) || !nearlyEqual(pt_out[2], 0.0f))
        {
            return failStep("5C packTernary rounding at +/-0.5");
        }
        std::vector<float> pt_s = {0.25f};
        std::vector<uint32_t> pt_sp(1, 0u);
        std::vector<float> pt_so(1, 0.0f);
        Cpu_Tensor_Impl::packTernary(pt_s.data(), pt_sp.data(), 1, 1, 2.0f);
        Cpu_Tensor_Impl::unpackTernary(pt_sp.data(), pt_so.data(), 1, 1, 1.0f);
        if (!nearlyEqual(pt_so[0], 1.0f)) return failStep("5C packTernary inv_gamma parameter");

        Tensor g3d(Shape{2, 2, 3}, std::vector<float>{1,2,3, 4,5,6, 7,8,9, 10,11,12}, Execution_Target::VULKAN_GPU);
        Tensor g11d(Shape{1, 1, 3}, std::vector<float>{10.f, 20.f, 30.f}, Execution_Target::VULKAN_GPU);
        Tensor g1d(Shape{3}, std::vector<float>{10.f, 20.f, 30.f}, Execution_Target::VULKAN_GPU);
        auto g_res1 = (g3d + g11d).getData();
        auto g_res2 = (g3d + g1d).getData();
        for (size_t i = 0; i < cadd_d.size(); ++i)
        {
            if (!nearlyEqual(g_res1[i], cadd_d[i], 1e-4f) || !nearlyEqual(g_res2[i], cadd_d[i], 1e-4f))
            {
                return failStep("5C GPU 3D suffix broadcast mismatch");
            }
        }

        bool threw_4d_gpu = false;
        try
        {
            Tensor g_bad_a(Shape{2, 3, 2, 2}, Execution_Target::VULKAN_GPU);
            Tensor g_bad_b(Shape{3, 2, 2, 2}, Execution_Target::VULKAN_GPU);
            (void)(g_bad_a * g_bad_b);
        }
        catch (const std::invalid_argument &) { threw_4d_gpu = true; }
        if (!threw_4d_gpu) return failStep("5C GPU 4D mismatched batch rejection");

        Tensor gt(Shape{2, 4}, Data_Type::TERNARY_INT2, Execution_Target::VULKAN_GPU);
        gt.uploadData({1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 1.f, -1.f});
        std::vector<uint32_t> before_packed = gt.getPackedData();
        gt.contiguous(gt);
        if (gt.isEmpty() || gt.getPackedData() != before_packed)
        {
            return failStep("5C GPU TERNARY_INT2 Tensor::contiguous(t)");
        }
        gt.getImplementation()->contiguous(*gt.getImplementation());
        if (gt.isEmpty() || gt.getPackedData() != before_packed)
        {
            return failStep("5C GPU TERNARY_INT2 direct impl::contiguous(self)");
        }

        Tensor gcache(Shape{2, 3}, std::vector<float>{1,2,3,4,5,6}, Execution_Target::VULKAN_GPU);
        (void)gcache.getData();
        auto &gcache_impl = Gpu_Tensor_Impl::castToGpu(*gcache.getImplementation());
        if (gcache_impl.getHostCache().empty()) return failStep("5C GPU host_cache populate check");
        gcache.uploadData({6,5,4,3,2,1});
        if (!gcache_impl.getHostCache().empty()) return failStep("5C GPU uploadData did not clear host_cache");
        (void)gcache.getData();
        gcache.reshape(1, 4);
        if (!Gpu_Tensor_Impl::castToGpu(*gcache.getImplementation()).getHostCache().empty())
        {
            return failStep("5C GPU reshape did not clear host_cache");
        }

        bool threw_rope_odd = false;
        try
        {
            Tensor rope_in(Shape{1, 3}, std::vector<float>{1.f, 2.f, 3.f}, Execution_Target::VULKAN_GPU);
            Tensor rope_out(Execution_Target::VULKAN_GPU);
            rope_in.applyRoPE(rope_out, 1, 3, 1, 10000.f, 1, 0, 0);
        }
        catch (const std::invalid_argument &) { threw_rope_odd = true; }
        if (!threw_rope_odd) return failStep("5C GPU applyRoPE odd head_dim guard");

        bool threw_swiglu = false;
        try
        {
            Tensor sw_a(Shape{2, 4}, Execution_Target::VULKAN_GPU);
            Tensor sw_b(Shape{2, 5}, Execution_Target::VULKAN_GPU);
            Tensor sw_out(Execution_Target::VULKAN_GPU);
            sw_a.swigluForward(sw_b, sw_out);
        }
        catch (const std::invalid_argument &) { threw_swiglu = true; }
        if (!threw_swiglu) return failStep("5C GPU swigluForward shape mismatch guard");

        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();
    }

    return true;
}

bool testWave5Phase5DFRegressions()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("Wave5 5D-5F", msg);
    };

    {
        Tensor x(Shape{2, 4}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}, Execution_Target::CPU);
        Tensor out(Execution_Target::CPU);
        Tensor mask(Execution_Target::CPU);

        if (!expectThrows<std::invalid_argument>([&]() { ops::dropout::dropout(x, out, mask, -0.1f, true, 123); })) return failStep("5D G08 dropout negative rate guard");
        if (!expectThrows<std::invalid_argument>([&]() { ops::dropout::dropout(x, out, mask, 1.0f, true, 123); })) return failStep("5D G08 dropout rate == 1.0 guard");

        Tensor sig_out(Shape{1, 4}, std::vector<float>{1000.0f, -1000.0f, 0.5f, 0.25f}, Execution_Target::CPU);
        Tensor sig_grad_in(Shape{1, 4}, std::vector<float>{1e38f, -1e38f, 1000.0f, -1000.0f}, Execution_Target::CPU);
        Tensor sig_dx(Execution_Target::CPU);
        ops::activation::sigmoidBackward(sig_out, sig_grad_in, sig_dx);
        const auto dx_vals = sig_dx.getData();
        if (dx_vals.size() != 4) return failStep("5D G08 sigmoidBackward size");
        for (float v : dx_vals)
        {
            if (!std::isfinite(v) || std::fabs(v) > 1e6f)
            {
                return failStep("5D G08 sigmoidBackward extreme finite/clamp");
            }
        }
    }

    {
        constexpr uint32_t H = 2, W = 2, C = 300, N = 2;
        std::vector<float> bn_in_data(N * H * W * C);
        std::vector<float> bn_dy_data(N * H * W * C);
        for (size_t i = 0; i < bn_in_data.size(); ++i)
        {
            bn_in_data[i] = 0.05f * static_cast<float>(static_cast<int>(i % 17) - 8);
            bn_dy_data[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5 + 1) % 13) - 6);
        }

        Batch_Norm_2d_Layer bn_cpu(H, W, C, 1e-5f, 0.1f, Execution_Target::CPU);
        Batch_Norm_2d_Layer bn_gpu(H, W, C, 1e-5f, 0.1f, Execution_Target::VULKAN_GPU);

        Tensor in_cpu(Shape{N, H, W, C}, bn_in_data, Execution_Target::CPU);
        Tensor dy_cpu(Shape{N, H, W, C}, bn_dy_data, Execution_Target::CPU);
        Tensor out_cpu = bn_cpu.forward(in_cpu);
        Tensor dx_cpu = bn_cpu.backward(dy_cpu);

        Tensor in_gpu(Shape{N, H, W, C}, bn_in_data, Execution_Target::VULKAN_GPU);
        Tensor dy_gpu(Shape{N, H, W, C}, bn_dy_data, Execution_Target::VULKAN_GPU);
        Tensor out_gpu = bn_gpu.forward(in_gpu);
        Tensor dx_gpu = bn_gpu.backward(dy_gpu);

        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();

        const auto out_c = out_cpu.getData();
        const auto out_g = out_gpu.getData();
        const auto dx_c = dx_cpu.getData();
        const auto dx_g = dx_gpu.getData();
        const auto dg_c = bn_cpu.getGammaGradient().getData();
        const auto dg_g = bn_gpu.getGammaGradient().getData();
        if (out_c.size() != out_g.size() || dx_c.size() != dx_g.size() || dg_c.size() != C || dg_g.size() != C)
        {
            return failStep("5D G09 BN2D C>256 size mismatch");
        }
        for (size_t i = 0; i < out_c.size(); ++i)
        {
            if (!std::isfinite(out_g[i]) || !std::isfinite(dx_g[i]) ||
                !nearlyEqual(out_c[i], out_g[i], 1e-2f) || !nearlyEqual(dx_c[i], dx_g[i], 1e-2f))
            {
                return failStep("5D G09 BN2D C=300 forward/backward CPU vs GPU mismatch");
            }
        }
        for (size_t c = 0; c < C; ++c)
        {
            if (!std::isfinite(dg_g[c]) || !nearlyEqual(dg_c[c], dg_g[c], 2e-2f))
            {
                return failStep("5D G09 BN2D C=300 dgamma CPU vs GPU mismatch");
            }
        }

        constexpr size_t RMS_D = 64;
        std::vector<float> rms_x(2 * 4 * RMS_D), rms_dy(2 * 4 * RMS_D);
        for (size_t i = 0; i < rms_x.size(); ++i)
        {
            rms_x[i] = 0.1f * static_cast<float>(static_cast<int>(i % 11) - 5) + 0.05f;
            rms_dy[i] = 0.08f * static_cast<float>(static_cast<int>((i * 3 + 2) % 9) - 4);
        }
        RMSNorm_Layer rms_cpu(RMS_D, 1e-5f, Execution_Target::CPU);
        RMSNorm_Layer rms_gpu(RMS_D, 1e-5f, Execution_Target::VULKAN_GPU);

        Tensor rx_c = rms_cpu.forward(Tensor(Shape{2, 4, RMS_D}, rms_x, Execution_Target::CPU));
        Tensor rdx_c = rms_cpu.backward(Tensor(Shape{2, 4, RMS_D}, rms_dy, Execution_Target::CPU));
        Tensor rx_g = rms_gpu.forward(Tensor(Shape{2, 4, RMS_D}, rms_x, Execution_Target::VULKAN_GPU));
        Tensor rdx_g = rms_gpu.backward(Tensor(Shape{2, 4, RMS_D}, rms_dy, Execution_Target::VULKAN_GPU));

        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();

        const auto r_out_c = rx_c.getData();
        const auto r_out_g = rx_g.getData();
        const auto r_dx_c = rdx_c.getData();
        const auto r_dx_g = rdx_g.getData();
        const auto r_dg_c = rms_cpu.getWeightsGradient().getData();
        const auto r_dg_g = rms_gpu.getWeightsGradient().getData();
        if (r_out_c.size() != r_out_g.size() || r_dx_c.size() != r_dx_g.size() || r_dg_c.size() != r_dg_g.size())
        {
            return failStep("5D G09 RMSNorm output/grad size mismatch");
        }
        for (size_t i = 0; i < r_dx_c.size(); ++i)
        {
            if (!nearlyEqual(r_out_c[i], r_out_g[i], 1e-3f) || !nearlyEqual(r_dx_c[i], r_dx_g[i], 1e-3f))
            {
                return failStep("5D G09 RMSNorm dx CPU vs GPU mismatch");
            }
        }
        for (size_t i = 0; i < r_dg_c.size(); ++i)
        {
            if (!nearlyEqual(r_dg_c[i], r_dg_g[i], 2e-3f))
            {
                return failStep("5D G09 RMSNorm dgamma CPU vs GPU mismatch");
            }
        }
    }

    {
        constexpr size_t LEN = 512;
        std::vector<float> pred_v(LEN), tgt_v(LEN);
        for (size_t i = 0; i < LEN; ++i)
        {
            pred_v[i] = 0.2f + 0.6f * static_cast<float>((i * 7 + 1) % 10) / 9.0f;
            tgt_v[i] = (i % 3 == 0) ? 1.0f : 0.0f;
        }
        Tensor p_c(Shape{1, LEN}, pred_v, Execution_Target::CPU);
        Tensor t_c(Shape{1, LEN}, tgt_v, Execution_Target::CPU);
        Tensor p_g(Shape{1, LEN}, pred_v, Execution_Target::VULKAN_GPU);
        Tensor t_g(Shape{1, LEN}, tgt_v, Execution_Target::VULKAN_GPU);

        Tensor mse_c(Execution_Target::CPU), mse_g(Execution_Target::VULKAN_GPU);
        Tensor dice_c(Execution_Target::CPU), dice_g(Execution_Target::VULKAN_GPU);
        Tensor dbce_c(Execution_Target::CPU), dbce_g(Execution_Target::VULKAN_GPU);
        Tensor dgrad_c(Execution_Target::CPU), dgrad_g(Execution_Target::VULKAN_GPU);

        ops::loss::mse(p_c, t_c, mse_c);
        ops::loss::mse(p_g, t_g, mse_g);
        ops::loss::dice(p_c, t_c, dice_c, 1e-6f);
        ops::loss::dice(p_g, t_g, dice_g, 1e-6f);
        ops::loss::diceBce(p_c, t_c, dbce_c, 1.0f, 1.0f, 1e-6f);
        ops::loss::diceBce(p_g, t_g, dbce_g, 1.0f, 1.0f, 1e-6f);
        ops::loss::diceBceGradient(p_c, t_c, dgrad_c, 1.0f, 1.0f, 1e-6f);
        ops::loss::diceBceGradient(p_g, t_g, dgrad_g, 1.0f, 1.0f, 1e-6f);

        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();

        if (mse_g.getShape() != Shape{1, 1} || dice_g.getShape() != Shape{1, 1} || dbce_g.getShape() != Shape{1, 4})
        {
            return failStep("5D G10 multi-WG loss output shape check");
        }
        const auto dbce_cv = dbce_c.getData();
        const auto dbce_gv = dbce_g.getData();
        if (dbce_cv.size() != 4 || dbce_gv.size() != 4 ||
            !nearlyEqual(mse_c.getData()[0], mse_g.getData()[0], 1e-3f) ||
            !nearlyEqual(dice_c.getData()[0], dice_g.getData()[0], 1e-3f) ||
            !nearlyEqual(dbce_cv[0], dbce_gv[0], 1e-3f) ||
            !nearlyEqual(dbce_cv[1], dbce_gv[1], 5e-3f) ||
            !nearlyEqual(dbce_cv[2], dbce_gv[2], 5e-3f) ||
            !nearlyEqual(dbce_cv[3], dbce_gv[3], 5e-3f))
        {
            return failStep("5D G10 2-pass GPU loss CPU vs GPU mismatch");
        }
        const auto dg_c_v = dgrad_c.getData();
        const auto dg_g_v = dgrad_g.getData();
        if (dg_c_v.size() != LEN || dg_g_v.size() != LEN) return failStep("5D G10 diceBceGradient size mismatch");
        for (size_t i = 0; i < LEN; ++i)
        {
            if (!nearlyEqual(dg_c_v[i], dg_g_v[i], 1e-3f))
            {
                return failStep("5D G10 diceBceGradient CPU vs GPU mismatch");
            }
        }

        Tensor emb_w(Shape{6, 4}, std::vector<float>{
            0.0f, 0.1f, 0.2f, 0.3f,
            1.0f, 1.1f, 1.2f, 1.3f,
            2.0f, 2.1f, 2.2f, 2.3f,
            3.0f, 3.1f, 3.2f, 3.3f,
            4.0f, 4.1f, 4.2f, 4.3f,
            5.0f, 5.1f, 5.2f, 5.3f
        }, Execution_Target::CPU);
        Tensor emb_w_fp16 = emb_w.toFp16();
        Tensor idx_3d(Shape{2, 3, 2}, std::vector<float>{
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f
        }, Execution_Target::CPU);

        Tensor emb_out_3d(Execution_Target::CPU);
        Tensor emb_out_fp16_auto(Execution_Target::CPU);
        Tensor emb_out_fp16(Execution_Target::CPU);
        ops::llm::embeddingForward(emb_w, idx_3d, emb_out_3d, false);
        ops::llm::embeddingForward(emb_w_fp16, idx_3d, emb_out_fp16_auto, false);
        ops::llm::embeddingForward(emb_w, idx_3d, emb_out_fp16, true);

        const Shape expected_emb_shape{2, 3, 2, 4};
        if (emb_out_3d.getShape() != expected_emb_shape ||
            emb_out_fp16_auto.getShape() != expected_emb_shape ||
            emb_out_fp16.getShape() != expected_emb_shape ||
            emb_out_3d.getDataType() != Data_Type::FLOAT32 ||
            emb_out_fp16_auto.getDataType() != Data_Type::FLOAT16 ||
            emb_out_fp16.getDataType() != Data_Type::FLOAT16)
        {
            return failStep("5D G10 embeddingForward 3D shape or FP16 dtype check");
        }
        const auto e32 = emb_out_3d.getData();
        const auto e16_auto = emb_out_fp16_auto.getData();
        const auto e16 = emb_out_fp16.getData();
        for (size_t i = 0; i < e32.size(); ++i)
        {
            if (!nearlyEqual(e32[i], e16_auto[i], 1e-2f) || !nearlyEqual(e32[i], e16[i], 1e-2f))
            {
                return failStep("5D G10 embeddingForward FP16 values");
            }
        }

        Tensor bad_idx(Shape{1, 2}, std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1e20f}, Execution_Target::CPU);
        Tensor emb_bad_out(Execution_Target::CPU);
        ops::llm::embeddingForward(emb_w, bad_idx, emb_bad_out, false);
        for (float v : emb_bad_out.getData())
        {
            if (v != 0.0f) return failStep("5D G10 embeddingForward non-finite/huge index zero fill");
        }
    }

    {
        static_assert(std::is_same_v<std::underlying_type_t<Layer_Type>, uint32_t>,
                      "Layer_Type underlying type must be uint32_t");

        Relu_Layer relu(Execution_Target::CPU);
        relu.setAccumulated(true);
        relu.setMixedPrecision(true);
        auto relu_c = relu.clone();
        if (!relu_c->isAccumulated() || !relu_c->isMixedPrecisionEnabled())
        {
            return failStep("5E G11 Relu_Layer::clone flag preservation");
        }

        Dropout_Layer drop(0.5f, Execution_Target::CPU);
        drop.setAccumulated(true);
        drop.setMixedPrecision(true);
        auto drop_c = drop.clone();
        if (!drop_c->isAccumulated() || !drop_c->isMixedPrecisionEnabled())
        {
            return failStep("5E G11 Dropout_Layer::clone flag preservation");
        }

        Up_Sample_2d_Layer up(2, 2, 1, 2, Execution_Target::CPU);
        up.setAccumulated(true);
        up.setMixedPrecision(true);
        auto up_c = up.clone();
        if (!up_c->isAccumulated() || !up_c->isMixedPrecisionEnabled())
        {
            return failStep("5E G11 Up_Sample_2d_Layer::clone flag preservation");
        }

        Avg_Pool_2d_Layer avg(4, 4, 1, 2, 2, 0, Execution_Target::CPU);

        auto checkLogicBeforeFwd = [](ILayer &layer) -> bool
        {
            return expectThrows<std::logic_error>([&]() {
                layer.backward(Tensor(Shape{1, 4}, std::vector<float>(4, 1.0f), Execution_Target::CPU));
            });
        };
        if (!checkLogicBeforeFwd(drop) || !checkLogicBeforeFwd(avg) || !checkLogicBeforeFwd(up))
        {
            return failStep("5E G11 backward-before-forward std::logic_error contract");
        }

        auto checkPopOutOfRange = [](ILayer &layer) -> bool
        {
            return expectThrows<std::out_of_range>([&]() { layer.setPopulationParameter(0, {}); });
        };
        if (!checkPopOutOfRange(drop) || !checkPopOutOfRange(avg) || !checkPopOutOfRange(up))
        {
            return failStep("5E G11 setPopulationParameter std::out_of_range contract");
        }

        if (!expectThrows<std::invalid_argument>([&]() {
                relu.forward(Tensor(Shape{1, 4}, std::vector<float>(4, 1.0f), Execution_Target::VULKAN_GPU));
            }))
        {
            return failStep("5E G11 Relu_Layer execution target guard");
        }

        Dropout_Layer drop_eval(0.5f, Execution_Target::CPU);
        drop_eval.setTrainingMode(false);
        Tensor eval_in(Shape{2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, Execution_Target::CPU);
        Tensor eval_out = drop_eval.forward(eval_in, std::vector<Tensor>{});
        if (!verifyMatrix(eval_out, {1, 2, 3, 4, 5, 6, 7, 8}))
        {
            return failStep("5E G11 Dropout_Layer eval batched forward");
        }
    }

    {
        Linear_Layer lin(4, 5, Execution_Target::CPU);
        for (float v : lin.getWeightsGradient().getData()) if (v != 0.0f) return failStep("5E G12 Linear w_grad zero init");
        for (float v : lin.getBiasesGradient().getData()) if (v != 0.0f) return failStep("5E G12 Linear b_grad zero init");

        Tensor lin_in_4d(Shape{2, 2, 3, 4}, std::vector<float>(2 * 2 * 3 * 4, 0.5f), Execution_Target::CPU);
        Tensor lin_out_4d = lin.forward(lin_in_4d);
        if (lin_out_4d.getShape() != Shape{2, 2, 3, 5}) return failStep("5E G12 Linear 4D forward shape");
        Tensor lin_dx_4d = lin.backward(Tensor(Shape{2, 2, 3, 5}, std::vector<float>(2 * 2 * 3 * 5, 1.0f), Execution_Target::CPU));
        if (lin_dx_4d.getShape() != Shape{2, 2, 3, 4}) return failStep("5E G12 Linear 4D backward shape");

        Conv2d_Layer conv(6, 6, 2, 3, 3, 1, 0, Execution_Target::CPU);
        for (float v : conv.getWeightsGradient().getData()) if (v != 0.0f) return failStep("5E G12 Conv2d w_grad zero init");
        for (float v : conv.getBiasesGradient().getData()) if (v != 0.0f) return failStep("5E G12 Conv2d b_grad zero init");

        conv.setInputHeight(8);
        conv.setInputWidth(8);
        if (conv.getOutputHeight() != 6 || conv.getOutputWidth() != 6)
        {
            return failStep("5E G12 Conv2d spatial setter recompute");
        }
        bool threw_small = false;
        try { conv.setInputHeight(1); }
        catch (const std::invalid_argument &) { threw_small = true; }
        if (!threw_small) return failStep("5E G12 Conv2d invalid height guard");

        conv.setInputHeight(4);
        conv.setInputWidth(4);
        if (conv.getOutputHeight() != 2 || conv.getOutputWidth() != 2)
        {
            return failStep("5E G12 Conv2d spatial setter 4x4 -> 2x2");
        }

        conv.setAccumulated(true);
        Tensor c_in(Shape{1, 4, 4, 2}, std::vector<float>(32, 0.25f), Execution_Target::CPU);
        Tensor c_dy(Shape{1, 2, 2, 3}, std::vector<float>(12, 0.5f), Execution_Target::CPU);
        conv.forward(c_in);
        conv.backward(c_dy);
        const auto g1 = conv.getWeightsGradient().getData();
        conv.forward(c_in);
        conv.backward(c_dy);
        const auto g2 = conv.getWeightsGradient().getData();
        if (g1.size() != g2.size() || g1.empty()) return failStep("5E G12 Conv2d accumulated grad size");
        for (size_t i = 0; i < g1.size(); ++i)
        {
            if (!nearlyEqual(g2[i], 2.0f * g1[i], 1e-4f)) return failStep("5E G12 Conv2d accumulated grad value");
        }

        Ternary_Linear_Layer tlin(4, 4, Execution_Target::CPU);
        if (tlin.getLayerType() != Layer_Type::TERNARY_LINEAR) return failStep("5E G12 Ternary_Linear_Layer initial type");
        tlin.dequantizeFromTernary();
        if (tlin.getLayerType() != Layer_Type::LINEAR) return failStep("5E G12 Ternary_Linear_Layer type after dequantize");
    }

    {
        Transformer_Block tblock(16, 2, 32, 1e-5f, 10000.0f, Execution_Target::CPU);
        KV_Cache_Manager kv(1, 2, 8, 8, Execution_Target::CPU, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);
        Tensor kv_out = tblock.forward(Tensor(Shape{1, 2, 16}, std::vector<float>(32, 0.1f), Execution_Target::CPU), &kv);
        bool threw_kv_bwd = false;
        try { tblock.backward(kv_out); }
        catch (const std::logic_error &) { threw_kv_bwd = true; }
        if (!threw_kv_bwd) return failStep("5E G13 Transformer_Block KV-cached backward guard");

        Embedding_Layer emb(8, 4, Execution_Target::CPU);
        ILayer *emb_base = &emb;
        emb.forward(std::vector<int32_t>{1, 3});
        if (emb_base->getWeights().getShape() != Shape{8, 4} ||
            emb_base->getWeightsGradient().getShape() != Shape{8, 4} ||
            emb_base->getInput().getTotalElements() != 2 ||
            emb_base->getOutput().getShape() != Shape{2, 4})
        {
            return failStep("5E G13 Embedding_Layer base getter overrides (1D)");
        }
        emb.forward(std::vector<std::vector<int32_t>>{{1, 3}});
        if (emb_base->getOutput().getShape() != Shape{1, 2, 4})
        {
            return failStep("5E G13 Embedding_Layer base getter overrides (2D batch)");
        }

        Batch_Norm_Layer bn(3, 1e-5f, 0.1f, Execution_Target::CPU);
        const std::vector<std::vector<float>> bad_stats{
            {1.0f, std::numeric_limits<float>::quiet_NaN(), -2.0f},
            {-0.5f, std::numeric_limits<float>::infinity(), 2.0f}
        };
        size_t cursor = 0;
        bn.restoreNonTrainableState(bad_stats, cursor);
        bn.sanitizeRunningStats();
        const auto rm = bn.getRunningMean().getData();
        const auto rv = bn.getRunningVariance().getData();
        if (rm.size() != 3 || rv.size() != 3 ||
            !nearlyEqual(rm[0], 1.0f) || !nearlyEqual(rm[1], 0.0f) || !nearlyEqual(rm[2], -2.0f) ||
            !nearlyEqual(rv[0], 1.0f) || !nearlyEqual(rv[1], 1.0f) || !nearlyEqual(rv[2], 2.0f))
        {
            return failStep("5E G13 Batch_Norm_Layer::sanitizeRunningStats");
        }

        bool threw_eps = false, threw_mom = false;
        try { bn.setEpsilon(0.0f); } catch (const std::invalid_argument &) { threw_eps = true; }
        try { bn.setMomentum(-0.1f); } catch (const std::invalid_argument &) { threw_mom = true; }
        if (!threw_eps || !threw_mom) return failStep("5E G13 Batch_Norm_Layer hyperparameter setter guards");

        SwiGLU_Layer swiglu(Execution_Target::CPU);
        swiglu.setAccumulated(true);
        swiglu.setMixedPrecision(true);
        auto sw_clone = swiglu.clone();
        if (!swiglu.supportsPopulationBatch() || !sw_clone->isAccumulated() || !sw_clone->isMixedPrecisionEnabled())
        {
            return failStep("5E G13 SwiGLU_Layer pop support or clone flags");
        }
        Tensor sw_out = swiglu.forward(Tensor(Shape{2, 3, 4}, std::vector<float>(24, 1.0f), Execution_Target::CPU), std::vector<Tensor>{});
        if (sw_out.getShape() != Shape{2, 3, 2}) return failStep("5E G13 SwiGLU_Layer batched forward shape");
        bool threw_sw_pop = false;
        try { swiglu.setPopulationParameter(0, {}); } catch (const std::out_of_range &) { threw_sw_pop = true; }
        if (!threw_sw_pop) return failStep("5E G13 SwiGLU_Layer setPopulationParameter out_of_range");

        PPO_Actor_Critic_Layer ppo(2, Execution_Target::CPU);
        ppo.addActorLayer<Relu_Layer>(Execution_Target::CPU);
        ppo.addCriticLayer<Relu_Layer>(Execution_Target::CPU);
        if (ppo.hasParameters()) return failStep("5E G13 PPO_Actor_Critic_Layer::hasParameters() false check");
        ppo.addActorLayer<Linear_Layer>(2, 2, Execution_Target::CPU);
        if (!ppo.hasParameters()) return failStep("5E G13 PPO_Actor_Critic_Layer::hasParameters() true check");
    }

    {
        Res_Net_Block_2d_Layer rb(Execution_Target::CPU);
        bool threw_null_main = false, threw_null_short = false;
        try { rb.addMainLayer(std::unique_ptr<ILayer>{}); } catch (const std::invalid_argument &) { threw_null_main = true; }
        try { rb.addShortcutLayer(std::unique_ptr<ILayer>{}); } catch (const std::invalid_argument &) { threw_null_short = true; }
        if (!threw_null_main || !threw_null_short) return failStep("5F G14a Res_Net_Block_2d_Layer null sublayer guard");

        rb.setPostActivation(std::make_unique<Relu_Layer>(Execution_Target::CPU));
        rb.setPostActivation(std::unique_ptr<ILayer>{});
        if (rb.getPostActivation() != nullptr) return failStep("5F G14a Res_Net_Block_2d_Layer clear post_activation");

        rb.setAccumulated(true);
        auto &added_relu = rb.addMainLayer<Relu_Layer>(Execution_Target::CPU);
        auto rb_c = rb.clone();
        if (!added_relu.isAccumulated() || !rb_c->isAccumulated())
        {
            return failStep("5F G14a Res_Net_Block_2d_Layer isAccumulated propagation & clone");
        }

        Res_Net_20_Layer rn20(16, 16, 1, 2, Execution_Target::CPU);
        rn20.setAccumulated(true);
        if (!rn20.isAccumulated()) return failStep("5F G14a Res_Net_20_Layer::isAccumulated()");

        U_Net_Layer unet(32, 32, 1, 1, 4, Execution_Target::CPU);
        bool threw_rank3 = false, threw_target = false;
        try { unet.forward(Tensor(Shape{1, 32, 32}, std::vector<float>(1024, 0.1f), Execution_Target::CPU)); }
        catch (const std::invalid_argument &) { threw_rank3 = true; }
        try { unet.forward(Tensor(Shape{1, 1, 32, 32}, std::vector<float>(1024, 0.1f), Execution_Target::VULKAN_GPU)); }
        catch (const std::invalid_argument &) { threw_target = true; }
        if (!threw_rank3 || !threw_target) return failStep("5F G14b U_Net_Layer input validation");

        std::vector<std::vector<float>> snap;
        unet.snapshotNonTrainableState(snap);
        if (snap.size() < 2 || snap[1].empty()) return failStep("5F G14b U_Net_Layer snapshot empty");
        snap[1][0] = -0.25f;
        size_t ucursor = 0;
        unet.restoreNonTrainableState(snap, ucursor);
        if (!unet.hasNanInRunningStats()) return failStep("5F G14b U_Net_Layer::hasNanInRunningStats() negative variance");

        NN_UNet_Layer nnunet(16, 16, 1, 1, 4, Execution_Target::CPU);
        const size_t n_params = nnunet.getParametersAndGradients().size();
        nnunet.setEnableDeepSupervision(false);
        if (nnunet.isDeepSupervisionEnabled() || nnunet.getParametersAndGradients().size() != n_params)
        {
            return failStep("5F G14b NN_UNet_Layer setEnableDeepSupervision / positional param count");
        }

        const auto tmp_dir = std::filesystem::temp_directory_path();
        const std::string cfg_path = (tmp_dir / "wave5_nnunet_cfg.tmp").string();
        const std::string inf_path = (tmp_dir / "wave5_nnunet_inf.tmp").string();
        auto cleanup = [&]()
        {
            std::error_code ec;
            std::filesystem::remove(cfg_path, ec);
            std::filesystem::remove(inf_path, ec);
        };
        cleanup();

        bool rt_ok = false;
        try
        {
            {
                std::ofstream out_cfg(cfg_path, std::ios::binary);
                nnunet.saveConfiguration(out_cfg);
            }
            {
                std::ifstream in_cfg(cfg_path, std::ios::binary);
                auto loaded_cfg = Training_Context::constructLayerFromConfig(in_cfg, Layer_Type::NN_UNET, Execution_Target::CPU);
                auto *nn_cfg = dynamic_cast<NN_UNet_Layer *>(loaded_cfg.get());
                if (!nn_cfg || nn_cfg->isDeepSupervisionEnabled())
                {
                    cleanup();
                    return failStep("5F G14b NN_UNet_Layer config round-trip ds flag");
                }
            }
            {
                std::ofstream out_inf(inf_path, std::ios::binary);
                nnunet.saveInference(out_inf);
            }
            {
                std::ifstream in_inf(inf_path, std::ios::binary);
                NN_UNet_Layer nn_loaded(16, 16, 1, 1, 4, Execution_Target::CPU);
                nn_loaded.setEnableDeepSupervision(true);
                nn_loaded.loadInference(in_inf);
                if (nn_loaded.isDeepSupervisionEnabled())
                {
                    cleanup();
                    return failStep("5F G14b NN_UNet_Layer inference round-trip ds flag");
                }
            }
            rt_ok = true;
        }
        catch (...)
        {
            cleanup();
            throw;
        }
        cleanup();
        if (!rt_ok) return failStep("5F G14b NN_UNet_Layer round-trip");
    }

    return true;
}

bool testWave5Phase5GIRegressions()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("Wave5 5G-5I", msg);
    };

    const auto tmp_dir = std::filesystem::temp_directory_path();
    const std::string poly_path = (tmp_dir / "wave5gi_poly_v2.ckpt").string();
    const std::string rop_path = (tmp_dir / "wave5gi_rop_v2.ckpt").string();
    const std::string dice_bad_path = (tmp_dir / "wave5gi_dice_bad.ckpt").string();
    const std::string causal_inf_path = (tmp_dir / "wave5gi_causal_inf.bin").string();
    const std::string bpe_good_path = (tmp_dir / "wave5gi_bpe_good.json").string();
    const std::string bpe_bad_path = (tmp_dir / "wave5gi_bpe_bad.json").string();
    const std::string ppo_ckpt_path = (tmp_dir / "wave5gi_ppo_v2.ckpt").string();
    const std::string ppo_inf_path = (tmp_dir / "wave5gi_ppo_inf.bin").string();
    const std::string prefs_bad_path = (tmp_dir / "wave5gi_prefs_bad.json").string();
    const std::filesystem::path clf_dir = tmp_dir / "wave5gi_clf_dir";
    const std::string clf_weights_path = (clf_dir / "weights.bin").string();

    Scoped_Temp_File_Guard temp_guard{
        {poly_path, rop_path, dice_bad_path, causal_inf_path, bpe_good_path, bpe_bad_path, ppo_ckpt_path, ppo_inf_path, prefs_bad_path},
        {clf_dir}};
    temp_guard.clean();

    {
        std::vector<float> logits_data = {
            2.0f, 0.5f, -0.5f, 0.1f,
            0.3f, 1.8f,  0.2f, 0.1f,
            0.1f, 2.2f,  0.4f, -0.3f,
            1.2f, -0.2f, 0.9f, 0.5f
        };
        std::vector<float> tgt_data = {0.0f, 2.0f, 1.0f, 2.0f};
        Fused_Cross_Entropy fce(2);

        Tensor logits_cpu(Shape{4, 4}, logits_data, Execution_Target::CPU);
        Tensor tgt_cpu(Shape{4}, tgt_data, Execution_Target::CPU);
        Tensor grad_cpu(Shape{4, 4}, Execution_Target::CPU);
        const float loss_cpu_only = fce.computeLoss(logits_cpu, tgt_cpu);
        const float loss_cpu_lag = fce.computeLossAndGradient(logits_cpu, tgt_cpu, grad_cpu);
        if (!std::isfinite(loss_cpu_only) || !nearlyEqual(loss_cpu_only, loss_cpu_lag, 1e-4f))
        {
            return failStep("5G G15a Fused_Cross_Entropy computeLoss CPU mismatch");
        }

        Tensor logits_gpu(Shape{4, 4}, logits_data, Execution_Target::VULKAN_GPU);
        Tensor tgt_gpu(Shape{4}, tgt_data, Execution_Target::VULKAN_GPU);
        Tensor grad_gpu(Shape{4, 4}, Execution_Target::VULKAN_GPU);
        const float loss_gpu_only = fce.computeLoss(logits_gpu, tgt_gpu);
        const float loss_gpu_lag = fce.computeLossAndGradient(logits_gpu, tgt_gpu, grad_gpu);
        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();
        if (!std::isfinite(loss_gpu_only) || !nearlyEqual(loss_gpu_only, loss_cpu_only, 1e-3f) ||
            !nearlyEqual(loss_gpu_only, loss_gpu_lag, 1e-3f))
        {
            return failStep("5G G15a Fused_Cross_Entropy computeLoss GPU mismatch");
        }

        Dice_Cost dc(1e-7f, Execution_Target::CPU);
        Tensor d_pred(Shape{1, 4}, std::vector<float>{0.2f, 0.8f, 0.4f, 0.9f}, Execution_Target::CPU);
        Tensor d_tgt(Shape{1, 4}, std::vector<float>{0.0f, 1.0f, 0.0f, 1.0f}, Execution_Target::CPU);
        const float dc_loss = dc.computeLoss(d_pred, d_tgt);
        const Tensor dc_g1 = dc.computeGradient(d_pred, d_tgt);
        Tensor dc_g2(Execution_Target::CPU);
        const float dc_lag = dc.computeLossAndGradient(d_pred, d_tgt, dc_g2);
        if (!nearlyEqual(dc_loss, dc_lag, 1e-5f) || !verifyMatrix(dc_g2, dc_g1.getData(), 1e-5f))
        {
            return failStep("5G G15a Dice_Cost single-pass computeLossAndGradient mismatch");
        }

        bool threw_eps = false;
        try { dc.setEpsilon(0.0f); }
        catch (const std::invalid_argument &) { threw_eps = true; }
        if (!threw_eps) return failStep("5G G15a Dice_Cost::setEpsilon(0) did not throw");

        {
            std::ofstream out_empty(dice_bad_path, std::ios::binary | std::ios::trunc);
        }
        bool threw_eof = false;
        try
        {
            std::ifstream in_empty(dice_bad_path, std::ios::binary);
            dc.loadCheckpoint(in_empty);
        }
        catch (const std::runtime_error &)
        {
            threw_eof = true;
        }
        if (!threw_eof) return failStep("5G G15a Dice_Cost::loadCheckpoint EOF did not throw");

        Tensor ext_pred(Shape{1, 4}, std::vector<float>{0.0f, 1.0f, 0.0f, 1.0f}, Execution_Target::CPU);
        Tensor ext_tgt(Shape{1, 4}, std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}, Execution_Target::CPU);
        Tensor ext_grad(Execution_Target::CPU);
        ops::loss::diceBceGradient(ext_pred, ext_tgt, ext_grad, 1.0f, 1.0f, 1e-7f);
        for (float v : ext_grad.getData())
        {
            if (!std::isfinite(v)) return failStep("5G G15a diceBceGradient extreme prediction non-finite");
        }
    }

    {
        Polynomial_Decay poly(0.01f, 0.001f, 10, 2.0f);
        poly.step();
        poly.step();
        if (poly.getCurrentEpoch() != 2 || !nearlyEqual(poly.getPower(), 2.0f, 1e-6f))
        {
            return failStep("5G G15b Polynomial_Decay initial state");
        }
        {
            std::ofstream out_poly(poly_path, std::ios::binary | std::ios::trunc);
            poly.saveCheckpoint(out_poly);
        }
        Polynomial_Decay poly_loaded;
        {
            std::ifstream in_poly(poly_path, std::ios::binary);
            poly_loaded.loadCheckpoint(in_poly);
        }
        if (poly_loaded.getCurrentEpoch() != poly.getCurrentEpoch() ||
            !nearlyEqual(poly_loaded.getPower(), 2.0f, 1e-6f) ||
            !nearlyEqual(poly_loaded.getCurrentRate(), poly.getCurrentRate(), 1e-6f))
        {
            return failStep("5G G15b Polynomial_Decay V2 round-trip");
        }

        Adam_Optimizer adam(poly);
        adam.setLearningRate(0.002f);
        if (adam.getLearningRateScheduler() != &poly || !nearlyEqual(adam.getLearningRate(), 0.002f, 1e-6f))
        {
            return failStep("5G G15b Adam_Optimizer::setLearningRate detached scheduler");
        }
        adam.detachLearningRateScheduler();
        if (adam.getLearningRateScheduler() != nullptr)
        {
            return failStep("5G G15b Adam_Optimizer::detachLearningRateScheduler");
        }

        Tensor p_reg(Shape{1, 2}, std::vector<float>{1.0f, 2.0f}, Execution_Target::CPU);
        Tensor g_reg(Shape{1, 2}, std::vector<float>{0.1f, 0.2f}, Execution_Target::CPU);
        Tensor p_unreg(Shape{1, 2}, std::vector<float>{3.0f, 4.0f}, Execution_Target::CPU);
        adam.step({{&p_reg, &g_reg}});
        adam.setParameterOrder({&p_reg, &p_unreg, nullptr});
        if (adam.getParameterOrder().size() != 1 || adam.getParameterOrder()[0] != &p_reg)
        {
            return failStep("5G G15b Adam_Optimizer::setParameterOrder filter");
        }

        Reduce_On_Plateau rop(0.01f, 1e-5f, 0.5f, 1, false, 0.05f, 2);
        rop.step(1.0f);
        rop.step(0.98f);
        rop.step(0.98f);
        if (rop.getReductionsCount() != 1 || rop.getCooldownCounter() != 1 || !nearlyEqual(rop.getCurrentRate(), 0.005f, 1e-6f))
        {
            return failStep("5G G15b Reduce_On_Plateau threshold/cooldown behavior");
        }
        {
            std::ofstream out_rop(rop_path, std::ios::binary | std::ios::trunc);
            rop.saveCheckpoint(out_rop);
        }
        Reduce_On_Plateau rop_loaded;
        {
            std::ifstream in_rop(rop_path, std::ios::binary);
            rop_loaded.loadCheckpoint(in_rop);
        }
        if (!nearlyEqual(rop_loaded.getThreshold(), 0.05f, 1e-6f) ||
            rop_loaded.getCooldown() != 2 ||
            rop_loaded.getCooldownCounter() != 1 ||
            rop_loaded.getReductionsCount() != 1 ||
            !nearlyEqual(rop_loaded.getCurrentRate(), 0.005f, 1e-6f))
        {
            return failStep("5G G15b Reduce_On_Plateau V2 round-trip");
        }
    }

    {
        Training_Context tc;
        bool threw_opt = false, threw_cost = false, threw_lr = false;
        try { tc.setOptimizer(nullptr); } catch (const std::invalid_argument &) { threw_opt = true; }
        try { tc.setCostFunction(nullptr); } catch (const std::invalid_argument &) { threw_cost = true; }
        try { tc.setLearningRateScheduler(nullptr); } catch (const std::invalid_argument &) { threw_lr = true; }
        if (!threw_opt || !threw_cost || !threw_lr)
        {
            return failStep("5G G16 Training_Context nullptr setter guards");
        }

        Neural_Network nn(Execution_Target::CPU);
        nn.addLayer<Linear_Layer>(2, 2, Execution_Target::CPU);
        nn.setCostFunction<Mse_Cost>(Execution_Target::CPU);
        nn.setOptimizer<Adam_Optimizer>(0.01f);
        Loss_Scaler custom_scaler(8.0f, 2.0f, 0.5f, 100, true, 64.0f, 2.0f);
        nn.setLossScaler(custom_scaler);
        nn.enableMixedPrecision(true);
        if (!nearlyEqual(nn.getLossScaler().getScaleFactor(), 8.0f, 1e-5f) ||
            !nearlyEqual(nn.getLossScaler().getMinScale(), 2.0f, 1e-5f))
        {
            return failStep("5G G16 Neural_Network custom Loss_Scaler overwritten");
        }
        nn.enableMixedPrecision(false);

        Neural_Network nn_moved(Execution_Target::CPU);
        nn_moved = std::move(nn);
        if (nn_moved.getLayerCount() != 1)
        {
            return failStep("5G G16 Neural_Network move-assign");
        }

        Tensor in_t(Shape{1, 2}, std::vector<float>{1.0f, -1.0f}, Execution_Target::CPU);
        Tensor tgt_t(Shape{1, 2}, std::vector<float>{0.5f, 0.5f}, Execution_Target::CPU);
        (void)nn_moved.forward(in_t);
        nn_moved.backwardAndStep(tgt_t);
        for (float w : nn_moved.getLayer(0).getWeights().getData())
        {
            if (!std::isfinite(w)) return failStep("5G G16 backwardAndStep produced non-finite weights");
        }

        Neural_Network tmpl(Execution_Target::CPU);
        tmpl.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
        tmpl.addLayer<Relu_Layer>(Execution_Target::CPU);
        Population pop(2, tmpl, 4, 2, Execution_Target::CPU, 42);
        Neural_Network ind = pop.getIndividual(0);
        if (ind.getLayerCount() != 2 || ind.isTrainingMode())
        {
            return failStep("5G G16 Population::getIndividual eval mode / layer count");
        }
    }

    {
        bool threw_vocab = false, threw_eps = false;
        try
        {
            Causal_LM_Config bad_v;
            bad_v.vocab_size = 0;
            bad_v.tokenizer_path = "";
            Causal_LM lm_bad(bad_v);
        }
        catch (const std::invalid_argument &) { threw_vocab = true; }
        try
        {
            Causal_LM_Config bad_e;
            bad_e.rms_norm_eps = 0.0f;
            bad_e.tokenizer_path = "";
            Causal_LM lm_bad(bad_e);
        }
        catch (const std::invalid_argument &) { threw_eps = true; }
        if (!threw_vocab || !threw_eps) return failStep("5H G17 Causal_LM::validateConfig");

        Causal_LM_Config lm_cfg;
        lm_cfg.vocab_size = 16;
        lm_cfg.hidden_dim = 16;
        lm_cfg.num_heads = 2;
        lm_cfg.intermediate_dim = 32;
        lm_cfg.num_layers = 1;
        lm_cfg.max_seq_len = 16;
        lm_cfg.execution_target = Execution_Target::CPU;
        lm_cfg.data_type = Data_Type::FLOAT32;
        lm_cfg.tokenizer_path = "";
        Causal_LM lm(lm_cfg);

        bool threw_ragged = false;
        try { (void)lm.forward(std::vector<std::vector<int32_t>>{{1, 2}, {3}}); }
        catch (const std::invalid_argument &) { threw_ragged = true; }
        if (!threw_ragged) return failStep("5H G17 Causal_LM ragged batch forward guard");

        const float empty_loss = lm.forwardLossAndBackward(std::vector<int32_t>{}, std::vector<int32_t>{});
        if (empty_loss != 0.0f) return failStep("5H G17 Causal_LM empty sequence forwardLossAndBackward");

        lm.setSamplingSeed(123);
        lm.clearSamplingSeed();
        const Causal_LM &const_lm = lm;
        const size_t param_cnt = const_lm.getParameterCount();
        const_lm.saveInference(causal_inf_path);
        std::error_code ec;
        if (param_cnt == 0 || !std::filesystem::exists(causal_inf_path, ec) ||
            std::filesystem::file_size(causal_inf_path, ec) == 0)
        {
            return failStep("5H G17 Causal_LM const getParameterCount / saveInference");
        }

        KV_Cache_Manager kv(1, 2, 8, 4, Execution_Target::CPU, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);
        Tensor k_empty(Shape{1, 2, 0, 4}, Execution_Target::CPU);
        Tensor v_empty(Shape{1, 2, 0, 4}, Execution_Target::CPU);
        kv.appendAt(k_empty, v_empty, 0);
        if (kv.getCurrentSeqLen() != 0) return failStep("5H G17 KV_Cache_Manager empty appendAt");

        Tensor k_gpu(Shape{1, 2, 1, 4}, Execution_Target::VULKAN_GPU);
        Tensor v_cpu(Shape{1, 2, 1, 4}, Execution_Target::CPU);
        bool threw_kv_tgt = false;
        try { kv.appendAt(k_gpu, v_cpu, 0); } catch (const std::exception &) { threw_kv_tgt = true; }

        Tensor k_bad_shape(Shape{1, 2, 1, 8}, Execution_Target::CPU);
        Tensor v_bad_shape(Shape{1, 2, 1, 8}, Execution_Target::CPU);
        bool threw_kv_shape = false;
        try { kv.appendAt(k_bad_shape, v_bad_shape, 0); } catch (const std::exception &) { threw_kv_shape = true; }

        Tensor k_tern(Shape{1, 2, 1, 4}, Data_Type::TERNARY_INT2, Execution_Target::CPU);
        Tensor v_ok(Shape{1, 2, 1, 4}, Execution_Target::CPU);
        bool threw_kv_tern = false;
        try { kv.appendAt(k_tern, v_ok, 0); } catch (const std::exception &) { threw_kv_tern = true; }
        if (!threw_kv_tgt || !threw_kv_shape || !threw_kv_tern)
        {
            return failStep("5H G17 KV_Cache_Manager::appendAt validation");
        }

        {
            std::ofstream out_tok(bpe_good_path, std::ios::binary | std::ios::trunc);
            out_tok << R"({"model":{"vocab":{"<unk>":0,"<s>":1,"</s>":2,"a":3,"b":4,"ab":5},"merges":[["a","b"]]}})";
        }
        Bpe_Tokenizer tok;
        if (!tok.load(bpe_good_path) || tok.getVocabSize() != 6)
        {
            return failStep("5H G17 Bpe_Tokenizer valid baseline load");
        }
        {
            std::ofstream out_neg(bpe_bad_path, std::ios::binary | std::ios::trunc);
            out_neg << R"({"model":{"vocab":{"<unk>":0,"<s>":1,"</s>":2,"bad":-5},"merges":[]}})";
        }
        if (tok.load(bpe_bad_path) || !tok.isLoaded() || tok.getVocabSize() != 6)
        {
            return failStep("5H G17 Bpe_Tokenizer negative ID rejection / rollback");
        }
        {
            std::ofstream out_huge(bpe_bad_path, std::ios::binary | std::ios::trunc);
            out_huge << R"({"model":{"vocab":{"<unk>":0,"<s>":1,"</s>":2,"huge":16777217},"merges":[]}})";
        }
        if (tok.load(bpe_bad_path) || !tok.isLoaded() || tok.getVocabSize() != 6)
        {
            return failStep("5H G17 Bpe_Tokenizer >16777216 ID rejection / rollback");
        }
    }

    {
        Cartpole_Env env(2, 123);
        env.reset();
        auto s1 = env.step(0);
        auto s2 = env.step(1);
        if (s1.is_terminal || s1.is_terminated || s1.is_truncated ||
            !s2.is_truncated || s2.is_terminated || !s2.is_terminal)
        {
            return failStep("5H G18 Cartpole_Env terminated vs truncated flags");
        }

        Replay_Buffer rb(8, 42);
        rb.push(Transition{{0.1f, 0.2f, 0.3f, 0.4f}, 1, 1.0f, {0.2f, 0.3f, 0.4f, 0.5f}, true, true});
        Transition_Batch tb = rb.sample(1);
        if (tb.terminals.size() != 1 || tb.truncated.size() != 1 || !tb.terminals[0] || !tb.truncated[0])
        {
            return failStep("5H G18 Replay_Buffer truncated field preservation");
        }

        Neural_Network proto(Execution_Target::CPU);
        proto.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
        proto.setCostFunction<Mse_Cost>(Execution_Target::CPU);
        proto.setOptimizer<Adam_Optimizer>(1e-3f);
        Dqn_Agent dqn(std::move(proto), 4, 2, 32, 0.99f, 0.0f, 0.0f, 1.0f, Execution_Target::CPU, 42);

        const std::vector<float> nan_state = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f};
        if (dqn.selectAction(nan_state, false) != 0)
        {
            return failStep("5H G18 Dqn_Agent::selectAction non-finite fallback");
        }
        dqn.storeTransition(Transition{{0.1f, 0.2f, 0.3f, 0.4f}, 0, 1.0f, {0.2f, 0.3f, 0.4f, 0.5f}, false, false});
        dqn.storeTransition(Transition{{0.2f, 0.3f, 0.4f, 0.5f}, 1, 0.5f, {0.3f, 0.4f, 0.5f, 0.6f}, true, true});
        dqn.trainStep(2);

        Ppo_Agent ppo(4, 2, 1e-3f, 0.99f, 0.95f, 0.2f, 0.5f, 0.01f, 1, 2, Execution_Target::CPU, 77);
        const std::vector<float> ppo_s = {0.1f, -0.2f, 0.3f, -0.1f};
        const float est_v = ppo.estimateValue(ppo_s);
        if (!std::isfinite(est_v)) return failStep("5H G18 Ppo_Agent::estimateValue");

        auto a1 = ppo.selectAction(ppo_s, false);
        auto a2 = ppo.selectAction(ppo_s, false);
        ppo.storeTransition(Ppo_Transition{ppo_s, a1.action, 1.0f, false, a1.log_prob, a1.value});
        ppo.storeTransition(Ppo_Transition{ppo_s, a2.action, 0.5f, false, a2.log_prob, a2.value});
        ppo.trainStep(0.5f);
        if (ppo.getTrajectorySize() != 0) return failStep("5H G18 Ppo_Agent::trainStep(last_value)");

        ppo.saveCheckpoint(ppo_ckpt_path, 7);
        Ppo_Agent ppo_loaded(4, 2, 1e-3f, 0.99f, 0.95f, 0.2f, 0.5f, 0.01f, 1, 2, Execution_Target::CPU, 99);
        ppo_loaded.loadCheckpoint(ppo_ckpt_path);
        if (ppo_loaded.getLastLoadedEpoch() != 7)
        {
            return failStep("5H G18 Ppo_Agent V2 loadCheckpoint last_loaded_epoch");
        }
        ppo.saveInference(ppo_inf_path);
        ppo_loaded.loadInference(ppo_inf_path);
    }

    {
        Input_Format fmt0{"x={{}}"};
        if (fmt0.toString() != "x={}") return failStep("5H G19 Input_Format 0-arg toString");

        {
            Scoped_Logger_Guard log_guard;

            Logger::setOnlyActiveFeatures(Log_Feature::NONE);
            Logger::setMinLogLevel(Log_Level::LOG_DEBUG);
            if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::NONE))
            {
                return failStep("5H G19 Logger LOG_DEBUG+NONE loophole");
            }
            Logger::setOnlyActiveFeatures(Log_Feature::ALL);
            Logger::setMinLogLevel(Log_Level::LOG_ERROR);
            if (Logger::shouldLog(Log_Level::LOG_WARNING, Log_Feature::DEVICE_MANAGEMENT))
            {
                return failStep("5H G19 Logger min_log_level filtering");
            }
            Logger::flush();
        }

        {
            struct ProfilerResetGuard
            {
                ~ProfilerResetGuard() { Step_Timings::getInstance().reset(); }
            } prof_guard;

            auto &st = Step_Timings::getInstance();
            st.reset();
            bool threw_oob = false;
            try { (void)st[static_cast<Timing_Stage>(999)]; }
            catch (const std::out_of_range &) { threw_oob = true; }
            if (!threw_oob) return failStep("5H G19 Step_Timings OOB did not throw");

            st.is_chained_bwd_opt = true;
            st.setNodeCount(Node_Stage::FWD_LOSS_GPU_NODES, 3);
            st.recordDuration(Timing_Stage::FWD_LOSS_EXEC, 2.0);
            st.recordDuration(Timing_Stage::OPT_EXEC, 1.5);
            st.recordDuration(Timing_Stage::OPT_WAITED_IN_FWD, 1.25);
            st.recordDuration(Timing_Stage::BWD_WAITED_IN_OPT, 0.75);
            std::ostringstream oss;
            st.printTable(1, 1, 100.0, 1, "tok/s", oss);
            const std::string tbl = oss.str();
            if (tbl.find("OPT_WAITED_IN_FWD") == std::string::npos ||
                tbl.find("BWD_WAITED_IN_OPT") == std::string::npos)
            {
                return failStep("5H G19 Step_Timings::printTable wait stages");
            }
        }

        auto &prefs = User_Preferences::getInstance();
        if (prefs.load(tmp_dir.string()))
        {
            return failStep("5H G19 User_Preferences::load directory");
        }
        {
            std::ofstream out_arr(prefs_bad_path, std::ios::binary | std::ios::trunc);
            out_arr << "[1, 2, 3]";
        }
        if (prefs.load(prefs_bad_path))
        {
            return failStep("5H G19 User_Preferences::load non-object JSON");
        }
        {
            std::ofstream out_zero(prefs_bad_path, std::ios::binary | std::ios::trunc);
            out_zero << R"({"execution":{"max_frames_in_flight":0}})";
        }
        if (prefs.load(prefs_bad_path))
        {
            return failStep("5H G19 User_Preferences::load max_frames_in_flight=0");
        }
    }

    {
        Neural_Network cnn(Execution_Target::CPU);
        cnn.addLayer<Conv2d_Layer>(8, 8, 1, 4, 3, 1, 1, Execution_Target::CPU);
        cnn.addLayer<Relu_Layer>(Execution_Target::CPU);
        cnn.addLayer<Global_Avg_Pool_2d_Layer>(8, 8, 4, Execution_Target::CPU);
        cnn.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);

        chexlocalize::GradcamConfig gcfg;
        gcfg.target_layer_index = 99;
        chexlocalize::Gradcam_Engine cam(cnn, gcfg);
        bool threw_cam_idx = false;
        try { (void)cam.getTargetLayerIndex(); }
        catch (const std::out_of_range &) { threw_cam_idx = true; }
        if (!threw_cam_idx) return failStep("5I G20a Gradcam_Engine OOB target_layer_index");

        bool threw_up = false;
        try { (void)chexlocalize::Gradcam_Engine::bilinearUpsample(std::vector<float>{1.0f, 2.0f}, 2, 2, 4, 4); }
        catch (const std::invalid_argument &) { threw_up = true; }
        if (!threw_up) return failStep("5I G20a Gradcam_Engine::bilinearUpsample undersize");

        std::vector<float> mm = {std::numeric_limits<float>::quiet_NaN(), 1.0f, 3.0f, std::numeric_limits<float>::infinity()};
        chexlocalize::Gradcam_Engine::minMaxNormalize(mm);
        if (mm[0] != 0.0f || mm[3] != 0.0f || !nearlyEqual(mm[2], 1.0f, 1e-5f))
        {
            return failStep("5I G20a Gradcam_Engine::minMaxNormalize NaN/Inf");
        }

        chexlocalize::Quaternion_Blender qb;
        std::vector<std::vector<float>> hms(10, std::vector<float>(160 * 160, 0.2f));
        std::vector<uint8_t> gray(160 * 160, 128);
        chexlocalize::DashboardAnnotation ann;
        ann.sample_key = "wave5_dash";
        ann.is_positive[0] = true;
        uint32_t dh = 0, dw = 0;
        const auto dash = qb.createAnnotatedDashboard(hms, gray, 160, 160, ann, dh, dw);
        if (dh != 348 || dash.size() != static_cast<size_t>(dh) * dw * 3)
        {
            return failStep("5I G20a Quaternion_Blender::createAnnotatedDashboard 160x160 height");
        }

        if (chexlocalize::CheXpert_Classifier::DEFAULT_FEATURE_LAYER_INDEX != 0)
        {
            return failStep("5I G20a CheXpert_Classifier::DEFAULT_FEATURE_LAYER_INDEX != 0");
        }
        auto clf = chexlocalize::CheXpert_Classifier::buildModel(Execution_Target::CPU, 2, 32, 32);
        if (!clf || clf->getLayerCount() != 6)
        {
            return failStep("5I G20a CheXpert_Classifier::buildModel");
        }
        chexlocalize::CheXpert_Classifier::saveWeights(*clf, clf_weights_path);
        auto clf2 = chexlocalize::CheXpert_Classifier::buildModel(Execution_Target::CPU, 2, 32, 32);
        if (!chexlocalize::CheXpert_Classifier::loadWeights(*clf2, clf_weights_path, Execution_Target::CPU) ||
            clf2->isTrainingMode())
        {
            return failStep("5I G20a CheXpert_Classifier::saveWeights/loadWeights");
        }
        if (chexlocalize::CheXpert_Classifier::loadWeights(*clf2, (clf_dir / "missing.bin").string(), Execution_Target::CPU) ||
            chexlocalize::CheXpert_Classifier::loadWeights(*clf2, clf_dir.string(), Execution_Target::CPU))
        {
            return failStep("5I G20a CheXpert_Classifier::loadWeights bad path");
        }

        const std::vector<uint8_t> zmask(4, 0);
        if (chexlocalize::Evaluator::computeIoU(zmask, zmask) != 0.0 ||
            chexlocalize::Evaluator::computeDice(zmask, zmask) != 0.0)
        {
            return failStep("5I G20a Evaluator empty-union default != 0.0");
        }
        bool threw_eval_empty = false, threw_eval_nan = false;
        try { (void)chexlocalize::Evaluator::thresholdHeatmap(std::vector<float>{}, 0.5f); }
        catch (const std::invalid_argument &) { threw_eval_empty = true; }
        try { (void)chexlocalize::Evaluator::thresholdHeatmap(std::vector<float>{0.5f}, std::numeric_limits<float>::quiet_NaN()); }
        catch (const std::invalid_argument &) { threw_eval_nan = true; }
        if (!threw_eval_empty || !threw_eval_nan) return failStep("5I G20a Evaluator thresholdHeatmap guards");

        if (chexlocalize::Evaluator::computeHitRate(std::vector<float>{0.1f, 0.9f, 0.2f, 0.0f},
                                                  std::vector<uint8_t>{0, 1, 0, 0}) != 1.0 ||
            chexlocalize::Evaluator::computeHitRate(std::vector<float>{0.1f, 0.9f, 0.2f, 0.0f},
                                                  std::vector<uint8_t>{1, 0, 0, 0}) != 0.0)
        {
            return failStep("5I G20a Evaluator::computeHitRate");
        }
        if (chexlocalize::Evaluator::computeAUROC(std::vector<float>{0.1f, 0.4f, 0.8f, 0.9f},
                                                std::vector<float>{0.0f, 0.0f, 1.0f, 1.0f}) != 1.0 ||
            chexlocalize::Evaluator::computeAUROC(std::vector<float>{0.2f, 0.5f},
                                                std::vector<float>{1.0f, 1.0f}) != 0.5)
        {
            return failStep("5I G20a Evaluator::computeAUROC");
        }
        chexlocalize::TaskMetric tm{"Test", 0.0, 0};
        tm.addCase(0.8);
        tm.addCase(0.4);
        const auto rep = chexlocalize::Evaluator::buildReport({tm});
        if (!nearlyEqual(static_cast<float>(rep.overall_mIoU), 0.6f, 1e-5f))
        {
            return failStep("5I G20a Evaluator::buildReport");
        }

        std::vector<uint8_t> canvas(16 * 16 * 3, 0);
        chexlocalize::Bitmap_Font::fillRect(canvas, 16, 16, -2, -2, 4, 4, chexlocalize::RgbColor{255, 0, 0});
        chexlocalize::Bitmap_Font::drawChar(canvas, 16, 16, -1, 0, 'A', chexlocalize::RgbColor{0, 255, 0}, 0);
        chexlocalize::Bitmap_Font::drawString(canvas, 16, 16, 0, 0, "Hi", chexlocalize::RgbColor{0, 0, 255}, 0);
        chexlocalize::Bitmap_Font::copyImage(canvas, 16, 16, -1, -1, std::vector<uint8_t>(2, 1), 2, 2);
        chexlocalize::Bitmap_Font::copyGrayscaleAsRgb(canvas, 16, 16, -1, -1, std::vector<uint8_t>{10, 20, 30, 40}, 2, 2);
        if (canvas[0] != 40 || canvas[1] != 40 || canvas[2] != 40)
        {
            return failStep("5I G20a Bitmap_Font signed coord / copyGrayscaleAsRgb");
        }
    }

    return true;
}

void runRegressionsW2W5Suite(Test_Result_Counter &counter)
{
    counter.record("  Wave 2 Regression Suite:           ", testWave2Regressions());
    counter.record("  Wave 3 Regression Suite:           ", testWave3Regressions());
    counter.record("  Wave 4 Regression Suite:           ", testWave4Regressions());
    counter.record("  Wave 5 (5A-5C) Regression Suite:   ", testWave5Phase5ACRegressions());
    counter.record("  Wave 5 (5D-5F) Regression Suite:   ", testWave5Phase5DFRegressions());
    counter.record("  Wave 5 (5G-5I) Regression Suite:   ", testWave5Phase5GIRegressions());
}
