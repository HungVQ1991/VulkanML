#include "test_common.h"

bool testWave6Regressions()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("Wave6", msg);
    };

    const auto tmp_dir = std::filesystem::temp_directory_path();
    const std::string bin_ds_path = (tmp_dir / "wave6_tokens.bin").string();
    const std::string prefs_w6_path = (tmp_dir / "wave6_user_prefs.json").string();

    Scoped_Temp_File_Guard temp_guard({bin_ds_path, prefs_w6_path});
    temp_guard.clean();

    {
        auto &ctx = Execution_Engine::getInstance().getContext();
        gpu::vector host_fp16(ctx);
        host_fp16.allocateHostVisible(8, Data_Type::FLOAT16);
        if (host_fp16.getDataType() != Data_Type::FLOAT16 || host_fp16.getSize() != 4)
        {
            return failStep("6 G01 gpu::vector::allocateHostVisible FLOAT16");
        }

        gpu::vector src_buf(ctx, 4, Data_Type::FLOAT32);
        gpu::vector dst_buf(ctx, 2, Data_Type::FLOAT32);
        src_buf.uploadData(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        dst_buf.uploadData(std::vector<float>{0.0f, 0.0f});
        ctx.executePendingTransfers();
        ctx.copyBufferImmediate(src_buf.getBuffer(), dst_buf.getBuffer(), sizeof(float) * 2, sizeof(float) * 2, 0);
        std::vector<float> copied;
        dst_buf.downloadData(copied);
        if (copied.size() != 2 || !nearlyEqual(copied[0], 3.0f) || !nearlyEqual(copied[1], 4.0f))
        {
            return failStep("6 G01 Vulkan_Context::copyBufferImmediate source offset");
        }

        const Compute_Graph const_graph;
        {
            Scoped_Logger_Guard log_silent;
            Logger::setOnlyActiveFeatures(Log_Feature::NONE);
            Logger::setConsoleOutput(false);
            Logger::enableFileLogging(false);
            const_graph.print();
        }

        Tensor gpu_t(Shape{2, 3}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
        gpu_t.fill(6.0f);
        if (!verifyMatrix(gpu_t, std::vector<float>(6, 6.0f)))
        {
            return failStep("6 G07 Gpu_Tensor_Impl::fill parity");
        }
        Tensor div_out(Shape{2, 3}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
        gpu_t.divScalar(2.0f, div_out);
        if (!verifyMatrix(div_out, std::vector<float>(6, 3.0f)))
        {
            return failStep("6 G07 Gpu_Tensor_Impl::divScalar parity");
        }
        bool threw_div0 = false;
        try { gpu_t.divScalar(0.0f, div_out); } catch (const std::runtime_error &) { threw_div0 = true; }
        if (!threw_div0) return failStep("6 G07 Gpu_Tensor_Impl::divScalar(0) did not throw");
    }

    {
        Multi_Scale_ResNet18_Balanced_Layer ms(32, 32, 2, Execution_Target::CPU, true);
        if (ms.getOutputDimension() != 560 || !ms.getUseResNetD() ||
            ms.getStage4Height() != 1 || ms.getStage4Width() != 1)
        {
            return failStep("6 G14a Multi_Scale_ResNet18_Balanced_Layer getters");
        }
        Tensor ms_in(1, 32 * 32 * 2, Execution_Target::CPU);
        ms_in.fill(0.1f);
        Tensor ms_out = ms.forward(ms_in);
        if (ms_out.getColumns() != 560) return failStep("6 G14a Multi_Scale_ResNet18 forward output dim");
        bool threw_ms_bwd = false;
        try { (void)ms.backward(Tensor(1, 320, Execution_Target::CPU)); }
        catch (const std::invalid_argument &) { threw_ms_bwd = true; }
        if (!threw_ms_bwd) return failStep("6 G14a Multi_Scale_ResNet18 backward column guard");

        NN_UNet_Layer nn_default;
        if (nn_default.getHeight() != 160 || nn_default.getWidth() != 160)
        {
            return failStep("6 G14b NN_UNet_Layer default 160x160 dims");
        }
    }

    {
        if (Neural_Network::findSubString("hello vulkan world", "vulkan") != 6 ||
            Neural_Network::findSubString("hello", "missing") != -1 ||
            Neural_Network::findSubString("hello", "") != 0)
        {
            return failStep("6 G16 Neural_Network::findSubString");
        }

        Neural_Network net(Execution_Target::VULKAN_GPU);
        net.addLayer<Linear_Layer>(4, 2, Execution_Target::VULKAN_GPU);
        Population pop(2, net, 4, 2, Execution_Target::VULKAN_GPU, 42);
        const std::vector<float> st = {0.1f, -0.2f, 0.3f, -0.4f};
        if (pop.selectAction(0, st) >= 2)
        {
            return failStep("6 G16 Population::selectAction(vec) CPU fast-path");
        }
        Tensor st_cpu(1, 4, st, Execution_Target::CPU);
        if (pop.selectAction(0, st_cpu) >= 2)
        {
            return failStep("6 G16 Population::selectAction(Tensor) CPU fast-path");
        }

        const std::vector<int32_t> tokens = {10, -2, 305, 4096, 123456};
        if (!Binary_Token_Dataset::save(bin_ds_path, tokens) ||
            !Binary_Token_Dataset::existsAndValid(bin_ds_path))
        {
            return failStep("6 G17 Binary_Token_Dataset save/existsAndValid");
        }
        std::vector<int32_t> loaded;
        if (!Binary_Token_Dataset::load(bin_ds_path, loaded) || loaded != tokens)
        {
            return failStep("6 G17 Binary_Token_Dataset load round-trip");
        }
    }

    {
        bool threw_zero_steps = false;
        try { Cartpole_Env bad_env(0, 1); } catch (const std::invalid_argument &) { threw_zero_steps = true; }
        if (!threw_zero_steps) return failStep("6 G18 Cartpole_Env max_steps==0");

        Cartpole_Env env(2, 123);
        bool threw_bad_act = false;
        try { (void)env.step(2); } catch (const std::invalid_argument &) { threw_bad_act = true; }
        if (!threw_bad_act) return failStep("6 G18 Cartpole_Env invalid action");

        (void)env.step(0);
        auto term_res = env.step(1);
        if (!term_res.is_terminal || !env.isDone())
        {
            return failStep("6 G18 Cartpole_Env terminal state");
        }
        bool threw_after_done = false;
        try { (void)env.step(0); } catch (const std::logic_error &) { threw_after_done = true; }
        if (!threw_after_done) return failStep("6 G18 Cartpole_Env step after terminal");

        auto makeProto = []() {
            Neural_Network p(Execution_Target::CPU);
            p.addLayer<Linear_Layer>(4, 2, Execution_Target::CPU);
            p.setCostFunction<Mse_Cost>(Execution_Target::CPU);
            p.setOptimizer<Adam_Optimizer>(1e-3f);
            return p;
        };
        bool threw_bad_gamma = false;
        try
        {
            Dqn_Agent bad_dqn(makeProto(), 4, 2, 16, 1.5f, 0.5f, 0.1f, 0.99f, Execution_Target::CPU, 1);
        }
        catch (const std::invalid_argument &)
        {
            threw_bad_gamma = true;
        }
        bool threw_nan_eps = false;
        try
        {
            Dqn_Agent nan_dqn(makeProto(), 4, 2, 16, 0.99f, std::numeric_limits<float>::quiet_NaN(), 0.1f, 0.99f, Execution_Target::CPU, 1);
        }
        catch (const std::invalid_argument &)
        {
            threw_nan_eps = true;
        }
        if (!threw_bad_gamma || !threw_nan_eps)
        {
            return failStep("6 G18 Dqn_Agent hyperparameter bounds");
        }
        Dqn_Agent eps_below_min(makeProto(), 4, 2, 16, 0.99f, 0.0f, 0.01f, 0.95f, Execution_Target::CPU, 1);
        if (!nearlyEqual(eps_below_min.getEpsilon(), 0.0f))
        {
            return failStep("6 G18 Dqn_Agent should allow epsilon < epsilon_min");
        }

        Ppo_Agent ppo(4, 2, 1e-3f, 0.99f, 0.95f, 0.2f, 0.5f, 0.01f, 1, 2, Execution_Target::CPU, 55);
        const std::vector<float> s = {0.1f, -0.1f, 0.2f, -0.2f};
        for (int i = 0; i < 5; ++i)
        {
            auto act = ppo.selectAction(s, false);
            ppo.storeTransition(Ppo_Transition{s, act.action, 1.0f, false, act.log_prob, act.value});
        }
        ppo.trainStep(0.0f);
        if (ppo.getTrajectorySize() != 0)
        {
            return failStep("6 G18 Ppo_Agent remainder mini-batch trainStep");
        }
    }

    {
        Log_Feature mask = Log_Feature::DEVICE_MANAGEMENT;
        mask |= Log_Feature::SYNCHRONIZATION;
        if ((mask & Log_Feature::DEVICE_MANAGEMENT) == Log_Feature::NONE ||
            (mask & Log_Feature::SYNCHRONIZATION) == Log_Feature::NONE)
        {
            return failStep("6 G19 Log_Feature::operator|=");
        }
        mask &= Log_Feature::DEVICE_MANAGEMENT;
        if (mask != Log_Feature::DEVICE_MANAGEMENT)
        {
            return failStep("6 G19 Log_Feature::operator&=");
        }

        {
            Scoped_Logger_Guard log_guard(true);

            Logger::resetLogCounters();
            Logger::setOnlyActiveFeatures(Log_Feature::DEVICE_MANAGEMENT);
            Logger::setMinLogLevel(Log_Level::LOG_INFO);
            Logger::setConsoleOutput(false);
            Logger::enableFileLogging(false);

            const std::source_location loc = std::source_location::current();
            if (!Logger::logMessage("w6_reset_probe", Log_Level::LOG_INFO, false, 1, Log_Feature::DEVICE_MANAGEMENT, loc))
                return failStep("6 G19 Logger first logMessage");
            if (Logger::logMessage("w6_reset_probe", Log_Level::LOG_INFO, false, 1, Log_Feature::DEVICE_MANAGEMENT, loc))
                return failStep("6 G19 Logger repeat throttle");
            Logger::resetSpecificLog(loc.file_name(), loc.line());
            if (!Logger::logMessage("w6_reset_probe", Log_Level::LOG_INFO, false, 1, Log_Feature::DEVICE_MANAGEMENT, loc))
                return failStep("6 G19 Logger resetSpecificLog(file, line)");
            if (Logger::logMessage("w6_reset_probe", Log_Level::LOG_INFO, false, 1, Log_Feature::DEVICE_MANAGEMENT, loc))
                return failStep("6 G19 Logger second throttle");
            Logger::resetLogCounters();
            if (!Logger::logMessage("w6_reset_probe", Log_Level::LOG_INFO, false, 1, Log_Feature::DEVICE_MANAGEMENT, loc))
                return failStep("6 G19 Logger resetLogCounters()");
        }

        {
            auto &prefs = User_Preferences::getInstance();
            struct PrefsRestoreGuard
            {
                User_Preferences &p;
                std::string orig_path;
                bool had_orig_file;
                Device_Preferences orig_dev;
                Execution_Preferences orig_exec;
                Memory_Preferences orig_mem;
                Logging_Preferences orig_log;
                ~PrefsRestoreGuard()
                {
                    if (had_orig_file && !orig_path.empty())
                    {
                        p.load(orig_path);
                    }
                    p.setDevicePreferences(orig_dev);
                    p.setExecutionPreferences(orig_exec);
                    p.setMemoryPreferences(orig_mem);
                    p.setLoggingPreferences(orig_log);
                }
            } prefs_guard{
                prefs,
                prefs.getConfigFilePath(),
                std::filesystem::exists(prefs.getConfigFilePath()),
                prefs.getDevicePreferences(),
                prefs.getExecutionPreferences(),
                prefs.getMemoryPreferences(),
                prefs.getLoggingPreferences()};

            if (!prefs.load(prefs_w6_path) || !prefs.isLoaded() || prefs.getConfigFilePath() != prefs_w6_path)
            {
                return failStep("6 G19 User_Preferences load / isLoaded / getConfigFilePath");
            }

            Execution_Preferences bad_exec = prefs.getExecutionPreferences();
            bad_exec.max_frames_in_flight = 0;
            bool threw_exec = false;
            try { prefs.setExecutionPreferences(bad_exec); } catch (const std::invalid_argument &) { threw_exec = true; }
            Memory_Preferences bad_mem = prefs.getMemoryPreferences();
            bad_mem.staging_pool_initial_size_mb = 0;
            bool threw_mem = false;
            try { prefs.setMemoryPreferences(bad_mem); } catch (const std::invalid_argument &) { threw_mem = true; }
            if (!threw_exec || !threw_mem)
            {
                return failStep("6 G19 User_Preferences setter validation");
            }

            Device_Preferences dev = prefs.getDevicePreferences();
            dev.preferred_device_index = 7;
            prefs.setDevicePreferences(dev);
            if (!prefs.save() || !prefs.load(prefs_w6_path) ||
                prefs.getDevicePreferences().preferred_device_index != 7)
            {
                return failStep("6 G19 User_Preferences 0-arg save() round-trip");
            }
        }
    }

    {
        const auto empty_cm = chexlocalize::Rle_Decoder::decodeToColumnMajor("", 2, 3);
        const auto empty_rm = chexlocalize::Rle_Decoder::decodeToRowMajor("", 2, 3);
        if (empty_cm != std::vector<uint8_t>(6, 0) || empty_rm != std::vector<uint8_t>(6, 0))
        {
            return failStep("6 G20a Rle_Decoder empty string all-zeros");
        }

        bool threw_cm_sum = false;
        try { (void)chexlocalize::Rle_Decoder::decodeToColumnMajor("12", 2, 3); }
        catch (const std::invalid_argument &) { threw_cm_sum = true; }
        bool threw_rm_sum = false;
        try { (void)chexlocalize::Rle_Decoder::decodeToRowMajor("12", 2, 3); }
        catch (const std::invalid_argument &) { threw_rm_sum = true; }
        if (!threw_cm_sum || !threw_rm_sum)
        {
            return failStep("6 G20a Rle_Decoder pixel sum mismatch guard");
        }

        const auto cm = chexlocalize::Rle_Decoder::decodeToColumnMajor("231", 2, 3);
        const auto rm = chexlocalize::Rle_Decoder::decodeToRowMajor("231", 2, 3);
        const std::vector<uint8_t> expected_cm = {0, 0, 1, 1, 1, 0};
        const std::vector<uint8_t> expected_rm = {0, 1, 1, 0, 1, 0};
        if (cm != expected_cm || rm != expected_rm)
        {
            return failStep("6 G20a Rle_Decoder valid decode");
        }

        std::vector<uint8_t> canvas_utf8(24 * 16 * 3, 0);
        std::vector<uint8_t> canvas_space(24 * 16 * 3, 0);
        const chexlocalize::RgbColor white{255, 255, 255};
        chexlocalize::Bitmap_Font::drawString(canvas_utf8, 24, 16, 0, 0, "A\xC3\xA9!", white, 1);
        chexlocalize::Bitmap_Font::drawString(canvas_space, 24, 16, 0, 0, "A !", white, 1);
        if (canvas_utf8 != canvas_space)
        {
            return failStep("6 G20a Bitmap_Font UTF-8 multi-byte single advance");
        }
    }

    return true;
}

bool testWave7DeferredRegressions()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("Wave7", msg);
    };

    const auto tmp_dir = std::filesystem::temp_directory_path();
    const std::string tb_gqa_path = (tmp_dir / "wave7_tb_gqa.inf").string();
    const std::string tb_mha_path = (tmp_dir / "wave7_tb_mha.inf").string();
    const std::string lm_ckpt_path = (tmp_dir / "wave7_causal_gqa.ckpt").string();
    const std::string lm_inf_path = (tmp_dir / "wave7_causal_gqa.inf").string();
    const std::string c_api_ckpt_path = (tmp_dir / "wave7_capi_best.ckpt").string();

    Scoped_Temp_File_Guard temp_guard({tb_gqa_path, tb_mha_path, lm_ckpt_path, lm_inf_path, c_api_ckpt_path});
    temp_guard.clean();

    Execution_Engine &engine = Execution_Engine::getInstance();

    {
        std::vector<float> rope_in_data(16);
        for (size_t i = 0; i < rope_in_data.size(); ++i)
        {
            rope_in_data[i] = 0.25f * static_cast<float>(i + 1) - 1.5f;
        }

        ops::llm::RoPE_Config rope_cfg;
        rope_cfg.seq_len = 2;
        rope_cfg.head_dim = 8;
        rope_cfg.direction = ops::llm::RoPE_Direction::FORWARD;
        rope_cfg.base = 10000.0f;
        rope_cfg.num_heads = 1;
        rope_cfg.mode = ops::llm::RoPE_Mode::STANDARD;
        rope_cfg.position_offset = 0;
        rope_cfg.rotary_dim = 4;
        rope_cfg.freq_scale = 0.5f;
        rope_cfg.engine = &engine;

        Tensor rope_in_cpu(Shape{1, 1, 2, 8}, rope_in_data, Execution_Target::CPU);
        Tensor rope_out_cpu(Execution_Target::CPU);
        ops::llm::applyRoPE(rope_in_cpu, rope_out_cpu, rope_cfg);

        Tensor rope_in_gpu(Shape{1, 1, 2, 8}, rope_in_data, Execution_Target::VULKAN_GPU);
        Tensor rope_out_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::applyRoPE(rope_in_gpu, rope_out_gpu, rope_cfg);
        engine.executeGraph();

        const auto rc = rope_out_cpu.getData();
        const auto rg = rope_out_gpu.getData();
        if (rc.size() != 16 || rg.size() != 16)
        {
            return failStep("7 G10 RoPE rotary_dim output size");
        }
        for (size_t i = 0; i < 16; ++i)
        {
            if (!nearlyEqual(rc[i], rg[i], 2e-3f))
            {
                std::cerr << " [i=" << i << " rc=" << rc[i] << " rg=" << rg[i] << "]";
                return failStep("7 G10 RoPE rotary_dim CPU/GPU mismatch");
            }
        }
        for (size_t d = 4; d < 8; ++d)
        {
            if (!nearlyEqual(rc[8 + d], rope_in_data[8 + d], 1e-5f))
            {
                return failStep("7 G10 RoPE trailing dims outside rotary_dim were modified");
            }
        }

        ops::llm::RoPE_Config rope_bwd_cfg = rope_cfg;
        rope_bwd_cfg.direction = ops::llm::RoPE_Direction::BACKWARD;
        Tensor rope_rec_cpu(Execution_Target::CPU);
        Tensor rope_rec_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::applyRoPE(rope_out_cpu, rope_rec_cpu, rope_bwd_cfg);
        ops::llm::applyRoPE(rope_out_gpu, rope_rec_gpu, rope_bwd_cfg);
        engine.executeGraph();
        const auto rec_c = rope_rec_cpu.getData();
        const auto rec_g = rope_rec_gpu.getData();
        for (size_t i = 0; i < 16; ++i)
        {
            if (!nearlyEqual(rec_c[i], rope_in_data[i], 1e-4f) || !nearlyEqual(rec_g[i], rope_in_data[i], 3e-3f))
            {
                std::cerr << " [i=" << i << " exp=" << rope_in_data[i] << " rec_c=" << rec_c[i] << " rec_g=" << rec_g[i] << "]";
                return failStep("7 G10 RoPE rotary_dim inverse recovery mismatch");
            }
        }

        bool threw_odd_rotary = false;
        try
        {
            ops::llm::RoPE_Config bad = rope_cfg;
            bad.rotary_dim = 3;
            ops::llm::applyRoPE(rope_in_cpu, rope_out_cpu, bad);
        }
        catch (const std::invalid_argument &)
        {
            threw_odd_rotary = true;
        }
        bool threw_bad_scale = false;
        try
        {
            ops::llm::RoPE_Config bad = rope_cfg;
            bad.freq_scale = 0.0f;
            ops::llm::applyRoPE(rope_in_cpu, rope_out_cpu, bad);
        }
        catch (const std::invalid_argument &)
        {
            threw_bad_scale = true;
        }
        if (!threw_odd_rotary || !threw_bad_scale)
        {
            return failStep("7 G10 RoPE invalid rotary_dim/freq_scale check");
        }

        const std::vector<float> emb_weights_data = {
            1.1f, 1.2f, 1.3f,
            9.1f, 9.2f, 9.3f,
            2.1f, 2.2f, 2.3f,
            3.1f, 3.2f, 3.3f
        };
        const std::vector<float> emb_indices_data = {1.0f, 0.0f, 1.0f, 2.0f};
        ops::llm::Embedding_Config emb_cfg;
        emb_cfg.padding_idx = 1;
        emb_cfg.engine = &engine;

        Tensor w_cpu(Shape{4, 3}, emb_weights_data, Execution_Target::CPU);
        Tensor idx_cpu(Shape{4}, emb_indices_data, Execution_Target::CPU);
        Tensor emb_out_cpu(Execution_Target::CPU);
        ops::llm::embeddingForward(w_cpu, idx_cpu, emb_out_cpu, emb_cfg);

        Tensor w_gpu(Shape{4, 3}, emb_weights_data, Execution_Target::VULKAN_GPU);
        Tensor idx_gpu(Shape{4}, emb_indices_data, Execution_Target::VULKAN_GPU);
        Tensor emb_out_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::embeddingForward(w_gpu, idx_gpu, emb_out_gpu, emb_cfg);
        engine.executeGraph();

        const auto efc = emb_out_cpu.getData();
        const auto efg = emb_out_gpu.getData();
        if (efc.size() != 12 || efg.size() != 12)
        {
            return failStep("7 G10 Embedding forward output size");
        }
        for (size_t d = 0; d < 3; ++d)
        {
            if (!nearlyEqual(efc[d], 0.0f, 1e-5f) || !nearlyEqual(efc[6 + d], 0.0f, 1e-5f) ||
                !nearlyEqual(efg[d], 0.0f, 1e-5f) || !nearlyEqual(efg[6 + d], 0.0f, 1e-5f) ||
                !nearlyEqual(efc[3 + d], emb_weights_data[d], 1e-5f) ||
                !nearlyEqual(efg[3 + d], emb_weights_data[d], 2e-3f) ||
                !nearlyEqual(efc[9 + d], emb_weights_data[6 + d], 1e-5f) ||
                !nearlyEqual(efg[9 + d], emb_weights_data[6 + d], 2e-3f))
            {
                return failStep("7 G10 Embedding padding_idx forward mismatch");
            }
        }

        const std::vector<float> emb_dy_data(12, 0.75f);
        Tensor emb_dy_cpu(Shape{4, 3}, emb_dy_data, Execution_Target::CPU);
        Tensor emb_dw_cpu(Execution_Target::CPU);
        ops::llm::embeddingBackward(w_cpu, idx_cpu, emb_dy_cpu, emb_dw_cpu, emb_cfg);

        Tensor emb_dy_gpu(Shape{4, 3}, emb_dy_data, Execution_Target::VULKAN_GPU);
        Tensor emb_dw_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::embeddingBackward(w_gpu, idx_gpu, emb_dy_gpu, emb_dw_gpu, emb_cfg);
        engine.executeGraph();

        const auto edwc = emb_dw_cpu.getData();
        const auto edwg = emb_dw_gpu.getData();
        for (size_t d = 0; d < 3; ++d)
        {
            if (!nearlyEqual(edwc[3 + d], 0.0f, 1e-5f) || !nearlyEqual(edwg[3 + d], 0.0f, 1e-5f) ||
                !nearlyEqual(edwc[d], 0.75f, 1e-5f) || !nearlyEqual(edwg[d], 0.75f, 2e-3f) ||
                !nearlyEqual(edwc[6 + d], 0.75f, 1e-5f) || !nearlyEqual(edwg[6 + d], 0.75f, 2e-3f))
            {
                return failStep("7 G10 Embedding padding_idx backward mismatch");
            }
        }

        const std::vector<float> sw_a = {0.5f, -0.3f, 1.2f, -0.8f, 0.2f, 0.9f, -1.1f, 0.4f};
        const std::vector<float> sw_b = {1.0f, 0.5f, -0.4f, 0.6f, -0.7f, 1.3f, 0.2f, -0.9f};
        const std::vector<float> sw_dy = {0.3f, -0.5f, 0.8f, 0.1f, -0.4f, 0.7f, 0.2f, -0.6f};
        ops::llm::SwiGLU_Config sw_cfg{&engine};
        for (Execution_Target tgt : {Execution_Target::CPU, Execution_Target::VULKAN_GPU})
        {
            Tensor ta(Shape{2, 4}, sw_a, tgt), tb(Shape{2, 4}, sw_b, tgt), tdy(Shape{2, 4}, sw_dy, tgt);
            Tensor ga1(tgt), gb1(tgt), ga2(tgt), gb2(tgt);
            ops::llm::swigluBackward(ta, tb, tdy, ga1, gb1, sw_cfg);
            ops::llm::swigluBackward(ta, tb, tdy, ga2, gb2);
            if (tgt == Execution_Target::VULKAN_GPU)
            {
                engine.executeGraph();
            }
            const auto vga1 = ga1.getData(), vga2 = ga2.getData();
            const auto vgb1 = gb1.getData(), vgb2 = gb2.getData();
            for (size_t i = 0; i < sw_a.size(); ++i)
            {
                if (!nearlyEqual(vga1[i], vga2[i], 1e-5f) || !nearlyEqual(vgb1[i], vgb2[i], 1e-5f))
                {
                    return failStep("7 G10 SwiGLU canonical backward parity");
                }
            }
        }

        const std::vector<float> loss_p = {1.0f, 3.0f, 2.0f, 4.0f};
        const std::vector<float> loss_t = {2.0f, 1.0f, 0.0f, 2.0f};
        for (Execution_Target tgt : {Execution_Target::CPU, Execution_Target::VULKAN_GPU})
        {
            Tensor tp(Shape{2, 2}, loss_p, tgt), tt(Shape{2, 2}, loss_t, tgt);
            Tensor gm(tgt), gc(tgt);
            ops::loss::mseGradient(tp, tt, gm, &engine);
            ops::loss::cceGradient(tp, tt, gc, &engine);
            if (tgt == Execution_Target::VULKAN_GPU)
            {
                engine.executeGraph();
            }
            const auto vgm = gm.getData(), vgc = gc.getData();
            for (size_t i = 0; i < loss_p.size(); ++i)
            {
                const float exp_mse = 2.0f * (loss_p[i] - loss_t[i]) / 4.0f;
                const float exp_cce = (loss_p[i] - loss_t[i]) / 2.0f;
                if (!nearlyEqual(vgm[i], exp_mse, 1e-5f) || !nearlyEqual(vgc[i], exp_cce, 1e-5f))
                {
                    return failStep("7 G10 mseGradient/cceGradient mismatch");
                }
            }
        }

        const std::vector<float> ce_logits = {1.2f, -0.4f, 0.3f, 2.1f, -1.0f, 0.8f, 0.4f, -0.2f};
        const std::vector<int32_t> ce_targets = {1, -100};
        ops::loss::Cross_Entropy_Config ce_cfg;
        ce_cfg.ignore_index = -100;
        ce_cfg.label_smoothing = 0.1f;
        ce_cfg.compute_grad = true;
        ce_cfg.readback_loss = true;
        ce_cfg.engine = &engine;

        Tensor ce_in_cpu(Shape{2, 4}, ce_logits, Execution_Target::CPU);
        Tensor ce_in_gpu(Shape{2, 4}, ce_logits, Execution_Target::VULKAN_GPU);
        Tensor ce_d_cpu(Execution_Target::CPU), ce_d_gpu(Execution_Target::VULKAN_GPU), ce_d_hard(Execution_Target::CPU);
        float ce_loss_cpu = ops::loss::fusedCrossEntropy(ce_in_cpu, ce_targets, ce_d_cpu, ce_cfg);
        float ce_loss_gpu = ops::loss::fusedCrossEntropy(ce_in_gpu, ce_targets, ce_d_gpu, ce_cfg);
        ops::loss::Cross_Entropy_Config ce_hard_cfg = ce_cfg;
        ce_hard_cfg.label_smoothing = 0.0f;
        float ce_loss_hard = ops::loss::fusedCrossEntropy(ce_in_cpu, ce_targets, ce_d_hard, ce_hard_cfg);

        if (!nearlyEqual(ce_loss_cpu, ce_loss_gpu, 1e-4f) || std::abs(ce_loss_cpu - ce_loss_hard) < 1e-3f)
        {
            return failStep("7 G10 fusedCrossEntropy label_smoothing loss mismatch");
        }
        const auto dce_c = ce_d_cpu.getData();
        const auto dce_g = ce_d_gpu.getData();
        for (size_t i = 0; i < 8; ++i)
        {
            if (!nearlyEqual(dce_c[i], dce_g[i], 1e-4f))
            {
                return failStep("7 G10 fusedCrossEntropy label_smoothing grad CPU/GPU mismatch");
            }
            if (i >= 4 && !nearlyEqual(dce_c[i], 0.0f, 1e-5f))
            {
                return failStep("7 G10 fusedCrossEntropy ignore_index grad should be zero");
            }
        }

        bool threw_bad_smooth = false;
        try
        {
            ops::loss::Cross_Entropy_Config bad_ce = ce_cfg;
            bad_ce.label_smoothing = 1.0f;
            ops::loss::fusedCrossEntropy(ce_in_cpu, ce_targets, ce_d_cpu, bad_ce);
        }
        catch (const std::invalid_argument &)
        {
            threw_bad_smooth = true;
        }
        if (!threw_bad_smooth)
        {
            return failStep("7 G10 fusedCrossEntropy invalid label_smoothing check");
        }

        if (static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::LOSS) != 0 ||
            static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::INTERSECTION) != 1 ||
            static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::PRED_SUM) != 2 ||
            static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::TARGET_SUM) != 3 ||
            static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::COUNT) != 4)
        {
            return failStep("7 G10 Dice_Bce_Stats_Index enum values");
        }
        const std::vector<float> db_p = {0.8f, 0.2f, 0.6f, 0.1f};
        const std::vector<float> db_t = {1.0f, 0.0f, 1.0f, 0.0f};
        ops::loss::Dice_Bce_Config db_cfg{1.0f, 1.0f, 1e-7f, &engine};
        Tensor db_out_cpu(Execution_Target::CPU), db_out_gpu(Execution_Target::VULKAN_GPU);
        ops::loss::diceBce(Tensor(Shape{1, 4}, db_p, Execution_Target::CPU), Tensor(Shape{1, 4}, db_t, Execution_Target::CPU), db_out_cpu, db_cfg);
        ops::loss::diceBce(Tensor(Shape{1, 4}, db_p, Execution_Target::VULKAN_GPU), Tensor(Shape{1, 4}, db_t, Execution_Target::VULKAN_GPU), db_out_gpu, db_cfg);
        engine.executeGraph();
        if (!nearlyEqual(db_out_cpu.getData()[static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::LOSS)],
                         db_out_gpu.getData()[static_cast<size_t>(ops::loss::Dice_Bce_Stats_Index::LOSS)], 1e-4f))
        {
            return failStep("7 G10 Dice_Bce_Config CPU/GPU parity");
        }
    }

    {
        const uint32_t B = 1, Q_H = 4, KV_H = 2, S = 6, D = 16;
        std::vector<float> q_data(B * Q_H * S * D);
        std::vector<float> do_data(B * Q_H * S * D);
        std::vector<float> k_gqa_data(B * KV_H * S * D);
        std::vector<float> v_gqa_data(B * KV_H * S * D);
        for (size_t i = 0; i < q_data.size(); ++i)
        {
            q_data[i] = 0.05f * static_cast<float>((static_cast<int>(i % 19)) - 9);
            do_data[i] = 0.04f * static_cast<float>((static_cast<int>((i * 3) % 17)) - 8);
        }
        for (size_t i = 0; i < k_gqa_data.size(); ++i)
        {
            k_gqa_data[i] = 0.05f * static_cast<float>((static_cast<int>((i * 5) % 23)) - 11);
            v_gqa_data[i] = 0.05f * static_cast<float>((static_cast<int>((i * 7) % 15)) - 7);
        }

        const size_t head_elems = static_cast<size_t>(S) * D;
        std::vector<float> k_mha_data(B * Q_H * S * D);
        std::vector<float> v_mha_data(B * Q_H * S * D);
        for (uint32_t qh = 0; qh < Q_H; ++qh)
        {
            const uint32_t kvh = qh / (Q_H / KV_H);
            std::copy_n(k_gqa_data.data() + kvh * head_elems, head_elems, k_mha_data.data() + qh * head_elems);
            std::copy_n(v_gqa_data.data() + kvh * head_elems, head_elems, v_mha_data.data() + qh * head_elems);
        }

        for (Execution_Target tgt : {Execution_Target::CPU, Execution_Target::VULKAN_GPU})
        {
            Tensor tq(Shape{B, Q_H, S, D}, q_data, tgt);
            Tensor tk_gqa(Shape{B, KV_H, S, D}, k_gqa_data, tgt);
            Tensor tv_gqa(Shape{B, KV_H, S, D}, v_gqa_data, tgt);
            Tensor tdo(Shape{B, Q_H, S, D}, do_data, tgt);
            Tensor o_gqa(tgt), l_gqa(tgt), dq_gqa(tgt), dk_gqa(tgt), dv_gqa(tgt);

            Tensor tk_mha(Shape{B, Q_H, S, D}, k_mha_data, tgt);
            Tensor tv_mha(Shape{B, Q_H, S, D}, v_mha_data, tgt);
            Tensor o_mha(tgt), l_mha(tgt), dq_mha(tgt), dk_mha(tgt), dv_mha(tgt);

            ops::llm::Attention_Config gqa_cfg;
            gqa_cfg.num_heads = Q_H;
            gqa_cfg.num_kv_heads = KV_H;
            gqa_cfg.seq_len = S;
            gqa_cfg.head_dim = D;
            gqa_cfg.is_causal = true;
            gqa_cfg.engine = &engine;

            ops::llm::Attention_Config mha_cfg = gqa_cfg;
            mha_cfg.num_kv_heads = Q_H;

            ops::llm::flashAttentionForward(tq, tk_gqa, tv_gqa, o_gqa, gqa_cfg, &l_gqa);
            ops::llm::flashAttentionForward(tq, tk_mha, tv_mha, o_mha, mha_cfg, &l_mha);
            ops::llm::flashAttentionBackward(tq, tk_gqa, tv_gqa, o_gqa, tdo, dq_gqa, dk_gqa, dv_gqa, gqa_cfg, &l_gqa);
            ops::llm::flashAttentionBackward(tq, tk_mha, tv_mha, o_mha, tdo, dq_mha, dk_mha, dv_mha, mha_cfg, &l_mha);
            if (tgt == Execution_Target::VULKAN_GPU)
            {
                engine.executeGraph();
            }

            const auto og = o_gqa.getData(), om = o_mha.getData();
            const auto dqg = dq_gqa.getData(), dqm = dq_mha.getData();
            const auto dkg = dk_gqa.getData(), dkm = dk_mha.getData();
            const auto dvg = dv_gqa.getData(), dvm = dv_mha.getData();

            const float gqa_fwd_tol = (tgt == Execution_Target::VULKAN_GPU) ? 3e-3f : 1e-4f;
            const float gqa_bwd_tol = (tgt == Execution_Target::VULKAN_GPU) ? 5e-3f : 1e-3f;
            for (size_t i = 0; i < og.size(); ++i)
            {
                if (!nearlyEqual(og[i], om[i], gqa_fwd_tol) || !nearlyEqual(dqg[i], dqm[i], gqa_bwd_tol))
                {
                    std::cerr << " [i=" << i << " og=" << og[i] << " om=" << om[i]
                              << " dqg=" << dqg[i] << " dqm=" << dqm[i] << "]";
                    return failStep("7 G13 GQA forward/dq parity vs expanded MHA");
                }
            }
            for (uint32_t kvh = 0; kvh < KV_H; ++kvh)
            {
                const size_t kv_off = kvh * head_elems;
                const size_t q0_off = (2 * kvh) * head_elems;
                const size_t q1_off = (2 * kvh + 1) * head_elems;
                for (size_t j = 0; j < head_elems; ++j)
                {
                    const float ref_dk = dkm[q0_off + j] + dkm[q1_off + j];
                    const float ref_dv = dvm[q0_off + j] + dvm[q1_off + j];
                    if (!nearlyEqual(dkg[kv_off + j], ref_dk, gqa_bwd_tol) ||
                        !nearlyEqual(dvg[kv_off + j], ref_dv, gqa_bwd_tol))
                    {
                        std::cerr << " [kvh=" << kvh << " j=" << j
                                  << " dkg=" << dkg[kv_off + j] << " ref_dk=" << ref_dk
                                  << " dvg=" << dvg[kv_off + j] << " ref_dv=" << ref_dv << "]";
                        return failStep("7 G13 GQA dk/dv sum parity vs expanded MHA");
                    }
                }
            }

            std::vector<float> sq_all_heads(B * Q_H * D);
            for (uint32_t qh = 0; qh < Q_H; ++qh)
            {
                std::copy_n(q_data.data() + qh * head_elems, D, sq_all_heads.data() + qh * D);
            }
            Tensor t_sq(Shape{B, Q_H, 1, D}, sq_all_heads, tgt);
            Tensor dec_gqa(tgt), dec_mha(tgt);
            ops::llm::singleTokenAttentionForward(t_sq, tk_gqa, tv_gqa, dec_gqa, Q_H, D, S, KV_H, &engine);
            ops::llm::singleTokenAttentionForward(t_sq, tk_mha, tv_mha, dec_mha, Q_H, D, S, Q_H, &engine);
            if (tgt == Execution_Target::VULKAN_GPU)
            {
                engine.executeGraph();
            }
            const auto dcg = dec_gqa.getData(), dcm = dec_mha.getData();
            for (size_t i = 0; i < dcg.size(); ++i)
            {
                if (!nearlyEqual(dcg[i], dcm[i], 1e-4f))
                {
                    return failStep("7 G13 singleTokenAttentionForward GQA parity");
                }
            }
        }

        for (Execution_Target tgt : {Execution_Target::CPU, Execution_Target::VULKAN_GPU})
        {
            Transformer_Block tb(32, 4, 64, 1e-5f, 10000.0f, tgt, Data_Type::FLOAT32, 1, 2);
            if (tb.getNumHeads() != 4 || tb.getNumKvHeads() != 2 || tb.getHeadDim() != 8 ||
                tb.getKProj().getWeights().getTotalElements() != 32 * 16 ||
                tb.getVProj().getWeights().getTotalElements() != 32 * 16)
            {
                return failStep("7 G13 Transformer_Block GQA dimensions");
            }

            std::vector<float> x_vals(3 * 32, 0.1f);
            Tensor x_in(Shape{1, 3, 32}, x_vals, tgt);
            Tensor out_tb = tb.forward(x_in);
            Tensor grad_in(Shape{1, 3, 32}, std::vector<float>(3 * 32, 0.05f), tgt);
            Tensor dx_tb = tb.backward(grad_in);
            if (tgt == Execution_Target::VULKAN_GPU)
            {
                engine.executeGraph();
            }
            if (out_tb.getShape() != Shape{1, 3, 32} || dx_tb.getShape() != Shape{1, 3, 32})
            {
                return failStep("7 G13 Transformer_Block GQA forward/backward shape");
            }

            KV_Cache_Manager bad_kv(1, 4, 16, 8, tgt, Data_Type::FLOAT32);
            bool threw_bad_kv = false;
            try
            {
                tb.forward(x_in, &bad_kv);
            }
            catch (const std::invalid_argument &)
            {
                threw_bad_kv = true;
            }
            if (!threw_bad_kv)
            {
                return failStep("7 G13 Transformer_Block should reject mismatched KV_Cache_Manager num_heads");
            }

            auto cloned_tb = tb.clone();
            auto *cloned_ptr = dynamic_cast<Transformer_Block *>(cloned_tb.get());
            if (!cloned_ptr || cloned_ptr->getNumKvHeads() != 2)
            {
                return failStep("7 G13 Transformer_Block::clone should preserve num_kv_heads");
            }
        }

        {
            Neural_Network gqa_net(Execution_Target::CPU);
            gqa_net.addLayer<Transformer_Block>(32, 4, 64, 1e-5f, 10000.0f, Execution_Target::CPU, Data_Type::FLOAT32, 1, 2);
            Tensor probe(Shape{1, 2, 32}, std::vector<float>(64, 0.2f), Execution_Target::CPU);
            Tensor pred_before = gqa_net.forward(probe);
            gqa_net.saveInference(tb_gqa_path);

            Neural_Network gqa_loaded(Execution_Target::CPU);
            gqa_loaded.loadInference(tb_gqa_path, Execution_Target::CPU);
            Tensor pred_after = gqa_loaded.forward(probe);
            if (!verifyMatrix(pred_after, pred_before.getData()))
            {
                return failStep("7 G13 Training_Context GQA Transformer_Block round-trip");
            }

            Neural_Network mha_net(Execution_Target::CPU);
            mha_net.addLayer<Transformer_Block>(32, 4, 64, 1e-5f, 10000.0f, Execution_Target::CPU, Data_Type::FLOAT32, 1, 0);
            Tensor mha_before = mha_net.forward(probe);
            mha_net.saveInference(tb_mha_path);

            Neural_Network mha_loaded(Execution_Target::CPU);
            mha_loaded.loadInference(tb_mha_path, Execution_Target::CPU);
            Tensor mha_after = mha_loaded.forward(probe);
            if (!verifyMatrix(mha_after, mha_before.getData()))
            {
                return failStep("7 G13 Training_Context MHA backward-compatible round-trip");
            }
        }

        {
            Causal_LM_Config bad_lm_cfg;
            bad_lm_cfg.vocab_size = 32;
            bad_lm_cfg.hidden_dim = 32;
            bad_lm_cfg.num_heads = 4;
            bad_lm_cfg.num_kv_heads = 3;
            bad_lm_cfg.intermediate_dim = 64;
            bad_lm_cfg.num_layers = 1;
            bad_lm_cfg.max_seq_len = 16;
            bad_lm_cfg.tokenizer_path = "";
            bool threw_bad_lm_gqa = false;
            try
            {
                Causal_LM bad_lm(bad_lm_cfg);
            }
            catch (const std::invalid_argument &)
            {
                threw_bad_lm_gqa = true;
            }
            if (!threw_bad_lm_gqa)
            {
                return failStep("7 G13 Causal_LM should reject non-divisible num_kv_heads");
            }

            for (Execution_Target tgt : {Execution_Target::CPU, Execution_Target::VULKAN_GPU})
            {
                Causal_LM_Config lm_cfg;
                lm_cfg.vocab_size = 32;
                lm_cfg.hidden_dim = 32;
                lm_cfg.num_heads = 4;
                lm_cfg.num_kv_heads = 2;
                lm_cfg.intermediate_dim = 64;
                lm_cfg.num_layers = 1;
                lm_cfg.max_seq_len = 16;
                lm_cfg.tokenizer_path = "";
                lm_cfg.execution_target = tgt;

                Causal_LM lm(lm_cfg);
                if (lm.getConfig().num_kv_heads != 2 || lm.getBlocks()[0]->getNumKvHeads() != 2)
                {
                    return failStep("7 G13 Causal_LM num_kv_heads initialization");
                }

                lm.resetKVCaches();
                Tensor full_logits = lm.forward(std::vector<int32_t>{1, 2, 3}, false);
                lm.resetKVCaches();
                (void)lm.forward(std::vector<int32_t>{1, 2}, true);
                Tensor dec_logits = lm.forward(std::vector<int32_t>{3}, true);
                const auto full_v = full_logits.getData();
                const auto dec_v = dec_logits.getData();
                for (size_t v = 0; v < 32; ++v)
                {
                    if (!nearlyEqual(full_v[2 * 32 + v], dec_v[v], 2e-3f))
                    {
                        std::cerr << " [v=" << v << " full=" << full_v[2 * 32 + v] << " dec=" << dec_v[v] << "]";
                        return failStep("7 G13 Causal_LM GQA KV-cache prefill+decode parity");
                    }
                }

                lm.setTrainingMode(true);
                (void)lm.forwardLossAndBackward(std::vector<int32_t>{1, 2, 3}, std::vector<int32_t>{2, 3, 4}, 1.0f, false);
                lm.resetGradients();
                lm.setTrainingMode(false);

                if (tgt == Execution_Target::CPU)
                {
                    lm.saveCheckpoint(lm_ckpt_path);
                    lm.saveInference(lm_inf_path);

                    Causal_LM lm_reloaded(lm_cfg);
                    lm_reloaded.loadCheckpoint(lm_ckpt_path);
                    Tensor ckpt_logits = lm_reloaded.forward(std::vector<int32_t>{1, 2, 3}, false);

                    Causal_LM lm_inf_loaded(lm_cfg);
                    lm_inf_loaded.loadInference(lm_inf_path);
                    Tensor inf_logits = lm_inf_loaded.forward(std::vector<int32_t>{1, 2, 3}, false);

                    if (!verifyMatrix(ckpt_logits, full_v) || !verifyMatrix(inf_logits, full_v))
                    {
                        return failStep("7 G13 Causal_LM GQA checkpoint/inference reload parity");
                    }

                    Causal_LM_Config mha_cfg = lm_cfg;
                    mha_cfg.num_kv_heads = 4;
                    Causal_LM mha_lm(mha_cfg);
                    bool threw_ckpt_mismatch = false;
                    try
                    {
                        mha_lm.loadCheckpoint(lm_ckpt_path);
                    }
                    catch (const std::runtime_error &)
                    {
                        threw_ckpt_mismatch = true;
                    }
                    bool threw_inf_mismatch = false;
                    try
                    {
                        mha_lm.loadInference(lm_inf_path);
                    }
                    catch (const std::runtime_error &)
                    {
                        threw_inf_mismatch = true;
                    }
                    if (!threw_ckpt_mismatch || !threw_inf_mismatch)
                    {
                        return failStep("7 G13 Causal_LM should reject loading GQA checkpoint into MHA model");
                    }
                }
            }
        }
    }

    {
        vulkanmlClearLastError();
        if (std::string(vulkanmlGetLastError()) != "")
        {
            return failStep("7 G18 vulkanmlClearLastError");
        }

        const std::array<size_t, 2> hidden_sizes = {8, 6};
        void *pop = createPopulation(4, 3, 2, hidden_sizes.data(), hidden_sizes.size(), VULKANML_ACT_GELU, VULKANML_TARGET_CPU);
        if (!pop || std::string(vulkanmlGetLastError()) != "")
        {
            return failStep("7 G18 createPopulation CPU");
        }

        const std::vector<float> states(4 * 3, 0.25f);
        std::array<size_t, 4> actions{};
        if (selectPopulationActions(pop, states.data(), nullptr, 4, actions.data()) != VULKANML_OK)
        {
            destroyPopulation(pop);
            return failStep("7 G18 selectPopulationActions dense");
        }
        const std::array<size_t, 2> subset_idx = {1, 3};
        std::array<size_t, 2> subset_actions{};
        if (selectPopulationActions(pop, states.data(), subset_idx.data(), 2, subset_actions.data()) != VULKANML_OK)
        {
            destroyPopulation(pop);
            return failStep("7 G18 selectPopulationActions subset");
        }
        for (size_t a : actions)
        {
            if (a >= 2)
            {
                destroyPopulation(pop);
                return failStep("7 G18 action index out of bounds");
            }
        }

        const std::array<float, 4> fitness = {1.0f, 3.0f, 0.5f, 2.0f};
        if (evolvePopulation(pop, fitness.data(), 0.25f, 0.1f, 0.05f) != VULKANML_OK)
        {
            destroyPopulation(pop);
            return failStep("7 G18 evolvePopulation");
        }
        if (saveBestIndividual(pop, c_api_ckpt_path.c_str(), fitness.data()) != VULKANML_OK ||
            !std::filesystem::exists(c_api_ckpt_path) ||
            std::filesystem::file_size(c_api_ckpt_path) == 0)
        {
            destroyPopulation(pop);
            return failStep("7 G18 saveBestIndividual");
        }

        void *pop_gpu = createPopulation(2, 3, 2, hidden_sizes.data(), 1, VULKANML_ACT_LEAKY_RELU, VULKANML_TARGET_VULKAN_GPU);
        if (!pop_gpu)
        {
            destroyPopulation(pop);
            return failStep("7 G18 createPopulation GPU");
        }
        std::array<size_t, 2> gpu_actions{};
        if (selectPopulationActions(pop_gpu, states.data(), nullptr, 2, gpu_actions.data()) != VULKANML_OK)
        {
            destroyPopulation(pop_gpu);
            destroyPopulation(pop);
            return failStep("7 G18 selectPopulationActions GPU");
        }
        destroyPopulation(pop_gpu);

        if (createPopulation(0, 3, 2, hidden_sizes.data(), hidden_sizes.size(), VULKANML_ACT_RELU, VULKANML_TARGET_CPU) != nullptr ||
            std::string(vulkanmlGetLastError()).empty())
        {
            destroyPopulation(pop);
            return failStep("7 G18 createPopulation pop_size=0 error check");
        }
        if (createPopulation(4, 3, 2, nullptr, 1, VULKANML_ACT_RELU, VULKANML_TARGET_CPU) != nullptr ||
            createPopulation(4, 3, 2, hidden_sizes.data(), 2, 99u, VULKANML_TARGET_CPU) != nullptr)
        {
            destroyPopulation(pop);
            return failStep("7 G18 createPopulation invalid args check");
        }
        const size_t bad_idx = 99;
        size_t dummy_act = 0;
        if (selectPopulationActions(nullptr, states.data(), nullptr, 1, &dummy_act) != VULKANML_ERR_INVALID_ARGUMENT ||
            selectPopulationActions(pop, states.data(), &bad_idx, 1, &dummy_act) != VULKANML_ERR_INVALID_ARGUMENT ||
            evolvePopulation(pop, fitness.data(), -0.1f, 0.1f, 0.1f) != VULKANML_ERR_INVALID_ARGUMENT ||
            saveBestIndividual(pop, "", fitness.data()) != VULKANML_ERR_INVALID_ARGUMENT)
        {
            destroyPopulation(pop);
            return failStep("7 G18 invalid argument status check");
        }
        const std::string bad_io_path = (tmp_dir / "wave7_nonexistent_dir_xyz" / "best.ckpt").string();
        if (saveBestIndividual(pop, bad_io_path.c_str(), fitness.data()) != VULKANML_ERR_IO ||
            std::string(vulkanmlGetLastError()).empty())
        {
            destroyPopulation(pop);
            return failStep("7 G18 saveBestIndividual IO error status check");
        }
        vulkanmlClearLastError();
        if (std::string(vulkanmlGetLastError()) != "")
        {
            destroyPopulation(pop);
            return failStep("7 G18 vulkanmlClearLastError after error");
        }
        destroyPopulation(pop);
        destroyPopulation(nullptr);
    }

    return true;
}

bool testGroup3EngineOptimizations(Execution_Target exec_target)
{
    auto failStep = [](const char* step) -> bool {
        return failTestStep("Group3", step);
    };

    {
        Memory_Planner planner;
        planner.registerTensor(10, 2048, 256, 4);
        planner.updateLastUsage(10, 6);
        planner.registerTensor(11, 1024, 256, 0);
        planner.updateLastUsage(11, 2);
        planner.registerTensor(12, 512, 256, 1);
        planner.updateLastUsage(12, 3);
        planner.planMemoryLayout();

        if (planner.getOffset(10) != 0) return failStep("Group3 Memory_Planner id10 offset");
        if (planner.getOffset(11) != 0) return failStep("Group3 Memory_Planner disjoint reuse id11");
        if (planner.getOffset(12) < 2048) return failStep("Group3 Memory_Planner overlap separation id12");
        if (planner.getTotalMemoryRequired() != 2560) return failStep("Group3 Memory_Planner total bytes");
    }

    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Vulkan_Sub_Allocator& allocator = Execution_Engine::getInstance().getContext().getAllocator();
        const size_t arena_idx = allocator.allocateDedicatedArena(4096, 0xFFFFFFFFu, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Memory_Allocation a0 = allocator.getArenaAllocation(arena_idx, 0, 1024);
        Memory_Allocation a1 = allocator.getArenaAllocation(arena_idx, 1024, 1024);
        if (a0.memory == VK_NULL_HANDLE || a0.memory != a1.memory || a1.offset != 1024 || a0.chunk_index != arena_idx)
        {
            allocator.releaseDedicatedArena(arena_idx);
            return failStep("Group3 Vulkan_Sub_Allocator arena sub-alloc");
        }
        bool threw_oob = false;
        try
        {
            (void)allocator.getArenaAllocation(arena_idx, 3072, 2048);
        }
        catch (const std::out_of_range&)
        {
            threw_oob = true;
        }
        allocator.releaseDedicatedArena(arena_idx);
        allocator.releaseDedicatedArena(arena_idx);
        if (!threw_oob) return failStep("Group3 Vulkan_Sub_Allocator arena OOB check");
    }

    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine& engine = Execution_Engine::getInstance();
        engine.executeGraph();

        const Data_Type dt = (engine.getContext().isFloat16Enabled() || engine.isCooperativeMatrixEnabled())
            ? Data_Type::FLOAT16
            : Data_Type::FLOAT32;

        Tensor a0(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor w0(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor w1(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor b0(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor v0(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor v1(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        a0.uploadData({1.0f, 2.0f, 3.0f, 4.0f});
        w0.uploadData({1.0f, 0.0f, 0.0f, 2.0f});
        w1.uploadData({2.0f, 1.0f, 0.0f, 1.0f});
        b0.uploadData({2.0f, 1.0f, 0.0f, 3.0f});
        v0.uploadData({1.0f, 1.0f, 2.0f, 0.0f});
        v1.uploadData({1.0f, -1.0f, 0.0f, 2.0f});

        Tensor a_mid(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor out_a1(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor b_mid(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);
        Tensor out_b1(Shape{2, 2}, dt, Execution_Target::VULKAN_GPU);

        a0.matmul(w0, a_mid);
        a_mid.matmul(w1, out_a1);
        b0.matmul(v0, b_mid);
        b_mid.matmul(v1, out_b1);

        Compute_Graph scheduled_copy = engine.getCurrentGraph();
        Graph_Optimizer::optimize(scheduled_copy, &engine.getShaderDictionary());
        const auto& scheduled = scheduled_copy.getNodes();
        size_t barrier_count = 0;
        for (const auto& n : scheduled)
        {
            if (n.is_barrier_required_after) ++barrier_count;
        }
        engine.executeGraph();

        if (scheduled.size() != 4 || barrier_count != 2 ||
            scheduled[0].is_barrier_required_after || !scheduled[1].is_barrier_required_after ||
            scheduled[2].is_barrier_required_after || !scheduled[3].is_barrier_required_after)
        {
            return failStep("Group3 DAG schedule level barriers");
        }
        if (!verifyMatrix(out_a1, {2.0f, 5.0f, 6.0f, 11.0f}, 1e-2f)) return failStep("Group3 DAG schedule out_a1");
        if (!verifyMatrix(out_b1, {4.0f, 0.0f, 6.0f, -6.0f}, 1e-2f)) return failStep("Group3 DAG schedule out_b1");

        Tensor q(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor k(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor v(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor o(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor d_o(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor dq(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor dk(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor dv(Shape{1, 1, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor l(Shape{1, 16}, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);

        auto buf_of = [](const Tensor& t) {
            return Gpu_Tensor_Impl::castToGpu(*t.getImplementation()).getStorageBuffer();
        };
        std::vector<std::shared_ptr<gpu::vector>> fa_bufs = {
            buf_of(q), buf_of(k), buf_of(v), buf_of(o), buf_of(d_o),
            buf_of(dq), buf_of(dk), buf_of(dv), buf_of(l)
        };
        Compute_Graph fa_graph;
        Compute_Node dq_node;
        dq_node.pipeline_id = Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16;
        dq_node.buffers = fa_bufs;
        dq_node.workgroup_count_x = 1;
        fa_graph.addNode(dq_node);

        Compute_Node dkv_node;
        dkv_node.pipeline_id = Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16;
        dkv_node.buffers = fa_bufs;
        dkv_node.workgroup_count_x = 1;
        fa_graph.addNode(dkv_node);

        Graph_Optimizer::optimize(fa_graph, &engine.getShaderDictionary());
        if (fa_graph.getNodeCount() != 2 || fa_graph.getNodes()[0].is_barrier_required_after)
        {
            return failStep("Group3 FlashAttn DQ/DKV false WAW barrier");
        }
    }

    if (exec_target == Execution_Target::VULKAN_GPU)
    {
        Tensor sw_a(Shape{2, 8}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        Tensor sw_b(Shape{2, 8}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
        sw_a.uploadData(std::vector<float>(16, 1.0f));
        sw_b.uploadData(std::vector<float>(16, 2.0f));
        Tensor sw_out(Execution_Target::VULKAN_GPU);

        ops::llm::swigluForward(sw_a, sw_b, sw_out);
        VkBuffer sw_buf1 = Gpu_Tensor_Impl::castToGpu(*sw_out.getImplementation()).getStorageBuffer()->getBuffer();
        ops::llm::swigluForward(sw_a, sw_b, sw_out);
        VkBuffer sw_buf2 = Gpu_Tensor_Impl::castToGpu(*sw_out.getImplementation()).getStorageBuffer()->getBuffer();
        Execution_Engine::getInstance().executeGraph();
        if (sw_buf1 == VK_NULL_HANDLE || sw_buf1 != sw_buf2)
        {
            return failStep("Group3 ensureUniqueTensorTarget SwiGLU VkBuffer reallocated");
        }

        Execution_Engine& engine = Execution_Engine::getInstance();
        if (engine.isCooperativeMatrixEnabled() && engine.getContext().isFloat16Enabled())
        {
            Tensor q(Shape{1, 2, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
            Tensor k(Shape{1, 2, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
            Tensor v(Shape{1, 2, 16, 16}, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
            q.uploadData(std::vector<float>(512, 0.1f));
            k.uploadData(std::vector<float>(512, 0.1f));
            v.uploadData(std::vector<float>(512, 0.2f));
            Tensor attn_out(Execution_Target::VULKAN_GPU);
            Tensor l_stats(Execution_Target::VULKAN_GPU);

            q.flashAttentionForward(k, v, attn_out, 2, 16, 16, true, 0.0f, &l_stats);
            VkBuffer attn_buf1 = Gpu_Tensor_Impl::castToGpu(*attn_out.getImplementation()).getStorageBuffer()->getBuffer();
            VkBuffer l_buf1 = Gpu_Tensor_Impl::castToGpu(*l_stats.getImplementation()).getStorageBuffer()->getBuffer();
            q.flashAttentionForward(k, v, attn_out, 2, 16, 16, true, 0.0f, &l_stats);
            VkBuffer attn_buf2 = Gpu_Tensor_Impl::castToGpu(*attn_out.getImplementation()).getStorageBuffer()->getBuffer();
            VkBuffer l_buf2 = Gpu_Tensor_Impl::castToGpu(*l_stats.getImplementation()).getStorageBuffer()->getBuffer();
            engine.executeGraph();
            if (attn_buf1 == VK_NULL_HANDLE || attn_buf1 != attn_buf2 || l_buf1 == VK_NULL_HANDLE || l_buf1 != l_buf2)
            {
                return failStep("Group3 ensureUniqueTensorTarget FlashAttn VkBuffer reallocated");
            }
        }
    }

    {
        Causal_LM_Config cfg;
        cfg.vocab_size = 64;
        cfg.hidden_dim = 32;
        cfg.num_heads = 2;
        cfg.intermediate_dim = 64;
        cfg.num_layers = 2;
        cfg.max_seq_len = 32;
        cfg.execution_target = exec_target;
        cfg.data_type = (exec_target == Execution_Target::VULKAN_GPU) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        cfg.use_loss_scaler = (exec_target == Execution_Target::VULKAN_GPU);
        cfg.initial_loss_scale = 128.0f;
        cfg.tokenizer_path = "tokenizer/tokenizer.json";

        std::vector<int32_t> input_tokens = { 5, 12, 23, 40, 8, 14, 25, 42 };
        std::vector<int32_t> target_tokens = { 12, 23, 40, 55, 14, 25, 42, 60 };

        if (exec_target == Execution_Target::CPU)
        {
            Causal_LM cpu_model(cfg);
            Adam_Optimizer opt(0.02f);
            float first_loss = cpu_model.trainStep(input_tokens, target_tokens, opt);
            float last_loss = first_loss;
            for (int step = 1; step < 6; ++step)
            {
                last_loss = cpu_model.trainStep(input_tokens, target_tokens, opt);
                if (std::isnan(last_loss) || std::isinf(last_loss) || last_loss <= 0.0f)
                {
                    return failStep("Group3 CPU multi-layer scratchpad non-finite loss");
                }
            }
            if (!(last_loss < first_loss)) return failStep("Group3 CPU multi-layer scratchpad convergence");
        }
        else
        {
            Execution_Engine& engine = Execution_Engine::getInstance();
            Scoped_Engine_Flags_Guard state_guard;

            Causal_LM gpu_model(cfg);
            const std::filesystem::path parity_ckpt = std::filesystem::temp_directory_path() / "group3_single_submit_parity.nnck";
            gpu_model.saveCheckpoint(parity_ckpt.string());
            Causal_LM gpu_dual_model(cfg);
            gpu_dual_model.loadCheckpoint(parity_ckpt.string());
            std::error_code rm_ec;
            std::filesystem::remove(parity_ckpt, rm_ec);

            engine.readPendingLoss(UINT32_MAX);
            engine.setLatestLoss(0.0f);
            engine.setSingleSubmitEnabled(false);
            Adam_Optimizer dual_opt(0.02f);
            float dual_loss0 = gpu_dual_model.trainStep(input_tokens, target_tokens, dual_opt);
            gpu_dual_model.setTrainingMode(false);
            Tensor dual_logits = gpu_dual_model.forward(input_tokens, false);
            engine.executeGraph();
            std::vector<float> dual_logits_data = dual_logits.getData();

            engine.readPendingLoss(UINT32_MAX);
            engine.setLatestLoss(0.0f);
            engine.setSingleSubmitEnabled(true);
            engine.enableStaticGraph(true);
            Adam_Optimizer single_opt(0.02f);
            float single_loss0 = gpu_model.trainStep(input_tokens, target_tokens, single_opt);
            gpu_model.setTrainingMode(false);
            Tensor single_logits = gpu_model.forward(input_tokens, false);
            engine.executeGraph();
            std::vector<float> single_logits_data = single_logits.getData();

            if (std::isnan(single_loss0) || std::isinf(single_loss0) || single_loss0 <= 0.0f ||
                std::fabs(single_loss0 - dual_loss0) > 1e-2f)
            {
                std::cerr << "\n[Group3 parity debug] dual_loss0=" << dual_loss0 << " single_loss0=" << single_loss0 << "\n";
                return failStep("Group3 GPU Single-Submit vs Dual-Submit step0 loss parity");
            }
            if (single_logits_data.size() != dual_logits_data.size())
            {
                return failStep("Group3 GPU Single-Submit logits size mismatch");
            }
            for (size_t i = 0; i < single_logits_data.size(); ++i)
            {
                if (std::fabs(single_logits_data[i] - dual_logits_data[i]) > 2e-2f)
                {
                    return failStep("Group3 GPU Single-Submit post-step logits parity");
                }
            }

            float prev_or_first = single_loss0;
            float last_loss = single_loss0;
            for (int step = 1; step < 6; ++step)
            {
                last_loss = gpu_model.trainStep(input_tokens, target_tokens, single_opt);
                if (std::isnan(last_loss) || std::isinf(last_loss) || last_loss <= 0.0f)
                {
                    return failStep("Group3 GPU multi-layer Single-Submit non-finite loss");
                }
            }
            if (!(last_loss < prev_or_first)) return failStep("Group3 GPU multi-layer Single-Submit convergence");
        }
    }

    return true;
}

bool testGroup4GpuTimestampProfiling()
{
    auto failStep = [](const char* step) -> bool {
        return failTestStep("Group4", step);
    };

    Execution_Engine& engine = Execution_Engine::getInstance();
    Scoped_Engine_Flags_Guard guard;
    struct ProfResetGuard
    {
        Execution_Engine& eng;
        ~ProfResetGuard() noexcept { eng.resetGpuProfileStats(); }
    } prof_reset{ engine };

    engine.flushGpuProfileResults();
    engine.resetGpuProfileStats();

    if (engine.isGpuProfilingEnabled())
    {
        return failStep("Group4 GPU profiling must be disabled by default");
    }

    Tensor a(Shape{ 128, 128 }, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    Tensor b(Shape{ 128, 128 }, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    Tensor c(Shape{ 128, 128 }, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    Tensor mid(Shape{ 128, 128 }, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    Tensor out(Shape{ 128, 128 }, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
    a.uploadData(std::vector<float>(128 * 128, 0.25f));
    b.uploadData(std::vector<float>(128 * 128, 0.50f));
    c.uploadData(std::vector<float>(128 * 128, 0.125f));

    a.matmul(b, mid);
    mid.matmul(c, out);
    engine.executeGraph();
    engine.flushGpuProfileResults();

    Gpu_Profile_Summary off_summary = engine.getGpuProfileSummary();
    if (off_summary.profiled_submits != 0 || off_summary.total_dispatches != 0 || off_summary.total_kernel_ms != 0.0)
    {
        return failStep("Group4 zero-overhead when disabled");
    }

    if (!engine.isTimestampQuerySupported())
    {
        return true;
    }

    engine.setGpuProfilingEnabled(true);
    if (!engine.isGpuProfilingEnabled())
    {
        return failStep("Group4 failed to enable GPU profiling");
    }
    engine.resetGpuProfileStats();

    engine.enableStaticGraph(false);
    a.matmul(b, mid);
    mid.matmul(c, out);
    engine.executeGraph();
    engine.flushGpuProfileResults();

    Gpu_Profile_Summary dyn_summary = engine.getGpuProfileSummary();
    if (dyn_summary.profiled_submits < 1 ||
        dyn_summary.total_dispatches < 2 ||
        dyn_summary.total_barriers < 1 ||
        dyn_summary.total_kernel_ms <= 0.0 ||
        dyn_summary.total_span_ms <= 0.0 ||
        dyn_summary.kernels.empty())
    {
        return failStep("Group4 dynamic graph timestamp capture");
    }
    for (size_t i = 1; i < dyn_summary.kernels.size(); ++i)
    {
        if (dyn_summary.kernels[i - 1].total_ms < dyn_summary.kernels[i].total_ms)
        {
            return failStep("Group4 kernel summary descending sort order");
        }
    }

    std::ostringstream table_oss;
    engine.printGpuProfileTable(table_oss, 10, 1);
    const std::string table_str = table_oss.str();
    if (table_str.find("Vulkan GPU Hardware Timestamp Profile") == std::string::npos ||
        table_str.find("Sum of Kernels") == std::string::npos)
    {
        return failStep("Group4 printGpuProfileTable output format");
    }
    if (engine.formatGpuProfileTable(10, 1) != table_str)
    {
        return failStep("Group4 formatGpuProfileTable mismatch");
    }

    {
        Scoped_Logger_Guard log_guard;
        Logger::setMinLogLevel(Log_Level::LOG_INFO);
        Logger::enableFeature(Log_Feature::PROFILER, true);
        Logger::setQuietMode(false);
        Logger::setConsoleOutput(false);

        Runtime::logGpuProfileTable(10, 1, Log_Level::LOG_INFO, false);
        auto &st = Step_Timings::getInstance();
        st.reset();
        st[Timing_Stage::TOTAL_BATCH] = 2.0;
        st.logTable(1, 1, 100.0, 1, "samples/s", Log_Level::LOG_INFO, false);
        st.reset();

        std::ostringstream dump_oss;
        Logger::dumpRecentLogs(dump_oss);
        const std::string dump_str = dump_oss.str();
        if (dump_str.find("Vulkan GPU Hardware Timestamp Profile") == std::string::npos ||
            dump_str.find("TOTAL_BATCH") == std::string::npos ||
            dump_str.find("Profiler") == std::string::npos)
        {
            return failStep("Group4 logGpuProfileTable / Step_Timings::logTable Logger routing");
        }
    }

    engine.resetGpuProfileStats();
    if (engine.getGpuProfileSummary().profiled_submits != 0)
    {
        return failStep("Group4 resetGpuProfileStats");
    }

    engine.enableStaticGraph(true);
    for (int iter = 0; iter < 4; ++iter)
    {
        a.matmul(b, mid);
        mid.matmul(c, out);
        engine.executeGraph();
    }
    engine.flushGpuProfileResults();

    Gpu_Profile_Summary static_summary = engine.getGpuProfileSummary();
    if (static_summary.profiled_submits < 4 ||
        static_summary.total_dispatches < 8 ||
        static_summary.total_barriers < 4 ||
        static_summary.total_kernel_ms <= 0.0 ||
        static_summary.total_span_ms <= 0.0)
    {
        return failStep("Group4 static graph bake + replay timestamp capture");
    }

    engine.setGpuProfilingEnabled(false);
    if (engine.isGpuProfilingEnabled())
    {
        return failStep("Group4 disable restore");
    }
    engine.resetGpuProfileStats();
    a.matmul(b, mid);
    mid.matmul(c, out);
    engine.executeGraph();
    engine.flushGpuProfileResults();

    Gpu_Profile_Summary post_off_summary = engine.getGpuProfileSummary();
    if (post_off_summary.profiled_submits != 0 || post_off_summary.total_dispatches != 0)
    {
        return failStep("Group4 no collection after disabling");
    }

    return true;
}

bool testConflictResolutions()
{
    auto failStep = [](const char *msg) {
        return failTestStep("ConflictResolutions", msg);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.waitIdle();

    if (engine.getContext().isFloat16Enabled() && engine.isCooperativeMatrixEnabled())
    {
        for (size_t dim : {size_t{16}, size_t{7}})
        {
            Tensor dy_fp16 = castTensorToFp16(Tensor(Shape{dim, dim}, std::vector<float>(dim * dim, 500.0f), Execution_Target::VULKAN_GPU));
            Tensor w_fp16 = castTensorToFp16(Tensor(Shape{dim, dim}, std::vector<float>(dim * dim, 500.0f), Execution_Target::VULKAN_GPU));
            Tensor dx_fp16(Execution_Target::VULKAN_GPU);
            ops::linear::linearBackwardInput(dy_fp16, w_fp16, dx_fp16);
            flushGpuEngine(true);
            const auto dx_vals = dx_fp16.getData();
            bool found_inf = false;
            for (float v : dx_vals)
            {
                if (std::isinf(v))
                {
                    found_inf = true;
                    break;
                }
            }
            if (!found_inf)
            {
                return failStep("Conflict1 CoopMat FP16 aligned/unaligned overflow must preserve Inf");
            }
        }
    }

    {
        Tensor p_adam(Shape{1, 4}, std::vector<float>{2.0f, -3.0f, 4.0f, -5.0f}, Execution_Target::VULKAN_GPU);
        Tensor g_nan(Shape{1, 4}, std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 1.0f}, Execution_Target::VULKAN_GPU);
        Adam_Optimizer opt_adam(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.25f);
        opt_adam.step({{&p_adam, &g_nan}}, 1.0f);
        engine.executeGraph();
        engine.waitIdle();
        if (!opt_adam.consumeGpuOverflowDetected())
        {
            return failStep("Conflict1 Adam GPU overflow_detected flag was not set");
        }
        if (opt_adam.consumeGpuOverflowDetected())
        {
            return failStep("Conflict1 Adam GPU overflow_detected flag was not cleared after consume");
        }
        const auto p_after = p_adam.getData();
        if (!nearlyEqual(p_after[0], 2.0f, 1e-6f))
        {
            return failStep("Conflict1 Adam GPU NaN element should leave parameter unchanged");
        }

        Tensor p_sgd(Shape{1, 4}, std::vector<float>{3.0f, 1.0f, 1.0f, 1.0f}, Execution_Target::VULKAN_GPU);
        Sgd_Optimizer opt_sgd(0.1f, 0.0f, 0.25f, 0.9f);
        opt_sgd.step({{&p_sgd, &g_nan}}, 1.0f);
        engine.executeGraph();
        engine.waitIdle();
        if (!opt_sgd.consumeGpuOverflowDetected())
        {
            return failStep("Conflict1 SGD GPU overflow_detected flag was not set");
        }
        const auto p_sgd_after = p_sgd.getData();
        if (!nearlyEqual(p_sgd_after[0], 3.0f, 1e-6f))
        {
            return failStep("Conflict1 SGD GPU NaN element should leave parameter unchanged");
        }
    }

    if (engine.getContext().isFloat16Enabled())
    {
        Scoped_Engine_Flags_Guard flags_guard;
        engine.enableFusedGemmAdam(true);
        Linear_Layer lin(16, 16, Execution_Target::VULKAN_GPU);
        lin.setMixedPrecision(true);
        Adam_Optimizer opt(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.5f);

        Tensor x(Shape{16, 16}, std::vector<float>(256, 0.25f), Execution_Target::VULKAN_GPU);
        Tensor dy_ok(Shape{16, 16}, std::vector<float>(256, 0.1f), Execution_Target::VULKAN_GPU);
        lin.forward(x);
        lin.backward(dy_ok);
        opt.step(lin.getParametersAndGradients(), 1.0f);
        engine.executeGraph();
        engine.waitIdle();
        (void)opt.consumeGpuOverflowDetected();

        const auto w_before_nan = lin.getWeights().getData();
        Tensor dy_nan(Shape{16, 16}, std::vector<float>(256, std::numeric_limits<float>::quiet_NaN()), Execution_Target::VULKAN_GPU);
        lin.forward(x);
        lin.backward(dy_nan);
        opt.step(lin.getParametersAndGradients(), 1.0f);
        engine.executeGraph();
        engine.waitIdle();

        const bool fused_overflow = opt.consumeGpuOverflowDetected();
        const auto w_after_nan = lin.getWeights().getData();

        if (!fused_overflow)
        {
            return failStep("Conflict2 Fused GEMM+Adam did not report GPU overflow_detected");
        }
        for (size_t i = 0; i < w_before_nan.size(); ++i)
        {
            if (!nearlyEqual(w_before_nan[i], w_after_nan[i], 1e-6f))
            {
                return failStep("Conflict2 Fused GEMM+Adam drifted weight on NaN gradient");
            }
        }
    }

    {
        Tensor a(Shape{4, 4}, std::vector<float>(16, 1.0f), Execution_Target::VULKAN_GPU);
        Tensor b(Shape{4, 4}, std::vector<float>(16, 2.0f), Execution_Target::VULKAN_GPU);
        Tensor c(Execution_Target::VULKAN_GPU);
        Tensor t_fill(Shape{8}, std::vector<float>(8, 9.0f), Execution_Target::VULKAN_GPU);
        Tensor t_zero(Shape{8}, std::vector<float>(8, 7.0f), Execution_Target::VULKAN_GPU);

        a.matmul(b, c);
        const size_t nodes_before = engine.getCurrentGraph().getNodeCount();
        if (nodes_before == 0)
        {
            return failStep("Conflict3 setup matmul did not add node to current_graph");
        }
        t_fill.fill(3.25f);
        t_zero.zero();
        const size_t nodes_after = engine.getCurrentGraph().getNodeCount();
        if (nodes_after < nodes_before + 2)
        {
            return failStep("Conflict3 in-graph fill/zero flushed or failed to append compute nodes");
        }
        engine.executeGraph();
        engine.waitIdle();
        const auto vf = t_fill.getData();
        const auto vz = t_zero.getData();
        for (size_t i = 0; i < 8; ++i)
        {
            if (!nearlyEqual(vf[i], 3.25f, 1e-5f) || !nearlyEqual(vz[i], 0.0f, 1e-6f))
            {
                return failStep("Conflict3 in-graph fill/zero output value mismatch");
            }
        }
    }

    {
        PPO_Actor_Critic_Layer ppo(4, Execution_Target::CPU);
        ppo.addActorLayer<Linear_Layer>(4, 4, Execution_Target::CPU);
        auto &drop_actor = ppo.addActorLayer<Dropout_Layer>(0.25f, Execution_Target::CPU);
        ppo.addCriticLayer<Linear_Layer>(4, 1, Execution_Target::CPU);
        drop_actor.setTrainingMode(true);
        if (ppo.supportsStaticGraph())
        {
            return failStep("Conflict4 PPO_Actor_Critic_Layer with training Dropout should disable static graph");
        }
        drop_actor.setTrainingMode(false);
        if (!ppo.supportsStaticGraph())
        {
            return failStep("Conflict4 PPO_Actor_Critic_Layer with eval Dropout should support static graph");
        }

        Res_Net_Block_2d_Layer res(Execution_Target::CPU);
        auto &drop_res = res.addMainLayer<Dropout_Layer>(0.2f, Execution_Target::CPU);
        drop_res.setTrainingMode(true);
        if (res.supportsStaticGraph())
        {
            return failStep("Conflict4 Res_Net_Block_2d_Layer with training Dropout should disable static graph");
        }
        drop_res.setTrainingMode(false);
        if (!res.supportsStaticGraph())
        {
            return failStep("Conflict4 Res_Net_Block_2d_Layer with eval Dropout should support static graph");
        }
    }

    {
        const bool prev_static = engine.isStaticGraphEnabled();
        engine.enableStaticGraph(true);

        Neural_Network net(Execution_Target::VULKAN_GPU);
        auto &lin = net.addLayer<Linear_Layer>(4, 4, Execution_Target::VULKAN_GPU);
        net.setCostFunction(std::make_unique<Mse_Cost>(Execution_Target::VULKAN_GPU));
        auto opt_ptr = std::make_unique<Adam_Optimizer>(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f);
        Adam_Optimizer *raw_opt = opt_ptr.get();
        net.setOptimizer(std::move(opt_ptr));

        Tensor x_zero(Shape{2, 4}, std::vector<float>(8, 0.0f), Execution_Target::VULKAN_GPU);
        Tensor y_zero(Shape{2, 4}, std::vector<float>(8, 0.0f), Execution_Target::VULKAN_GPU);
        lin.setWeights(Tensor(Shape{4, 4}, std::vector<float>(16, 1.0f), Execution_Target::VULKAN_GPU));
        lin.setBiases(Tensor(Shape{1, 4}, std::vector<float>(4, 0.0f), Execution_Target::VULKAN_GPU));

        net.trainStep(x_zero, y_zero);
        net.trainStep(x_zero, y_zero);
        engine.waitIdle();
        const auto w_no_decay = lin.getWeights().getData();
        if (!nearlyEqual(w_no_decay[0], 1.0f, 1e-4f))
        {
            engine.enableStaticGraph(prev_static);
            return failStep("Conflict5 baseline zero-grad zero-decay static step altered weight");
        }

        const uint64_t gen_before = raw_opt->getConfigGeneration();
        raw_opt->setWeightDecay(0.5f);
        if (raw_opt->getConfigGeneration() == gen_before)
        {
            engine.enableStaticGraph(prev_static);
            return failStep("Conflict5 Adam_Optimizer::setWeightDecay did not bump config_generation");
        }
        net.trainStep(x_zero, y_zero);
        engine.waitIdle();
        const auto w_with_decay = lin.getWeights().getData();
        engine.enableStaticGraph(prev_static);
        if (!(w_with_decay[0] < 0.98f))
        {
            return failStep("Conflict5 Static Graph Replay ignored updated weight_decay push constant");
        }
    }

    {
        Causal_LM_Config cfg{};
        cfg.vocab_size = 32;
        cfg.hidden_dim = 16;
        cfg.num_heads = 2;
        cfg.num_kv_heads = 2;
        cfg.num_layers = 1;
        cfg.intermediate_dim = 32;
        cfg.max_seq_len = 16;
        cfg.execution_target = Execution_Target::VULKAN_GPU;
        cfg.use_loss_scaler = engine.getContext().isFloat16Enabled();

        Causal_LM lm(cfg);
        Adam_Optimizer opt(1e-3f);
        std::vector<int32_t> inp = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int32_t> tgt = {2, 3, 4, 5, 6, 7, 8, 9};
        lm.resetForwardState();
        float l1 = lm.trainStep(inp, tgt, opt, 1.0f);
        float l2 = lm.trainStep(inp, tgt, opt, 1.0f);
        engine.waitIdle();
        if (!std::isfinite(l1) || !std::isfinite(l2) || l1 <= 0.0f || l2 <= 0.0f)
        {
            return failStep("Conflict6 Causal_LM GPU trainStep with resetForwardState produced invalid loss");
        }
    }

    {
        Tensor a(Shape{1, 2}, std::vector<float>{1.0f, -0.5f}, Execution_Target::CPU);
        Tensor b(Shape{1, 2}, std::vector<float>{2.0f, 3.0f}, Execution_Target::CPU);
        Tensor dy(Shape{1, 2}, std::vector<float>{0.5f, -1.0f}, Execution_Target::CPU);
        Tensor ga(Execution_Target::CPU), gb(Execution_Target::CPU);
        ops::llm::swigluBackward(a, b, dy, ga, gb);
        const auto vga = ga.getData();
        const auto vgb = gb.getData();
        auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        const float s0 = sig(1.0f);
        const float expected_gb0 = 0.5f * (1.0f * s0);
        const float expected_ga0 = 0.5f * 2.0f * (s0 + 1.0f * s0 * (1.0f - s0));
        if (!nearlyEqual(vga[0], expected_ga0, 1e-5f) || !nearlyEqual(vgb[0], expected_gb0, 1e-5f))
        {
            return failStep("Conflict7 canonical ops::llm::swigluBackward default-config math mismatch");
        }
    }

    {
        Tensor sw_ref(Execution_Target::VULKAN_GPU);
        Tensor sw_in(Shape{2, 4}, {1.0f, -0.5f, 0.25f, 2.0f, -1.5f, 0.5f, 1.5f, -0.25f}, Execution_Target::VULKAN_GPU);
        ops::llm::swigluForward(sw_in, sw_in, sw_ref);
        Tensor sw_alias = sw_in;
        ops::llm::swigluForward(sw_alias, sw_alias, sw_alias);
        flushGpuEngine(true);
        if (!verifyTensorsClose(sw_ref, sw_alias, 1e-5f))
        {
            return failStep("P2 H11 self-aliased swiglu output mismatch");
        }

        Tensor m_in(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, Execution_Target::VULKAN_GPU);
        Tensor m_ref(Execution_Target::VULKAN_GPU);
        m_in.matmul(m_in, m_ref);
        Tensor m_alias = m_in;
        m_alias.matmul(m_alias, m_alias);
        flushGpuEngine(true);
        if (!verifyTensorsClose(m_ref, m_alias, 1e-5f))
        {
            return failStep("P2 H11 self-aliased matmul output mismatch");
        }

        Tensor rn_in(Shape{2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -1.5f}, Execution_Target::VULKAN_GPU);
        Tensor rn_w(Shape{1, 4}, {1.0f, 1.2f, 0.8f, 1.5f}, Execution_Target::VULKAN_GPU);
        Tensor rn_inv_ref(Execution_Target::VULKAN_GPU);
        Tensor rn_inv_alias(Execution_Target::VULKAN_GPU);
        Tensor rn_ref(Execution_Target::VULKAN_GPU);
        ops::normalization::rmsNormForward(rn_in, rn_w, rn_inv_ref, rn_ref, 1e-5f);
        Tensor rn_alias = rn_in;
        ops::normalization::rmsNormForward(rn_alias, rn_w, rn_inv_alias, rn_alias, 1e-5f);
        flushGpuEngine(true);
        if (!verifyTensorsClose(rn_ref, rn_alias, 1e-4f))
        {
            return failStep("P2 H11 self-aliased rmsnorm output mismatch");
        }
    }

    if (engine.getContext().isFloat16Enabled())
    {
        Tensor base(Shape{2, 4}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Execution_Target::VULKAN_GPU);
        base.prewarmFp16Cache();
        auto gpu_base = std::dynamic_pointer_cast<Gpu_Tensor_Impl>(base.getImplementation());
        if (!gpu_base)
        {
            return failStep("P2 H15 failed to retrieve Gpu_Tensor_Impl");
        }

        Tensor patch(Shape{1, 4}, std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}, Execution_Target::VULKAN_GPU);
        base.updateSlice(0, 1, patch);
        flushGpuEngine(true);

        std::shared_ptr<gpu::vector> shadow_a;
        std::shared_ptr<gpu::vector> shadow_b;
        std::thread t1([&]() { shadow_a = gpu_base->getEffectiveFp16Storage(); });
        std::thread t2([&]() { shadow_b = gpu_base->getEffectiveFp16Storage(); });
        t1.join();
        t2.join();
        flushGpuEngine(true);
        if (!shadow_a || !shadow_b || shadow_a.get() != shadow_b.get())
        {
            return failStep("P2 H15 concurrent getEffectiveFp16Storage did not return shared shadow storage");
        }
        std::vector<float> fp16_vals;
        shadow_a->downloadData(fp16_vals);
        if (fp16_vals.size() != 8 || !nearlyEqual(fp16_vals[4], 10.0f, 1e-2f) || !nearlyEqual(fp16_vals[7], 40.0f, 1e-2f))
        {
            return failStep("P2 H15 FP16 shadow cache held stale data after updateSlice");
        }
    }

    {
        const std::vector<float> a_vals = generateRandomVector(24, -1.0f, 1.0f, 77);
        const std::vector<float> b_vals = generateRandomVector(16, -1.0f, 1.0f, 88);
        const std::vector<float> c_vals = generateRandomVector(4, -0.5f, 0.5f, 99);

        Tensor a_cpu(Shape{1, 3, 2, 4}, a_vals, Execution_Target::CPU);
        Tensor b_cpu(Shape{2, 1, 4, 2}, b_vals, Execution_Target::CPU);
        Tensor bias_cpu(Shape{1, 2}, c_vals.data() ? std::vector<float>{c_vals[0], c_vals[1]} : std::vector<float>{0.0f, 0.0f}, Execution_Target::CPU);
        Tensor out_cpu(Execution_Target::CPU);
        Tensor out_add_cpu(Execution_Target::CPU);
        a_cpu.matmul(b_cpu, out_cpu);
        a_cpu.matmulAdd(b_cpu, bias_cpu, out_add_cpu);

        Tensor a_gpu(Shape{1, 3, 2, 4}, a_vals, Execution_Target::VULKAN_GPU);
        Tensor b_gpu(Shape{2, 1, 4, 2}, b_vals, Execution_Target::VULKAN_GPU);
        Tensor bias_gpu(Shape{1, 2}, std::vector<float>{c_vals[0], c_vals[1]}, Execution_Target::VULKAN_GPU);
        Tensor out_gpu(Execution_Target::VULKAN_GPU);
        Tensor out_add_gpu(Execution_Target::VULKAN_GPU);
        a_gpu.matmul(b_gpu, out_gpu);
        a_gpu.matmulAdd(b_gpu, bias_gpu, out_add_gpu);
        flushGpuEngine(true);

        const float m12_tol = engine.isCooperativeMatrixEnabled() ? 5e-3f : 1e-4f;
        if (out_cpu.getShape() != Shape{2, 3, 2, 2} ||
            out_gpu.getShape() != Shape{2, 3, 2, 2} ||
            out_add_gpu.getShape() != Shape{2, 3, 2, 2} ||
            !verifyTensorsClose(out_cpu, out_gpu, m12_tol) ||
            !verifyTensorsClose(out_add_cpu, out_add_gpu, m12_tol))
        {
            return failStep("P2 M12 4D broadcast matmul/matmulAdd CPU/GPU parity mismatch");
        }
    }

    return true;
}

void runRegressionsW6W7Suite(Test_Result_Counter &counter)
{
    counter.record("  Wave 6 Regression Suite:           ", testWave6Regressions());
    counter.record("  Wave 7 Deferred Regression Suite:  ", testWave7DeferredRegressions());
    counter.record("  Group 3 Engine Optimization (CPU): ", testGroup3EngineOptimizations(Execution_Target::CPU));
    counter.record("  Group 3 Engine Optimization (GPU): ", testGroup3EngineOptimizations(Execution_Target::VULKAN_GPU));
    counter.record("  Group 4 GPU Timestamp Profiling:   ", testGroup4GpuTimestampProfiling());
    counter.record("  Conflict Resolutions Suite:        ", testConflictResolutions());
}
