#include "test_common.h"
#include "engine/pipeline_cache_manager.h"
#include "helper/resource_path.h"

bool testDxAndLoggerSuite()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("DX", msg);
    };
    namespace fs = std::filesystem;
    const fs::path temp_root = fs::path("temp") / "test_dx_suite";
    Scoped_Temp_File_Guard temp_cleanup({}, {temp_root});
    temp_cleanup.clean();

    {
        Scoped_Logger_Guard log_cfg_guard(true);

        Log_Level parsed_lvl = Log_Level::LOG_INFO;
        if (!Logger::parseLevel("debug", parsed_lvl) || parsed_lvl != Log_Level::LOG_DEBUG)
            return failStep("DX Logger::parseLevel debug");
        if (!Logger::parseLevel("warn", parsed_lvl) || parsed_lvl != Log_Level::LOG_WARNING)
            return failStep("DX Logger::parseLevel warn");
        Log_Color_Mode parsed_color = Log_Color_Mode::AUTO;
        if (!Logger::parseColorMode("never", parsed_color) || parsed_color != Log_Color_Mode::NEVER)
            return failStep("DX Logger::parseColorMode never");

        const Log_Feature masked = Logger::parseFeatureMask("ALL,-GRAPH_RECORDING", Log_Feature::NONE);
        if ((masked & Log_Feature::STATIC_GRAPH) == Log_Feature::NONE ||
            (masked & Log_Feature::GRAPH_RECORDING) != Log_Feature::NONE)
        {
            return failStep("DX Logger::parseFeatureMask +/- modifiers");
        }

        Logging_Preferences lp;
        lp.enable_logging = true;
        lp.level = "LOG_WARNING";
        lp.console_enabled = true;
        lp.console_level = "LOG_ERROR";
        lp.file_enabled = false;
        lp.file_level = "LOG_DEBUG";
        lp.color_mode = "never";
        lp.quiet_mode = true;
        lp.errors_to_stderr = false;
        lp.filter_info_by_feature = true;
        lp.include_timestamp_ms = true;
        lp.include_thread_id = true;
        lp.include_source_location = true;
        lp.features = {"STATIC_GRAPH", "CHECKPOINT_IO"};
        Logger::applyPreferences(lp);
        Logger_Config applied = Logger::getConfig();
        if (applied.min_level != Log_Level::LOG_WARNING ||
            applied.console_min_level != Log_Level::LOG_ERROR ||
            !applied.quiet_mode ||
            !applied.filter_info_by_feature ||
            !applied.include_timestamp_ms ||
            !applied.include_thread_id ||
            !applied.include_source_location ||
            applied.color_mode != Log_Color_Mode::NEVER ||
            (applied.active_features & Log_Feature::STATIC_GRAPH) == Log_Feature::NONE ||
            (applied.active_features & Log_Feature::TRANSFORMER) != Log_Feature::NONE)
        {
            return failStep("DX Logger::applyPreferences state mismatch");
        }

        const fs::path log_dir = temp_root / "logs";
        std::string arg0 = "test.exe";
        std::string arg1 = "--log-level";
        std::string arg2 = "debug";
        std::string arg3 = "--log-console-level";
        std::string arg4 = "error";
        std::string arg5 = "--log-file-level";
        std::string arg6 = "debug";
        std::string arg7 = "--log-features";
        std::string arg8 = "STATIC_GRAPH,TRANSFORMER,CHECKPOINT_IO";
        std::string arg9 = "--log-file";
        std::string arg10 = "--log-dir";
        std::string arg11 = log_dir.string();
        std::string arg12 = "--log-quiet";
        std::string arg13 = "--no-color";
        std::string arg14 = "--log-filter-info";
        char *cli_argv[] = {
            arg0.data(), arg1.data(), arg2.data(), arg3.data(), arg4.data(),
            arg5.data(), arg6.data(), arg7.data(), arg8.data(), arg9.data(),
            arg10.data(), arg11.data(), arg12.data(), arg13.data(), arg14.data()
        };
        if (Logger::loggerCliFlagArity("--log-level") != 2 ||
            Logger::loggerCliFlagArity("--no-color") != 1 ||
            Logger::loggerCliFlagArity("--log-level=debug") != 1 ||
            Logger::loggerCliFlagArity("--unknown-flag") != 0 ||
            !Logger::isLoggerCliFlag("--log-features") ||
            Logger::isLoggerCliFlag("--unknown-flag"))
        {
            return failStep("DX Logger CLI flag arity/predicate mismatch");
        }
        Logger::applyCliArgs(15, cli_argv);
        Logger_Config cli_cfg = Logger::getConfig();
        if (cli_cfg.min_level != Log_Level::LOG_DEBUG ||
            cli_cfg.console_min_level != Log_Level::LOG_ERROR ||
            cli_cfg.file_min_level != Log_Level::LOG_DEBUG ||
            !cli_cfg.file_enabled ||
            !cli_cfg.quiet_mode ||
            cli_cfg.color_mode != Log_Color_Mode::NEVER ||
            !cli_cfg.filter_info_by_feature ||
            (cli_cfg.active_features & Log_Feature::TRANSFORMER) == Log_Feature::NONE)
        {
            return failStep("DX Logger::applyCliArgs config mismatch");
        }

        int side_effect = 0;
        Logger::setMinLogLevel(Log_Level::LOG_WARNING);
        VML_LOG_DEBUG(Log_Feature::STATIC_GRAPH, "Skipped counter={}", ++side_effect);
        if (side_effect != 0)
        {
            return failStep("DX VML_LOG_DEBUG did not short-circuit argument evaluation");
        }

        Logger::setMinLogLevel(Log_Level::LOG_DEBUG);
        VML_LOG_DEBUG(Log_Feature::STATIC_GRAPH, "DX log separation marker {}", 42);
        Logger::flush();
        const std::string written_log_path = Logger::getCurrentLogFilepath();
        Logger::enableFileLogging(false);
        if (written_log_path.empty() || !fs::exists(written_log_path))
        {
            return failStep("DX Logger did not create log file in configured directory");
        }
        std::ifstream ifs(written_log_path);
        std::string log_contents((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        if (log_contents.find("DX log separation marker 42") == std::string::npos ||
            log_contents.find("\033[") != std::string::npos)
        {
            return failStep("DX Logger log file missing expected content or contained ANSI escapes");
        }
    }

    {
        Scoped_Engine_Flags_Guard flags_guard;

        Runtime_Config rt_cfg;
        rt_cfg.load_preferences = false;
        rt_cfg.apply_env_logging = false;
        rt_cfg.enable_pipeline_cache = false;
        rt_cfg.save_pipeline_cache_on_exit = false;
        rt_cfg.validate_hardware = false;
        rt_cfg.enable_static_graph = true;
        rt_cfg.enable_single_submit = true;

        Runtime rt(rt_cfg);
        Runtime_Info info = rt.info();
        if (info.device_name.empty() || !info.static_graph_enabled || !Runtime::isStaticGraphEnabled())
        {
            return failStep("DX Runtime::queryInfo device/flag mismatch");
        }
        Runtime::setStaticGraphEnabled(false);
        if (Runtime::isStaticGraphEnabled())
        {
            return failStep("DX Runtime::setStaticGraphEnabled did not update state");
        }
    }

    {
        Neural_Network nn;
        nn.withTarget(Execution_Target::CPU)
          .withStepLrPerBatch(true);
        nn.addLayer<Linear_Layer>(2, 4, Execution_Target::CPU);
        nn.addLayer<Relu_Layer>(Execution_Target::CPU);
        nn.addLayer<Linear_Layer>(4, 1, Execution_Target::CPU);
        nn.setCostFunction<Mse_Cost>(Execution_Target::CPU);
        nn.setLearningRate<No_Decay>(0.01f);
        nn.setOptimizer<Adam_Optimizer>(nn.getLearningRate());

        Lambda_Data_Pipeline pipeline(
            4,
            [](size_t, std::vector<float> &inputs, std::vector<float> &targets)
            {
                inputs = {1.0f, 0.5f, 0.25f, 1.0f};
                targets = {1.5f, 1.25f};
            },
            Execution_Target::CPU);
        pipeline.initializeBuffers(2, 2, 1, Execution_Target::CPU);

        size_t step_cb_count = 0;
        size_t epoch_cb_count = 0;
        Fit_Config fcfg;
        fcfg.epochs = 3;
        fcfg.steps_per_epoch = 4;
        fcfg.batch_size = 2;
        fcfg.input_dim = 2;
        fcfg.output_dim = 1;
        fcfg.compute_step_loss = true;

        Fit_Callbacks fcbs;
        fcbs.on_step = [&](const Fit_Progress &p)
        {
            if (std::isfinite(p.batch_loss))
            {
                ++step_cb_count;
            }
        };
        fcbs.on_epoch_end = [&](const Fit_Progress &)
        {
            ++epoch_cb_count;
        };
        fcbs.should_stop = [](const Fit_Progress &p)
        {
            return (p.epoch == 1 && p.step >= 2);
        };

        nn.fit(pipeline, fcfg, fcbs);
        if (epoch_cb_count != 2 || step_cb_count != 6)
        {
            return failStep("DX Neural_Network::fit Fit_Callbacks / should_stop mismatch");
        }

        nn.setTrainingMode(true);
        Tensor sample_in(1, 2, std::vector<float>{1.0f, 0.5f}, Execution_Target::CPU);
        std::vector<float> pred = nn.predictData(sample_in);
        if (pred.size() != 1 || !std::isfinite(pred[0]) || !nn.isTrainingMode())
        {
            return failStep("DX Neural_Network::predictData output or training-mode restore mismatch");
        }
    }

    {
        if (Causal_LM::alignVocabSize(13, 16) != 16 || Causal_LM::alignVocabSize(32, 16) != 32)
        {
            return failStep("DX Causal_LM::alignVocabSize mismatch");
        }

        Causal_LM_Config lcfg;
        lcfg.vocab_size = 13;
        lcfg.auto_align_vocab = true;
        lcfg.hidden_dim = 8;
        lcfg.num_heads = 2;
        lcfg.intermediate_dim = 16;
        lcfg.num_layers = 1;
        lcfg.max_seq_len = 16;
        lcfg.execution_target = Execution_Target::CPU;
        lcfg.tokenizer_path = "";

        Causal_LM lm(lcfg);
        if (lm.getConfig().vocab_size != 16)
        {
            return failStep("DX Causal_LM auto_align_vocab did not align 13 -> 16");
        }

        const fs::path ckpt_path = temp_root / "sub" / "model.nnck";
        const fs::path infer_path = temp_root / "sub" / "model.bin";
        lm.saveCheckpoint(ckpt_path.string());
        lm.saveInference(infer_path.string());

        Causal_LM_Config load_cfg = lcfg;
        load_cfg.vocab_size = 16;
        load_cfg.auto_align_vocab = false;
        Causal_LM lm_loaded(load_cfg);
        if (!lm_loaded.loadAuto(ckpt_path.string()))
        {
            return failStep("DX Causal_LM::loadAuto did not return true for checkpoint");
        }
        if (lm_loaded.loadAuto(infer_path.string()))
        {
            return failStep("DX Causal_LM::loadAuto did not return false for inference file");
        }

        No_Decay sched(1e-3f);
        Adam_Optimizer opt(sched);
        std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8};
        auto samples = LM_Trainer::buildSamples(tokens, 8, 4);
        if (samples.empty())
        {
            return failStep("DX LM_Trainer::buildSamples returned empty samples");
        }
        LM_Trainer trainer(lm);
        LM_Train_Config tcfg;
        tcfg.epochs = 2;
        tcfg.batch_size = 2;
        tcfg.seq_len = 8;
        tcfg.stride = 4;
        size_t lm_epochs = 0;
        float last_avg_loss = 0.0f;
        LM_Train_Callbacks tcbs;
        tcbs.on_epoch_end = [&](const LM_Epoch_Metrics &m)
        {
            ++lm_epochs;
            last_avg_loss = m.avg_loss;
        };
        trainer.fit(samples, opt, &sched, tcfg, tcbs);
        if (lm_epochs != 2 || !std::isfinite(last_avg_loss))
        {
            return failStep("DX LM_Trainer::fit epoch count or loss invalid");
        }

        const std::string tok_path = fs::exists("tokenizer/syllable_tokenizer.json")
            ? "tokenizer/syllable_tokenizer.json"
            : (fs::exists("tokenizer/tokenizer.json") ? "tokenizer/tokenizer.json" : "");
        if (!tok_path.empty())
        {
            Causal_LM_Config gcfg;
            gcfg.vocab_size = 0;
            gcfg.auto_align_vocab = true;
            gcfg.hidden_dim = 8;
            gcfg.num_heads = 2;
            gcfg.intermediate_dim = 16;
            gcfg.num_layers = 1;
            gcfg.max_seq_len = 16;
            gcfg.execution_target = Execution_Target::CPU;
            gcfg.tokenizer_path = tok_path;
            Causal_LM lm_gen(gcfg);
            if (lm_gen.getConfig().vocab_size == 0 || (lm_gen.getConfig().vocab_size % 16) != 0)
            {
                return failStep("DX Causal_LM vocab_size==0 auto-sizing failed");
            }
            Generate_Config gen_opts;
            gen_opts.max_new_tokens = 2;
            gen_opts.temperature = 0.0f;
            (void)lm_gen.generate("a b", gen_opts);
        }
    }

    return true;
}

bool testAttentionEnginePointerOverload()
{
    std::vector<float> q_data(512, 0.0f);
    std::vector<float> k_data(512, 0.0f);
    std::vector<float> v_data(512, 0.0f);
    std::vector<float> do_data(512, 0.0f);
    for (size_t i = 0; i < 512; ++i)
    {
        q_data[i] = 0.04f * static_cast<float>((i % 7) + 1);
        k_data[i] = 0.04f * static_cast<float>((i % 5) + 1);
        v_data[i] = 0.08f * static_cast<float>((i % 11) + 1);
        do_data[i] = 0.02f * static_cast<float>((i % 9) + 1);
    }

    Tensor q_fp16 = castTensorToFp16(Tensor(Shape{1, 2, 16, 16}, q_data, Execution_Target::VULKAN_GPU));
    Tensor k_fp16 = castTensorToFp16(Tensor(Shape{1, 2, 16, 16}, k_data, Execution_Target::VULKAN_GPU));
    Tensor v_fp16 = castTensorToFp16(Tensor(Shape{1, 2, 16, 16}, v_data, Execution_Target::VULKAN_GPU));
    Tensor do_fp16 = castTensorToFp16(Tensor(Shape{1, 2, 16, 16}, do_data, Execution_Target::VULKAN_GPU));

    Execution_Engine &engine = Execution_Engine::getInstance();

    Tensor out_cfg(Execution_Target::VULKAN_GPU);
    Tensor l_cfg(Execution_Target::VULKAN_GPU);
    ops::llm::Attention_Config cfg{
        .num_heads = 2,
        .num_kv_heads = 2,
        .seq_len = 16,
        .head_dim = 16,
        .is_causal = true,
        .scale = 0.0f,
        .kv_seq_len = 16,
        .causal_offset = 0,
        .is_bshd = false,
        .engine = &engine};
    ops::llm::flashAttentionForward(q_fp16, k_fp16, v_fp16, out_cfg, cfg, &l_cfg);

    Tensor out_pos(Execution_Target::VULKAN_GPU);
    Tensor l_pos(Execution_Target::VULKAN_GPU);
    ops::llm::flashAttentionForward(q_fp16, k_fp16, v_fp16, out_pos,
                                    2u, 16u, 16u, true, 0.0f, &l_pos,
                                    16u, 0u, 2u, &engine);

    Tensor dq_cfg(Execution_Target::VULKAN_GPU);
    Tensor dk_cfg(Execution_Target::VULKAN_GPU);
    Tensor dv_cfg(Execution_Target::VULKAN_GPU);
    ops::llm::flashAttentionBackward(q_fp16, k_fp16, v_fp16, out_cfg, do_fp16,
                                     dq_cfg, dk_cfg, dv_cfg, cfg, &l_cfg);

    Tensor dq_pos(Execution_Target::VULKAN_GPU);
    Tensor dk_pos(Execution_Target::VULKAN_GPU);
    Tensor dv_pos(Execution_Target::VULKAN_GPU);
    ops::llm::flashAttentionBackward(q_fp16, k_fp16, v_fp16, out_pos, do_fp16,
                                     dq_pos, dk_pos, dv_pos,
                                     2u, 16u, 16u, true, 0.0f, &l_pos,
                                     2u, &engine);

    engine.executeGraph();
    engine.waitIdle();

    return verifyMatrix(out_pos, out_cfg.getData(), 1e-4f) &&
           verifyMatrix(dq_pos, dq_cfg.getData(), 1e-4f) &&
           verifyMatrix(dk_pos, dk_cfg.getData(), 1e-4f) &&
           verifyMatrix(dv_pos, dv_cfg.getData(), 1e-4f);
}

bool testVramEmptyChunkReclamation()
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.waitIdle();
    Vulkan_Sub_Allocator &allocator = engine.getContext().getAllocator();

    Runtime::reclaimEmptyVram(false);
    const Allocator_Stats baseline = allocator.getStats();
    if (baseline.empty_chunk_count != 0)
    {
        return false;
    }

    const VkDeviceSize oversized_bytes = allocator.getDefaultChunkSize() + 4096;
    VkMemoryRequirements req{
        .size = oversized_bytes,
        .alignment = 256,
        .memoryTypeBits = 0xFFFFFFFF};
    Memory_Allocation oversized_alloc = allocator.allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (oversized_alloc.memory == VK_NULL_HANDLE)
    {
        return false;
    }
    const Allocator_Stats during = allocator.getStats();
    if (during.active_chunk_count < baseline.active_chunk_count + 1)
    {
        allocator.free(oversized_alloc);
        return false;
    }
    allocator.free(oversized_alloc);

    const Allocator_Stats after_free = allocator.getStats();
    const Runtime_Info rt_info = Runtime::queryInfo();
    return (after_free.empty_chunk_count == 0) &&
           (after_free.reclaimed_chunk_count >= baseline.reclaimed_chunk_count + 1) &&
           (after_free.reclaimed_bytes >= baseline.reclaimed_bytes + oversized_bytes) &&
           (rt_info.empty_vram_chunk_count == 0) &&
           (rt_info.active_vram_chunk_count == after_free.active_chunk_count);
}

bool testResourcePathResolver()
{
    namespace fs = std::filesystem;

    const std::string exe_dir = Resource_Path::getExecutableDirectory();
    const std::string shader_dir = Resource_Path::resolveDirectory(
        "compute_shader/spv", "VULKANML_SHADER_DIR", "compute_shader/spv");
    const std::string dict_file = Resource_Path::resolveFile(
        "compute_shader/shader_dictionary.json", "VULKANML_SHADER_DICTIONARY", "compute_shader/shader_dictionary.json");
    const std::string fallback_dir = Resource_Path::resolveDirectory(
        "", nullptr, "compute_shader/spv");
    const std::string missing_file = Resource_Path::resolveFile(
        "nonexistent_file_xyz_test.json", nullptr, "compute_shader/shader_dictionary.json");
    const std::string writable_tmp = Resource_Path::resolveWritableDirectory(
        "temp", "VULKANML_PIPELINE_CACHE_DIR", "temp");

    std::error_code ec;
    return !exe_dir.empty() && fs::is_directory(exe_dir, ec) &&
           !shader_dir.empty() && fs::is_directory(shader_dir, ec) &&
           !dict_file.empty() && fs::is_regular_file(dict_file, ec) &&
           !fallback_dir.empty() && fs::is_directory(fallback_dir, ec) &&
           !fs::exists(missing_file, ec) &&
           !writable_tmp.empty();
}

bool testUnifiedDeviceKeyedPipelineCache()
{
    auto &engine = Execution_Engine::getInstance();
    const std::string keyed_path = Pipeline_Cache_Manager::buildDeviceKeyedCachePath(engine.getContext(), "");
    const std::string explicit_path = Pipeline_Cache_Manager::buildDeviceKeyedCachePath(engine.getContext(), "temp/custom_test.bin");
    if (keyed_path.find("pipeline_cache_") == std::string::npos ||
        !keyed_path.ends_with(".bin") ||
        keyed_path == "pipeline_cache.bin" ||
        explicit_path != "temp/custom_test.bin")
    {
        return false;
    }

    VkPipeline relu_pipe = engine.getNetwork().getPipeline(Compute_Pipeline::RELU);
    if (relu_pipe == VK_NULL_HANDLE)
    {
        return false;
    }

    auto &cache_mgr = engine.getPipelineCacheManager();
    cache_mgr.savePipelineCache();
    const std::string active_cache_path = cache_mgr.getCacheFilePath();
    std::error_code ec;
    return !active_cache_path.empty() &&
           std::filesystem::is_regular_file(active_cache_path, ec) &&
           std::filesystem::file_size(active_cache_path, ec) > 0;
}

bool testSpirvPushConstantContractGuard()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("DX_SPV_PC", msg);
    };

    if (Vulkan_Network::reflectSpirvPushConstantBytes({}) != 0u)
    {
        return failStep("empty SPIR-V should reflect 0 bytes");
    }

    auto &engine = Execution_Engine::getInstance();
    auto &net = engine.getNetwork();
    if (net.getPipeline(Compute_Pipeline::RELU) == VK_NULL_HANDLE ||
        net.getPipelinePushConstantSize(Compute_Pipeline::RELU) != 4u)
    {
        return failStep("RELU push constant reflection expected 4 bytes");
    }

    if (engine.getContext().isFloat16Enabled())
    {
        if (net.getPipeline(Compute_Pipeline::CONV2D_IM2COL_FP16) == VK_NULL_HANDLE ||
            net.getPipelinePushConstantSize(Compute_Pipeline::CONV2D_IM2COL_FP16) != 40u)
        {
            return failStep("CONV2D_IM2COL_FP16 push constant reflection expected 40 bytes");
        }
    }

    {
        Scoped_Engine_Flags_Guard flags_guard;
        if (Runtime::isFloat16Enabled() && Runtime::isCooperativeMatrixSupported())
        {
            Runtime::setCooperativeMatrixEnabled(true);
            if (net.getPipeline(Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16) == VK_NULL_HANDLE ||
                net.getPipelinePushConstantSize(Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16) != 80u)
            {
                return failStep("MATMUL_ADD_COOPMAT_FP16 push constant reflection expected 80 bytes");
            }
        }
    }

    Tensor a(1, 4, std::vector<float>{1.0f, -1.0f, 2.0f, 3.0f}, Execution_Target::VULKAN_GPU);
    Tensor b(1, 4, Execution_Target::VULKAN_GPU);
    engine.getContext().executePendingTransfers();

    auto buf_a = Gpu_Tensor_Impl::castToGpu(*a.getImplementation()).getStorageBuffer();
    auto buf_b = Gpu_Tensor_Impl::castToGpu(*b.getImplementation()).getStorageBuffer();

    Compute_Node bad_node;
    bad_node.pipeline_id = Compute_Pipeline::RELU;
    bad_node.buffers = {buf_a, buf_b};
    bad_node.workgroup_count_x = 1;
    bad_node.is_barrier_required_after = true;
    engine.getCurrentGraph().addNode(std::move(bad_node));

    if (!expectThrows<std::runtime_error>([&]() { engine.executeGraph(); }))
    {
        engine.getCurrentGraph().clear();
        return failStep("undersized push constants did not throw runtime_error");
    }
    if (!engine.getCurrentGraph().getNodes().empty())
    {
        engine.getCurrentGraph().clear();
        return failStep("current_graph was not cleared after rejected dispatch");
    }
    return true;
}

bool testGpuFitStaticGraphAndProfiler()
{
    auto failStep = [](const char *msg) -> bool
    {
        return failTestStep("DX_FIT_GPU", msg);
    };

    struct Profiler_Reset_Guard
    {
        ~Profiler_Reset_Guard()
        {
            Step_Timings::getInstance().reset();
        }
    } profiler_guard;

    Scoped_Engine_Flags_Guard flags_guard;
    const auto &ctx = Execution_Engine::getInstance().getContext();
    if (!ctx.isFloat16Supported() || !ctx.isFloat16Enabled())
    {
        return true;
    }

    Neural_Network nn(Execution_Target::VULKAN_GPU);
    nn.setTrainingMode(true);
    nn.setLearningRate<Cosine_Annealing>(0.001f, 1e-5f, 6);
    nn.setOptimizer<Adam_Optimizer>(nn.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f);
    nn.setCostFunction<Cce_Cost>();
    nn.setStepLearningRatePerBatch(true);

    nn.addLayer<Conv2d_Layer>(8, 8, 1, 16, 3, 1, 1);
    nn.addLayer<Batch_Norm_2d_Layer>(8, 8, 16, 1e-5f, 0.1f);
    nn.addLayer<Gelu_Layer>();
    nn.addLayer<Max_Pool_2d_Layer>(8, 8, 16, 2, 2, 0);
    nn.addLayer<Linear_Layer>(4 * 4 * 16, 4);
    nn.addLayer<Softmax_Layer>(true);

    nn.withMixedPrecision().withStaticGraph();
    if (Runtime::isCooperativeMatrixSupported())
    {
        nn.withCooperativeMatrix(true);
    }

    constexpr size_t BATCH = 4;
    constexpr size_t IN_DIM = 64;
    constexpr size_t OUT_DIM = 4;
    constexpr size_t STEPS = 6;

    Lambda_Data_Pipeline pipeline(
        STEPS,
        [](size_t step, std::vector<float> &in_out, std::vector<float> &tgt_out)
        {
            in_out.resize(BATCH * IN_DIM);
            for (size_t i = 0; i < in_out.size(); ++i)
            {
                in_out[i] = 0.05f * static_cast<float>((i + step) % 13);
            }
            tgt_out.assign(BATCH * OUT_DIM, 0.0f);
            for (size_t b = 0; b < BATCH; ++b)
            {
                tgt_out[b * OUT_DIM + ((step + b) % OUT_DIM)] = 1.0f;
            }
        },
        Execution_Target::VULKAN_GPU);
    pipeline.initializeBuffers(BATCH, IN_DIM, OUT_DIM, Execution_Target::VULKAN_GPU);

    Fit_Config cfg;
    cfg.epochs = 1;
    cfg.steps_per_epoch = STEPS;
    cfg.batch_size = BATCH;
    cfg.input_dim = IN_DIM;
    cfg.output_dim = OUT_DIM;
    cfg.compute_step_loss = false;

    Step_Timings::getInstance().reset();
    size_t step_count = 0;
    bool all_zero_loss = true;
    uint64_t gen0_step2 = 0;
    uint64_t gen1_step2 = 0;
    uint64_t gen0_step6 = 0;
    uint64_t gen1_step6 = 0;

    Fit_Callbacks cbs;
    cbs.on_step = [&](const Fit_Progress &p)
    {
        ++step_count;
        if (p.batch_loss != 0.0f)
        {
            all_zero_loss = false;
        }
        const auto &exec = Execution_Engine::getInstance().getGraphExecutor();
        if (p.step == 2)
        {
            gen0_step2 = exec.getStaticBakeGeneration(0);
            gen1_step2 = exec.getStaticBakeGeneration(1);
        }
        if (p.step == STEPS)
        {
            gen0_step6 = exec.getStaticBakeGeneration(0);
            gen1_step6 = exec.getStaticBakeGeneration(1);
        }
    };

    nn.fit(pipeline, cfg, cbs);
    flushGpuEngine(true);

    if (step_count != STEPS || !all_zero_loss)
    {
        return failStep("fit should skip per-step computeLoss when compute_step_loss=false");
    }
    if (gen0_step2 == 0 || gen1_step2 == 0 || gen0_step6 != gen0_step2 || gen1_step6 != gen1_step2)
    {
        return failStep("static graph re-baked instead of hitting fast replay across steps 3..6");
    }

    const auto &st = Step_Timings::getInstance();
    if (st.getDuration(Timing_Stage::TOTAL_BATCH) <= 0.0)
    {
        return failStep("Step_Timings TOTAL_BATCH was not recorded by fit()");
    }
    if (st.getNodeCount(Node_Stage::FWD_LOSS_GPU_NODES) == 0 ||
        st.getNodeCount(Node_Stage::BWD_GPU_NODES) == 0 ||
        st.getNodeCount(Node_Stage::OPT_GPU_NODES) == 0)
    {
        return failStep("Step_Timings GPU node counts remained 0");
    }

    const std::string formatted = st.formatTable(STEPS, STEPS, 100.0, STEPS, "samples/s");
    if (formatted.find("samples/s") == std::string::npos || formatted.find("tok/s") != std::string::npos)
    {
        return failStep("Step_Timings formatTable did not honor samples/s unit");
    }
    return true;
}

void runDxInfraSuite(Test_Result_Counter &counter)
{
    counter.record("  DX Facades & Logger Suite:         ", testDxAndLoggerSuite());
    counter.record("  Attention Engine Ptr Overload:     ", testAttentionEnginePointerOverload());
    counter.record("  VRAM Empty Chunk Reclamation:      ", testVramEmptyChunkReclamation());
    counter.record("  Resource Path Layered Resolver:    ", testResourcePathResolver());
    counter.record("  Device-Keyed Unified Pipe Cache:   ", testUnifiedDeviceKeyedPipelineCache());
    counter.record("  SPIR-V Push-Constant Guard:        ", testSpirvPushConstantContractGuard());
    counter.record("  GPU Fit StaticGraph & Profiler:    ", testGpuFitStaticGraphAndProfiler());
}
