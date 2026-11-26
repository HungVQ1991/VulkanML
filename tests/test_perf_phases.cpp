#include "test_common.h"

bool testPhase1KernelOptimizations()
{
    auto failStep = [](const char *name) -> bool {
        return failTestStep("Phase1KernelOptimizations", name);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    const bool prev_coop = engine.isCooperativeMatrixEnabled();
    Scoped_Coop_Mat_Guard coop_guard;

    struct RmsCase { size_t batch; size_t dim; };
    const RmsCase rms_cases[] = {{67, 7}, {130, 256}, {70, 300}};
    for (bool use_fp16 : {false, true})
    {
        engine.setCooperativeMatrixEnabled(use_fp16 ? prev_coop : false);
        const float tol_fwd = use_fp16 ? 2e-2f : 1e-4f;
        const float tol_dx  = use_fp16 ? 3e-2f : 1e-4f;
        const float tol_dg  = use_fp16 ? 5e-2f : 2e-3f;

        for (const auto &rc : rms_cases)
        {
            std::vector<float> x_vals(rc.batch * rc.dim);
            std::vector<float> dy_vals(rc.batch * rc.dim);
            std::vector<float> g_vals(rc.dim);
            for (size_t i = 0; i < x_vals.size(); ++i)
            {
                x_vals[i] = 0.05f * static_cast<float>(static_cast<int>(i % 19) - 9);
                dy_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 7) % 17) - 8);
            }
            for (size_t c = 0; c < rc.dim; ++c)
            {
                g_vals[c] = 0.75f + 0.02f * static_cast<float>(c % 13);
            }

            Tensor x_cpu(Shape{rc.batch, rc.dim}, x_vals, Execution_Target::CPU);
            Tensor dy_cpu(Shape{rc.batch, rc.dim}, dy_vals, Execution_Target::CPU);
            Tensor g_cpu(Shape{1, rc.dim}, g_vals, Execution_Target::CPU);
            Tensor inv_cpu(Execution_Target::CPU), out_cpu(Execution_Target::CPU);
            Tensor dx_cpu(Execution_Target::CPU), dg_cpu(Execution_Target::CPU);
            ops::normalization::rmsNormForward(x_cpu, g_cpu, inv_cpu, out_cpu, 1e-5f);
            ops::normalization::rmsNormBackward(x_cpu, dy_cpu, g_cpu, inv_cpu, dg_cpu, dx_cpu, false);
            ops::normalization::rmsNormBackward(x_cpu, dy_cpu, g_cpu, inv_cpu, dg_cpu, dx_cpu, true);

            Tensor x_gpu(Shape{rc.batch, rc.dim}, x_vals, Execution_Target::VULKAN_GPU);
            Tensor dy_gpu(Shape{rc.batch, rc.dim}, dy_vals, Execution_Target::VULKAN_GPU);
            Tensor g_gpu(Shape{1, rc.dim}, g_vals, Execution_Target::VULKAN_GPU);
            if (use_fp16)
            {
                convertTensorToFp16InPlace(x_gpu);
                convertTensorToFp16InPlace(dy_gpu);
                convertTensorToFp16InPlace(g_gpu);
            }
            Tensor inv_gpu(Execution_Target::VULKAN_GPU), out_gpu(Execution_Target::VULKAN_GPU);
            Tensor dx_gpu(Execution_Target::VULKAN_GPU), dg_gpu(Execution_Target::VULKAN_GPU);
            ops::normalization::rmsNormForward(x_gpu, g_gpu, inv_gpu, out_gpu, 1e-5f);
            ops::normalization::rmsNormBackward(x_gpu, dy_gpu, g_gpu, inv_gpu, dg_gpu, dx_gpu, false);
            ops::normalization::rmsNormBackward(x_gpu, dy_gpu, g_gpu, inv_gpu, dg_gpu, dx_gpu, true);
            flushGpuEngine(true);

            if (!verifyMatrix(out_gpu, out_cpu.getData(), tol_fwd))
            {
                return failStep(use_fp16 ? "Phase1 RMSNorm FP16 forward mismatch" : "Phase1 RMSNorm FP32 forward mismatch");
            }
            if (!verifyMatrix(dx_gpu, dx_cpu.getData(), tol_dx))
            {
                return failStep(use_fp16 ? "Phase1 RMSNorm FP16 backward dx mismatch" : "Phase1 RMSNorm FP32 backward dx mismatch");
            }
            if (!verifyMatrix(dg_gpu, dg_cpu.getData(), tol_dg))
            {
                return failStep(use_fp16 ? "Phase1 RMSNorm FP16 gamma_grad accumulate mismatch" : "Phase1 RMSNorm FP32 gamma_grad accumulate mismatch");
            }
        }
    }

    {
        constexpr size_t N = 1539;
        std::vector<float> p_init(N), g_init(N), m_init(N, 0.0f), v_init(N, 0.0f);
        for (size_t i = 0; i < N; ++i)
        {
            p_init[i] = 0.25f * static_cast<float>(static_cast<int>(i % 11) - 5);
            g_init[i] = 0.04f * static_cast<float>(static_cast<int>((i * 5) % 13) - 6);
        }

        for (bool use_fp16 : {false, true})
        {
            engine.setCooperativeMatrixEnabled(use_fp16 ? prev_coop : false);
            Tensor p_cpu(Shape{1, N}, p_init, Execution_Target::CPU);
            Tensor g_cpu(Shape{1, N}, g_init, Execution_Target::CPU);
            Tensor m_cpu(Shape{1, N}, m_init, Execution_Target::CPU);
            Tensor v_cpu(Shape{1, N}, v_init, Execution_Target::CPU);
            ops::optimizer::adamStep(p_cpu, g_cpu, m_cpu, v_cpu, 1e-2f, 0.9f, 0.999f, 1e-8f, 3, 1.0f, 1.0f, 0.01f);

            Tensor p_gpu(Shape{1, N}, p_init, Execution_Target::VULKAN_GPU);
            Tensor g_gpu(Shape{1, N}, g_init, Execution_Target::VULKAN_GPU);
            Tensor m_gpu(Shape{1, N}, m_init, Execution_Target::VULKAN_GPU);
            Tensor v_gpu(Shape{1, N}, v_init, Execution_Target::VULKAN_GPU);
            ops::optimizer::adamStep(p_gpu, g_gpu, m_gpu, v_gpu, 1e-2f, 0.9f, 0.999f, 1e-8f, 3, 1.0f, 1.0f, 0.01f);
            flushGpuEngine(true);

            if (!verifyMatrix(p_gpu, p_cpu.getData(), 1e-4f))
            {
                return failStep(use_fp16 ? "Phase1 AdamW FP16 boundary parity mismatch" : "Phase1 AdamW FP32 boundary parity mismatch");
            }
        }

        engine.setCooperativeMatrixEnabled(prev_coop);
        std::vector<float> g_ovf = g_init;
        g_ovf[257] = std::numeric_limits<float>::infinity();
        Tensor p_gpu(Shape{1, N}, p_init, Execution_Target::VULKAN_GPU);
        Tensor g_gpu(Shape{1, N}, g_ovf, Execution_Target::VULKAN_GPU);
        Adam_Optimizer opt_ovf(1e-2f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.0f);
        opt_ovf.step({{&p_gpu, &g_gpu}}, 1.0f);
        engine.executeGraph();
        engine.waitIdle();
        if (!opt_ovf.consumeGpuOverflowDetected())
        {
            return failStep("Phase1 AdamW single-element overflow flag not set");
        }
        const auto p_after = p_gpu.getData();
        if (!nearlyEqual(p_after[257], p_init[257], 1e-6f))
        {
            return failStep("Phase1 AdamW overflowed element was modified");
        }
        if (nearlyEqual(p_after[1], p_init[1], 1e-6f) || nearlyEqual(p_after[513], p_init[513], 1e-6f))
        {
            return failStep("Phase1 AdamW finite sibling elements were not updated");
        }
    }

    {
        constexpr size_t N = 1539;
        std::vector<float> a_vals(N), b_vals(N), dy_vals(N);
        for (size_t i = 0; i < N; ++i)
        {
            a_vals[i] = 0.2f * static_cast<float>(static_cast<int>(i % 17) - 8);
            b_vals[i] = 0.15f * static_cast<float>(static_cast<int>((i * 3) % 15) - 7);
            dy_vals[i] = 0.1f * static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
        }

        Tensor a_cpu(Shape{1, N}, a_vals, Execution_Target::CPU);
        Tensor b_cpu(Shape{1, N}, b_vals, Execution_Target::CPU);
        Tensor dy_cpu(Shape{1, N}, dy_vals, Execution_Target::CPU);
        Tensor y_cpu(Execution_Target::CPU), da_cpu(Execution_Target::CPU), db_cpu(Execution_Target::CPU);
        ops::llm::swigluForward(a_cpu, b_cpu, y_cpu);
        ops::llm::swigluBackward(a_cpu, b_cpu, dy_cpu, da_cpu, db_cpu);

        for (bool use_fp16 : {false, true})
        {
            engine.setCooperativeMatrixEnabled(use_fp16 ? prev_coop : false);
            const float tol = use_fp16 ? 2e-2f : 1e-4f;
            Tensor a_gpu(Shape{1, N}, a_vals, Execution_Target::VULKAN_GPU);
            Tensor b_gpu(Shape{1, N}, b_vals, Execution_Target::VULKAN_GPU);
            Tensor dy_gpu(Shape{1, N}, dy_vals, Execution_Target::VULKAN_GPU);
            if (use_fp16)
            {
                convertTensorToFp16InPlace(a_gpu);
                convertTensorToFp16InPlace(b_gpu);
                convertTensorToFp16InPlace(dy_gpu);
            }
            Tensor y_gpu(Execution_Target::VULKAN_GPU), da_gpu(Execution_Target::VULKAN_GPU), db_gpu(Execution_Target::VULKAN_GPU);
            ops::llm::swigluForward(a_gpu, b_gpu, y_gpu);
            ops::llm::swigluBackward(a_gpu, b_gpu, dy_gpu, da_gpu, db_gpu);
            flushGpuEngine(true);

            if (!verifyMatrix(y_gpu, y_cpu.getData(), tol) ||
                !verifyMatrix(da_gpu, da_cpu.getData(), tol) ||
                !verifyMatrix(db_gpu, db_cpu.getData(), tol))
            {
                return failStep(use_fp16 ? "Phase1 SwiGLU FP16 boundary parity mismatch" : "Phase1 SwiGLU FP32 boundary parity mismatch");
            }
        }
    }

    return true;
}

bool testPhaseBCoopMatGemm()
{
    auto failStep = [](const char *name) -> bool {
        return failTestStep("PhaseBCoopMatGemm", name);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    if (!engine.isCooperativeMatrixSupported())
    {
        return true;
    }

    Scoped_Engine_Flags_Guard guard;
    engine.setCooperativeMatrixEnabled(true);

    struct GemmCase { size_t M; size_t K; size_t N; float tol_fp16; };
    const GemmCase cases[] = {
        {128, 256, 192, 6e-2f},
        {130, 128, 128, 5e-2f},
        {67,  71,  73,  5e-2f},
        {5,   7,   9,   2e-2f}
    };

    for (const auto &gc : cases)
    {
        std::vector<float> x_vals(gc.M * gc.K);
        std::vector<float> w_vals(gc.K * gc.N);
        std::vector<float> b1_vals(gc.N);
        std::vector<float> b2_vals(gc.M * gc.N);
        std::vector<float> dy_vals(gc.M * gc.N);

        for (size_t i = 0; i < x_vals.size(); ++i)
            x_vals[i] = 0.04f * static_cast<float>(static_cast<int>(i % 13) - 6);
        for (size_t i = 0; i < w_vals.size(); ++i)
            w_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5) % 11) - 5);
        for (size_t i = 0; i < b1_vals.size(); ++i)
            b1_vals[i] = 0.02f * static_cast<float>(static_cast<int>(i % 9) - 4);
        for (size_t i = 0; i < b2_vals.size(); ++i)
            b2_vals[i] = 0.015f * static_cast<float>(static_cast<int>((i * 3) % 7) - 3);
        for (size_t i = 0; i < dy_vals.size(); ++i)
            dy_vals[i] = 0.035f * static_cast<float>(static_cast<int>((i * 7) % 15) - 7);

        Tensor x_cpu(Shape{gc.M, gc.K}, x_vals, Execution_Target::CPU);
        Tensor w_cpu(Shape{gc.K, gc.N}, w_vals, Execution_Target::CPU);
        Tensor b1_cpu(Shape{1, gc.N}, b1_vals, Execution_Target::CPU);
        Tensor b2_cpu(Shape{gc.M, gc.N}, b2_vals, Execution_Target::CPU);
        Tensor dy_cpu(Shape{gc.M, gc.N}, dy_vals, Execution_Target::CPU);

        Tensor mm_cpu(Execution_Target::CPU);
        x_cpu.matmul(w_cpu, mm_cpu);
        Tensor y1_cpu(Execution_Target::CPU), y2_cpu(Execution_Target::CPU);
        ops::linear::linearForward(x_cpu, w_cpu, b1_cpu, y1_cpu);
        y2_cpu = x_cpu.matmulAdd(w_cpu, b2_cpu);
        Tensor dx_cpu(Execution_Target::CPU), dw_cpu(Execution_Target::CPU), db_cpu(Execution_Target::CPU);
        ops::linear::linearBackwardInput(dy_cpu, w_cpu, dx_cpu);
        ops::linear::linearBackwardWeightBias(x_cpu, dy_cpu, dw_cpu, db_cpu, false);
        ops::linear::linearBackwardWeightBias(x_cpu, dy_cpu, dw_cpu, db_cpu, true);

        Tensor x16 = castTensorToFp16(Tensor(Shape{gc.M, gc.K}, x_vals, Execution_Target::VULKAN_GPU));
        Tensor w16 = castTensorToFp16(Tensor(Shape{gc.K, gc.N}, w_vals, Execution_Target::VULKAN_GPU));
        Tensor b1_16 = castTensorToFp16(Tensor(Shape{1, gc.N}, b1_vals, Execution_Target::VULKAN_GPU));
        Tensor b2_16 = castTensorToFp16(Tensor(Shape{gc.M, gc.N}, b2_vals, Execution_Target::VULKAN_GPU));
        Tensor dy16 = castTensorToFp16(Tensor(Shape{gc.M, gc.N}, dy_vals, Execution_Target::VULKAN_GPU));

        Tensor mm_gpu(Execution_Target::VULKAN_GPU);
        x16.matmul(w16, mm_gpu);
        Tensor y1_gpu(Execution_Target::VULKAN_GPU), y2_gpu(Execution_Target::VULKAN_GPU);
        ops::linear::linearForward(x16, w16, b1_16, y1_gpu);
        y2_gpu = x16.matmulAdd(w16, b2_16);
        Tensor dx_gpu(Execution_Target::VULKAN_GPU), dw_gpu(Execution_Target::VULKAN_GPU), db_gpu(Execution_Target::VULKAN_GPU);
        ops::linear::linearBackwardInput(dy16, w16, dx_gpu);
        ops::linear::linearBackwardWeightBias(x16, dy16, dw_gpu, db_gpu, false);
        ops::linear::linearBackwardWeightBias(x16, dy16, dw_gpu, db_gpu, true);

        flushGpuEngine(true);

        if (gc.K % 8 == 0 && gc.N % 8 == 0)
        {
            if (!verifyMatrix(mm_gpu, mm_cpu.getData(), gc.tol_fp16))
            {
                return failStep("PhaseB 2D matmul CoopMat mismatch");
            }
        }
        if (!verifyMatrix(y1_gpu, y1_cpu.getData(), gc.tol_fp16))
        {
            return failStep("PhaseB linearForward 1D bias CoopMat mismatch");
        }
        if (!verifyMatrix(y2_gpu, y2_cpu.getData(), gc.tol_fp16))
        {
            return failStep("PhaseB matmulAdd 2D bias CoopMat mismatch");
        }
        if (!verifyMatrix(dx_gpu, dx_cpu.getData(), gc.tol_fp16))
        {
            return failStep("PhaseB linearBackwardInput CoopMat mismatch");
        }
        if (!verifyMatrix(dw_gpu, dw_cpu.getData(), gc.tol_fp16 * 2.0f))
        {
            return failStep("PhaseB linearBackwardWeightBias dW (accumulate=true) CoopMat mismatch");
        }
        if (!verifyMatrix(db_gpu, db_cpu.getData(), gc.tol_fp16 * 2.0f))
        {
            return failStep("PhaseB linearBackwardWeightBias dB (accumulate=true) CoopMat mismatch");
        }
    }

    {
        constexpr size_t B = 2, M = 70, K = 64, N = 72;
        std::vector<float> a_vals(B * M * K), w_vals(B * K * N), b1_vals(N), b3_vals(B * M * N);
        for (size_t i = 0; i < a_vals.size(); ++i) a_vals[i] = 0.03f * static_cast<float>(static_cast<int>(i % 11) - 5);
        for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = 0.025f * static_cast<float>(static_cast<int>((i * 5) % 13) - 6);
        for (size_t i = 0; i < b1_vals.size(); ++i) b1_vals[i] = 0.04f * static_cast<float>(static_cast<int>(i % 7) - 3);
        for (size_t i = 0; i < b3_vals.size(); ++i) b3_vals[i] = 0.02f * static_cast<float>(static_cast<int>((i * 3) % 9) - 4);

        Tensor a_cpu(Shape{B, M, K}, a_vals, Execution_Target::CPU);
        Tensor w_cpu(Shape{B, K, N}, w_vals, Execution_Target::CPU);
        Tensor b1_cpu(Shape{1, N}, b1_vals, Execution_Target::CPU);
        Tensor b3_cpu(Shape{B, M, N}, b3_vals, Execution_Target::CPU);

        Tensor mm_cpu(Execution_Target::CPU);
        a_cpu.matmul(w_cpu, mm_cpu);
        Tensor y1_cpu = a_cpu.matmulAdd(w_cpu, b1_cpu);
        Tensor y3_cpu = a_cpu.matmulAdd(w_cpu, b3_cpu);

        Tensor a16 = castTensorToFp16(Tensor(Shape{B, M, K}, a_vals, Execution_Target::VULKAN_GPU));
        Tensor w16 = castTensorToFp16(Tensor(Shape{B, K, N}, w_vals, Execution_Target::VULKAN_GPU));
        Tensor b1_16 = castTensorToFp16(Tensor(Shape{1, N}, b1_vals, Execution_Target::VULKAN_GPU));
        Tensor b3_16 = castTensorToFp16(Tensor(Shape{B, M, N}, b3_vals, Execution_Target::VULKAN_GPU));

        Tensor mm_gpu(Execution_Target::VULKAN_GPU);
        a16.matmul(w16, mm_gpu);
        Tensor y1_gpu = a16.matmulAdd(w16, b1_16);
        Tensor y3_gpu = a16.matmulAdd(w16, b3_16);
        flushGpuEngine(true);

        if (!verifyMatrix(mm_gpu, mm_cpu.getData(), 5e-2f) ||
            !verifyMatrix(y1_gpu, y1_cpu.getData(), 5e-2f) ||
            !verifyMatrix(y3_gpu, y3_cpu.getData(), 5e-2f))
        {
            return failStep("PhaseB 3D Batched matmul/matmulAdd CoopMat mismatch");
        }
    }

    for (const auto &fc : {GemmCase{128, 128, 128, 2e-3f}, GemmCase{70, 67, 71, 2e-3f}})
    {
        std::vector<float> x_vals(fc.M * fc.K), dy_vals(fc.M * fc.N), w_vals(fc.K * fc.N), b_vals(fc.N);
        for (size_t i = 0; i < x_vals.size(); ++i) x_vals[i] = 0.05f * static_cast<float>(static_cast<int>(i % 9) - 4);
        for (size_t i = 0; i < dy_vals.size(); ++i) dy_vals[i] = 0.04f * static_cast<float>(static_cast<int>((i * 5) % 11) - 5);
        for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = 0.1f * static_cast<float>(static_cast<int>((i * 3) % 7) - 3);
        for (size_t i = 0; i < b_vals.size(); ++i) b_vals[i] = 0.05f * static_cast<float>(static_cast<int>(i % 5) - 2);

        engine.enableFusedGemmAdam(false);
        Linear_Layer lin_ref(fc.K, fc.N, Execution_Target::VULKAN_GPU);
        lin_ref.setMixedPrecision(true);
        lin_ref.setWeights(Tensor(Shape{fc.K, fc.N}, w_vals, Execution_Target::VULKAN_GPU));
        lin_ref.setBiases(Tensor(Shape{1, fc.N}, b_vals, Execution_Target::VULKAN_GPU));
        Adam_Optimizer opt_ref(1e-2f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        Tensor x_ref(Shape{fc.M, fc.K}, x_vals, Execution_Target::VULKAN_GPU);
        Tensor dy_ref(Shape{fc.M, fc.N}, dy_vals, Execution_Target::VULKAN_GPU);
        lin_ref.forward(x_ref);
        lin_ref.backward(dy_ref);
        opt_ref.step(lin_ref.getParametersAndGradients(), 1.0f);
        flushGpuEngine(true);

        engine.enableFusedGemmAdam(true);
        Linear_Layer lin_fused(fc.K, fc.N, Execution_Target::VULKAN_GPU);
        lin_fused.setMixedPrecision(true);
        lin_fused.setWeights(Tensor(Shape{fc.K, fc.N}, w_vals, Execution_Target::VULKAN_GPU));
        lin_fused.setBiases(Tensor(Shape{1, fc.N}, b_vals, Execution_Target::VULKAN_GPU));
        Adam_Optimizer opt_fused(1e-2f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        Tensor x_fused(Shape{fc.M, fc.K}, x_vals, Execution_Target::VULKAN_GPU);
        Tensor dy_fused(Shape{fc.M, fc.N}, dy_vals, Execution_Target::VULKAN_GPU);
        lin_fused.forward(x_fused);
        lin_fused.backward(dy_fused);
        opt_fused.step(lin_fused.getParametersAndGradients(), 1.0f);
        flushGpuEngine(true);

        if (!verifyMatrix(lin_fused.getWeights(), lin_ref.getWeights().getData(), fc.tol_fp16) ||
            !verifyMatrix(lin_fused.getBiases(), lin_ref.getBiases().getData(), fc.tol_fp16))
        {
            return failStep("PhaseB Fused GEMM+AdamW parity vs unfused mismatch");
        }

        const auto w_before_nan = lin_fused.getWeights().getData();
        Tensor dy_nan(Shape{fc.M, fc.N}, std::vector<float>(fc.M * fc.N, std::numeric_limits<float>::quiet_NaN()), Execution_Target::VULKAN_GPU);
        lin_fused.forward(x_fused);
        lin_fused.backward(dy_nan);
        opt_fused.step(lin_fused.getParametersAndGradients(), 1.0f);
        flushGpuEngine(true);

        if (!opt_fused.consumeGpuOverflowDetected())
        {
            return failStep("PhaseB Fused GEMM+AdamW did not report GPU overflow_detected");
        }
        if (!verifyVectorsClose(w_before_nan, lin_fused.getWeights().getData(), 1e-6f))
        {
            return failStep("PhaseB Fused GEMM+AdamW drifted weight on NaN gradient");
        }
    }

    return true;
}

bool testPhaseCFlashAttention2()
{
    auto failStep = [](const char *name) -> bool {
        return failTestStep("PhaseCFlashAttention2", name);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    if (!engine.isCooperativeMatrixSupported())
    {
        return true;
    }

    Scoped_Engine_Flags_Guard guard;
    engine.setCooperativeMatrixEnabled(true);
    engine.enableFusedGemmAdam(false);

    struct AttnCase
    {
        uint32_t B;
        uint32_t H;
        uint32_t H_kv;
        uint32_t S;
        uint32_t D;
        bool is_causal;
        bool is_bshd;
        bool recompute_l;
    };

    const std::vector<AttnCase> cases = {
        {1, 2, 2,  19, 16, false, false, false},
        {1, 2, 2,  19, 16, true,  true,  false},
        {2, 4, 4,  70, 32, true,  false, false},
        {2, 8, 2,  70, 32, true,  true,  false},
        {2, 8, 2,  70, 32, true,  false, false},
        {2, 8, 2,  70, 32, true,  true,  true},
        {2, 8, 2,  70, 32, true,  false, true},
        {1, 2, 2, 128, 64, true,  true,  false},
        {1, 2, 2,  70, 64, false, false, false},
    };

    for (size_t ci = 0; ci < cases.size(); ++ci)
    {
        const auto &tc = cases[ci];
        Shape q_shape = tc.is_bshd ? Shape{tc.B, tc.S, tc.H, tc.D} : Shape{tc.B, tc.H, tc.S, tc.D};
        Shape kv_shape = tc.is_bshd ? Shape{tc.B, tc.S, tc.H_kv, tc.D} : Shape{tc.B, tc.H_kv, tc.S, tc.D};

        std::vector<float> q_vals(q_shape.getTotalElements());
        std::vector<float> k_vals(kv_shape.getTotalElements());
        std::vector<float> v_vals(kv_shape.getTotalElements());
        std::vector<float> do_vals(q_shape.getTotalElements());

        for (size_t i = 0; i < q_vals.size(); ++i)
        {
            q_vals[i] = 0.08f * static_cast<float>(static_cast<int>((i * 7 + ci) % 11) - 5);
            do_vals[i] = 0.07f * static_cast<float>(static_cast<int>((i * 5 + 3) % 9) - 4);
        }
        for (size_t i = 0; i < k_vals.size(); ++i)
        {
            k_vals[i] = 0.08f * static_cast<float>(static_cast<int>((i * 3 + 2) % 11) - 5);
            v_vals[i] = 0.08f * static_cast<float>(static_cast<int>((i * 11 + 1) % 9) - 4);
        }

        Tensor q_cpu(q_shape, q_vals, Execution_Target::CPU);
        Tensor k_cpu(kv_shape, k_vals, Execution_Target::CPU);
        Tensor v_cpu(kv_shape, v_vals, Execution_Target::CPU);
        Tensor do_cpu(q_shape, do_vals, Execution_Target::CPU);

        ops::llm::Attention_Config cpu_cfg{};
        cpu_cfg.num_heads = tc.H;
        cpu_cfg.num_kv_heads = tc.H_kv;
        cpu_cfg.seq_len = tc.S;
        cpu_cfg.head_dim = tc.D;
        cpu_cfg.is_causal = tc.is_causal;
        cpu_cfg.is_bshd = tc.is_bshd;

        Tensor o_cpu(Execution_Target::CPU), l_cpu(Execution_Target::CPU);
        Tensor dq_cpu(Execution_Target::CPU), dk_cpu(Execution_Target::CPU), dv_cpu(Execution_Target::CPU);
        ops::llm::flashAttentionForward(q_cpu, k_cpu, v_cpu, o_cpu, cpu_cfg, &l_cpu);
        ops::llm::flashAttentionBackward(q_cpu, k_cpu, v_cpu, o_cpu, do_cpu, dq_cpu, dk_cpu, dv_cpu, cpu_cfg, &l_cpu);

        Tensor q_gpu = castTensorToFp16(Tensor(q_shape, q_vals, Execution_Target::VULKAN_GPU));
        Tensor k_gpu = castTensorToFp16(Tensor(kv_shape, k_vals, Execution_Target::VULKAN_GPU));
        Tensor v_gpu = castTensorToFp16(Tensor(kv_shape, v_vals, Execution_Target::VULKAN_GPU));
        Tensor do_gpu = castTensorToFp16(Tensor(q_shape, do_vals, Execution_Target::VULKAN_GPU));

        ops::llm::Attention_Config gpu_cfg = cpu_cfg;
        Tensor o_gpu(Execution_Target::VULKAN_GPU), l_gpu(Execution_Target::VULKAN_GPU);
        Tensor dq_gpu(Execution_Target::VULKAN_GPU), dk_gpu(Execution_Target::VULKAN_GPU), dv_gpu(Execution_Target::VULKAN_GPU);

        ops::llm::flashAttentionForward(q_gpu, k_gpu, v_gpu, o_gpu, gpu_cfg, &l_gpu);
        ops::llm::flashAttentionBackward(q_gpu, k_gpu, v_gpu, o_gpu, do_gpu,
                                         dq_gpu, dk_gpu, dv_gpu, gpu_cfg,
                                         tc.recompute_l ? nullptr : &l_gpu);
        flushGpuEngine(true);

        if (o_gpu.getShape() != q_shape ||
            dq_gpu.getShape() != q_shape ||
            dk_gpu.getShape() != kv_shape ||
            dv_gpu.getShape() != kv_shape)
        {
            return failStep("PhaseC FlashAttention2 output/gradient shape mismatch");
        }
        if (!verifyMatrix(o_gpu, o_cpu.getData(), 2e-2f) ||
            !verifyMatrix(l_gpu, l_cpu.getData(), 2e-2f))
        {
            return failStep("PhaseC FlashAttention2 forward O/L FP16 parity mismatch");
        }
        if (!verifyMatrix(dq_gpu, dq_cpu.getData(), 5e-2f) ||
            !verifyMatrix(dk_gpu, dk_cpu.getData(), 5e-2f) ||
            !verifyMatrix(dv_gpu, dv_cpu.getData(), 5e-2f))
        {
            return failStep("PhaseC FlashAttention2 backward dQ/dK/dV FP16 parity mismatch");
        }
    }

    {
        Shape rope_shape{2, 70, 4, 32};
        std::vector<float> x_vals(rope_shape.getTotalElements());
        for (size_t i = 0; i < x_vals.size(); ++i)
        {
            x_vals[i] = 0.05f * static_cast<float>(static_cast<int>((i * 13 + 7) % 17) - 8);
        }

        Tensor x_cpu(rope_shape, x_vals, Execution_Target::CPU);
        Tensor y_cpu(Execution_Target::CPU), x_rec_cpu(Execution_Target::CPU);
        ops::llm::RoPE_Config rope_fwd{70, 32, ops::llm::RoPE_Direction::FORWARD, 10000.0f, 4, ops::llm::RoPE_Mode::BSHD_INPLACE, 5u, 0u, 1.0f, nullptr};
        ops::llm::RoPE_Config rope_bwd{70, 32, ops::llm::RoPE_Direction::BACKWARD, 10000.0f, 4, ops::llm::RoPE_Mode::BSHD_INPLACE, 5u, 0u, 1.0f, nullptr};
        ops::llm::applyRoPE(x_cpu, y_cpu, rope_fwd);
        ops::llm::applyRoPE(y_cpu, x_rec_cpu, rope_bwd);

        Tensor x_gpu = castTensorToFp16(Tensor(rope_shape, x_vals, Execution_Target::VULKAN_GPU));
        Tensor y_gpu(Execution_Target::VULKAN_GPU), x_rec_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::applyRoPE(x_gpu, y_gpu, rope_fwd);
        ops::llm::applyRoPE(y_gpu, x_rec_gpu, rope_bwd);
        flushGpuEngine(true);

        if (y_gpu.getShape() != rope_shape || x_rec_gpu.getShape() != rope_shape ||
            !verifyMatrix(y_gpu, y_cpu.getData(), 1e-2f) ||
            !verifyMatrix(x_rec_gpu, x_rec_cpu.getData(), 1e-2f) ||
            !verifyMatrix(x_rec_gpu, x_vals, 1e-2f))
        {
            return failStep("PhaseC RoPE_Mode::BSHD_INPLACE forward/backward parity mismatch");
        }
    }

    {
        constexpr size_t B = 2;
        constexpr size_t S = 70;
        constexpr size_t hidden_dim = 128;
        constexpr size_t num_heads = 4;
        constexpr size_t num_kv_heads = 2;
        constexpr size_t intermediate_dim = 256;

        Transformer_Block tb_cpu(hidden_dim, num_heads, intermediate_dim, 1e-5f, 10000.0f,
                                 Execution_Target::CPU, Data_Type::FLOAT32, 1, num_kv_heads);
        Transformer_Block tb_gpu(hidden_dim, num_heads, intermediate_dim, 1e-5f, 10000.0f,
                                 Execution_Target::VULKAN_GPU, Data_Type::FLOAT32, 1, num_kv_heads);

        auto cpu_params = tb_cpu.getParametersAndGradients();
        auto gpu_params = tb_gpu.getParametersAndGradients();
        if (cpu_params.size() != gpu_params.size())
        {
            return failStep("PhaseC Transformer_Block parameter count mismatch");
        }
        for (size_t i = 0; i < cpu_params.size(); ++i)
        {
            *gpu_params[i].first = Tensor(cpu_params[i].first->getShape(),
                                          cpu_params[i].first->getData(),
                                          Execution_Target::VULKAN_GPU);
        }
        tb_gpu.setMixedPrecision(true);

        std::vector<float> x_vals(B * S * hidden_dim);
        std::vector<float> dy_vals(B * S * hidden_dim);
        for (size_t i = 0; i < x_vals.size(); ++i)
        {
            x_vals[i] = 0.04f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
            dy_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5 + 1) % 11) - 5);
        }

        Tensor x_cpu(Shape{B, S, hidden_dim}, x_vals, Execution_Target::CPU);
        Tensor dy_cpu(Shape{B, S, hidden_dim}, dy_vals, Execution_Target::CPU);
        Tensor out_cpu = tb_cpu.forward(x_cpu);
        Tensor dx_cpu = tb_cpu.backward(dy_cpu);

        Tensor x_gpu(Shape{B, S, hidden_dim}, x_vals, Execution_Target::VULKAN_GPU);
        Tensor dy_gpu(Shape{B, S, hidden_dim}, dy_vals, Execution_Target::VULKAN_GPU);
        Tensor out_gpu = tb_gpu.forward(x_gpu);
        Tensor dx_gpu = tb_gpu.backward(dy_gpu);
        flushGpuEngine(true);

        if (out_gpu.getShape() != Shape{B, S, hidden_dim} || dx_gpu.getShape() != Shape{B, S, hidden_dim} ||
            !verifyMatrix(out_gpu, out_cpu.getData(), 5e-2f) ||
            !verifyMatrix(dx_gpu, dx_cpu.getData(), 5e-2f))
        {
            return failStep("PhaseC Transformer_Block BSHD FP16 forward/backward mismatch");
        }
        for (size_t i = 0; i < cpu_params.size(); ++i)
        {
            if (cpu_params[i].second && gpu_params[i].second &&
                !verifyMatrix(*gpu_params[i].second, cpu_params[i].second->getData(), 8e-2f))
            {
                return failStep("PhaseC Transformer_Block BSHD parameter gradient mismatch");
            }
        }
    }

    return true;
}

bool testPhaseDOptimizations()
{
    auto failStep = [](const char *name) -> bool {
        return failTestStep("PhaseDOptimizations", name);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    if (!engine.isCooperativeMatrixSupported())
    {
        return true;
    }

    Scoped_Engine_Flags_Guard guard;
    struct AsyncStaticGuard
    {
        Execution_Engine &eng;
        bool async_l;
        ~AsyncStaticGuard()
        {
            eng.invalidateStaticGraph();
            eng.setAsyncLossEnabled(async_l);
        }
    } async_guard{engine, engine.isAsyncLossEnabled()};

    Scoped_Engine_Flags_Guard flags_guard;
    engine.invalidateStaticGraph();
    engine.setCooperativeMatrixEnabled(true);
    engine.enableFusedGemmAdam(false);
    engine.enableStaticGraph(false);

    {
        const std::vector<size_t> tail_sizes = {1, 3, 7, 13, 256, 259};
        for (size_t n : tail_sizes)
        {
            std::vector<float> p_vals(n), g_vals(n), m_vals(n), v_vals(n);
            for (size_t i = 0; i < n; ++i)
            {
                p_vals[i] = 0.05f * static_cast<float>(static_cast<int>(i % 11) - 5);
                g_vals[i] = 0.25f * static_cast<float>(static_cast<int>((i * 7 + 1) % 9) - 4);
                m_vals[i] = 0.01f * static_cast<float>(static_cast<int>(i % 5) - 2);
                v_vals[i] = 0.02f * static_cast<float>((i % 4) + 1);
            }

            Tensor p_cpu(Shape{1, n}, p_vals, Execution_Target::CPU);
            Tensor g_cpu(Shape{1, n}, g_vals, Execution_Target::CPU);
            Tensor m_cpu(Shape{1, n}, m_vals, Execution_Target::CPU);
            Tensor v_cpu(Shape{1, n}, v_vals, Execution_Target::CPU);
            ops::optimizer::adamStep(p_cpu, g_cpu, m_cpu, v_cpu, 1e-2f, 0.9f, 0.999f, 1e-8f, 3, 0.5f, 1.0f, 0.01f);

            Tensor p_gpu(Shape{1, n}, p_vals, Execution_Target::VULKAN_GPU);
            Tensor g_gpu(Shape{1, n}, g_vals, Execution_Target::VULKAN_GPU);
            Tensor m_gpu(Shape{1, n}, m_vals, Execution_Target::VULKAN_GPU);
            Tensor v_gpu(Shape{1, n}, v_vals, Execution_Target::VULKAN_GPU);
            p_gpu.prewarmFp16Cache();
            engine.executeGraph();

            ops::optimizer::adamStep(p_gpu, g_gpu, m_gpu, v_gpu, 1e-2f, 0.9f, 0.999f, 1e-8f, 3, 0.5f, 1.0f, 0.01f);
            flushGpuEngine(true);

            if (!verifyMatrix(p_gpu, p_cpu.getData(), 1e-4f) ||
                !verifyMatrix(m_gpu, m_cpu.getData(), 1e-4f) ||
                !verifyMatrix(v_gpu, v_cpu.getData(), 1e-4f))
            {
                return failStep("D1 vectorized adamStep CPU vs GPU parity mismatch");
            }
            const auto &p_gpu_impl = Gpu_Tensor_Impl::castToGpu(*p_gpu.getImplementation());
            if (!p_gpu_impl.getCachedFp16Storage() || p_gpu_impl.isFp16CacheDirty())
            {
                return failStep("D1 adamStep did not sync FP16 shadow cache");
            }
            std::vector<float> fp16_shadow(n, 0.0f);
            p_gpu_impl.getCachedFp16Storage()->downloadData(fp16_shadow);
            for (size_t i = 0; i < n; ++i)
            {
                if (!nearlyEqual(fp16_shadow[i], p_cpu.getData()[i], 2e-3f))
                {
                    return failStep("D1 adamStep FP16 shadow cache value mismatch");
                }
            }
        }

        const std::array<size_t, 4> m4_sizes = {13, 256, 7, 259};
        std::array<Tensor, 4> ps_cpu = {Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU)};
        std::array<Tensor, 4> gs_cpu = {Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU)};
        std::array<Tensor, 4> ms_cpu = {Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU)};
        std::array<Tensor, 4> vs_cpu = {Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU), Tensor(Execution_Target::CPU)};
        std::array<Tensor, 4> ps_gpu = {Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU)};
        std::array<Tensor, 4> gs_gpu = {Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU)};
        std::array<Tensor, 4> ms_gpu = {Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU)};
        std::array<Tensor, 4> vs_gpu = {Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU), Tensor(Execution_Target::VULKAN_GPU)};
        std::array<ops::optimizer::Adam_Multi_Item, 4> multi_items{};

        for (size_t s = 0; s < 4; ++s)
        {
            const size_t n = m4_sizes[s];
            std::vector<float> p_vals(n), g_vals(n), m_vals(n), v_vals(n);
            for (size_t i = 0; i < n; ++i)
            {
                p_vals[i] = 0.04f * static_cast<float>(static_cast<int>((i + s) % 9) - 4);
                g_vals[i] = 0.15f * static_cast<float>(static_cast<int>((i * 3 + s) % 7) - 3);
                m_vals[i] = 0.005f * static_cast<float>(static_cast<int>(i % 5) - 2);
                v_vals[i] = 0.01f * static_cast<float>((i % 3) + 1);
            }
            ps_cpu[s] = Tensor(Shape{1, n}, p_vals, Execution_Target::CPU);
            gs_cpu[s] = Tensor(Shape{1, n}, g_vals, Execution_Target::CPU);
            ms_cpu[s] = Tensor(Shape{1, n}, m_vals, Execution_Target::CPU);
            vs_cpu[s] = Tensor(Shape{1, n}, v_vals, Execution_Target::CPU);
            ops::optimizer::adamStep(ps_cpu[s], gs_cpu[s], ms_cpu[s], vs_cpu[s], 1e-2f, 0.9f, 0.999f, 1e-8f, 4, 0.5f, 1.0f, 0.02f);

            ps_gpu[s] = Tensor(Shape{1, n}, p_vals, Execution_Target::VULKAN_GPU);
            gs_gpu[s] = Tensor(Shape{1, n}, g_vals, Execution_Target::VULKAN_GPU);
            ms_gpu[s] = Tensor(Shape{1, n}, m_vals, Execution_Target::VULKAN_GPU);
            vs_gpu[s] = Tensor(Shape{1, n}, v_vals, Execution_Target::VULKAN_GPU);
            multi_items[s] = ops::optimizer::Adam_Multi_Item{&ps_gpu[s], &gs_gpu[s], &ms_gpu[s], &vs_gpu[s]};
        }

        const uint32_t frame = engine.getContext().getCurrentFrame();
        auto dyn_buf = engine.allocateDynamicOptimizerParamsBuffer(1e-2f, 1.0f / (1.0f - std::pow(0.9f, 4.0f)), 1.0f / std::sqrt(1.0f - std::pow(0.999f, 4.0f)), 1.0f, frame);
        ops::optimizer::adamStepMulti4Fp16(multi_items, 4, 1e-2f, 0.9f, 0.999f, 1e-8f, 4, 0.5f, 1.0f, 0.02f, dyn_buf);
        flushGpuEngine(true);

        for (size_t s = 0; s < 4; ++s)
        {
            if (!verifyMatrix(ps_gpu[s], ps_cpu[s].getData(), 1e-4f) ||
                !verifyMatrix(ms_gpu[s], ms_cpu[s].getData(), 1e-4f) ||
                !verifyMatrix(vs_gpu[s], vs_cpu[s].getData(), 1e-4f))
            {
                return failStep("D1 ADAM_UPDATE_MULTI4_FP16 parity mismatch");
            }
        }

        std::array<std::vector<float>, 4> before_nan;
        for (size_t s = 0; s < 4; ++s)
        {
            before_nan[s] = ps_gpu[s].getData();
        }
        std::vector<float> nan_grad(m4_sizes[2], 0.0f);
        nan_grad[m4_sizes[2] - 1] = std::numeric_limits<float>::quiet_NaN();
        gs_gpu[2].uploadData(nan_grad);
        const uint32_t nan_frame = engine.getContext().getCurrentFrame();
        auto nan_dyn = engine.allocateDynamicOptimizerParamsBuffer(1e-2f, 1.0f, 1.0f, 1.0f, nan_frame);
        ops::optimizer::adamStepMulti4Fp16(multi_items, 4, 1e-2f, 0.9f, 0.999f, 1e-8f, 5, 0.5f, 1.0f, 0.02f, nan_dyn);
        flushGpuEngine(true);

        if (!nan_dyn->isHostMapped())
        {
            return failStep("D1 MULTI4 dynamic buffer not host mapped");
        }
        Graph_Executor::Adam_Dynamic_Params snapshot{};
        std::memcpy(&snapshot, nan_dyn->getHostMappedPointer(), sizeof(snapshot));
        if (snapshot.overflow_detected == 0u)
        {
            return failStep("D1 MULTI4 did not set overflow_detected on NaN slot");
        }
        const std::vector<float> after_nan_2 = ps_gpu[2].getData();
        if (after_nan_2.size() != before_nan[2].size())
        {
            return failStep("D1 MULTI4 NaN slot size mismatch");
        }
        for (float v : after_nan_2)
        {
            if (!std::isfinite(v))
            {
                return failStep("D1 MULTI4 produced non-finite weight on partial-NaN slot");
            }
        }
        if (std::fabs(after_nan_2[m4_sizes[2] - 1] - before_nan[2][m4_sizes[2] - 1]) > 1e-6f)
        {
            return failStep("D1 MULTI4 modified weight at NaN element");
        }
        for (size_t s : {size_t{0}, size_t{1}, size_t{3}})
        {
            const std::vector<float> after_other = ps_gpu[s].getData();
            if (after_other.size() != m4_sizes[s])
            {
                return failStep("D1 MULTI4 non-NaN slot size mismatch");
            }
            for (float v : after_other)
            {
                if (!std::isfinite(v))
                {
                    return failStep("D1 MULTI4 produced non-finite weight on finite slot");
                }
            }
        }

        const std::vector<float> before_full_nan_2 = after_nan_2;
        std::fill(nan_grad.begin(), nan_grad.end(), std::numeric_limits<float>::quiet_NaN());
        gs_gpu[2].uploadData(nan_grad);
        const uint32_t full_nan_frame = engine.getContext().getCurrentFrame();
        auto full_nan_dyn = engine.allocateDynamicOptimizerParamsBuffer(1e-2f, 1.0f, 1.0f, 1.0f, full_nan_frame);
        ops::optimizer::adamStepMulti4Fp16(multi_items, 4, 1e-2f, 0.9f, 0.999f, 1e-8f, 6, 0.5f, 1.0f, 0.02f, full_nan_dyn);
        flushGpuEngine(true);

        Graph_Executor::Adam_Dynamic_Params full_snapshot{};
        std::memcpy(&full_snapshot, full_nan_dyn->getHostMappedPointer(), sizeof(full_snapshot));
        if (full_snapshot.overflow_detected == 0u)
        {
            return failStep("D1 MULTI4 did not set overflow_detected on full-NaN slot");
        }
        if (!verifyMatrix(ps_gpu[2], before_full_nan_2, 1e-6f))
        {
            return failStep("D1 MULTI4 modified weights on full-NaN slot");
        }
    }

    {
        constexpr size_t M = 64, K = 128, N = 128;
        std::vector<float> w_init(K * N), b_init(N);
        for (size_t i = 0; i < w_init.size(); ++i) w_init[i] = 0.05f * static_cast<float>(static_cast<int>(i % 9) - 4);
        for (size_t i = 0; i < b_init.size(); ++i) b_init[i] = 0.02f * static_cast<float>(static_cast<int>(i % 7) - 3);

        Linear_Layer lin_unfused(K, N, Execution_Target::VULKAN_GPU);
        Linear_Layer lin_fused(K, N, Execution_Target::VULKAN_GPU);
        lin_unfused.setMixedPrecision(true);
        lin_fused.setMixedPrecision(true);
        lin_unfused.setWeights(Tensor(Shape{K, N}, w_init, Execution_Target::VULKAN_GPU));
        lin_unfused.setBiases(Tensor(Shape{1, N}, b_init, Execution_Target::VULKAN_GPU));
        lin_fused.setWeights(Tensor(Shape{K, N}, w_init, Execution_Target::VULKAN_GPU));
        lin_fused.setBiases(Tensor(Shape{1, N}, b_init, Execution_Target::VULKAN_GPU));

        Adam_Optimizer opt_unfused(5e-3f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        Adam_Optimizer opt_fused(5e-3f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        Tensor x_g(Shape{M, K}, Execution_Target::VULKAN_GPU);
        Tensor dy_g(Shape{M, N}, Execution_Target::VULKAN_GPU);
        std::vector<float> xv(M * K), dyv(M * N);

        for (int step = 0; step < 3; ++step)
        {
            for (size_t i = 0; i < xv.size(); ++i) xv[i] = 0.04f * static_cast<float>(static_cast<int>((i + step * 3) % 11) - 5);
            for (size_t i = 0; i < dyv.size(); ++i) dyv[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5 + step) % 9) - 4);
            x_g.uploadData(xv);
            dy_g.uploadData(dyv);

            engine.enableFusedGemmAdam(false);
            lin_unfused.forward(x_g);
            lin_unfused.backward(dy_g);
            opt_unfused.step(lin_unfused.getParametersAndGradients(), 1.0f);
            engine.executeGraph();

            engine.enableFusedGemmAdam(true);
            lin_fused.forward(x_g);
            lin_fused.backward(dy_g);
            opt_fused.step(lin_fused.getParametersAndGradients(), 1.0f);
            engine.executeGraph();
        }
        engine.waitIdle();
        engine.enableFusedGemmAdam(false);

        if (!verifyMatrix(lin_fused.getWeights(), lin_unfused.getWeights().getData(), 2e-2f) ||
            !verifyMatrix(lin_fused.getBiases(), lin_unfused.getBiases().getData(), 2e-2f))
        {
            return failStep("D1 multi-step cached fused GEMM+AdamW parity mismatch");
        }
    }

    {
        for (size_t dim : {size_t{19}, size_t{33}, size_t{256}})
        {
            const size_t rows = 4;
            std::vector<float> a_vals(rows * dim), b_vals(rows * dim), dy_vals(rows * dim);
            std::vector<float> ab_vals(rows * dim * 2);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < dim; ++c)
                {
                    float a = 0.05f * static_cast<float>(static_cast<int>((r * 7 + c * 3) % 11) - 5);
                    float b = 0.05f * static_cast<float>(static_cast<int>((r * 5 + c * 7) % 13) - 6);
                    float d = 0.04f * static_cast<float>(static_cast<int>((r * 11 + c) % 9) - 4);
                    a_vals[r * dim + c] = a;
                    b_vals[r * dim + c] = b;
                    dy_vals[r * dim + c] = d;
                    ab_vals[r * dim * 2 + c] = a;
                    ab_vals[r * dim * 2 + dim + c] = b;
                }
            }

            Tensor a_cpu(Shape{rows, dim}, a_vals, Execution_Target::CPU);
            Tensor b_cpu(Shape{rows, dim}, b_vals, Execution_Target::CPU);
            Tensor dy_cpu(Shape{rows, dim}, dy_vals, Execution_Target::CPU);
            Tensor out_cpu(Execution_Target::CPU), da_cpu(Execution_Target::CPU), db_cpu(Execution_Target::CPU);
            ops::llm::swigluForward(a_cpu, b_cpu, out_cpu);
            ops::llm::swigluBackward(a_cpu, b_cpu, dy_cpu, da_cpu, db_cpu);

            Tensor a_gpu = castTensorToFp16(Tensor(Shape{rows, dim}, a_vals, Execution_Target::VULKAN_GPU));
            Tensor b_gpu = castTensorToFp16(Tensor(Shape{rows, dim}, b_vals, Execution_Target::VULKAN_GPU));
            Tensor dy_gpu = castTensorToFp16(Tensor(Shape{rows, dim}, dy_vals, Execution_Target::VULKAN_GPU));

            Tensor out_gpu(Execution_Target::VULKAN_GPU), da_gpu(Execution_Target::VULKAN_GPU), db_gpu(Execution_Target::VULKAN_GPU);
            ops::llm::swigluForward(a_gpu, b_gpu, out_gpu);
            ops::llm::swigluBackward(a_gpu, b_gpu, dy_gpu, da_gpu, db_gpu);

            Tensor ab_gpu = castTensorToFp16(Tensor(Shape{rows, dim * 2}, ab_vals, Execution_Target::VULKAN_GPU));
            Tensor fout_gpu(Execution_Target::VULKAN_GPU), dab_gpu(Execution_Target::VULKAN_GPU);
            ops::llm::fusedSwiGLUForward(ab_gpu, fout_gpu);
            ops::llm::fusedSwiGLUBackward(ab_gpu, dy_gpu, dab_gpu);

            flushGpuEngine(true);

            if (!verifyMatrix(out_gpu, out_cpu.getData(), 1e-2f) ||
                !verifyMatrix(da_gpu, da_cpu.getData(), 1e-2f) ||
                !verifyMatrix(db_gpu, db_cpu.getData(), 1e-2f) ||
                !verifyMatrix(fout_gpu, out_cpu.getData(), 1e-2f))
            {
                return failStep("D2 vectorized SwiGLU / FusedSwiGLU forward/backward mismatch");
            }

            const std::vector<float> dab_actual = dab_gpu.getData();
            const std::vector<float> da_exp = da_cpu.getData();
            const std::vector<float> db_exp = db_cpu.getData();
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < dim; ++c)
                {
                    if (!nearlyEqual(dab_actual[r * dim * 2 + c], da_exp[r * dim + c], 1e-2f) ||
                        !nearlyEqual(dab_actual[r * dim * 2 + dim + c], db_exp[r * dim + c], 1e-2f))
                    {
                        return failStep("D2 vectorized FusedSwiGLU backward grad mismatch");
                    }
                }
            }
        }

        struct QkvCase { uint32_t B, S, H, H_kv, D, off; bool bhsd; };
        const std::vector<QkvCase> qkv_cases = {
            {2, 17, 4, 4, 32, 0, false},
            {2, 17, 4, 2, 32, 5, true},
            {2, 17, 4, 2, 32, 0, false},
        };
        for (const auto &qc : qkv_cases)
        {
            const uint32_t q_dim = qc.H * qc.D;
            const uint32_t kv_dim = qc.H_kv * qc.D;
            const uint32_t tot_dim = q_dim + 2 * kv_dim;
            std::vector<float> qkv_vals(qc.B * qc.S * tot_dim);
            std::vector<float> dq_vals(qc.B * qc.S * q_dim);
            std::vector<float> dk_vals(qc.B * qc.S * kv_dim);
            std::vector<float> dv_vals(qc.B * qc.S * kv_dim);
            for (size_t i = 0; i < qkv_vals.size(); ++i) qkv_vals[i] = 0.04f * static_cast<float>(static_cast<int>((i * 7 + 2) % 13) - 6);
            for (size_t i = 0; i < dq_vals.size(); ++i) dq_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5 + 1) % 11) - 5);
            for (size_t i = 0; i < dk_vals.size(); ++i) dk_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 3 + 4) % 11) - 5);
            for (size_t i = 0; i < dv_vals.size(); ++i) dv_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 9 + 2) % 11) - 5);

            Tensor qkv_cpu(Shape{qc.B * qc.S, tot_dim}, qkv_vals, Execution_Target::CPU);
            Tensor q_cpu(Execution_Target::CPU), k_cpu(Execution_Target::CPU), v_cpu(Execution_Target::CPU);
            ops::llm::qkvSplitRopeForward(qkv_cpu, q_cpu, k_cpu, v_cpu, qc.B, qc.S, qc.H, qc.H_kv, qc.D, 10000.0f, qc.bhsd, qc.off);

            Shape q_shape = qc.bhsd ? Shape{qc.B, qc.H, qc.S, qc.D} : Shape{qc.B, qc.S, qc.H, qc.D};
            Shape kv_shape = qc.bhsd ? Shape{qc.B, qc.H_kv, qc.S, qc.D} : Shape{qc.B, qc.S, qc.H_kv, qc.D};
            Tensor dq_cpu(q_shape, dq_vals, Execution_Target::CPU);
            Tensor dk_cpu(kv_shape, dk_vals, Execution_Target::CPU);
            Tensor dv_cpu(kv_shape, dv_vals, Execution_Target::CPU);
            Tensor dqkv_cpu(Execution_Target::CPU);
            ops::llm::qkvPackUnropeBackward(dq_cpu, dk_cpu, dv_cpu, dqkv_cpu, qc.B, qc.S, qc.H, qc.H_kv, qc.D, 10000.0f, qc.bhsd, qc.off);

            Tensor qkv_gpu = castTensorToFp16(Tensor(Shape{qc.B * qc.S, tot_dim}, qkv_vals, Execution_Target::VULKAN_GPU));
            Tensor q_gpu(Execution_Target::VULKAN_GPU), k_gpu(Execution_Target::VULKAN_GPU), v_gpu(Execution_Target::VULKAN_GPU);
            ops::llm::qkvSplitRopeForward(qkv_gpu, q_gpu, k_gpu, v_gpu, qc.B, qc.S, qc.H, qc.H_kv, qc.D, 10000.0f, qc.bhsd, qc.off);

            Tensor dq_gpu = castTensorToFp16(Tensor(q_shape, dq_vals, Execution_Target::VULKAN_GPU));
            Tensor dk_gpu = castTensorToFp16(Tensor(kv_shape, dk_vals, Execution_Target::VULKAN_GPU));
            Tensor dv_gpu = castTensorToFp16(Tensor(kv_shape, dv_vals, Execution_Target::VULKAN_GPU));
            Tensor dqkv_gpu(Execution_Target::VULKAN_GPU);
            ops::llm::qkvPackUnropeBackward(dq_gpu, dk_gpu, dv_gpu, dqkv_gpu, qc.B, qc.S, qc.H, qc.H_kv, qc.D, 10000.0f, qc.bhsd, qc.off);

            flushGpuEngine(true);

            if (!verifyMatrix(q_gpu, q_cpu.getData(), 1e-2f) ||
                !verifyMatrix(k_gpu, k_cpu.getData(), 1e-2f) ||
                !verifyMatrix(v_gpu, v_cpu.getData(), 1e-2f) ||
                !verifyMatrix(dqkv_gpu, dqkv_cpu.getData(), 1e-2f))
            {
                return failStep("D2 QKV_SPLIT_ROPE / QKV_PACK_UNROPE FP16 vs CPU mismatch");
            }
        }

        for (Execution_Target tgt : {Execution_Target::CPU, Execution_Target::VULKAN_GPU})
        {
            constexpr size_t B = 2, S = 16, H_dim = 128, H = 4, H_kv = 2, I_dim = 256;
            const bool is_gpu = (tgt == Execution_Target::VULKAN_GPU);
            const Data_Type dt = is_gpu ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

            Transformer_Block tb_unfused(H_dim, H, I_dim, 1e-5f, 10000.0f, tgt, dt, 1, H_kv, false);
            auto cloned = tb_unfused.clone();
            auto &tb_fused = dynamic_cast<Transformer_Block &>(*cloned);
            tb_fused.enableFusedProjections(true);

            std::vector<float> xv(B * S * H_dim), dyv(B * S * H_dim);
            for (size_t i = 0; i < xv.size(); ++i)
            {
                xv[i] = 0.04f * static_cast<float>(static_cast<int>((i * 7 + 1) % 11) - 5);
                dyv[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5 + 3) % 9) - 4);
            }
            Tensor x_t = Tensor(Shape{B, S, H_dim}, xv, tgt);
            Tensor dy_t = Tensor(Shape{B, S, H_dim}, dyv, tgt);
            Tensor x_in = is_gpu ? castTensorToFp16(x_t) : x_t;
            Tensor dy_in = is_gpu ? castTensorToFp16(dy_t) : dy_t;

            Tensor y_u = tb_unfused.forward(x_in);
            Tensor dx_u = tb_unfused.backward(dy_in);
            Tensor y_f = tb_fused.forward(x_in);
            Tensor dx_f = tb_fused.backward(dy_in);
            flushIfGpu(tgt, true);

            const std::vector<float> y_u_ref = y_u.getData();
            const std::vector<float> dx_u_ref = dx_u.getData();
            const float tol = is_gpu ? 5e-2f : 1e-4f;
            if (!verifyMatrix(y_f, y_u_ref, tol) ||
                !verifyMatrix(dx_f, dx_u_ref, tol))
            {
                return failStep("D2 Transformer_Block fused vs unfused forward/backward mismatch");
            }
            if (!verifyMatrix(tb_fused.getInputLayernorm().getGammaGradient(), tb_unfused.getInputLayernorm().getGammaGradient().getData(), tol) ||
                !verifyMatrix(tb_fused.getPostAttentionLayernorm().getGammaGradient(), tb_unfused.getPostAttentionLayernorm().getGammaGradient().getData(), tol) ||
                !verifyMatrix(tb_fused.getOProj().getWeightsGradient(), tb_unfused.getOProj().getWeightsGradient().getData(), tol) ||
                !verifyMatrix(tb_fused.getDownProj().getWeightsGradient(), tb_unfused.getDownProj().getWeightsGradient().getData(), tol))
            {
                return failStep("D2 Transformer_Block fused vs unfused param grad mismatch");
            }

            if (is_gpu)
            {
                KV_Cache_Manager kv_u(1, H_kv, 32, H_dim / H, tgt, Data_Type::FLOAT16, KV_Cache_Mode::RING_BUFFER);
                KV_Cache_Manager kv_f(1, H_kv, 32, H_dim / H, tgt, Data_Type::FLOAT16, KV_Cache_Mode::RING_BUFFER);
                Tensor x_p16 = castTensorToFp16(Tensor(Shape{1, 4, H_dim}, std::vector<float>(xv.begin(), xv.begin() + 4 * H_dim), tgt));
                Tensor x_d16 = castTensorToFp16(Tensor(Shape{1, 1, H_dim}, std::vector<float>(xv.begin() + 4 * H_dim, xv.begin() + 5 * H_dim), tgt));

                Tensor p_u = tb_unfused.forward(x_p16, &kv_u);
                Tensor p_f = tb_fused.forward(x_p16, &kv_f);
                flushGpuEngine(true);
                if (!verifyMatrix(p_f, p_u.getData(), 5e-2f))
                {
                    return failStep("PhaseB/D2 Transformer_Block KV-cache prefill fused vs unfused mismatch");
                }

                Tensor d_u = tb_unfused.forward(x_d16, &kv_u);
                Tensor d_f = tb_fused.forward(x_d16, &kv_f);
                flushGpuEngine(true);
                if (!verifyMatrix(d_f, d_u.getData(), 5e-2f))
                {
                    return failStep("D2 Transformer_Block KV-cache fused vs unfused mismatch");
                }

                const auto tmp_dir = std::filesystem::temp_directory_path();
                const std::string ckpt_path = (tmp_dir / "phaseD_tb_ckpt.bin").string();
                const std::string inf_path = (tmp_dir / "phaseD_tb_inf.bin").string();
                Scoped_Temp_File_Guard cleaner({ckpt_path, inf_path});

                {
                    std::ofstream ofs(ckpt_path, std::ios::binary | std::ios::trunc);
                    tb_fused.saveCheckpoint(ofs);
                }
                {
                    Transformer_Block tb_loaded_u(H_dim, H, I_dim, 1e-5f, 10000.0f, tgt, dt, 1, H_kv, false);
                    std::ifstream ifs(ckpt_path, std::ios::binary);
                    tb_loaded_u.loadCheckpoint(ifs);
                    Tensor y_lu = tb_loaded_u.forward(x_in);
                    flushGpuEngine(true);
                    if (!verifyMatrix(y_lu, y_u_ref, 5e-2f))
                    {
                        return failStep("D2 Transformer_Block cross-load fused->unfused checkpoint mismatch");
                    }
                }
                {
                    std::ofstream ofs(inf_path, std::ios::binary | std::ios::trunc);
                    tb_unfused.saveInference(ofs);
                }
                {
                    Transformer_Block tb_loaded_f(H_dim, H, I_dim, 1e-5f, 10000.0f, tgt, dt, 1, H_kv, true);
                    std::ifstream ifs(inf_path, std::ios::binary);
                    tb_loaded_f.loadInference(ifs);
                    Tensor y_lf = tb_loaded_f.forward(x_in);
                    flushGpuEngine(true);
                    if (!verifyMatrix(y_lf, y_u_ref, 5e-2f))
                    {
                        return failStep("D2 Transformer_Block cross-load unfused->fused inference mismatch");
                    }
                }
            }
        }
    }

    {
        for (size_t dim : {size_t{256}, size_t{384}, size_t{35}})
        {
            const size_t rows = 5;
            const float scale = 0.25f;
            std::vector<float> res_v(rows * dim), br_v(rows * dim), dy_v(rows * dim), rg_v(rows * dim), g_v(dim);
            for (size_t i = 0; i < res_v.size(); ++i)
            {
                res_v[i] = 0.05f * static_cast<float>(static_cast<int>((i * 3 + 1) % 11) - 5);
                br_v[i]  = 0.06f * static_cast<float>(static_cast<int>((i * 7 + 2) % 9) - 4);
                dy_v[i]  = 0.04f * static_cast<float>(static_cast<int>((i * 5 + 3) % 13) - 6);
                rg_v[i]  = 0.03f * static_cast<float>(static_cast<int>((i * 11 + 1) % 7) - 3);
            }
            for (size_t c = 0; c < dim; ++c)
            {
                g_v[c] = 0.8f + 0.01f * static_cast<float>(c % 21);
            }

            Tensor res_c(Shape{rows, dim}, res_v, Execution_Target::CPU);
            Tensor br_c(Shape{rows, dim}, br_v, Execution_Target::CPU);
            Tensor g_c(Shape{1, dim}, g_v, Execution_Target::CPU);
            Tensor dy_c(Shape{rows, dim}, dy_v, Execution_Target::CPU);
            Tensor rg_c(Shape{rows, dim}, rg_v, Execution_Target::CPU);
            Tensor out_res_c(Execution_Target::CPU), inv_c(Execution_Target::CPU), out_norm_c(Execution_Target::CPU);
            Tensor dg_c(Execution_Target::CPU), sum_c(Execution_Target::CPU), scaled_c(Execution_Target::CPU);
            ops::normalization::residualScaleRmsNormForward(res_c, br_c, g_c, scale, out_res_c, inv_c, out_norm_c, 1e-5f);
            ops::normalization::rmsNormBackwardResidual(out_res_c, dy_c, g_c, inv_c, rg_c, dg_c, sum_c, &scaled_c, scale, false);

            Tensor res_g = castTensorToFp16(Tensor(Shape{rows, dim}, res_v, Execution_Target::VULKAN_GPU));
            Tensor br_g = castTensorToFp16(Tensor(Shape{rows, dim}, br_v, Execution_Target::VULKAN_GPU));
            Tensor g_g = castTensorToFp16(Tensor(Shape{1, dim}, g_v, Execution_Target::VULKAN_GPU));
            Tensor dy_g = castTensorToFp16(Tensor(Shape{rows, dim}, dy_v, Execution_Target::VULKAN_GPU));
            Tensor rg_g = castTensorToFp16(Tensor(Shape{rows, dim}, rg_v, Execution_Target::VULKAN_GPU));

            Tensor out_res_g(Execution_Target::VULKAN_GPU), inv_g(Execution_Target::VULKAN_GPU), out_norm_g(Execution_Target::VULKAN_GPU);
            Tensor dg_g(Execution_Target::VULKAN_GPU), sum_g(Execution_Target::VULKAN_GPU), scaled_g(Execution_Target::VULKAN_GPU);
            Tensor dg_g2(Execution_Target::VULKAN_GPU), sum_g2(Execution_Target::VULKAN_GPU);

            ops::normalization::residualScaleRmsNormForward(res_g, br_g, g_g, scale, out_res_g, inv_g, out_norm_g, 1e-5f);
            ops::normalization::rmsNormBackwardResidual(out_res_g, dy_g, g_g, inv_g, rg_g, dg_g, sum_g, &scaled_g, scale, false);
            ops::normalization::rmsNormBackwardResidual(out_res_g, dy_g, g_g, inv_g, rg_g, dg_g2, sum_g2, nullptr, 1.0f, false);
            flushGpuEngine(true);

            if (!verifyMatrix(out_res_g, out_res_c.getData(), 1e-2f) ||
                !verifyMatrix(inv_g, inv_c.getData(), 1e-2f) ||
                !verifyMatrix(out_norm_g, out_norm_c.getData(), 1e-2f) ||
                !verifyMatrix(dg_g, dg_c.getData(), 2e-2f) ||
                !verifyMatrix(sum_g, sum_c.getData(), 2e-2f) ||
                !verifyMatrix(scaled_g, scaled_c.getData(), 2e-2f) ||
                !verifyMatrix(dg_g2, dg_c.getData(), 2e-2f) ||
                !verifyMatrix(sum_g2, sum_c.getData(), 2e-2f))
            {
                return failStep("D3 Residual+Scale+RMSNorm forward/backward FP16 parity mismatch");
            }
        }
    }

    {
        Causal_LM_Config cfg{};
        cfg.vocab_size = 64;
        cfg.hidden_dim = 64;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.intermediate_dim = 128;
        cfg.num_layers = 2;
        cfg.max_seq_len = 32;
        cfg.execution_target = Execution_Target::VULKAN_GPU;
        cfg.data_type = Data_Type::FLOAT16;
        cfg.use_fused_projections = true;
        cfg.use_loss_scaler = false;
        cfg.tokenizer_path.clear();

        Causal_LM lm_ref(cfg);
        Causal_LM lm_static(cfg);
        auto ref_params = lm_ref.getParametersAndGradients();
        auto st_params = lm_static.getParametersAndGradients();
        if (ref_params.size() != st_params.size())
        {
            return failStep("D4 Causal_LM parameter count mismatch");
        }
        for (size_t i = 0; i < ref_params.size(); ++i)
        {
            *st_params[i].first = Tensor(ref_params[i].first->getShape(),
                                         ref_params[i].first->getData(),
                                         Execution_Target::VULKAN_GPU);
        }
        lm_static.warmupWeightCaches();

        Adam_Optimizer opt_ref(1e-3f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        Adam_Optimizer opt_static(1e-3f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);

        Tensor in_ref(Shape{2, 8}, Execution_Target::VULKAN_GPU);
        Tensor tgt_ref(Shape{2, 8}, Execution_Target::VULKAN_GPU);
        Tensor in_st(Shape{2, 8}, Execution_Target::VULKAN_GPU);
        Tensor tgt_st(Shape{2, 8}, Execution_Target::VULKAN_GPU);
        std::vector<float> in_v(16), tgt_v(16);

        Scoped_Engine_Flags_Guard flags_guard;
        lm_ref.setTrainingMode(true);
        engine.enableStaticGraph(false);
        for (int step = 0; step < 5; ++step)
        {
            for (size_t i = 0; i < 16; ++i)
            {
                in_v[i] = static_cast<float>((i * 5 + step * 3) % 63);
                tgt_v[i] = (i == 15) ? -1.0f : static_cast<float>((i * 5 + step * 3 + 1) % 63);
            }
            {
                uint32_t cur_f = engine.getContext().getCurrentFrame();
                if (!engine.getContext().isFrameReady(cur_f))
                {
                    engine.prepareCurrentFrame();
                }
            }
            in_ref.uploadData(in_v);
            tgt_ref.uploadData(tgt_v);
            bool applied_ref = false;
            float l_ref = lm_ref.trainStep(in_ref, tgt_ref, opt_ref, 1.0f, &applied_ref);
            if (!applied_ref || std::isnan(l_ref) || std::isinf(l_ref))
            {
                return failStep("D4 Causal_LM non-static trainStep failed");
            }
        }
        engine.readPendingLoss(UINT32_MAX);
        engine.waitIdle();

        lm_static.setTrainingMode(true);
        engine.enableStaticGraph(true);
        uint64_t replay_hits = 0;
        for (int step = 0; step < 5; ++step)
        {
            for (size_t i = 0; i < 16; ++i)
            {
                in_v[i] = static_cast<float>((i * 5 + step * 3) % 63);
                tgt_v[i] = (i == 15) ? -1.0f : static_cast<float>((i * 5 + step * 3 + 1) % 63);
            }
            {
                uint32_t cur_f = engine.getContext().getCurrentFrame();
                if (!engine.getContext().isFrameReady(cur_f))
                {
                    engine.prepareCurrentFrame();
                }
            }
            in_st.uploadData(in_v);
            tgt_st.uploadData(tgt_v);
            const uint32_t f_before = engine.getContext().getCurrentFrame();
            const uint64_t gen_before = engine.getGraphExecutor().getStaticBakeGeneration(f_before);
            const bool baked_before = engine.getGraphExecutor().isStaticBaked(f_before);
            bool applied_st = false;
            float l_st = lm_static.trainStep(in_st, tgt_st, opt_static, 1.0f, &applied_st);
            const uint64_t gen_after = engine.getGraphExecutor().getStaticBakeGeneration(f_before);
            if (!applied_st || std::isnan(l_st) || std::isinf(l_st))
            {
                return failStep("D4 Causal_LM static trainStep failed or produced non-finite loss");
            }
            if (step >= 2 && baked_before && gen_before == gen_after)
            {
                replay_hits++;
            }
        }
        engine.readPendingLoss(UINT32_MAX);
        engine.waitIdle();

        if (replay_hits != 3)
        {
            return failStep("D4 Causal_LM static graph fast replay did not reuse baked graphs on steps 2..4");
        }

        for (size_t i = 0; i < ref_params.size(); ++i)
        {
            if (!verifyMatrix(*st_params[i].first, ref_params[i].first->getData(), 2e-2f))
            {
                return failStep("D4 Causal_LM static replay weights diverged from non-static");
            }
        }

        lm_ref.setTrainingMode(false);
        lm_static.setTrainingMode(false);
        Tensor eval_in(Shape{1, 8}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, Execution_Target::VULKAN_GPU);
        Tensor logits_ref = lm_ref.forward(eval_in, false);
        Tensor logits_st = lm_static.forward(eval_in, false);
        engine.waitIdle();
        if (!verifyMatrix(logits_st, logits_ref.getData(), 5e-2f))
        {
            return failStep("D4 Causal_LM eval forward after static replay mismatch");
        }
    }

    return true;
}

bool testPhase2KernelOptimizations()
{
    auto failStep = [](const char *name) -> bool {
        return failTestStep("Phase2KernelOptimizations", name);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    Scoped_Engine_Flags_Guard flag_guard;
    if (engine.isCooperativeMatrixSupported())
    {
        engine.setCooperativeMatrixEnabled(true);
    }

    struct SwiGluCase { size_t rows; size_t half_dim; const char *label; };
    const SwiGluCase swiglu_cases[] = {
        {4, 64, "div8"},
        {1, 12, "div4"},
        {1, 13, "odd"}
    };
    for (const auto &sc : swiglu_cases)
    {
        std::vector<float> in_vals(sc.rows * 2 * sc.half_dim);
        std::vector<float> dy_vals(sc.rows * sc.half_dim);
        for (size_t i = 0; i < in_vals.size(); ++i)
        {
            in_vals[i] = 0.12f * static_cast<float>(static_cast<int>(i % 15) - 7);
        }
        for (size_t i = 0; i < dy_vals.size(); ++i)
        {
            dy_vals[i] = 0.1f * static_cast<float>(static_cast<int>((i * 5) % 13) - 6);
        }

        Tensor in_cpu(Shape{sc.rows, 2 * sc.half_dim}, in_vals, Execution_Target::CPU);
        Tensor dy_cpu(Shape{sc.rows, sc.half_dim}, dy_vals, Execution_Target::CPU);
        Tensor dx_cpu(Execution_Target::CPU);
        ops::llm::fusedSwiGLUBackward(in_cpu, dy_cpu, dx_cpu);

        Tensor in_gpu = castTensorToFp16(Tensor(Shape{sc.rows, 2 * sc.half_dim}, in_vals, Execution_Target::VULKAN_GPU));
        Tensor dy_gpu = castTensorToFp16(Tensor(Shape{sc.rows, sc.half_dim}, dy_vals, Execution_Target::VULKAN_GPU));
        Tensor dx_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::fusedSwiGLUBackward(in_gpu, dy_gpu, dx_gpu);
        flushGpuEngine(true);

        if (!verifyMatrix(dx_gpu, dx_cpu.getData(), 3e-2f))
        {
            return failStep(sc.label);
        }
    }

    {
        constexpr size_t half_dim = 8;
        std::vector<float> in_vals(2 * half_dim, 0.25f);
        std::vector<float> dy_vals(half_dim, 0.2f);
        in_vals[0] = std::numeric_limits<float>::infinity();
        dy_vals[1] = std::numeric_limits<float>::quiet_NaN();

        Tensor in_cpu(Shape{1, 2 * half_dim}, in_vals, Execution_Target::CPU);
        Tensor dy_cpu(Shape{1, half_dim}, dy_vals, Execution_Target::CPU);
        Tensor dx_cpu(Execution_Target::CPU);
        ops::llm::fusedSwiGLUBackward(in_cpu, dy_cpu, dx_cpu);
        const auto dx_cpu_data = dx_cpu.getData();

        Tensor in_gpu = castTensorToFp16(Tensor(Shape{1, 2 * half_dim}, in_vals, Execution_Target::VULKAN_GPU));
        Tensor dy_gpu = castTensorToFp16(Tensor(Shape{1, half_dim}, dy_vals, Execution_Target::VULKAN_GPU));
        Tensor dx_gpu(Execution_Target::VULKAN_GPU);
        ops::llm::fusedSwiGLUBackward(in_gpu, dy_gpu, dx_gpu);
        flushGpuEngine(true);

        const auto dx_gpu_data = dx_gpu.getData();
        if (dx_gpu_data.size() != 2 * half_dim)
        {
            return failStep("fusedSwiGLUBackward NaN/Inf output size mismatch");
        }
        for (float v : dx_gpu_data)
        {
            if (!std::isfinite(v))
            {
                return failStep("fusedSwiGLUBackward FP16 produced non-finite value");
            }
        }
        for (size_t k = 2; k < half_dim; ++k)
        {
            if (!nearlyEqual(dx_gpu_data[k], dx_cpu_data[k], 3e-2f) ||
                !nearlyEqual(dx_gpu_data[half_dim + k], dx_cpu_data[half_dim + k], 3e-2f))
            {
                return failStep("fusedSwiGLUBackward finite sibling lane mismatch");
            }
        }
    }

    {
        VkPipeline p32 = engine.getNetwork().getSpecializedPipeline(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16, 32u);
        VkPipeline p64 = engine.getNetwork().getSpecializedPipeline(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16, 64u);
        if (p32 == VK_NULL_HANDLE || p64 == VK_NULL_HANDLE || p32 == p64)
        {
            return failStep("Specialized FlashAttention backward dQ pipelines invalid or identical");
        }

        for (uint32_t hd : {32u, 64u})
        {
            Shape q_shape{1, 24, 2, hd};
            std::vector<float> q_v(q_shape.getTotalElements()), k_v(q_shape.getTotalElements()), v_v(q_shape.getTotalElements()), do_v(q_shape.getTotalElements());
            for (size_t i = 0; i < q_v.size(); ++i)
            {
                q_v[i] = 0.08f * static_cast<float>(static_cast<int>((i * 7 + hd) % 11) - 5);
                k_v[i] = 0.07f * static_cast<float>(static_cast<int>((i * 3 + 2) % 11) - 5);
                v_v[i] = 0.08f * static_cast<float>(static_cast<int>((i * 11 + 1) % 9) - 4);
                do_v[i] = 0.06f * static_cast<float>(static_cast<int>((i * 5 + 3) % 9) - 4);
            }
            ops::llm::Attention_Config cfg{};
            cfg.num_heads = 2;
            cfg.num_kv_heads = 2;
            cfg.seq_len = 24;
            cfg.head_dim = hd;
            cfg.is_causal = true;
            cfg.is_bshd = true;

            Tensor q_cpu(q_shape, q_v, Execution_Target::CPU);
            Tensor k_cpu(q_shape, k_v, Execution_Target::CPU);
            Tensor v_cpu(q_shape, v_v, Execution_Target::CPU);
            Tensor do_cpu(q_shape, do_v, Execution_Target::CPU);
            Tensor o_cpu(Execution_Target::CPU), l_cpu(Execution_Target::CPU);
            Tensor dq_cpu(Execution_Target::CPU), dk_cpu(Execution_Target::CPU), dv_cpu(Execution_Target::CPU);
            ops::llm::flashAttentionForward(q_cpu, k_cpu, v_cpu, o_cpu, cfg, &l_cpu);
            ops::llm::flashAttentionBackward(q_cpu, k_cpu, v_cpu, o_cpu, do_cpu, dq_cpu, dk_cpu, dv_cpu, cfg, &l_cpu);

            Tensor q_gpu = castTensorToFp16(Tensor(q_shape, q_v, Execution_Target::VULKAN_GPU));
            Tensor k_gpu = castTensorToFp16(Tensor(q_shape, k_v, Execution_Target::VULKAN_GPU));
            Tensor v_gpu = castTensorToFp16(Tensor(q_shape, v_v, Execution_Target::VULKAN_GPU));
            Tensor do_gpu = castTensorToFp16(Tensor(q_shape, do_v, Execution_Target::VULKAN_GPU));
            Tensor o_gpu(Execution_Target::VULKAN_GPU), l_gpu(Execution_Target::VULKAN_GPU);
            Tensor dq_gpu(Execution_Target::VULKAN_GPU), dk_gpu(Execution_Target::VULKAN_GPU), dv_gpu(Execution_Target::VULKAN_GPU);
            ops::llm::flashAttentionForward(q_gpu, k_gpu, v_gpu, o_gpu, cfg, &l_gpu);
            ops::llm::flashAttentionBackward(q_gpu, k_gpu, v_gpu, o_gpu, do_gpu, dq_gpu, dk_gpu, dv_gpu, cfg, &l_gpu);
            flushGpuEngine(true);

            if (!verifyMatrix(dq_gpu, dq_cpu.getData(), 5e-2f) ||
                !verifyMatrix(dk_gpu, dk_cpu.getData(), 5e-2f) ||
                !verifyMatrix(dv_gpu, dv_cpu.getData(), 5e-2f))
            {
                return failStep(hd == 32u ? "Specialized FlashAttention backward D=32 mismatch" : "Specialized FlashAttention backward D=64 mismatch");
            }
        }
    }

    if (engine.isCooperativeMatrixSupported())
    {
        constexpr size_t M = 256, K = 64, N = 64;
        std::vector<float> x_vals(M * K), dy_vals(M * N);
        for (size_t i = 0; i < x_vals.size(); ++i) x_vals[i] = 0.04f * static_cast<float>(static_cast<int>(i % 11) - 5);
        for (size_t i = 0; i < dy_vals.size(); ++i) dy_vals[i] = 0.03f * static_cast<float>(static_cast<int>((i * 5) % 13) - 6);

        Tensor x_cpu(Shape{M, K}, x_vals, Execution_Target::CPU);
        Tensor dy_cpu(Shape{M, N}, dy_vals, Execution_Target::CPU);
        Tensor dw_cpu(Execution_Target::CPU), db_cpu(Execution_Target::CPU);
        ops::linear::linearBackwardWeightBias(x_cpu, dy_cpu, dw_cpu, db_cpu, false);

        Tensor x_gpu = castTensorToFp16(Tensor(Shape{M, K}, x_vals, Execution_Target::VULKAN_GPU));
        Tensor dy_gpu = castTensorToFp16(Tensor(Shape{M, N}, dy_vals, Execution_Target::VULKAN_GPU));
        Tensor dw_gpu(Execution_Target::VULKAN_GPU), db_gpu(Execution_Target::VULKAN_GPU);
        ops::linear::linearBackwardWeightBias(x_gpu, dy_gpu, dw_gpu, db_gpu, false);
        flushGpuEngine(true);

        if (!verifyMatrix(dw_gpu, dw_cpu.getData(), 5e-2f) ||
            !verifyMatrix(db_gpu, db_cpu.getData(), 5e-2f))
        {
            return failStep("Split-K linearBackwardWeightBias (accumulate=false) mismatch");
        }

        ops::linear::linearBackwardWeightBias(x_cpu, dy_cpu, dw_cpu, db_cpu, true);
        ops::linear::linearBackwardWeightBias(x_gpu, dy_gpu, dw_gpu, db_gpu, true);
        flushGpuEngine(true);

        if (!verifyMatrix(dw_gpu, dw_cpu.getData(), 1e-1f) ||
            !verifyMatrix(db_gpu, db_cpu.getData(), 1e-1f))
        {
            return failStep("Split-K linearBackwardWeightBias (accumulate=true) mismatch");
        }
    }

    return true;
}

bool testPhase3And4UnifiedFusion()
{
    auto failStep = [](const char *name) -> bool {
        return failTestStep("Phase3And4UnifiedFusion", name);
    };

    Execution_Engine &engine = Execution_Engine::getInstance();
    Scoped_Engine_Flags_Guard flag_guard;
    auto &pcm = engine.getPipelineCacheManager();

    struct SpirvDirResetGuard
    {
        Pipeline_Cache_Manager &mgr;
        Execution_Engine &eng;
        ~SpirvDirResetGuard()
        {
            eng.waitIdle();
            eng.invalidateGraphCache();
            mgr.clearMemoryPipelines();
            mgr.setSpirvCacheDir("");
        }
    };

    {
        const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "vml_phase3_spirv_test";
        std::error_code ec;
        std::filesystem::remove_all(tmp_dir, ec);
        std::filesystem::create_directories(tmp_dir, ec);
        Scoped_Temp_File_Guard temp_guard({}, {tmp_dir});
        SpirvDirResetGuard spirv_reset{pcm, engine};

        constexpr size_t N = 515;
        std::vector<float> a_v(N, 0.25f), b_v(N, 0.5f);
        Tensor a_g = castTensorToFp16(Tensor(Shape{1, N}, a_v, Execution_Target::VULKAN_GPU));
        Tensor b_g = castTensorToFp16(Tensor(Shape{1, N}, b_v, Execution_Target::VULKAN_GPU));
        flushGpuEngine(true);

        engine.waitIdle();
        engine.invalidateGraphCache();
        pcm.clearMemoryPipelines();
        pcm.setSpirvCacheDir(tmp_dir.string());
        pcm.resetCacheStats();
        {
            Tensor t1(Execution_Target::VULKAN_GPU), t2(Execution_Target::VULKAN_GPU), t3(Execution_Target::VULKAN_GPU);
            a_g.add(b_g, t1);
            t1.mulScalar(0.5f, t2);
            ops::activation::gelu(t2, t3);
            flushGpuEngine(true);
        }
        const uint64_t comp1 = pcm.getShadercCompileCount();
        const uint64_t hits1 = pcm.getSpirvDiskHits();
        if (comp1 == 0 || hits1 != 0)
        {
            return failStep("SPIR-V disk cache miss pass did not compile via shaderc");
        }

        engine.waitIdle();
        engine.invalidateGraphCache();
        pcm.clearMemoryPipelines();
        pcm.resetCacheStats();
        {
            Tensor t1(Execution_Target::VULKAN_GPU), t2(Execution_Target::VULKAN_GPU), t3(Execution_Target::VULKAN_GPU);
            a_g.add(b_g, t1);
            t1.mulScalar(0.5f, t2);
            ops::activation::gelu(t2, t3);
            flushGpuEngine(true);
        }
        const uint64_t comp2 = pcm.getShadercCompileCount();
        const uint64_t hits2 = pcm.getSpirvDiskHits();
        if (hits2 == 0 || comp2 != 0)
        {
            size_t spv_count = 0;
            for (const auto &entry : std::filesystem::directory_iterator(tmp_dir, ec))
            {
                if (entry.is_regular_file()) ++spv_count;
            }
            const std::string detail = std::format("SPIR-V disk cache hit pass mismatch (c1={}, h1={}, c2={}, h2={}, files={})",
                                                   comp1, hits1, comp2, hits2, spv_count);
            return failStep(detail.c_str());
        }
    }

    if (engine.isCooperativeMatrixSupported())
    {
        engine.setCooperativeMatrixEnabled(true);

        constexpr size_t M = 64, K = 64, N = 64;
        std::vector<float> x_v(M * K), w_v(K * N), b_v(N), res_v(M * N);
        for (size_t i = 0; i < x_v.size(); ++i) x_v[i] = 0.04f * static_cast<float>(static_cast<int>(i % 9) - 4);
        for (size_t i = 0; i < w_v.size(); ++i) w_v[i] = 0.03f * static_cast<float>(static_cast<int>((i * 3) % 7) - 3);
        for (size_t i = 0; i < b_v.size(); ++i) b_v[i] = 0.02f * static_cast<float>(static_cast<int>(i % 5) - 2);
        for (size_t i = 0; i < res_v.size(); ++i) res_v[i] = 0.05f * static_cast<float>(static_cast<int>((i * 7) % 11) - 5);

        Tensor x_c(Shape{M, K}, x_v, Execution_Target::CPU), w_c(Shape{K, N}, w_v, Execution_Target::CPU);
        Tensor b_c(Shape{1, N}, b_v, Execution_Target::CPU), r_c(Shape{M, N}, res_v, Execution_Target::CPU);
        Tensor yg_c(Execution_Target::CPU), yo_c(Execution_Target::CPU), yr_c(Execution_Target::CPU);
        ops::linear::linearForward(x_c, w_c, b_c, yg_c);
        yg_c.add(r_c, yo_c);
        ops::activation::relu(yg_c, yr_c);

        Tensor x_g = castTensorToFp16(Tensor(Shape{M, K}, x_v, Execution_Target::VULKAN_GPU));
        Tensor w_g = castTensorToFp16(Tensor(Shape{K, N}, w_v, Execution_Target::VULKAN_GPU));
        Tensor b_g = castTensorToFp16(Tensor(Shape{1, N}, b_v, Execution_Target::VULKAN_GPU));
        Tensor r_g = castTensorToFp16(Tensor(Shape{M, N}, res_v, Execution_Target::VULKAN_GPU));
        flushGpuEngine(true);

        engine.waitIdle();
        engine.invalidateGraphCache();
        pcm.clearMemoryPipelines();
        pcm.resetCacheStats();

        Tensor yg_g(Execution_Target::VULKAN_GPU), yo_g(Execution_Target::VULKAN_GPU);
        ops::linear::linearForward(x_g, w_g, b_g, yg_g);
        yg_g.add(r_g, yo_g);
        {
            Compute_Graph copy = engine.getCurrentGraph();
            Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
            if (copy.getNodeCount() != 1 || !copy.getNode(0).is_fused || !copy.getNode(0).fusion_plan ||
                copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::GEMM_COOPMAT ||
                copy.getNode(0).fusion_plan->materialized_buffer_indices.empty())
            {
                return failStep("Phase3A GEMM+ADD fusion plan or materialization invalid");
            }
        }
        flushGpuEngine(true);

        if ((pcm.getShadercCompileCount() + pcm.getSpirvDiskHits()) == 0)
        {
            return failStep("Phase3A GEMM+ADD did not compile or load fused pipeline");
        }
        if (!verifyMatrix(yg_g, yg_c.getData(), 5e-2f) || !verifyMatrix(yo_g, yo_c.getData(), 5e-2f))
        {
            return failStep("Phase3A GEMM+ADD numeric parity mismatch");
        }

        Tensor yg2_g(Execution_Target::VULKAN_GPU), yr_g(Execution_Target::VULKAN_GPU);
        ops::linear::linearForward(x_g, w_g, b_g, yg2_g);
        ops::activation::relu(yg2_g, yr_g);
        {
            Compute_Graph copy = engine.getCurrentGraph();
            Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
            if (copy.getNodeCount() != 1 || !copy.getNode(0).fusion_plan ||
                copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::GEMM_COOPMAT)
            {
                return failStep("Phase3A GEMM+RELU fusion plan invalid");
            }
        }
        flushGpuEngine(true);

        if (!verifyMatrix(yg2_g, yg_c.getData(), 5e-2f) || !verifyMatrix(yr_g, yr_c.getData(), 5e-2f))
        {
            return failStep("Phase3A GEMM+RELU numeric parity mismatch");
        }
    }

    {
        constexpr size_t N = 1027;
        std::vector<float> a_v(N), b_v(N);
        for (size_t i = 0; i < N; ++i)
        {
            a_v[i] = 0.12f * static_cast<float>(static_cast<int>(i % 15) - 7);
            b_v[i] = 0.09f * static_cast<float>(static_cast<int>((i * 5) % 13) - 6);
        }

        Tensor a_c(Shape{1, N}, a_v, Execution_Target::CPU), b_c(Shape{1, N}, b_v, Execution_Target::CPU);
        Tensor t1_c(Execution_Target::CPU), t2_c(Execution_Target::CPU), t3_c(Execution_Target::CPU);
        a_c.add(b_c, t1_c);
        t1_c.mulScalar(0.5f, t2_c);
        ops::activation::gelu(t2_c, t3_c);

        Tensor a_g = castTensorToFp16(Tensor(Shape{1, N}, a_v, Execution_Target::VULKAN_GPU));
        Tensor b_g = castTensorToFp16(Tensor(Shape{1, N}, b_v, Execution_Target::VULKAN_GPU));
        flushGpuEngine(true);

        Tensor t1_g(Execution_Target::VULKAN_GPU), t2_g(Execution_Target::VULKAN_GPU), t3_g(Execution_Target::VULKAN_GPU);
        a_g.add(b_g, t1_g);
        t1_g.mulScalar(0.5f, t2_g);
        ops::activation::gelu(t2_g, t3_g);
        {
            Compute_Graph copy = engine.getCurrentGraph();
            Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
            if (copy.getNodeCount() != 1 || !copy.getNode(0).fusion_plan ||
                copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::POINTWISE_CHAIN ||
                !copy.getNode(0).fusion_plan->use_vec4)
            {
                return failStep("Phase3B Pointwise vec4 tail fusion plan invalid");
            }
        }
        flushGpuEngine(true);

        if (!verifyMatrix(t1_g, t1_c.getData(), 2e-2f) ||
            !verifyMatrix(t2_g, t2_c.getData(), 2e-2f) ||
            !verifyMatrix(t3_g, t3_c.getData(), 2e-2f))
        {
            return failStep("Phase3B Pointwise vec4 tail numeric parity mismatch");
        }
    }

    {
        constexpr size_t rows = 17, cols = 32;
        std::vector<float> m_v(rows * cols), rv_v(cols);
        for (size_t i = 0; i < m_v.size(); ++i) m_v[i] = 0.08f * static_cast<float>(static_cast<int>(i % 11) - 5);
        for (size_t c = 0; c < cols; ++c) rv_v[c] = 0.05f * static_cast<float>(static_cast<int>(c % 9) - 4);

        Tensor m_c(Shape{rows, cols}, m_v, Execution_Target::CPU), rv_c(Shape{1, cols}, rv_v, Execution_Target::CPU);
        Tensor b1_c(Execution_Target::CPU), b2_c(Execution_Target::CPU), b3_c(Execution_Target::CPU);
        m_c.add(rv_c, b1_c);
        b1_c.mulScalar(0.75f, b2_c);
        ops::activation::relu(b2_c, b3_c);

        Tensor m_g = castTensorToFp16(Tensor(Shape{rows, cols}, m_v, Execution_Target::VULKAN_GPU));
        Tensor rv_g = castTensorToFp16(Tensor(Shape{1, cols}, rv_v, Execution_Target::VULKAN_GPU));
        flushGpuEngine(true);

        Tensor b1_g(Execution_Target::VULKAN_GPU), b2_g(Execution_Target::VULKAN_GPU), b3_g(Execution_Target::VULKAN_GPU);
        m_g.add(rv_g, b1_g);
        b1_g.mulScalar(0.75f, b2_g);
        ops::activation::relu(b2_g, b3_g);
        {
            Compute_Graph copy = engine.getCurrentGraph();
            Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
            if (copy.getNodeCount() != 1 || !copy.getNode(0).fusion_plan ||
                copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::POINTWISE_CHAIN ||
                copy.getNode(0).fusion_plan->use_vec4)
            {
                return failStep("Phase3B Pointwise broadcast fusion plan invalid");
            }
        }
        flushGpuEngine(true);

        if (!verifyMatrix(b1_g, b1_c.getData(), 2e-2f) ||
            !verifyMatrix(b2_g, b2_c.getData(), 2e-2f) ||
            !verifyMatrix(b3_g, b3_c.getData(), 2e-2f))
        {
            return failStep("Phase3B Pointwise broadcast numeric parity mismatch");
        }
    }

    for (size_t dim : {size_t{128}, size_t{384}})
    {
        constexpr size_t batch = 16;
        std::vector<float> a_v(batch * dim), b_v(batch * dim), g_v(dim);
        for (size_t i = 0; i < a_v.size(); ++i)
        {
            a_v[i] = 0.06f * static_cast<float>(static_cast<int>(i % 13) - 6);
            b_v[i] = 0.05f * static_cast<float>(static_cast<int>((i * 3) % 11) - 5);
        }
        for (size_t c = 0; c < dim; ++c) g_v[c] = 0.7f + 0.02f * static_cast<float>(c % 11);

        {
            Tensor a_c(Shape{batch, dim}, a_v, Execution_Target::CPU), b_c(Shape{batch, dim}, b_v, Execution_Target::CPU), gam_c(Shape{1, dim}, g_v, Execution_Target::CPU);
            Tensor res_c(Execution_Target::CPU), inv_c(Execution_Target::CPU), out_c(Execution_Target::CPU), sc_c(Execution_Target::CPU);
            a_c.add(b_c, res_c);
            ops::normalization::rmsNormForward(res_c, gam_c, inv_c, out_c, 1e-5f);
            out_c.mulScalar(1.25f, sc_c);

            Tensor a_g = castTensorToFp16(Tensor(Shape{batch, dim}, a_v, Execution_Target::VULKAN_GPU));
            Tensor b_g = castTensorToFp16(Tensor(Shape{batch, dim}, b_v, Execution_Target::VULKAN_GPU));
            Tensor gam_g = castTensorToFp16(Tensor(Shape{1, dim}, g_v, Execution_Target::VULKAN_GPU));
            flushGpuEngine(true);

            Tensor res_g(Execution_Target::VULKAN_GPU), inv_g(Execution_Target::VULKAN_GPU), out_g(Execution_Target::VULKAN_GPU), sc_g(Execution_Target::VULKAN_GPU);
            a_g.add(b_g, res_g);
            ops::normalization::rmsNormForward(res_g, gam_g, inv_g, out_g, 1e-5f);
            out_g.mulScalar(1.25f, sc_g);
            {
                Compute_Graph copy = engine.getCurrentGraph();
                Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
                if (copy.getNodeCount() != 1 || !copy.getNode(0).fusion_plan ||
                    copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::ROW_REDUCTION)
                {
                    return failStep("Phase4A RowReduction out-of-place fusion plan invalid");
                }
            }
            flushGpuEngine(true);

            if (!verifyMatrix(res_g, res_c.getData(), 2e-2f) ||
                !verifyMatrix(inv_g, inv_c.getData(), 2e-2f) ||
                !verifyMatrix(out_g, out_c.getData(), 2e-2f) ||
                !verifyMatrix(sc_g, sc_c.getData(), 2e-2f))
            {
                return failStep("Phase4A RowReduction out-of-place parity mismatch");
            }
        }

        {
            Tensor res_c(Shape{batch, dim}, a_v, Execution_Target::CPU), b_c(Shape{batch, dim}, b_v, Execution_Target::CPU), gam_c(Shape{1, dim}, g_v, Execution_Target::CPU);
            Tensor inv_c(Execution_Target::CPU), out_c(Execution_Target::CPU);
            res_c.add(b_c, res_c);
            ops::normalization::rmsNormForward(res_c, gam_c, inv_c, out_c, 1e-5f);

            Tensor res_g = castTensorToFp16(Tensor(Shape{batch, dim}, a_v, Execution_Target::VULKAN_GPU));
            Tensor b_g = castTensorToFp16(Tensor(Shape{batch, dim}, b_v, Execution_Target::VULKAN_GPU));
            Tensor gam_g = castTensorToFp16(Tensor(Shape{1, dim}, g_v, Execution_Target::VULKAN_GPU));
            flushGpuEngine(true);

            Tensor inv_g(Execution_Target::VULKAN_GPU), out_g(Execution_Target::VULKAN_GPU);
            res_g.add(b_g, res_g);
            ops::normalization::rmsNormForward(res_g, gam_g, inv_g, out_g, 1e-5f);
            {
                Compute_Graph copy = engine.getCurrentGraph();
                Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
                if (copy.getNodeCount() != 1 || !copy.getNode(0).fusion_plan ||
                    copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::ROW_REDUCTION)
                {
                    return failStep("Phase4A RowReduction in-place fusion plan invalid");
                }
            }
            flushGpuEngine(true);

            if (!verifyMatrix(res_g, res_c.getData(), 2e-2f) ||
                !verifyMatrix(inv_g, inv_c.getData(), 2e-2f) ||
                !verifyMatrix(out_g, out_c.getData(), 2e-2f))
            {
                return failStep("Phase4A RowReduction in-place parity mismatch");
            }
        }
    }

    {
        Scoped_Engine_Flags_Guard flags_guard;
        if (engine.isCooperativeMatrixSupported())
        {
            engine.setCooperativeMatrixEnabled(true);
        }
        engine.waitIdle();
        engine.invalidateGraphCache();
        engine.enableStaticGraph(true);

        const std::array<Shape, 6> lane_shapes = {
            Shape{8, 130},
            Shape{5, 37},
            Shape{4, 64},
            Shape{6, 51},
            Shape{3, 45},
            Shape{7, 19}
        };

        std::vector<Tensor> p_cpu, g_cpu, p_gpu, g_gpu;
        p_cpu.reserve(6);
        g_cpu.reserve(6);
        p_gpu.reserve(6);
        g_gpu.reserve(6);

        for (size_t k = 0; k < 6; ++k)
        {
            const size_t n = lane_shapes[k].getTotalElements();
            std::vector<float> pv(n), gv(n);
            for (size_t i = 0; i < n; ++i)
            {
                pv[i] = 0.05f * static_cast<float>(static_cast<int>((i + k * 3) % 13) - 6);
                gv[i] = 0.04f * static_cast<float>(static_cast<int>((i * 5 + k) % 11) - 5);
            }
            p_cpu.emplace_back(lane_shapes[k], pv, Execution_Target::CPU);
            g_cpu.emplace_back(lane_shapes[k], gv, Execution_Target::CPU);
            p_gpu.emplace_back(lane_shapes[k], pv, Execution_Target::VULKAN_GPU);
            g_gpu.emplace_back(lane_shapes[k], gv, Execution_Target::VULKAN_GPU);
            auto &pg = Gpu_Tensor_Impl::castToGpu(*p_gpu.back().getImplementation());
            if (!pg.getCachedFp16Storage() || pg.getCachedFp16Storage()->getSize() < n)
            {
                pg.setCachedFp16Storage(std::make_shared<gpu::vector>(engine.getContext(), n, Data_Type::FLOAT16));
            }
        }

        std::vector<std::pair<Tensor *, Tensor *>> cpu_pairs, gpu_pairs;
        for (size_t k = 0; k < 6; ++k)
        {
            cpu_pairs.emplace_back(&p_cpu[k], &g_cpu[k]);
            gpu_pairs.emplace_back(&p_gpu[k], &g_gpu[k]);
        }

        Adam_Optimizer opt_cpu(1e-3f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        Adam_Optimizer opt_gpu(1e-3f, 0.9f, 0.999f, 1e-8f, 1.0f, 0.01f);
        opt_cpu.addNoDecayParameter(&p_cpu[1]);
        opt_cpu.addNoDecayParameter(&p_cpu[4]);
        opt_gpu.addNoDecayParameter(&p_gpu[1]);
        opt_gpu.addNoDecayParameter(&p_gpu[4]);

        uint64_t replay_hits = 0;
        for (int step = 0; step < 4; ++step)
        {
            opt_cpu.step(cpu_pairs, 1.0f);
            opt_gpu.step(gpu_pairs, 1.0f);
            if (step == 0)
            {
                Compute_Graph copy = engine.getCurrentGraph();
                Graph_Optimizer::optimize(copy, &engine.getShaderDictionary());
                if (copy.getNodeCount() != 1 || !copy.getNode(0).is_fused || !copy.getNode(0).fusion_plan ||
                    copy.getNode(0).fusion_plan->anchor_role != Fused_Anchor_Role::HORIZONTAL_MULTI_LANE ||
                    copy.getNode(0).fusion_plan->horizontal_lanes.size() != 6)
                {
                    return failStep("Phase4B Horizontal Multi-Lane Adam did not fuse 6 lanes into 1 node");
                }
            }
            const uint32_t cur_f = engine.getContext().getCurrentFrame();
            const bool baked_before = engine.getGraphExecutor().isStaticBaked(cur_f);
            const uint64_t gen_before = engine.getGraphExecutor().getStaticBakeGeneration(cur_f);
            engine.executeGraph();
            const uint64_t gen_after = engine.getGraphExecutor().getStaticBakeGeneration(cur_f);
            if (step >= 2 && baked_before && gen_before == gen_after)
            {
                ++replay_hits;
            }
        }
        engine.waitIdle();

        if (replay_hits != 2)
        {
            return failStep("Phase4B Horizontal Multi-Lane Adam static graph replay failed on steps 2..3");
        }

        for (size_t k = 0; k < 6; ++k)
        {
            const auto expected = p_cpu[k].getData();
            if (!verifyMatrix(p_gpu[k], expected, 1e-3f))
            {
                return failStep("Phase4B Horizontal Multi-Lane Adam FP32 parameter parity mismatch");
            }
            const auto &pg = Gpu_Tensor_Impl::castToGpu(*p_gpu[k].getImplementation());
            const auto fp16_buf = pg.getCachedFp16Storage();
            if (!fp16_buf)
            {
                return failStep("Phase4B Horizontal Multi-Lane Adam missing FP16 cache buffer");
            }
            std::vector<float16_t> h16(expected.size());
            fp16_buf->downloadData(h16);
            std::vector<float> h32(expected.size());
            convertFp16ToFp32(h16.data(), h32.data(), expected.size());
            if (!verifyVectorsClose(h32, expected, 5e-3f))
            {
                return failStep("Phase4B Horizontal Multi-Lane Adam FP16 cache parity mismatch");
            }
        }
    }

    return true;
}

void runPerfPhasesSuite(Test_Result_Counter &counter)
{
    counter.record("  Phase 1 Kernel Optimizations:      ", testPhase1KernelOptimizations());
    counter.record("  Phase B CoopMat GEMM (64x64x32):   ", testPhaseBCoopMatGemm());
    counter.record("  Phase C FlashAttention-2 & BSHD:   ", testPhaseCFlashAttention2());
    counter.record("  Phase D Optimizations Suite:       ", testPhaseDOptimizations());
    counter.record("  Phase 2 Kernel Optimizations:      ", testPhase2KernelOptimizations());
    counter.record("  Phase 3&4 Unified Fusion & Cache:  ", testPhase3And4UnifiedFusion());
}
