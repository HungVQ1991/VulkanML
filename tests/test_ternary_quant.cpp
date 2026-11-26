#include "test_common.h"

bool testTernaryComputingPhase1()
{
    size_t N = 3;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 1})) != 12) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 15})) != 12) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 16})) != 12) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 17})) != 24) return false;
    if (getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape({N, 33})) != 36) return false;

    size_t test_rows = 2;
    size_t test_cols = 33;
    std::vector<float> original(test_rows * test_cols);
    for (size_t i = 0; i < original.size(); ++i)
    {
        int mod = static_cast<int>(i % 3);
        original[i] = (mod == 0) ? -1.0f : ((mod == 1) ? 0.0f : 1.0f);
    }
    size_t words_per_row = (test_cols + 15) / 16;
    std::vector<uint32_t> packed(test_rows * words_per_row, 0);
    Cpu_Tensor_Impl::packTernary(original.data(), packed.data(), test_rows, test_cols);

    std::vector<float> unpacked(test_rows * test_cols, 0.0f);
    float gamma = 0.75f;
    Cpu_Tensor_Impl::unpackTernary(packed.data(), unpacked.data(), test_rows, test_cols, gamma);

    for (size_t i = 0; i < original.size(); ++i)
    {
        float expected = original[i] * gamma;
        if (std::fabs(unpacked[i] - expected) > 1e-6f)
        {
            return false;
        }
    }

    size_t M = 2, K = 32, N_out = 4;
    std::vector<float> A(M * K);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i % 7) - 3.0f;

    std::vector<float> W_raw(N_out * K);
    for (size_t i = 0; i < W_raw.size(); ++i)
    {
        int mod = static_cast<int>(i % 3);
        W_raw[i] = (mod == 0) ? -1.0f : ((mod == 1) ? 0.0f : 1.0f);
    }
    std::vector<float> biases = {0.5f, -0.5f, 1.0f, -1.0f};

    size_t w_words_per_row = (K + 15) / 16;
    std::vector<uint32_t> B_packed(N_out * w_words_per_row);
    Cpu_Tensor_Impl::packTernary(W_raw.data(), B_packed.data(), N_out, K);

    std::vector<float> C_ternary(M * N_out, 0.0f);
    float test_gamma = 0.4f;
    Cpu_Tensor_Impl::ternaryMatmul(A.data(), B_packed.data(), C_ternary.data(), M, K, N_out, test_gamma, biases.data());

    std::vector<float> C_ref(M * N_out, 0.0f);
    for (size_t m = 0; m < M; ++m)
    {
        for (size_t n = 0; n < N_out; ++n)
        {
            float sum = biases[n];
            for (size_t k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * (W_raw[n * K + k] * test_gamma);
            }
            C_ref[m * N_out + n] = sum;
        }
    }

    for (size_t i = 0; i < C_ternary.size(); ++i)
    {
        if (std::fabs(C_ternary[i] - C_ref[i]) > 1e-4f)
        {
            return false;
        }
    }

    Tensor t_ternary(Shape({N_out, K}), Data_Type::TERNARY_INT2);
    bool caught_add = false;
    try { auto res = t_ternary + t_ternary; (void)res; } catch (const std::invalid_argument&) { caught_add = true; }
    if (!caught_add) return false;

    bool caught_relu = false;
    try { auto res = t_ternary.relu(); (void)res; } catch (const std::invalid_argument&) { caught_relu = true; }
    if (!caught_relu) return false;

    bool caught_transpose = false;
    try { auto res = t_ternary.transpose(); (void)res; } catch (const std::invalid_argument&) { caught_transpose = true; }
    if (!caught_transpose) return false;

    return true;
}

bool testTernaryComputingPhase2()
{
    auto &engine = Execution_Engine::getInstance();
    auto &context = engine.getContext();
    if (!context.isFloat16Supported() || !context.isFloat16Enabled())
    {
        std::cout << "[INFO: GPU does not enable/support FP16, skipping Ternary Phase 2 test] ";
        return true;
    }

    auto pipe_p1 = engine.getNetwork().getPipeline(Compute_Pipeline::REDUCE_ABSMEAN_PASS1);
    auto pipe_p2 = engine.getNetwork().getPipeline(Compute_Pipeline::REDUCE_ABSMEAN_PASS2);
    auto pipe_fq = engine.getNetwork().getPipeline(Compute_Pipeline::FAKEQUANT_TERNARY);
    auto pipe_qp = engine.getNetwork().getPipeline(Compute_Pipeline::QUANTIZE_PACK_TERNARY);
    auto pipe_tg = engine.getNetwork().getPipeline(Compute_Pipeline::MATMUL_TERNARY_FP16);

    if (pipe_p1 == VK_NULL_HANDLE || pipe_p2 == VK_NULL_HANDLE || pipe_fq == VK_NULL_HANDLE ||
        pipe_qp == VK_NULL_HANDLE || pipe_tg == VK_NULL_HANDLE)
    {
        std::cout << "[ERROR: One or more Ternary pipelines failed to compile] ";
        return false;
    }

    uint32_t K = 64;
    uint32_t N = 128;
    uint32_t total_elements = K * N;

    std::vector<float> host_w(total_elements);
    double cpu_sum_abs = 0.0;
    for (size_t i = 0; i < total_elements; ++i)
    {
        float val = std::sin(static_cast<float>(i + 1) * 0.123f);
        host_w[i] = val;
        cpu_sum_abs += std::fabs(val);
    }
    float cpu_gamma = std::max(static_cast<float>(cpu_sum_abs / static_cast<double>(total_elements)), 1e-5f);
    float cpu_inv_gamma = 1.0f / cpu_gamma;

    auto w_buf = std::make_shared<gpu::vector>(context, total_elements, Data_Type::FLOAT32);
    w_buf->uploadData(host_w);

    uint32_t num_workgroups = std::min((total_elements + 255u) / 256u, 1024u);
    auto partials_buf = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);

    auto gamma_buf = std::make_shared<gpu::vector>(context);
    gamma_buf->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS1;
        node.buffers = { w_buf, partials_buf };
        struct PushConsts { uint32_t total; } pcs{ total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = num_workgroups;
        engine.getCurrentGraph().addNode(std::move(node));
    }
    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS2;
        node.buffers = { partials_buf, gamma_buf };
        struct PushConsts { uint32_t num_partials; uint32_t total; } pcs{ num_workgroups, total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = 1;
        engine.getCurrentGraph().addNode(std::move(node));
    }
    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<float> host_gamma(2, 0.0f);
    gamma_buf->downloadData(host_gamma);
    float gpu_gamma = host_gamma[0];
    float gpu_inv_gamma = host_gamma[1];

    if (std::fabs(gpu_gamma - cpu_gamma) / cpu_gamma > 1e-4f)
    {
        std::cout << "[ERROR: Gamma mismatch: GPU=" << gpu_gamma << ", CPU=" << cpu_gamma << "] ";
        return false;
    }
    if (std::fabs(gpu_inv_gamma - cpu_inv_gamma) / cpu_inv_gamma > 1e-4f)
    {
        std::cout << "[ERROR: InvGamma mismatch: GPU=" << gpu_inv_gamma << ", CPU=" << cpu_inv_gamma << "] ";
        return false;
    }

    auto w_eff_buf = std::make_shared<gpu::vector>(context, total_elements, Data_Type::FLOAT16);
    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::FAKEQUANT_TERNARY;
        node.buffers = { w_buf, gamma_buf, w_eff_buf };
        struct PushConsts { uint32_t total; } pcs{ total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (total_elements + 255u) / 256u;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<float16_t> host_w_eff(total_elements);
    w_eff_buf->downloadData(host_w_eff);

    for (size_t i = 0; i < total_elements; ++i)
    {
        float w = host_w[i];
        float q = std::clamp(std::floor(std::fabs(w) * cpu_inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
        float expected = q * cpu_gamma;
        float actual = static_cast<float>(host_w_eff[i]);
        if (std::fabs(actual - expected) > 1e-3f * (std::fabs(expected) + 1.0f))
        {
            std::cout << "[ERROR: FakeQuant mismatch at " << i << ": expected " << expected << ", got " << actual << "] ";
            return false;
        }
    }

    uint32_t words_per_row = (K + 15u) / 16u;
    uint32_t total_words = N * words_per_row;
    auto packed_buf = std::make_shared<gpu::vector>(context);
    packed_buf->allocateMemoryByBytes(total_words * sizeof(uint32_t), Data_Type::TERNARY_INT2, total_words);

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::QUANTIZE_PACK_TERNARY;
        node.buffers = { w_buf, gamma_buf, packed_buf };
        struct PushConsts { uint32_t K; uint32_t N; uint32_t words_per_row; } pcs{ K, N, words_per_row };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (words_per_row + 15u) / 16u;
        node.workgroup_count_y = (N + 15u) / 16u;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<uint32_t> gpu_packed(total_words, 0);
    packed_buf->downloadData(gpu_packed);

    std::vector<uint32_t> cpu_packed(total_words, 0);
    for (uint32_t n = 0; n < N; ++n)
    {
        for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
        {
            uint32_t word = 0u;
            for (uint32_t b = 0; b < 16; ++b)
            {
                uint32_t k = w_idx * 16 + b;
                if (k < K)
                {
                    float w = host_w[k * N + n];
                    float q = std::clamp(std::floor(std::fabs(w) * cpu_inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
                    uint32_t code = static_cast<uint32_t>(static_cast<int>(q)) & 3u;
                    word |= (code << (b * 2u));
                }
            }
            cpu_packed[n * words_per_row + w_idx] = word;
        }
    }

    for (size_t i = 0; i < total_words; ++i)
    {
        if (gpu_packed[i] != cpu_packed[i])
        {
            std::cout << "[ERROR: QuantizePack bit-exact mismatch at word " << i << ": GPU=" << gpu_packed[i] << ", CPU=" << cpu_packed[i] << "] ";
            return false;
        }
    }

    uint32_t M = 4;
    std::vector<float16_t> host_x(M * K);
    for (size_t i = 0; i < host_x.size(); ++i)
    {
        host_x[i] = static_cast<float16_t>(std::cos(static_cast<float>(i) * 0.2f) * 2.0f);
    }
    std::vector<float16_t> host_bias(N);
    for (size_t i = 0; i < N; ++i)
    {
        host_bias[i] = static_cast<float16_t>(0.1f * static_cast<float>(i % 5));
    }

    auto x_buf = std::make_shared<gpu::vector>(context, M * K, Data_Type::FLOAT16);
    x_buf->uploadData(host_x);

    auto bias_buf = std::make_shared<gpu::vector>(context, N, Data_Type::FLOAT16);
    bias_buf->uploadData(host_bias);

    auto y_buf = std::make_shared<gpu::vector>(context, M * N, Data_Type::FLOAT16);

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::MATMUL_TERNARY_FP16;
        node.buffers = { x_buf, packed_buf, bias_buf, gamma_buf, y_buf };
        struct PushConsts { uint32_t M; uint32_t K; uint32_t N; uint32_t words_per_row; uint32_t has_bias; }
            pcs{ M, K, N, words_per_row, 1u };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (N + 15u) / 16u;
        node.workgroup_count_y = (M + 15u) / 16u;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    engine.waitIdle();

    std::vector<float16_t> gpu_y(M * N);
    y_buf->downloadData(gpu_y);

    for (uint32_t m = 0; m < M; ++m)
    {
        for (uint32_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
            {
                uint32_t pval = cpu_packed[n * words_per_row + w_idx];
                for (uint32_t b = 0; b < 16; ++b)
                {
                    uint32_t k = w_idx * 16 + b;
                    if (k < K)
                    {
                        uint32_t code = (pval >> (b * 2u)) & 3u;
                        int weight = static_cast<int>(code & 1u) * (1 - static_cast<int>(code & 2u));
                        acc += static_cast<float>(host_x[m * K + k]) * static_cast<float>(weight);
                    }
                }
            }
            acc = acc * cpu_gamma + static_cast<float>(host_bias[n]);
            float actual = static_cast<float>(gpu_y[m * N + n]);
            if (std::fabs(actual - acc) > 5e-2f * (std::fabs(acc) + 1.0f))
            {
                std::cout << "[ERROR: TGEMM mismatch at (" << m << ", " << n << "): GPU=" << actual << ", CPU=" << acc << "] ";
                return false;
            }
        }
    }

    return true;
}

bool testTernaryComputingPhase4()
{
    {
        size_t K = 32;
        size_t N = 16;
        size_t M = 4;
        Ternary_Linear_Layer layer(K, N, Execution_Target::CPU);

        auto params_and_grads = layer.getParametersAndGradients();
        if (params_and_grads.size() != 2) return false;
        if (params_and_grads[0].first->getShape() != Shape{ K, N }) return false;
        if (params_and_grads[1].first->getShape() != Shape{ 1, N }) return false;

        std::vector<float> input_data(M * K, 1.0f);
        Tensor in(M, K, input_data, Execution_Target::CPU);
        Tensor out = layer.forward(in);
        if (out.getShape() != Shape{ M, N }) return false;

        std::vector<float> grad_data(M * N, 0.5f);
        Tensor grad_out(M, N, grad_data, Execution_Target::CPU);
        Tensor grad_in = layer.backward(grad_out);
        if (grad_in.getShape() != Shape{ M, K }) return false;

        const auto &w_grad = layer.getWeightsGradient().getData();
        bool has_grad = false;
        for (float g : w_grad)
        {
            if (std::abs(g) > 1e-6f)
            {
                has_grad = true;
                break;
            }
        }
        if (!has_grad) return false;

        layer.prepareInference();
        if (!layer.hasPackedInference()) return false;
        Tensor out_infer = layer.forward(in);
        if (out_infer.getShape() != Shape{ M, N }) return false;

        layer.setTrainingMode(true);
        if (layer.hasPackedInference()) return false;
        Tensor out_train = layer.forward(in);
        if (out_train.getShape() != Shape{ M, N }) return false;

        const std::string ckpt_path = makeUniqueTestTempPath("test_ternary_ckpt").string();
        Scoped_Temp_File_Guard ckpt_guard({ckpt_path});
        {
            std::ofstream ofs(ckpt_path, std::ios::binary);
            layer.saveCheckpoint(ofs);
        }
        Ternary_Linear_Layer loaded_ckpt(K, N, Execution_Target::CPU);
        {
            std::ifstream ifs(ckpt_path, std::ios::binary);
            loaded_ckpt.loadCheckpoint(ifs);
        }
        if (loaded_ckpt.getWeights().getShape() != Shape{ K, N }) return false;

        layer.prepareInference();
        const std::string infer_path = makeUniqueTestTempPath("test_ternary_infer").string();
        Scoped_Temp_File_Guard infer_guard({infer_path});
        {
            std::ofstream ofs(infer_path, std::ios::binary);
            layer.saveInference(ofs);
        }
        Ternary_Linear_Layer loaded_infer(K, N, Execution_Target::CPU);
        {
            std::ifstream ifs(infer_path, std::ios::binary);
            loaded_infer.loadInference(ifs);
        }
        if (!loaded_infer.hasPackedInference()) return false;
        Tensor out_loaded = loaded_infer.forward(in);
        if (out_loaded.getShape() != Shape{ M, N }) return false;

        bool threw_on_backward = false;
        try
        {
            loaded_infer.backward(grad_out);
        }
        catch (const std::runtime_error &)
        {
            threw_on_backward = true;
        }
        if (!threw_on_backward) return false;

        bool threw_on_train = false;
        try
        {
            loaded_infer.setTrainingMode(true);
        }
        catch (const std::runtime_error &)
        {
            threw_on_train = true;
        }
        if (!threw_on_train) return false;
    }

    {
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            size_t K = 64;
            size_t N = 32;
            size_t M = 4;
            Ternary_Linear_Layer gpu_layer(K, N, Execution_Target::VULKAN_GPU);

            std::vector<float> input_data(M * K, 0.5f);
            Tensor in(M, K, input_data, Execution_Target::VULKAN_GPU);
            Tensor out = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out.getShape() != Shape{ M, N }) return false;

            std::vector<float> grad_data(M * N, 0.2f);
            Tensor grad_out(M, N, grad_data, Execution_Target::VULKAN_GPU);
            Tensor grad_in = gpu_layer.backward(grad_out);
            engine.executeGraph();
            engine.waitIdle();
            if (grad_in.getShape() != Shape{ M, K }) return false;

            gpu_layer.prepareInference();
            engine.executeGraph();
            engine.waitIdle();
            if (!gpu_layer.hasPackedInference()) return false;

            Tensor out_infer = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_infer.getShape() != Shape{ M, N }) return false;

            const std::string gpu_infer_path = makeUniqueTestTempPath("test_ternary_gpu_infer").string();
            Scoped_Temp_File_Guard gpu_infer_guard({gpu_infer_path});
            {
                std::ofstream ofs(gpu_infer_path, std::ios::binary);
                gpu_layer.saveInference(ofs);
            }
            Ternary_Linear_Layer loaded_gpu(K, N, Execution_Target::VULKAN_GPU);
            {
                std::ifstream ifs(gpu_infer_path, std::ios::binary);
                loaded_gpu.loadInference(ifs);
            }
            if (!loaded_gpu.hasPackedInference()) return false;

            Tensor out_loaded_gpu = loaded_gpu.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_loaded_gpu.getShape() != Shape{ M, N }) return false;
        }
    }

    return true;
}

bool testTernaryComputingPhase5()
{
    {
        size_t K = 128;
        size_t N = 64;
        size_t M = 8;

        {
            Ternary_Linear_Layer cpu_layer(K, N, Execution_Target::CPU);
            std::vector<float> in_data(M * K);
            for (size_t i = 0; i < in_data.size(); ++i)
            {
                in_data[i] = std::sin(static_cast<float>(i) * 0.1f);
            }
            Tensor in(M, K, in_data, Execution_Target::CPU);

            Tensor out_fake = cpu_layer.forward(in);
            auto data_fake = out_fake.getData();

            cpu_layer.prepareInference();
            Tensor out_infer = cpu_layer.forward(in);
            auto data_infer = out_infer.getData();

            double diff_sq = 0.0;
            double norm_sq = 0.0;
            for (size_t i = 0; i < data_fake.size(); ++i)
            {
                double d = data_infer[i] - data_fake[i];
                diff_sq += d * d;
                norm_sq += data_fake[i] * data_fake[i];
            }
            double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
            if (rel_err > 1e-3)
            {
                std::cout << "[ERROR: CPU Train -> Infer drift error: " << rel_err << "] ";
                return false;
            }
        }

        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            Ternary_Linear_Layer gpu_layer(K, N, Execution_Target::VULKAN_GPU);
            std::vector<float> in_data(M * K);
            for (size_t i = 0; i < in_data.size(); ++i)
            {
                in_data[i] = std::sin(static_cast<float>(i) * 0.1f);
            }
            Tensor in(M, K, in_data, Execution_Target::VULKAN_GPU);

            Tensor out_fake = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            auto data_fake = out_fake.getData();

            gpu_layer.prepareInference();
            engine.executeGraph();
            engine.waitIdle();

            Tensor out_infer = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            auto data_infer = out_infer.getData();

            double diff_sq = 0.0;
            double norm_sq = 0.0;
            for (size_t i = 0; i < data_fake.size(); ++i)
            {
                double d = data_infer[i] - data_fake[i];
                diff_sq += d * d;
                norm_sq += data_fake[i] * data_fake[i];
            }
            double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
            if (rel_err > 0.01)
            {
                std::cout << "[ERROR: GPU Train -> Infer drift error: " << rel_err << "] ";
                return false;
            }
        }
    }

    {
        std::vector<size_t> test_k = { 1, 15, 17, 33 };
        size_t N = 8;
        size_t M = 2;

        for (size_t K : test_k)
        {
            size_t expected_words_per_row = (K + 15) / 16;
            size_t expected_bytes = N * expected_words_per_row * sizeof(uint32_t);
            size_t actual_bytes = getTensorStorageBytes(Data_Type::TERNARY_INT2, Shape{ N, K });
            if (actual_bytes != expected_bytes)
            {
                std::cout << "[ERROR: getTensorStorageBytes mismatch for K=" << K << "] ";
                return false;
            }

            {
                Ternary_Linear_Layer layer(K, N, Execution_Target::CPU);
                std::vector<float> in_data(M * K, 0.3f);
                Tensor in(M, K, in_data, Execution_Target::CPU);
                Tensor out = layer.forward(in);
                if (out.getShape() != Shape{ M, N }) return false;

                std::vector<float> g_data(M * N, 0.1f);
                Tensor g_out(M, N, g_data, Execution_Target::CPU);
                Tensor g_in = layer.backward(g_out);
                if (g_in.getShape() != Shape{ M, K }) return false;

                layer.prepareInference();
                Tensor out_inf = layer.forward(in);
                if (out_inf.getShape() != Shape{ M, N }) return false;
            }

            auto &engine = Execution_Engine::getInstance();
            auto &context = engine.getContext();
            if (context.isFloat16Supported() && context.isFloat16Enabled())
            {
                Ternary_Linear_Layer layer(K, N, Execution_Target::VULKAN_GPU);
                std::vector<float> in_data(M * K, 0.3f);
                Tensor in(M, K, in_data, Execution_Target::VULKAN_GPU);
                Tensor out = layer.forward(in);
                engine.executeGraph();
                engine.waitIdle();
                if (out.getShape() != Shape{ M, N }) return false;

                std::vector<float> g_data(M * N, 0.1f);
                Tensor g_out(M, N, g_data, Execution_Target::VULKAN_GPU);
                Tensor g_in = layer.backward(g_out);
                engine.executeGraph();
                engine.waitIdle();
                if (g_in.getShape() != Shape{ M, K }) return false;

                layer.prepareInference();
                engine.executeGraph();
                engine.waitIdle();
                Tensor out_inf = layer.forward(in);
                engine.executeGraph();
                engine.waitIdle();
                if (out_inf.getShape() != Shape{ M, N }) return false;
            }
        }
    }

    return true;
}

bool testUnifiedLinearLayer()
{
    {
        Linear_Layer fp32_l(64, 32, Execution_Target::CPU, 2.0f, Data_Type::FLOAT32);
        if (fp32_l.isTernary() || fp32_l.getDataType() != Data_Type::FLOAT32) return false;
        if (!fp32_l.getMasterWeights().isEmpty()) return false;

        Linear_Layer fp16_l(64, 32, Execution_Target::CPU, 2.0f, Data_Type::FLOAT16);
        if (fp16_l.isTernary() || !fp16_l.isMixedPrecisionEnabled()) return false;
        if (!fp16_l.getMasterWeights().isEmpty()) return false;

        Linear_Layer ternary_l(64, 32, Execution_Target::CPU, 2.0f, Data_Type::TERNARY_INT2);
        if (!ternary_l.isTernary() || ternary_l.getDataType() != Data_Type::TERNARY_INT2) return false;
        if (ternary_l.getMasterWeights().isEmpty()) return false;
        if (ternary_l.getLayerType() != Layer_Type::TERNARY_LINEAR) return false;
    }

    {
        size_t K = 64;
        size_t N = 32;
        size_t M = 4;
        Linear_Layer layer(K, N, Execution_Target::CPU, 2.0f, Data_Type::TERNARY_INT2);

        std::vector<float> in_data(M * K);
        for (size_t i = 0; i < in_data.size(); ++i)
        {
            in_data[i] = std::sin(static_cast<float>(i) * 0.15f);
        }
        Tensor in(M, K, in_data, Execution_Target::CPU);

        Tensor out_fake = layer.forward(in);
        if (out_fake.getShape() != Shape{ M, N }) return false;
        auto fake_data = out_fake.getData();

        std::vector<float> grad_data(M * N, 0.1f);
        Tensor grad_out(M, N, grad_data, Execution_Target::CPU);
        Tensor grad_in = layer.backward(grad_out);
        if (grad_in.getShape() != Shape{ M, K }) return false;

        auto params = layer.getParametersAndGradients();
        if (params.size() != 2) return false;
        if (params[0].first != &layer.getMasterWeights()) return false;

        layer.prepareInference();
        if (!layer.hasPackedInference()) return false;

        Tensor out_infer = layer.forward(in);
        if (out_infer.getShape() != Shape{ M, N }) return false;
        auto infer_data = out_infer.getData();

        double diff_sq = 0.0, norm_sq = 0.0;
        for (size_t i = 0; i < fake_data.size(); ++i)
        {
            double d = infer_data[i] - fake_data[i];
            diff_sq += d * d;
            norm_sq += fake_data[i] * fake_data[i];
        }
        double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
        if (rel_err > 1e-3) return false;
    }

    {
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            size_t K = 64;
            size_t N = 32;
            size_t M = 4;
            Linear_Layer gpu_layer(K, N, Execution_Target::VULKAN_GPU, 2.0f, Data_Type::TERNARY_INT2);

            std::vector<float> in_data(M * K);
            for (size_t i = 0; i < in_data.size(); ++i)
            {
                in_data[i] = std::sin(static_cast<float>(i) * 0.15f);
            }
            Tensor in(M, K, in_data, Execution_Target::VULKAN_GPU);

            Tensor out_fake = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_fake.getShape() != Shape{ M, N }) return false;
            auto fake_data = out_fake.getData();

            std::vector<float> grad_data(M * N, 0.1f);
            Tensor grad_out(M, N, grad_data, Execution_Target::VULKAN_GPU);
            Tensor grad_in = gpu_layer.backward(grad_out);
            engine.executeGraph();
            engine.waitIdle();
            if (grad_in.getShape() != Shape{ M, K }) return false;

            gpu_layer.prepareInference();
            engine.executeGraph();
            engine.waitIdle();
            if (!gpu_layer.hasPackedInference()) return false;

            Tensor out_infer = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_infer.getShape() != Shape{ M, N }) return false;
            auto infer_data = out_infer.getData();

            double diff_sq = 0.0, norm_sq = 0.0;
            for (size_t i = 0; i < fake_data.size(); ++i)
            {
                double d = infer_data[i] - fake_data[i];
                diff_sq += d * d;
                norm_sq += fake_data[i] * fake_data[i];
            }
            double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
            if (rel_err > 0.01) return false;
        }
    }

    {
        size_t K = 32;
        size_t N = 16;
        size_t M = 2;
        Linear_Layer src_layer(K, N, Execution_Target::CPU, 2.0f, Data_Type::TERNARY_INT2);
        std::vector<float> in_data(M * K, 0.4f);
        Tensor in(M, K, in_data, Execution_Target::CPU);

        src_layer.prepareInference();
        Tensor expected_out = src_layer.forward(in);

        const std::string infer_file = makeUniqueTestTempPath("test_unified_infer").string();
        Scoped_Temp_File_Guard infer_guard({infer_file});
        {
            std::ofstream ofs(infer_file, std::ios::binary);
            src_layer.saveInference(ofs);
        }
        Linear_Layer loaded_layer(K, N, Execution_Target::CPU);
        {
            std::ifstream ifs(infer_file, std::ios::binary);
            loaded_layer.loadInference(ifs);
        }

        if (!loaded_layer.isTernary() || !loaded_layer.hasPackedInference()) return false;
        Tensor actual_out = loaded_layer.forward(in);
        auto exp_d = expected_out.getData();
        auto act_d = actual_out.getData();
        for (size_t i = 0; i < exp_d.size(); ++i)
        {
            if (std::abs(exp_d[i] - act_d[i]) > 1e-5f) return false;
        }

        const std::string ckpt_file = makeUniqueTestTempPath("test_unified_ckpt").string();
        Scoped_Temp_File_Guard ckpt_guard({ckpt_file});
        Linear_Layer train_src(K, N, Execution_Target::CPU, 2.0f, Data_Type::TERNARY_INT2);
        {
            std::ofstream ofs(ckpt_file, std::ios::binary);
            train_src.saveCheckpoint(ofs);
        }
        Linear_Layer loaded_ckpt(K, N, Execution_Target::CPU);
        {
            std::ifstream ifs(ckpt_file, std::ios::binary);
            loaded_ckpt.loadCheckpoint(ifs);
        }

        if (!loaded_ckpt.isTernary() || !loaded_ckpt.hasMasterWeights()) return false;
    }

    {
        size_t K = 32;
        size_t N = 16;
        size_t M = 2;
        Linear_Layer ptq_layer(K, N, Execution_Target::CPU, 2.0f, Data_Type::FLOAT32);
        if (ptq_layer.isTernary()) return false;

        ptq_layer.quantizeToTernary();
        if (!ptq_layer.isTernary() || !ptq_layer.hasPackedInference()) return false;

        std::vector<float> in_data(M * K, 0.5f);
        Tensor in(M, K, in_data, Execution_Target::CPU);
        Tensor out = ptq_layer.forward(in);
        if (out.getShape() != Shape{ M, N }) return false;
    }

    {
        std::vector<size_t> test_k = { 1, 15, 17, 33 };
        size_t N = 8;
        size_t M = 2;
        for (size_t K : test_k)
        {
            Linear_Layer layer(K, N, Execution_Target::CPU, 2.0f, Data_Type::TERNARY_INT2);
            std::vector<float> in_data(M * K, 0.3f);
            Tensor in(M, K, in_data, Execution_Target::CPU);
            Tensor out = layer.forward(in);
            if (out.getShape() != Shape{ M, N }) return false;

            std::vector<float> g_data(M * N, 0.1f);
            Tensor g_out(M, N, g_data, Execution_Target::CPU);
            Tensor g_in = layer.backward(g_out);
            if (g_in.getShape() != Shape{ M, K }) return false;

            layer.prepareInference();
            Tensor out_inf = layer.forward(in);
            if (out_inf.getShape() != Shape{ M, N }) return false;
        }
    }

    return true;
}

bool testUnifiedConv2dLayer()
{
    {
        Conv2d_Layer fp32_c(8, 8, 4, 8, 3, 1, 1, Execution_Target::CPU, Data_Type::FLOAT32);
        if (fp32_c.isTernary() || fp32_c.getDataType() != Data_Type::FLOAT32) return false;
        if (!fp32_c.getMasterWeights().isEmpty()) return false;

        Conv2d_Layer ternary_c(8, 8, 4, 8, 3, 1, 1, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        if (!ternary_c.isTernary() || ternary_c.getDataType() != Data_Type::TERNARY_INT2) return false;
        if (ternary_c.getMasterWeights().isEmpty()) return false;
    }

    {
        uint32_t H = 6, W = 6, C_in = 4, C_out = 8, K = 3, stride = 1, pad = 1;
        Conv2d_Layer layer(H, W, C_in, C_out, K, stride, pad, Execution_Target::CPU, Data_Type::TERNARY_INT2);

        size_t batch_size = 2;
        std::vector<float> in_data(batch_size * H * W * C_in);
        for (size_t i = 0; i < in_data.size(); ++i)
        {
            in_data[i] = std::sin(static_cast<float>(i) * 0.12f);
        }
        Tensor in(batch_size, H * W * C_in, in_data, Execution_Target::CPU);

        Tensor out_fake = layer.forward(in);
        uint32_t out_h = layer.getOutputHeight();
        uint32_t out_w = layer.getOutputWidth();
        if (out_fake.getShape() != Shape{ batch_size, out_h * out_w * C_out }) return false;
        auto fake_data = out_fake.getData();

        std::vector<float> grad_data(batch_size * out_h * out_w * C_out, 0.05f);
        Tensor grad_out(batch_size, out_h * out_w * C_out, grad_data, Execution_Target::CPU);
        Tensor grad_in = layer.backward(grad_out);
        if (grad_in.getShape() != Shape{ batch_size, H * W * C_in }) return false;

        auto params = layer.getParametersAndGradients();
        if (params.size() != 2) return false;
        if (params[0].first != &layer.getMasterWeights()) return false;

        layer.prepareInference();
        if (!layer.hasPackedInference()) return false;

        Tensor out_infer = layer.forward(in);
        if (out_infer.getShape() != Shape{ batch_size, out_h * out_w * C_out }) return false;
        auto infer_data = out_infer.getData();

        double diff_sq = 0.0, norm_sq = 0.0;
        for (size_t i = 0; i < fake_data.size(); ++i)
        {
            double d = infer_data[i] - fake_data[i];
            diff_sq += d * d;
            norm_sq += fake_data[i] * fake_data[i];
        }
        double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
        if (rel_err > 1e-3) return false;
    }

    {
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            uint32_t H = 6, W = 6, C_in = 4, C_out = 8, K = 3, stride = 1, pad = 1;
            Conv2d_Layer gpu_layer(H, W, C_in, C_out, K, stride, pad, Execution_Target::VULKAN_GPU, Data_Type::TERNARY_INT2);

            size_t batch_size = 2;
            std::vector<float> in_data(batch_size * H * W * C_in);
            for (size_t i = 0; i < in_data.size(); ++i)
            {
                in_data[i] = std::sin(static_cast<float>(i) * 0.12f);
            }
            Tensor in(batch_size, H * W * C_in, in_data, Execution_Target::VULKAN_GPU);

            Tensor out_fake = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            uint32_t out_h = gpu_layer.getOutputHeight();
            uint32_t out_w = gpu_layer.getOutputWidth();
            if (out_fake.getShape() != Shape{ batch_size, out_h * out_w * C_out }) return false;
            auto fake_data = out_fake.getData();

            std::vector<float> grad_data(batch_size * out_h * out_w * C_out, 0.05f);
            Tensor grad_out(batch_size, out_h * out_w * C_out, grad_data, Execution_Target::VULKAN_GPU);
            Tensor grad_in = gpu_layer.backward(grad_out);
            engine.executeGraph();
            engine.waitIdle();
            if (grad_in.getShape() != Shape{ batch_size, H * W * C_in }) return false;

            gpu_layer.prepareInference();
            engine.executeGraph();
            engine.waitIdle();
            if (!gpu_layer.hasPackedInference()) return false;

            Tensor out_infer = gpu_layer.forward(in);
            engine.executeGraph();
            engine.waitIdle();
            if (out_infer.getShape() != Shape{ batch_size, out_h * out_w * C_out }) return false;
            auto infer_data = out_infer.getData();

            double diff_sq = 0.0, norm_sq = 0.0;
            for (size_t i = 0; i < fake_data.size(); ++i)
            {
                double d = infer_data[i] - fake_data[i];
                diff_sq += d * d;
                norm_sq += fake_data[i] * fake_data[i];
            }
            double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
            if (rel_err > 0.02) return false;
        }
    }

    {
        uint32_t H = 4, W = 4, C_in = 2, C_out = 4, K = 3, stride = 1, pad = 1;
        Conv2d_Layer src_layer(H, W, C_in, C_out, K, stride, pad, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        size_t batch_size = 1;
        std::vector<float> in_data(batch_size * H * W * C_in, 0.35f);
        Tensor in(batch_size, H * W * C_in, in_data, Execution_Target::CPU);

        src_layer.prepareInference();
        Tensor expected_out = src_layer.forward(in);

        const std::string infer_file = makeUniqueTestTempPath("test_conv2d_infer").string();
        Scoped_Temp_File_Guard infer_guard({infer_file});
        {
            std::ofstream ofs(infer_file, std::ios::binary);
            src_layer.saveInference(ofs);
        }
        Conv2d_Layer loaded_layer(H, W, C_in, C_out, K, stride, pad, Execution_Target::CPU);
        {
            std::ifstream ifs(infer_file, std::ios::binary);
            loaded_layer.loadInference(ifs);
        }

        if (!loaded_layer.isTernary() || !loaded_layer.hasPackedInference()) return false;
        Tensor actual_out = loaded_layer.forward(in);
        auto exp_d = expected_out.getData();
        auto act_d = actual_out.getData();
        for (size_t i = 0; i < exp_d.size(); ++i)
        {
            if (std::abs(exp_d[i] - act_d[i]) > 1e-5f) return false;
        }

        const std::string ckpt_file = makeUniqueTestTempPath("test_conv2d_ckpt").string();
        Scoped_Temp_File_Guard ckpt_guard({ckpt_file});
        Conv2d_Layer train_src(H, W, C_in, C_out, K, stride, pad, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        {
            std::ofstream ofs(ckpt_file, std::ios::binary);
            train_src.saveCheckpoint(ofs);
        }
        Conv2d_Layer loaded_ckpt(H, W, C_in, C_out, K, stride, pad, Execution_Target::CPU);
        {
            std::ifstream ifs(ckpt_file, std::ios::binary);
            loaded_ckpt.loadCheckpoint(ifs);
        }

        if (!loaded_ckpt.isTernary() || !loaded_ckpt.hasMasterWeights()) return false;
    }

    {
        uint32_t H = 4, W = 4, C_in = 2, C_out = 4, K = 3, stride = 1, pad = 1;
        Conv2d_Layer ptq_layer(H, W, C_in, C_out, K, stride, pad, Execution_Target::CPU, Data_Type::FLOAT32);
        if (ptq_layer.isTernary()) return false;

        ptq_layer.quantizeToTernary();
        if (!ptq_layer.isTernary() || !ptq_layer.hasPackedInference()) return false;

        size_t batch_size = 1;
        std::vector<float> in_data(batch_size * H * W * C_in, 0.4f);
        Tensor in(batch_size, H * W * C_in, in_data, Execution_Target::CPU);
        Tensor out = ptq_layer.forward(in);
        if (out.getShape() != Shape{ batch_size, ptq_layer.getOutputHeight() * ptq_layer.getOutputWidth() * C_out }) return false;
    }

    return true;
}

bool testBitNetCausalLM()
{
    {
        constexpr size_t hidden_dim = 32;
        constexpr size_t num_heads = 2;
        constexpr size_t intermediate_dim = 64;
        constexpr size_t S = 4;

        Transformer_Block block(hidden_dim, num_heads, intermediate_dim, 1e-5f, 10000.0f, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        if (!block.isTernary() || block.getDataType() != Data_Type::TERNARY_INT2)
        if (!block.isTernary() || block.getDataType() != Data_Type::TERNARY_INT2) return false;

        std::vector<float> in_data(S * hidden_dim, 0.2f);
        for (size_t i = 0; i < in_data.size(); ++i)
        {
            in_data[i] = 0.05f * static_cast<float>((i % 7) + 1);
        }
        Tensor in_tensor(Shape{ 1, S, hidden_dim }, in_data, Execution_Target::CPU);

        Tensor out_fake = block.forward(in_tensor);
        if (out_fake.getTotalElements() != S * hidden_dim)
        if (out_fake.getTotalElements() != S * hidden_dim) return false;
        auto fake_data = out_fake.getData();
        for (float val : fake_data)
        {
            if (std::isnan(val) || std::isinf(val))
            if (std::isnan(val) || std::isinf(val)) return false;
        }

        KV_Cache_Manager cache(1, num_heads, 16, hidden_dim / num_heads, Execution_Target::CPU, Data_Type::FLOAT32, KV_Cache_Mode::STATIC);
        Tensor prefill_in(Shape{ 1, 3, hidden_dim }, std::vector<float>(3 * hidden_dim, 0.1f), Execution_Target::CPU);
        Tensor prefill_out = block.forward(prefill_in, &cache);
        if (cache.getCurrentSeqLen() != 3 || prefill_out.getTotalElements() != 3 * hidden_dim)
        if (cache.getCurrentSeqLen() != 3 || prefill_out.getTotalElements() != 3 * hidden_dim) return false;

        Tensor decode_in(Shape{ 1, 1, hidden_dim }, std::vector<float>(1 * hidden_dim, 0.2f), Execution_Target::CPU);
        Tensor decode_out = block.forward(decode_in, &cache);
        if (cache.getCurrentSeqLen() != 4 || decode_out.getTotalElements() != hidden_dim)
        if (cache.getCurrentSeqLen() != 4 || decode_out.getTotalElements() != hidden_dim) return false;
        block.setKVCache(nullptr);
        out_fake = block.forward(in_tensor);

        Tensor d_out(Shape{ S, hidden_dim }, std::vector<float>(S * hidden_dim, 0.01f), Execution_Target::CPU);
        Tensor d_in = block.backward(d_out);
        if (d_in.getTotalElements() != S * hidden_dim)
        if (d_in.getTotalElements() != S * hidden_dim) return false;

        block.prepareInference();
        Tensor out_infer = block.forward(in_tensor);
        if (out_infer.getTotalElements() != S * hidden_dim)
        if (out_infer.getTotalElements() != S * hidden_dim) return false;
        auto infer_data = out_infer.getData();

        double diff_sq = 0.0, norm_sq = 0.0;
        for (size_t i = 0; i < fake_data.size(); ++i)
        {
            double d = infer_data[i] - fake_data[i];
            diff_sq += d * d;
            norm_sq += fake_data[i] * fake_data[i];
        }
        double rel_err = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-6);
        if (rel_err > 0.02)
        if (rel_err > 0.02) return false;

        Transformer_Block fp_block(hidden_dim, num_heads, intermediate_dim, 1e-5f, 10000.0f, Execution_Target::CPU, Data_Type::FLOAT32);
        if (fp_block.isTernary())
        if (fp_block.isTernary()) return false;
        fp_block.quantizeToTernary();
        if (!fp_block.isTernary())
        if (!fp_block.isTernary()) return false;
    }

    {
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            constexpr size_t hidden_dim = 32;
            constexpr size_t num_heads = 2;
            constexpr size_t intermediate_dim = 64;
            constexpr size_t S = 4;

            Transformer_Block gpu_block(hidden_dim, num_heads, intermediate_dim, 1e-5f, 10000.0f, Execution_Target::VULKAN_GPU, Data_Type::TERNARY_INT2);
            if (!gpu_block.isTernary() || gpu_block.getDataType() != Data_Type::TERNARY_INT2)
            if (!gpu_block.isTernary() || gpu_block.getDataType() != Data_Type::TERNARY_INT2) return false;

            std::vector<float> in_data(S * hidden_dim, 0.2f);
            Tensor in_tensor(Shape{ 1, S, hidden_dim }, in_data, Execution_Target::VULKAN_GPU);

            Tensor out_fake = gpu_block.forward(in_tensor);
            engine.executeGraph();
            engine.waitIdle();
            if (out_fake.getTotalElements() != S * hidden_dim)
            if (out_fake.getTotalElements() != S * hidden_dim) return false;

            Tensor d_out(Shape{ S, hidden_dim }, std::vector<float>(S * hidden_dim, 0.01f), Execution_Target::VULKAN_GPU);
            Tensor d_in = gpu_block.backward(d_out);
            engine.executeGraph();
            engine.waitIdle();
            if (d_in.getTotalElements() != S * hidden_dim)
            if (d_in.getTotalElements() != S * hidden_dim) return false;

            gpu_block.prepareInference();
            engine.executeGraph();
            engine.waitIdle();

            Tensor out_infer = gpu_block.forward(in_tensor);
            engine.executeGraph();
            engine.waitIdle();
            if (out_infer.getTotalElements() != S * hidden_dim)
            if (out_infer.getTotalElements() != S * hidden_dim) return false;
        }
    }

    {
        Causal_LM_Config cfg;
        cfg.vocab_size = 128;
        cfg.hidden_dim = 32;
        cfg.num_heads = 2;
        cfg.intermediate_dim = 64;
        cfg.num_layers = 2;
        cfg.max_seq_len = 32;
        cfg.execution_target = Execution_Target::CPU;
        cfg.data_type = Data_Type::TERNARY_INT2;
        cfg.tokenizer_path = "tokenizer/tokenizer.json";

        Causal_LM model(cfg);
        if (!model.isTernary())
        if (!model.isTernary()) return false;
        if (model.getBlocks().empty() || !model.getBlocks()[0]->isTernary())
        if (model.getBlocks().empty() || !model.getBlocks()[0]->isTernary()) return false;
        if (model.getTokenEmbedding().isMixedPrecisionEnabled())
        if (model.getTokenEmbedding().isMixedPrecisionEnabled()) return false;

        std::vector<int32_t> input_tokens = { 1, 10, 25, 40 };
        std::vector<int32_t> target_tokens = { 10, 25, 40, 50 };
        float loss = model.forwardLossAndBackward(input_tokens, target_tokens);
        if (std::isnan(loss) || std::isinf(loss) || loss <= 0.0f)
        if (std::isnan(loss) || std::isinf(loss) || loss <= 0.0f) return false;

        model.prepareInference();

        Tensor logits = model.forward(input_tokens, false);
        if (logits.getTotalElements() != input_tokens.size() * cfg.vocab_size)
        if (logits.getTotalElements() != input_tokens.size() * cfg.vocab_size) return false;
        auto logits_data = logits.getData();
        for (float val : logits_data)
        {
            if (std::isnan(val) || std::isinf(val))
            if (std::isnan(val) || std::isinf(val)) return false;
        }

        const std::string test_file = makeUniqueTestTempPath("test_bitnet_causal_lm").string();
        Scoped_Temp_File_Guard test_guard({test_file});
        model.saveInference(test_file);

        Causal_LM loaded_model(cfg);
        loaded_model.loadInference(test_file);

        if (!loaded_model.isTernary())
        if (!loaded_model.isTernary()) return false;
        Tensor loaded_logits = loaded_model.forward(input_tokens, false);
        auto loaded_data = loaded_logits.getData();

        if (loaded_data.size() != logits_data.size())
        if (loaded_data.size() != logits_data.size()) return false;
        for (size_t i = 0; i < logits_data.size(); ++i)
        {
            if (std::abs(logits_data[i] - loaded_data[i]) > 1e-4f)
            if (std::abs(logits_data[i] - loaded_data[i]) > 1e-4f) return false;
        }

        std::string gen_text = model.generate("Hello", 4, 0.0f, 0.9f);
        if (gen_text.empty())
        if (gen_text.empty()) return false;
    }

    {
        auto &engine = Execution_Engine::getInstance();
        auto &context = engine.getContext();
        if (context.isFloat16Supported() && context.isFloat16Enabled())
        {
            Causal_LM_Config cfg;
            cfg.vocab_size = 128;
            cfg.hidden_dim = 32;
            cfg.num_heads = 2;
            cfg.intermediate_dim = 64;
            cfg.num_layers = 1;
            cfg.max_seq_len = 32;
            cfg.execution_target = Execution_Target::VULKAN_GPU;
            cfg.data_type = Data_Type::TERNARY_INT2;
            cfg.tokenizer_path = "tokenizer/tokenizer.json";

            Causal_LM gpu_model(cfg);
            if (!gpu_model.isTernary())
            if (!gpu_model.isTernary()) return false;

            std::vector<int32_t> input_tokens = { 1, 10, 25, 40 };
            std::vector<int32_t> target_tokens = { 10, 25, 40, 50 };
            float loss = gpu_model.forwardLossAndBackward(input_tokens, target_tokens);
            if (std::isnan(loss) || std::isinf(loss) || loss <= 0.0f)
            if (std::isnan(loss) || std::isinf(loss) || loss <= 0.0f) return false;

            gpu_model.prepareInference();
            Tensor logits = gpu_model.forward(input_tokens, false);
            engine.executeGraph();
            engine.waitIdle();
            if (logits.getTotalElements() != input_tokens.size() * cfg.vocab_size)
            if (logits.getTotalElements() != input_tokens.size() * cfg.vocab_size) return false;
        }
    }

    return true;
}

bool testUnifiedVisionTernarySuite()
{
    {
        Res_Net_Block_2d_Layer blk(8, 8, 4, 8, 1, Execution_Target::CPU, true, Data_Type::TERNARY_INT2);
        if (!blk.isTernary() || blk.getDataType() != Data_Type::TERNARY_INT2) return false;

        Tensor in_blk(Shape{1, 4 * 8 * 8}, Execution_Target::CPU);
        std::vector<float> d(4 * 8 * 8, 0.5f);
        in_blk.uploadData(d);
        Tensor out_blk = blk.forward(in_blk);
        if (out_blk.getTotalElements() != 1 * 8 * 8 * 8) return false;

        blk.prepareInference();
        Tensor out_blk_inf = blk.forward(in_blk);
        if (out_blk_inf.getTotalElements() != 1 * 8 * 8 * 8) return false;
    }

    {
        Res_Net_20_Layer res20(16, 16, 3, 5, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        if (!res20.isTernary() || res20.getDataType() != Data_Type::TERNARY_INT2) return false;

        Tensor in_res(Shape{1, 3 * 16 * 16}, Execution_Target::CPU);
        std::vector<float> d(3 * 16 * 16, 0.1f);
        in_res.uploadData(d);
        Tensor out_res = res20.forward(in_res);
        if (out_res.getTotalElements() != 5) return false;

        res20.prepareInference();
        Tensor out_res_inf = res20.forward(in_res);
        if (out_res_inf.getTotalElements() != 5) return false;
        auto inf_data = out_res_inf.getData();
        for (float v : inf_data) {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        Res_Net_20_Layer res20_fp(16, 16, 3, 5, Execution_Target::CPU, Data_Type::FLOAT32);
        if (res20_fp.isTernary()) return false;
        res20_fp.quantizeToTernary();
        if (!res20_fp.isTernary()) return false;
    }

    {
        U_Net_Layer unet(16, 16, 1, 1, 4, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        if (!unet.isTernary() || unet.getDataType() != Data_Type::TERNARY_INT2) return false;

        Tensor in_unet(Shape{1, 1 * 16 * 16}, Execution_Target::CPU);
        std::vector<float> d(1 * 16 * 16, 0.2f);
        in_unet.uploadData(d);
        Tensor out_unet = unet.forward(in_unet);
        if (out_unet.getTotalElements() != 1 * 16 * 16) return false;

        unet.prepareInference();
        Tensor out_unet_inf = unet.forward(in_unet);
        if (out_unet_inf.getTotalElements() != 1 * 16 * 16) return false;
        auto udata = out_unet_inf.getData();
        for (float v : udata) {
            if (std::isnan(v) || std::isinf(v)) return false;
        }

        U_Net_Layer unet_fp(16, 16, 1, 1, 4, Execution_Target::CPU, Data_Type::FLOAT32);
        if (unet_fp.isTernary()) return false;
        unet_fp.quantizeToTernary();
        if (!unet_fp.isTernary()) return false;
    }

    {
        NN_UNet_Layer nn_unet(16, 16, 1, 1, 4, Execution_Target::CPU, Data_Type::TERNARY_INT2);
        if (!nn_unet.isTernary() || nn_unet.getDataType() != Data_Type::TERNARY_INT2) return false;

        nn_unet.prepareInference();
        Tensor in_nn(Shape{1, 1 * 16 * 16}, Execution_Target::CPU);
        std::vector<float> d(1 * 16 * 16, 0.3f);
        in_nn.uploadData(d);
        Tensor out_nn = nn_unet.forward(in_nn);
        if (out_nn.getTotalElements() != 1 * 16 * 16) return false;
        auto nndata = out_nn.getData();
        for (float v : nndata) {
            if (std::isnan(v) || std::isinf(v)) return false;
        }
    }

    return true;
}

bool testPPOActorCriticTernarySuite()
{
    PPO_Actor_Critic_Layer ppo(2, Execution_Target::CPU);
    ppo.addActorLayer(std::make_unique<Linear_Layer>(4, 8, Execution_Target::CPU));
    ppo.addActorLayer(std::make_unique<Relu_Layer>(Execution_Target::CPU));
    ppo.addActorLayer(std::make_unique<Linear_Layer>(8, 2, Execution_Target::CPU));

    ppo.addCriticLayer(std::make_unique<Linear_Layer>(4, 8, Execution_Target::CPU));
    ppo.addCriticLayer(std::make_unique<Relu_Layer>(Execution_Target::CPU));
    ppo.addCriticLayer(std::make_unique<Linear_Layer>(8, 1, Execution_Target::CPU));

    if (ppo.isTernary()) return false;
    ppo.quantizeToTernary();
    if (!ppo.isTernary()) return false;

    ppo.prepareInference();

    Tensor in_ppo(Shape{1, 4}, Execution_Target::CPU);
    std::vector<float> d = { 0.1f, 0.2f, -0.3f, 0.4f };
    in_ppo.uploadData(d);
    Tensor out_ppo = ppo.forward(in_ppo);

    if (out_ppo.getTotalElements() != 3) return false;
    auto pdata = out_ppo.getData();
    for (float v : pdata) {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    return true;
}

bool testModelQuantizerSuite()
{
    Neural_Network net(Execution_Target::CPU);
    net.addLayer<Linear_Layer>(4, 8, Execution_Target::CPU);
    net.addLayer<Relu_Layer>(Execution_Target::CPU);
    net.addLayer<Linear_Layer>(8, 2, Execution_Target::CPU);

    auto &lin1 = dynamic_cast<Linear_Layer&>(net.getLayer(0));
    if (lin1.isTernary()) return false;

    Model_Quantizer::quantizeToTernary(net);
    if (!lin1.isTernary()) return false;
    auto &lin2 = dynamic_cast<Linear_Layer&>(net.getLayer(2));
    if (!lin2.isTernary()) return false;

    net.prepareInference();

    Tensor in_net(Shape{1, 4}, Execution_Target::CPU);
    std::vector<float> d = { 0.5f, -0.2f, 0.1f, 0.9f };
    in_net.uploadData(d);
    Tensor out_net = net.forward(in_net);

    if (out_net.getTotalElements() != 2) return false;
    auto ndata = out_net.getData();
    for (float v : ndata) {
        if (std::isnan(v) || std::isinf(v)) return false;
    }

    return true;
}

void runTernaryQuantSuite(Test_Result_Counter &counter)
{
    counter.record("  Ternary Computing Phase 1 Suite:   ", testTernaryComputingPhase1());
    counter.record("  Ternary Computing Phase 2 Suite:   ", testTernaryComputingPhase2());
    counter.record("  Ternary Computing Phase 4 Suite:   ", testTernaryComputingPhase4());
    counter.record("  Ternary Computing Phase 5 Suite:   ", testTernaryComputingPhase5());
    counter.record("  Unified Linear Layer Suite:        ", testUnifiedLinearLayer());
    counter.record("  BitNet Causal LM Suite:            ", testBitNetCausalLM());
    counter.record("  Unified Conv2D Layer Suite:        ", testUnifiedConv2dLayer());
    counter.record("  Unified Vision Ternary Suite:      ", testUnifiedVisionTernarySuite());
    counter.record("  PPO Actor-Critic Ternary Suite:    ", testPPOActorCriticTernarySuite());
    counter.record("  Model Quantizer Suite:             ", testModelQuantizerSuite());
}
