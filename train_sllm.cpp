#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/vulkan_context.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"
#include "learning_rate/cosine_annealing.h"
#include "llm/binary_dataset.h"
#include "llm/causal_lm.h"
#include "optimizer/adam_optimizer.h"
#include "tokenizer/bpe_tokenizer.h"

namespace
{

    struct Train_Config
    {
        std::string data_path = "data/lucbat/lucbat.txt";
        std::string bin_cache_path = "data/lucbat/lucbat.bin";
        std::string tokenizer_path = "tokenizer/syllable_tokenizer.json";
        std::string output_dir = "output/lucbat";
        std::string ckpt_best_name = "checkpoint_best.nnck";
        std::string ckpt_latest_name = "checkpoint_latest.nnck";
        std::string model_name = "model.bin";

        std::vector<std::string> eval_prompts = {
                    "Trăm năm trong cõi người ta\n",
                    "Trước đèn xem truyện Tây minh\n",
                    "Đầu lòng hai ả tố nga\n",
                    "Vân Tiên vào tạ tôn sư\n"
        };
        std::vector<std::string> stop_sequences = {};

        std::string getBestCkptPath() const { return output_dir + "/" + ckpt_best_name; }
        std::string getLatestCkptPath() const { return output_dir + "/" + ckpt_latest_name; }
        std::string getModelPath() const { return output_dir + "/" + model_name; }

        Execution_Target target = Execution_Target::VULKAN_GPU;
        size_t hidden_dim = 256;
        size_t num_heads = 8;
        size_t intermediate_dim = 1024;
        size_t num_layers = 6;
        size_t seq_len = 128;
        size_t stride = 64;
        size_t epochs = 40;
        size_t eval_interval = 1;
        size_t batch_size = 8;
        size_t grad_accum = 1;
        float learning_rate = 0.0004f;
        float min_lr = 0.00004f;
        float warmup_ratio = 0.05f;
        float weight_decay = 0.1f;
        float grad_clip = 1.0f;
        bool use_fp16 = true;
        bool use_coop_matrix = true;
        bool use_static_graph = true;
        bool use_fused_gemm_adam = false;
        float initial_loss_scale = 1.0f;

        float temperature = 0.1f;
        float top_p = 0.9f;
        size_t top_k = 20;
        float repetition_penalty = 1.1f;
        size_t max_gen_tokens = 128;
    };

    [[maybe_unused]] void setCooperativeMatrixEnabled(bool enable)
    {
        auto& engine = Execution_Engine::getInstance();
        if (enable && !engine.isCooperativeMatrixSupported())
        {
            std::cerr << "[Warning] Cooperative Matrix is not supported on this device. Keeping disabled.\n";
            engine.setCooperativeMatrixEnabled(false);
            return;
        }
        engine.setCooperativeMatrixEnabled(enable);
    }

    [[maybe_unused]] void setCooperativeMatrixEnabled(Train_Config& config, bool enable)
    {
        auto& engine = Execution_Engine::getInstance();
        if (enable && !engine.isCooperativeMatrixSupported())
        {
            std::cerr << "[Warning] Cooperative Matrix is not supported on this device. Keeping disabled.\n";
            config.use_coop_matrix = false;
            engine.setCooperativeMatrixEnabled(false);
            return;
        }
        if (enable && !config.use_fp16)
        {
            std::cout << "[Config] Cooperative Matrix requires FP16 precision. Automatically enabling FP16.\n";
            config.use_fp16 = true;
            if (engine.getContext().isFloat16Supported())
            {
                engine.getContext().setFloat16Enabled(true);
            }
        }
        config.use_coop_matrix = enable;
        engine.setCooperativeMatrixEnabled(enable);
    }

    [[maybe_unused]] void setFp16Enabled(bool enable)
    {
        auto& context = Execution_Engine::getInstance().getContext();
        if (context.isFloat16Supported())
        {
            context.setFloat16Enabled(enable);
        }
    }

    [[maybe_unused]] void setFp16Enabled(Train_Config& config, bool enable)
    {
        auto& engine = Execution_Engine::getInstance();
        if (enable && !engine.getContext().isFloat16Supported())
        {
            std::cerr << "[Warning] FP16 is not supported on this device. Falling back to FP32.\n";
            config.use_fp16 = false;
            config.use_coop_matrix = false;
            engine.setCooperativeMatrixEnabled(false);
            return;
        }
        config.use_fp16 = enable;
        if (engine.getContext().isFloat16Supported())
        {
            engine.getContext().setFloat16Enabled(enable);
        }
        if (!enable && config.use_coop_matrix)
        {
            std::cout << "[Config] Disabling FP16 also disables Cooperative Matrix (requires FP16).\n";
            config.use_coop_matrix = false;
            engine.setCooperativeMatrixEnabled(false);
        }
    }

    [[maybe_unused]] void setStaticGraphEnabled(bool enable)
    {
        Execution_Engine::getInstance().setStaticGraphEnabled(enable);
    }

    [[maybe_unused]] void setStaticGraphEnabled(Train_Config& config, bool enable)
    {
        config.use_static_graph = enable;
        Execution_Engine::getInstance().setStaticGraphEnabled(enable);
    }

    [[maybe_unused]] void setFusedGemmAdamEnabled(bool enable)
    {
        Execution_Engine::getInstance().setFusedGemmAdamEnabled(enable);
    }

    [[maybe_unused]] void setFusedGemmAdamEnabled(Train_Config& config, bool enable)
    {
        config.use_fused_gemm_adam = enable;
        Execution_Engine::getInstance().setFusedGemmAdamEnabled(enable);
    }

    [[maybe_unused]] void configureHardwareAcceleration(Train_Config& config, bool use_coop_matrix, bool use_fp16, bool use_static_graph, bool use_fused_gemm_adam = false)
    {
        setFp16Enabled(config, use_fp16);
        setCooperativeMatrixEnabled(config, use_coop_matrix);
        setStaticGraphEnabled(config, use_static_graph);
        setFusedGemmAdamEnabled(config, use_fused_gemm_adam);
    }

    [[maybe_unused]] void printAccelerationStatus(const Train_Config& config)
    {
        std::cout << "\n[Hardware Acceleration Features]\n"
            << "  - Cooperative Matrix (Tensor Cores) : "
            << (config.use_coop_matrix && Execution_Engine::getInstance().isCooperativeMatrixEnabled() ? "ENABLED" : "DISABLED") << "\n"
            << "  - FP16 Precision                     : "
            << (config.use_fp16 ? "ENABLED (float16)" : "DISABLED (float32)") << "\n"
            << "  - Static Command Buffer Graph        : "
            << (config.use_static_graph ? "ENABLED" : "DISABLED") << "\n"
            << "  - Fused GEMM + AdamW Optimizer       : "
            << (config.use_fused_gemm_adam && Execution_Engine::getInstance().isFusedGemmAdamEnabled() ? "ENABLED" : "DISABLED") << "\n\n";
    }

    size_t countParameters(Causal_LM& model)
    {
        size_t total = 0;
        for (const auto& [param, grad] : model.getParametersAndGradients())
        {
            if (param)
            {
                total += param->getShape().getTotalElements();
            }
        }
        return total;
    }

    [[maybe_unused]] void printModelArchitecture(const Train_Config& config, size_t total_params)
    {
        size_t head_dim = config.hidden_dim / config.num_heads;
        double params_m = static_cast<double>(total_params) / 1'000'000.0;
        double fp16_mb = static_cast<double>(total_params * sizeof(uint16_t)) / (1024.0 * 1024.0);
        double adam_mb = static_cast<double>(total_params * 12) / (1024.0 * 1024.0);

        std::cout << "[Model Architecture Configuration]\n"
            << "  - Total Parameters                   : " << total_params << " (~" << std::fixed << std::setprecision(2) << params_m << "M params)\n"
            << "  - Hidden Dimension (d_model)         : " << config.hidden_dim << "\n"
            << "  - Attention Heads                    : " << config.num_heads << " (head_dim = " << head_dim << ")\n"
            << "  - Intermediate Dim (FFN / SwiGLU)    : " << config.intermediate_dim << "\n"
            << "  - Transformer Layers                 : " << config.num_layers << "\n"
            << "  - Context Sequence Length (seq_len)  : " << config.seq_len << " tokens (stride: " << config.stride << ")\n"
            << "  - Estimated Memory (FP16 Weights)    : ~" << std::fixed << std::setprecision(2) << fp16_mb << " MB\n"
            << "  - Estimated Memory (AdamW States)    : ~" << std::fixed << std::setprecision(2) << adam_mb << " MB\n\n";
    }

    void safeSaveCheckpoint(Causal_LM& model, const std::string& path)
    {
        std::filesystem::path p(path);
        if (p.has_parent_path())
        {
            std::filesystem::create_directories(p.parent_path());
        }
        if (std::filesystem::exists(path))
        {
            std::error_code ec;
            std::filesystem::copy_file(path, path + ".bak", std::filesystem::copy_options::overwrite_existing, ec);
        }
        model.saveCheckpoint(path);
    }

    void safeSaveInference(Causal_LM& model, const std::string& path)
    {
        std::filesystem::path p(path);
        if (p.has_parent_path())
        {
            std::filesystem::create_directories(p.parent_path());
        }
        if (std::filesystem::exists(path))
        {
            std::error_code ec;
            std::filesystem::copy_file(path, path + ".bak", std::filesystem::copy_options::overwrite_existing, ec);
        }
        model.saveInference(path);
    }

    std::string findDefaultInferenceModel(const Train_Config& config)
    {
        std::vector<std::string> candidates = {
            config.getModelPath(),
            config.getBestCkptPath(),
            config.getLatestCkptPath(),
            "output/lucbat/model.bin",
            "output/lucbat/checkpoint_best.nnck",
            "output/truyenkieu/model.bin",
            "output/truyenkieu/checkpoint_best.nnck",
            "checkpoint_best.nnck",
            "model.bin"
        };
        for (const auto& path : candidates)
        {
            if (std::filesystem::exists(path)) return path;
        }
        return config.getModelPath();
    }

    std::string findDefaultResumeCheckpoint(const Train_Config& config)
    {
        std::vector<std::string> candidates = {
            config.getBestCkptPath(),
            config.getLatestCkptPath(),
            config.getModelPath(),
            "output/lucbat/checkpoint_best.nnck",
            "output/lucbat/checkpoint_latest.nnck",
            "output/truyenkieu/checkpoint_best.nnck",
            "output/truyenkieu/checkpoint_latest.nnck",
            "checkpoint_best.nnck",
            "model.bin"
        };
        for (const auto& path : candidates)
        {
            if (std::filesystem::exists(path)) return path;
        }
        return config.getBestCkptPath();
    }

    struct Training_Sample
    {
        std::vector<int32_t> inputs;
        std::vector<int32_t> targets;
    };

    struct Cli_Args
    {
        enum class Execution_Mode { UNSET, TRAIN, INFER };
        enum class Train_Source { UNSET, SCRATCH, RESUME };

        Execution_Mode mode = Execution_Mode::UNSET;
        Train_Source train_source = Train_Source::UNSET;
        std::string checkpoint_path = "";
        std::string custom_prompt = "";
        std::string output_dir = "";
        std::string data_path = "";
        std::string tokenizer_path = "";
        int override_epochs = -1;
        int override_batch_size = -1;
        int override_max_batches = -1;
        bool show_help = false;
        std::optional<bool> override_coop;
        std::optional<bool> override_fp16;
        std::optional<bool> override_static_graph;
        std::optional<bool> override_fused_gemm_adam;

        // Model architecture overrides
        std::optional<size_t> override_hidden_dim;
        std::optional<size_t> override_num_heads;
        std::optional<size_t> override_intermediate_dim;
        std::optional<size_t> override_num_layers;
        std::optional<size_t> override_seq_len;
        std::optional<size_t> override_stride;
        std::optional<float> override_learning_rate;
        std::optional<float> override_min_lr;
        std::optional<float> override_repetition_penalty;
    };

    void printHelp()
    {
        std::cout << "\nUsage: train_sllm [options]\n\n"
            << "Mode Options:\n"
            << "  -t, --train                 Training mode\n"
            << "  -i, --infer                 Inference mode\n"
            << "  --scratch                   Train from scratch\n"
            << "  -r, --resume <file>         Resume training from checkpoint\n"
            << "  -m, --model <file>          Model file for inference\n"
            << "  -p, --prompt <text>         Prompt for direct inference\n"
            << "  -o, --output-dir <path>     Output directory (default: output/lucbat)\n"
            << "  -d, --data <file>           Text corpus file for training (default: data/lucbat/lucbat.txt)\n"
            << "  --tokenizer <file>          Tokenizer config file (default: tokenizer/tokenizer_2.json)\n\n"
            << "Training Hyperparameters:\n"
            << "  -e, --epochs <N>            Number of epochs (default: 40)\n"
            << "  -b, --batch-size <B>        Training batch size (default: 8)\n"
            << "  --max-batches <N>           Maximum batches per epoch (0 = all)\n"
            << "  --lr <float>                Initial learning rate (default: 0.0004)\n"
            << "  --min-lr <float>            Minimum learning rate (default: 0.00004)\n"
            << "  --rep-penalty <float>       Repetition penalty for inference (default: 1.05)\n\n"
            << "Model Architecture Options:\n"
            << "  --hidden-dim <N>            Embedding / hidden dimension (default: 256)\n"
            << "  --heads <N>                 Number of attention heads (default: 8)\n"
            << "  --intermediate-dim <N>      FFN intermediate dimension (default: 1024)\n"
            << "  --layers <N>                Number of transformer layers (default: 6)\n"
            << "  --seq-len <N>               Context sequence length (default: 128)\n"
            << "  --stride <N>                Dataset sliding window stride (default: 64)\n\n"
            << "Hardware Acceleration Options:\n"
            << "  --coop                      Enable Cooperative Matrix (Tensor Cores)\n"
            << "  --no-coop                   Disable Cooperative Matrix\n"
            << "  --fp16                      Enable FP16 precision\n"
            << "  --fp32, --no-fp16           Disable FP16 precision (use FP32)\n"
            << "  --static-graph              Enable Static Command Buffer Graph\n"
            << "  --no-static-graph           Disable Static Command Buffer Graph\n"
            << "  --fused-gemm-adam           Enable Fused GEMM + AdamW backward kernel\n"
            << "  --no-fused-gemm-adam        Disable Fused GEMM + AdamW backward kernel (default)\n"
            << "  -h, --help                  Show help message\n\n";
    }

    Cli_Args parseArgs(int argc, char* argv[])
    {
        Cli_Args args;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                args.show_help = true;
            }
            else if (arg == "-t" || arg == "--train")
            {
                args.mode = Cli_Args::Execution_Mode::TRAIN;
            }
            else if (arg == "-i" || arg == "--infer")
            {
                args.mode = Cli_Args::Execution_Mode::INFER;
            }
            else if (arg == "--scratch")
            {
                args.train_source = Cli_Args::Train_Source::SCRATCH;
            }
            else if (arg == "-r" || arg == "--resume")
            {
                args.train_source = Cli_Args::Train_Source::RESUME;
                if (i + 1 < argc && argv[i + 1][0] != '-')
                {
                    args.checkpoint_path = argv[++i];
                }
            }
            else if (arg == "-m" || arg == "--model")
            {
                if (i + 1 < argc && argv[i + 1][0] != '-')
                {
                    args.checkpoint_path = argv[++i];
                }
            }
            else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc)
            {
                args.custom_prompt = argv[++i];
            }
            else if ((arg == "-o" || arg == "--output-dir") && i + 1 < argc)
            {
                args.output_dir = argv[++i];
            }
            else if ((arg == "-d" || arg == "--data") && i + 1 < argc)
            {
                args.data_path = argv[++i];
            }
            else if (arg == "--tokenizer" && i + 1 < argc)
            {
                args.tokenizer_path = argv[++i];
            }
            else if ((arg == "-e" || arg == "--epochs") && i + 1 < argc)
            {
                args.override_epochs = std::stoi(argv[++i]);
            }
            else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc)
            {
                args.override_batch_size = std::stoi(argv[++i]);
            }
            else if (arg == "--max-batches" && i + 1 < argc)
            {
                args.override_max_batches = std::stoi(argv[++i]);
            }
            else if (arg == "--hidden-dim" && i + 1 < argc)
            {
                args.override_hidden_dim = std::stoul(argv[++i]);
            }
            else if (arg == "--heads" && i + 1 < argc)
            {
                args.override_num_heads = std::stoul(argv[++i]);
            }
            else if (arg == "--intermediate-dim" && i + 1 < argc)
            {
                args.override_intermediate_dim = std::stoul(argv[++i]);
            }
            else if (arg == "--layers" && i + 1 < argc)
            {
                args.override_num_layers = std::stoul(argv[++i]);
            }
            else if (arg == "--seq-len" && i + 1 < argc)
            {
                args.override_seq_len = std::stoul(argv[++i]);
            }
            else if (arg == "--stride" && i + 1 < argc)
            {
                args.override_stride = std::stoul(argv[++i]);
            }
            else if (arg == "--lr" && i + 1 < argc)
            {
                args.override_learning_rate = std::stof(argv[++i]);
            }
            else if (arg == "--min-lr" && i + 1 < argc)
            {
                args.override_min_lr = std::stof(argv[++i]);
            }
            else if (arg == "--rep-penalty" && i + 1 < argc)
            {
                args.override_repetition_penalty = std::stof(argv[++i]);
            }
            else if (arg == "--coop")
            {
                args.override_coop = true;
            }
            else if (arg == "--no-coop")
            {
                args.override_coop = false;
            }
            else if (arg == "--fp16")
            {
                args.override_fp16 = true;
            }
            else if (arg == "--fp32" || arg == "--no-fp16")
            {
                args.override_fp16 = false;
            }
            else if (arg == "--static-graph")
            {
                args.override_static_graph = true;
            }
            else if (arg == "--no-static-graph")
            {
                args.override_static_graph = false;
            }
            else if (arg == "--fused-gemm-adam")
            {
                args.override_fused_gemm_adam = true;
            }
            else if (arg == "--no-fused-gemm-adam")
            {
                args.override_fused_gemm_adam = false;
            }
        }
        return args;
    }

    bool loadModelWeights(Causal_LM& model, const std::string& path)
    {
        if (!std::filesystem::exists(path))
        {
            std::cerr << "[Error] File not found: " << path << "\n";
            return false;
        }

        try
        {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "[Error] Cannot open file: " << path << "\n";
                return false;
            }

            uint32_t magic = 0;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            file.close();

            if (magic == 0x434D4C43)
            {
                std::cout << "[Model] Loading checkpoint (.nnck): " << path << "\n";
                model.loadCheckpoint(path);
                return true;
            }
            else if (magic == 0x494D4C43)
            {
                std::cout << "[Model] Loading inference format (.bin): " << path << "\n";
                model.loadInference(path);
                return true;
            }
            else
            {
                try
                {
                    model.loadCheckpoint(path);
                    return true;
                }
                catch (...)
                {
                    model.loadInference(path);
                    return true;
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "[Error] Exception while loading weights: " << e.what() << "\n";
            return false;
        }
    }

    std::vector<int32_t> loadOrCacheTokens(const Train_Config& config, Bpe_Tokenizer& tokenizer)
    {
        std::vector<int32_t> tokens;
        if (std::filesystem::exists(config.bin_cache_path))
        {
            if (Binary_Token_Dataset::load(config.bin_cache_path, tokens) && !tokens.empty())
            {
                std::cout << "[Dataset] Loaded " << tokens.size() << " tokens from binary cache: " << config.bin_cache_path << "\n";
                return tokens;
            }
        }

        std::ifstream file(config.data_path);
        if (!file.is_open())
        {
            std::cerr << "[Error] Cannot open text corpus: " << config.data_path << "\n";
            return tokens;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string text = buffer.str();

        tokens = tokenizer.encode(text, true);
        if (tokens.size() >= 2)
        {
            Binary_Token_Dataset::save(config.bin_cache_path, tokens);
            std::cout << "[Dataset] Cached " << tokens.size() << " tokens to file: " << config.bin_cache_path << "\n";
        }
        return tokens;
    }

    std::vector<Training_Sample> createDataset(const std::vector<int32_t>& tokens, size_t seq_len, size_t stride)
    {
        std::vector<Training_Sample> dataset;
        if (tokens.size() <= seq_len)
        {
            if (tokens.size() >= 2)
            {
                dataset.push_back({
                    std::vector<int32_t>(tokens.begin(), tokens.end() - 1),
                    std::vector<int32_t>(tokens.begin() + 1, tokens.end())
                    });
            }
            return dataset;
        }

        for (size_t start = 0; start + seq_len < tokens.size(); start += stride)
        {
            dataset.push_back({
                std::vector<int32_t>(tokens.begin() + start, tokens.begin() + start + seq_len),
                std::vector<int32_t>(tokens.begin() + start + 1, tokens.begin() + start + seq_len + 1)
                });
        }

        return dataset;
    }

    void runInferenceInteractive(Causal_LM& model, const Train_Config& config, const std::string& single_prompt = "")
    {
        model.setTrainingMode(false);
        std::cout << "\n==============================================================\n";
        std::cout << "               INFERENCE MODE\n";
        std::cout << "==============================================================\n";
        std::cout << " - Commands: /temp <float>, /tokens <int>, /top_p <float>, /exit\n";
        std::cout << "==============================================================\n\n";

        float cur_temp = config.temperature;
        size_t cur_max_tokens = config.max_gen_tokens;
        float cur_top_p = config.top_p;

        if (!single_prompt.empty())
        {
            std::cout << "[Prompt] " << single_prompt << "\n[Output] ";
            model.generate(single_prompt, cur_max_tokens, cur_temp, cur_top_p, 2, true,
                [](const std::string& token_str) {
                    std::cout << token_str << std::flush;
                }, config.repetition_penalty, config.top_k, config.stop_sequences);
            std::cout << "\n";
            return;
        }

        while (true)
        {
            std::cout << "\nPrompt > ";
            std::string line;
            if (!std::getline(std::cin, line))
            {
                break;
            }

            auto start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
            {
                continue;
            }
            auto end = line.find_last_not_of(" \t\r\n");
            std::string trimmed = line.substr(start, end - start + 1);

            if (trimmed == "exit" || trimmed == "quit" || trimmed == "/exit" || trimmed == "/quit")
            {
                break;
            }

            if (trimmed.starts_with("/temp "))
            {
                try { cur_temp = std::stof(trimmed.substr(6)); }
                catch (...) {}
                continue;
            }

            if (trimmed.starts_with("/tokens "))
            {
                try { cur_max_tokens = std::stoul(trimmed.substr(8)); }
                catch (...) {}
                continue;
            }

            if (trimmed.starts_with("/top_p "))
            {
                try { cur_top_p = std::stof(trimmed.substr(7)); }
                catch (...) {}
                continue;
            }

            std::cout << "[Generating...] \"";
            auto gen_start = std::chrono::high_resolution_clock::now();
            model.generate(trimmed, cur_max_tokens, cur_temp, cur_top_p, 2, true,
                [](const std::string& token_str) {
                    std::cout << token_str << std::flush;
                }, config.repetition_penalty, config.top_k, config.stop_sequences);
            auto gen_end = std::chrono::high_resolution_clock::now();
            double gen_sec = std::chrono::duration<double>(gen_end - gen_start).count();
            std::cout << "\"\n[" << std::fixed << std::setprecision(2) << gen_sec << "s]\n";
        }
    }

    void runTraining(Causal_LM& model, Bpe_Tokenizer& tokenizer, Train_Config& config,
        bool resume_from_ckpt, const std::string& ckpt_path, size_t max_batches_per_epoch)
    {
        std::filesystem::create_directories(config.output_dir);

        if (resume_from_ckpt && !ckpt_path.empty())
        {
            if (!loadModelWeights(model, ckpt_path))
            {
                std::cout << "Could not load requested checkpoint. Continue from scratch? (y/n): ";
                std::string answer;
                std::getline(std::cin, answer);
                if (answer != "y" && answer != "Y")
                {
                    return;
                }
            }
        }

        std::vector<int32_t> all_tokens = loadOrCacheTokens(config, tokenizer);
        if (all_tokens.size() < 2)
        {
            std::cerr << "[Error] Insufficient tokens for training.\n";
            return;
        }

        std::vector<Training_Sample> dataset = createDataset(all_tokens, config.seq_len, config.stride);
        std::cout << "[Dataset] Tokens: " << all_tokens.size() << " | Samples: " << dataset.size() << "\n";

        size_t actual_batch_size = std::min(config.batch_size, dataset.size());
        size_t num_batches = (actual_batch_size > 0) ? (dataset.size() / actual_batch_size) : 0;
        if (num_batches == 0)
        {
            num_batches = 1;
        }
        size_t effective_batches = (max_batches_per_epoch > 0) ? std::min(max_batches_per_epoch, num_batches) : num_batches;
        size_t total_training_steps = config.epochs * effective_batches;
        int calculated_warmup_steps = static_cast<int>(std::round(config.warmup_ratio * static_cast<float>(total_training_steps)));
        calculated_warmup_steps = std::min<int>(calculated_warmup_steps, static_cast<int>(std::max<size_t>(1, total_training_steps)) - 1);

        Cosine_Annealing scheduler(config.learning_rate, config.min_lr, static_cast<int>(std::max<size_t>(1, total_training_steps)), calculated_warmup_steps);
        Adam_Optimizer optimizer(scheduler);
        optimizer.setWeightDecay(config.weight_decay);

        float best_loss = std::numeric_limits<float>::infinity();
        if (resume_from_ckpt && !dataset.empty())
        {
            size_t actual_b = std::min(config.batch_size, dataset.size());
            Tensor b_in(Shape{ actual_b, config.seq_len }, config.target);
            Tensor b_tgt(Shape{ actual_b, config.seq_len }, config.target);
            std::vector<float> in_d(actual_b * config.seq_len, 0.0f);
            std::vector<float> tgt_d(actual_b * config.seq_len, -1.0f);
            for (size_t b = 0; b < actual_b; ++b)
            {
                for (size_t s = 0; s < config.seq_len && s < dataset[b].inputs.size(); ++s) in_d[b * config.seq_len + s] = static_cast<float>(dataset[b].inputs[s]);
                for (size_t s = 0; s < config.seq_len && s < dataset[b].targets.size(); ++s) tgt_d[b * config.seq_len + s] = static_cast<float>(dataset[b].targets[s]);
            }
            b_in.uploadData(in_d);
            b_tgt.uploadData(tgt_d);
            Tensor logits = model.forward(b_in, false);
            Tensor dummy_grad;
            float base_loss = model.computeLossAndGradient(logits, b_tgt, dummy_grad);
            std::cout << "[Checkpoint] Baseline Loss (initial sample batch): " << std::fixed << std::setprecision(4) << base_loss << "\n";
        }

        // Persistent batch GPU tensors and host staging buffers for Zero-Allocation Replay
        Tensor batch_inputs(Shape{ actual_batch_size, config.seq_len }, config.target);
        Tensor batch_targets(Shape{ actual_batch_size, config.seq_len }, config.target);
        std::vector<float> in_data(actual_batch_size * config.seq_len, 0.0f);
        std::vector<float> tgt_data(actual_batch_size * config.seq_len, -1.0f);

        if (config.target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().setCooperativeMatrixEnabled(config.use_coop_matrix);
        }

        std::cout << "\n[Training] Starting: " << config.epochs << " epochs, "
            << effective_batches << " batches/epoch (Batch Size: " << actual_batch_size
            << ", Seq Len: " << config.seq_len << ")...\n";

        for (size_t epoch = 1; epoch <= config.epochs; ++epoch)
        {
            if (config.target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().setStaticGraphEnabled(config.use_static_graph);
            }
            model.setTrainingMode(true);
            auto epoch_start = std::chrono::high_resolution_clock::now();
            float epoch_loss_sum = 0.0f;
            size_t total_samples_processed = 0;
            size_t interval_batch_count = 0;
            Step_Timings::getInstance().reset();

            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
            {
                if (max_batches_per_epoch > 0 && batch_idx >= max_batches_per_epoch)
                {
                    break;
                }

                auto batch_step_start = std::chrono::high_resolution_clock::now();

                size_t start_idx = batch_idx * actual_batch_size;
                size_t actual_b = actual_batch_size;
                total_samples_processed += actual_b;

                // [1] Data Prep & Upload
                auto data_prep_start = std::chrono::high_resolution_clock::now();
                for (size_t b = 0; b < actual_b; ++b)
                {
                    const auto& sample = dataset[start_idx + b];
                    for (size_t s = 0; s < config.seq_len && s < sample.inputs.size(); ++s)
                    {
                        in_data[b * config.seq_len + s] = static_cast<float>(sample.inputs[s]);
                    }
                    for (size_t s = 0; s < config.seq_len && s < sample.targets.size(); ++s)
                    {
                        tgt_data[b * config.seq_len + s] = static_cast<float>(sample.targets[s]);
                    }
                }

                batch_inputs.uploadData(in_data);
                batch_targets.uploadData(tgt_data);
                auto data_prep_end = std::chrono::high_resolution_clock::now();
                double data_prep_ms = std::chrono::duration<double, std::milli>(data_prep_end - data_prep_start).count();
                Step_Timings::getInstance()[Timing_Stage::DATA_PREP] += data_prep_ms;

                bool is_start_of_accum = (batch_idx % config.grad_accum == 0);
                bool is_end_of_accum = ((batch_idx + 1) % config.grad_accum == 0) ||
                    (batch_idx + 1 == num_batches) ||
                    (max_batches_per_epoch > 0 && batch_idx + 1 == max_batches_per_epoch);

                // [2] Reset Gradients
                if (is_start_of_accum)
                {
                    model.setAccumulated(false);
                    auto reset_start = std::chrono::high_resolution_clock::now();
                    model.resetGradients();
                    auto reset_end = std::chrono::high_resolution_clock::now();
                    Step_Timings::getInstance()[Timing_Stage::RESET_GRAD] += std::chrono::duration<double, std::milli>(reset_end - reset_start).count();
                }
                else
                {
                    model.setAccumulated(true);
                }

                // [3] Forward + Loss + Backward
                float current_scale = (model.getLossScaler() && model.getLossScaler()->isEnabled()) ? model.getLossScaler()->getScale() : 1.0f;
                bool defer_bwd = is_end_of_accum && (model.getConfig().execution_target == Execution_Target::VULKAN_GPU);
                float step_loss = model.forwardLossAndBackward(batch_inputs, batch_targets, current_scale / static_cast<float>(config.grad_accum), defer_bwd);
                epoch_loss_sum += step_loss * static_cast<float>(actual_b);

                // [4] Optimizer Step & [5] LR Scheduler Step
                if (is_end_of_accum)
                {
                    auto opt_start = std::chrono::high_resolution_clock::now();
                    bool overflow_hint = std::isnan(step_loss) || std::isinf(step_loss);
                    bool success = model.stepOptimizer(optimizer, config.grad_clip, current_scale, overflow_hint, defer_bwd);
                    auto opt_end = std::chrono::high_resolution_clock::now();
                    Step_Timings::getInstance()[Timing_Stage::OPT_EXEC] += std::chrono::duration<double, std::milli>(opt_end - opt_start).count();

                    if (success)
                    {
                        auto sched_start = std::chrono::high_resolution_clock::now();
                        scheduler.step();
                        auto sched_end = std::chrono::high_resolution_clock::now();
                        Step_Timings::getInstance()[Timing_Stage::SCHED_STEP] += std::chrono::duration<double, std::milli>(sched_end - sched_start).count();
                    }
                    else
                    {
                        auto reset_start = std::chrono::high_resolution_clock::now();
                        model.resetGradients();
                        auto reset_end = std::chrono::high_resolution_clock::now();
                        Step_Timings::getInstance()[Timing_Stage::RESET_GRAD] += std::chrono::duration<double, std::milli>(reset_end - reset_start).count();
                    }
                }

                auto batch_step_end = std::chrono::high_resolution_clock::now();
                double batch_total_ms = std::chrono::duration<double, std::milli>(batch_step_end - batch_step_start).count();
                Step_Timings::getInstance()[Timing_Stage::TOTAL_BATCH] += batch_total_ms;
                interval_batch_count++;

                size_t log_interval = std::max<size_t>(5, effective_batches / 10);
                if ((batch_idx + 1) % log_interval == 0 || batch_idx + 1 == effective_batches)
                {
                    auto now_t = std::chrono::high_resolution_clock::now();
                    double cur_sec = std::chrono::duration<double>(now_t - epoch_start).count();
                    double cur_tok_s = (cur_sec > 0.0) ? (static_cast<double>(total_samples_processed * config.seq_len) / cur_sec) : 0.0;
                    float cur_avg_loss = epoch_loss_sum / static_cast<float>(total_samples_processed);
                    std::cout << "  [Batch " << (batch_idx + 1) << "/" << effective_batches << "] Step Loss: "
                        << std::fixed << std::setprecision(4) << step_loss << " | Avg: " << cur_avg_loss << " | Speed: "
                        << std::setprecision(0) << cur_tok_s << " tok/s\n";

                    Step_Timings::getInstance().printTable(batch_idx + 1, effective_batches, cur_tok_s, interval_batch_count);
                    Step_Timings::getInstance().reset();
                    interval_batch_count = 0;
                }
            }

            if (model.getConfig().execution_target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().readPendingLoss(0);
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
            double tokens_per_sec = (elapsed_sec > 0.0) ? (static_cast<double>(total_samples_processed * config.seq_len) / elapsed_sec) : 0.0;
            float avg_loss = (total_samples_processed > 0) ? (epoch_loss_sum / static_cast<float>(total_samples_processed)) : 0.0f;

            std::cout << "\n>>> Epoch " << epoch << "/" << config.epochs << " - Loss: " << std::fixed
                << std::setprecision(4) << avg_loss << " | PPL: " << std::exp(std::min(avg_loss, 20.0f))
                << " | Speed: " << static_cast<int>(tokens_per_sec) << " tok/s\n";

            if (avg_loss < best_loss)
            {
                best_loss = avg_loss;
                std::string best_path = config.getBestCkptPath();
                safeSaveCheckpoint(model, best_path);
                std::cout << "  [Checkpoint] Saved best model to " << best_path << " (Loss: " << best_loss << ")\n";
            }

            if (config.target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().setStaticGraphEnabled(false);
            }

            model.setTrainingMode(false);
            std::cout << "\n--------------------------------------------------------------\n";
            std::cout << "  [Automated Inference Check - Epoch " << epoch << "/" << config.epochs << "]\n";
            for (const auto& eval_prompt : config.eval_prompts)
            {
                std::cout << "  Prompt: \"" << eval_prompt << "\"\n";
                std::cout << "  Output: \"";
                model.generate(eval_prompt, config.max_gen_tokens, config.temperature, config.top_p, 2, true,
                    [](const std::string& token_str) {
                        std::cout << token_str << std::flush;
                    }, config.repetition_penalty, config.top_k, config.stop_sequences);
                std::cout << "\"\n\n";
            }
            std::cout << "--------------------------------------------------------------\n\n";
        }

        if (config.target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().setStaticGraphEnabled(false);
        }

        std::string latest_path = config.getLatestCkptPath();
        std::string model_path = config.getModelPath();
        safeSaveCheckpoint(model, latest_path);
        safeSaveInference(model, model_path);
        std::cout << "[Complete] Checkpoint saved: " << latest_path
            << " | Model saved: " << model_path << "\n";
    }

} // namespace

int main(int argc, char* argv[])
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Logger::setFileLogging(true);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);

    setCooperativeMatrixEnabled(true);
    setFp16Enabled(true);
    setStaticGraphEnabled(true);
    setFusedGemmAdamEnabled(true);

    Train_Config config_args;
    Cli_Args cli_args = parseArgs(argc, argv);

    if (cli_args.show_help)
    {
        printHelp();
        return 0;
    }

    if (!cli_args.output_dir.empty())
    {
        config_args.output_dir = cli_args.output_dir;
    }
    if (!cli_args.data_path.empty())
    {
        config_args.data_path = cli_args.data_path;
        std::filesystem::path dp(cli_args.data_path);
        config_args.bin_cache_path = (dp.parent_path() / (dp.stem().string() + ".bin")).string();
    }
    if (!cli_args.tokenizer_path.empty())
    {
        config_args.tokenizer_path = cli_args.tokenizer_path;
    }

    // Apply model architecture & training hyperparameters CLI overrides
    if (cli_args.override_epochs > 0)
    {
        config_args.epochs = static_cast<size_t>(cli_args.override_epochs);
    }
    if (cli_args.override_batch_size > 0)
    {
        config_args.batch_size = static_cast<size_t>(cli_args.override_batch_size);
    }
    if (cli_args.override_hidden_dim.has_value())
    {
        config_args.hidden_dim = cli_args.override_hidden_dim.value();
    }
    if (cli_args.override_num_heads.has_value())
    {
        config_args.num_heads = cli_args.override_num_heads.value();
    }
    if (cli_args.override_intermediate_dim.has_value())
    {
        config_args.intermediate_dim = cli_args.override_intermediate_dim.value();
    }
    if (cli_args.override_num_layers.has_value())
    {
        config_args.num_layers = cli_args.override_num_layers.value();
    }
    if (cli_args.override_seq_len.has_value())
    {
        config_args.seq_len = cli_args.override_seq_len.value();
    }
    if (cli_args.override_stride.has_value())
    {
        config_args.stride = cli_args.override_stride.value();
    }
    if (cli_args.override_learning_rate.has_value())
    {
        config_args.learning_rate = cli_args.override_learning_rate.value();
    }
    if (cli_args.override_min_lr.has_value())
    {
        config_args.min_lr = cli_args.override_min_lr.value();
    }
    if (cli_args.override_repetition_penalty.has_value())
    {
        config_args.repetition_penalty = cli_args.override_repetition_penalty.value();
    }

    if (config_args.num_heads == 0 || config_args.hidden_dim % config_args.num_heads != 0)
    {
        std::cerr << "[Error] hidden_dim (" << config_args.hidden_dim
                  << ") must be divisible by num_heads (" << config_args.num_heads << ").\n";
        return 1;
    }
    size_t head_dim = config_args.hidden_dim / config_args.num_heads;
    if (config_args.use_coop_matrix && (head_dim % 16 != 0))
    {
        std::cerr << "[Error] head_dim (" << head_dim
                  << ") must be a multiple of 16 for Cooperative Matrix acceleration.\n";
        return 1;
    }

    // Apply hardware acceleration settings & CLI overrides
    if (cli_args.override_fp16.has_value())
    {
        setFp16Enabled(config_args, cli_args.override_fp16.value());
    }
    else
    {
        setFp16Enabled(config_args, config_args.use_fp16);
    }

    if (cli_args.override_coop.has_value())
    {
        setCooperativeMatrixEnabled(config_args, cli_args.override_coop.value());
    }
    else
    {
        setCooperativeMatrixEnabled(config_args, config_args.use_coop_matrix);
    }

    if (cli_args.override_static_graph.has_value())
    {
        setStaticGraphEnabled(config_args, cli_args.override_static_graph.value());
    }
    else
    {
        setStaticGraphEnabled(config_args, config_args.use_static_graph);
    }

    if (cli_args.override_fused_gemm_adam.has_value())
    {
        setFusedGemmAdamEnabled(config_args, cli_args.override_fused_gemm_adam.value());
    }
    else
    {
        setFusedGemmAdamEnabled(config_args, config_args.use_fused_gemm_adam);
    }

    printAccelerationStatus(config_args);

    Bpe_Tokenizer tokenizer;
    if (!tokenizer.load(config_args.tokenizer_path))
    {
        std::cerr << "[Error] Cannot load tokenizer: " << config_args.tokenizer_path << "\n";
        return 1;
    }

    Causal_LM_Config model_config;
    model_config.vocab_size = tokenizer.getVocabSize();
    model_config.hidden_dim = config_args.hidden_dim;
    model_config.num_heads = config_args.num_heads;
    model_config.intermediate_dim = config_args.intermediate_dim;
    model_config.num_layers = config_args.num_layers;
    model_config.max_seq_len = std::max(config_args.seq_len * 2, static_cast<size_t>(128));
    model_config.execution_target = config_args.target;
    model_config.data_type = config_args.use_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    model_config.use_loss_scaler = config_args.use_fp16;
    model_config.initial_loss_scale = config_args.initial_loss_scale;
    model_config.tokenizer_path = config_args.tokenizer_path;

    Causal_LM model(model_config);

    size_t total_params = countParameters(model);
    printModelArchitecture(config_args, total_params);

    if (cli_args.mode == Cli_Args::Execution_Mode::INFER)
    {
        std::string model_path = cli_args.checkpoint_path;
        if (model_path.empty())
        {
            model_path = findDefaultInferenceModel(config_args);
        }
        if (!loadModelWeights(model, model_path))
        {
            return 1;
        }
        runInferenceInteractive(model, config_args, cli_args.custom_prompt);
        return 0;
    }
    else if (cli_args.mode == Cli_Args::Execution_Mode::TRAIN)
    {
        if (cli_args.override_epochs > 0) config_args.epochs = cli_args.override_epochs;
        if (cli_args.override_batch_size > 0) config_args.batch_size = cli_args.override_batch_size;
        bool resume = (cli_args.train_source == Cli_Args::Train_Source::RESUME);
        size_t max_batches = (cli_args.override_max_batches >= 0) ? cli_args.override_max_batches : 0;
        std::string ckpt_path = cli_args.checkpoint_path;
        if (resume && ckpt_path.empty())
        {
            ckpt_path = findDefaultResumeCheckpoint(config_args);
        }
        runTraining(model, tokenizer, config_args, resume, ckpt_path, max_batches);
        return 0;
    }

    std::cout << "\n======================================================\n";
    std::cout << "       Small LLM - Vulkan Native Engine (C++23)\n";
    std::cout << "======================================================\n";
    std::cout << " [1] Training Mode\n";
    std::cout << " [2] Inference Mode\n";
    std::cout << " [0] Exit\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "Select mode [1/2/0] (Default: 1): ";

    std::string choice_str;
    std::getline(std::cin, choice_str);
    int choice = 1;
    if (!choice_str.empty())
    {
        try { choice = std::stoi(choice_str); }
        catch (...) { choice = 1; }
    }

    if (choice == 0)
    {
        return 0;
    }
    else if (choice == 2)
    {
        std::string default_model = findDefaultInferenceModel(config_args);
        std::cout << "\nEnter model path [Default: " << default_model << "]: ";
        std::string model_path;
        std::getline(std::cin, model_path);
        if (model_path.empty()) model_path = default_model;

        if (!loadModelWeights(model, model_path))
        {
            return 1;
        }

        runInferenceInteractive(model, config_args);
    }
    else
    {
        std::cout << "\n--- TRAINING SOURCE ---\n";
        std::cout << " [1] Train from scratch\n";
        std::cout << " [2] Resume from checkpoint\n";
        std::cout << "Select source [1/2] (Default: 1): ";

        std::string src_str;
        std::getline(std::cin, src_str);
        int src_choice = 1;
        if (!src_str.empty())
        {
            try { src_choice = std::stoi(src_str); }
            catch (...) { src_choice = 1; }
        }

        bool resume = (src_choice == 2);
        std::string ckpt_path = "";
        if (resume)
        {
            std::string default_ckpt = findDefaultResumeCheckpoint(config_args);
            std::cout << "Enter checkpoint path [Default: " << default_ckpt << "]: ";
            std::getline(std::cin, ckpt_path);
            if (ckpt_path.empty()) ckpt_path = default_ckpt;
        }

        std::cout << "Output directory [Default: " << config_args.output_dir << "]: ";
        std::string out_dir_input;
        std::getline(std::cin, out_dir_input);
        if (!out_dir_input.empty())
        {
            config_args.output_dir = out_dir_input;
        }

        std::cout << "Epochs [Default: " << config_args.epochs << "]: ";
        std::string ep_str;
        std::getline(std::cin, ep_str);
        if (!ep_str.empty())
        {
            try { config_args.epochs = std::stoul(ep_str); }
            catch (...) {}
        }

        std::cout << "Batch size [Default: " << config_args.batch_size << "]: ";
        std::string bs_str;
        std::getline(std::cin, bs_str);
        if (!bs_str.empty())
        {
            try { config_args.batch_size = std::stoul(bs_str); }
            catch (...) {}
        }

        std::cout << "Max batches per epoch (0 = full dataset) [Default: 0]: ";
        std::string mb_str;
        std::getline(std::cin, mb_str);
        size_t max_batches = 0;
        if (!mb_str.empty())
        {
            try { max_batches = std::stoul(mb_str); }
            catch (...) {}
        }

        runTraining(model, tokenizer, config_args, resume, ckpt_path, max_batches);
    }

    return 0;
}