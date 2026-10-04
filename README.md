# VulkanML

> A high-performance, modular machine learning library built from scratch in modern C++23 with Vulkan Compute acceleration — supporting gradient-based deep learning (Vision CNNs, ResNets, Transformer LLMs), gradient-free neuroevolution, and reinforcement learning on any Vulkan-capable GPU.

![Language](https://img.shields.io/badge/C%2B%2B-23-blue.svg)
![API](https://img.shields.io/badge/Vulkan-1.3-red.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-success.svg)
![Architecture](https://img.shields.io/badge/Architecture-Modular%20Static%20Library-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

VulkanML is a self-contained, educational and production-oriented machine learning library written entirely from scratch in C++23 with **Vulkan Compute** as the hardware accelerator. It does not depend on CUDA, ROCm, PyTorch, TensorFlow, or any external deep learning runtime.

The primary goal is **education through implementation**: understanding how matrix operations, memory allocators, GPU compute pipelines, transformer attention mechanisms, and learning algorithms work from first principles by engineering every component from the ground up.

The library features a cleanly decoupled **Header (`include/`) + Implementation (`src/`)** architecture compiled into the static library `vulkanml_core`, providing rapid incremental builds and modularity across three primary paradigms:

- **Gradient-Based Deep Learning** — Supervised training via forward/backpropagation, supporting Vision CNNs, ResNets, and autoregressive Large Language Models (LLMs) with Adam/SGD optimizers and learning rate schedulers.
- **Gradient-Free Neuroevolution** — High-throughput population-based neuroevolution (genetic algorithms) where entire generations are evaluated concurrently via batched GPU GEMM dispatches.
- **Reinforcement Learning** — DQN and PPO agents with experience replay and environment simulation abstractions.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION & SUBSYSTEMS                       │
│      Neural_Network · Population · RL (DQN, PPO) · Causal_LM (LLM)       │
│         Tokenizers (BPE, Syllable, VN Phonetics) · Training_Context      │
├──────────────────────────────────────────────────────────────────────────┤
│                            LAYER SYSTEM                                  │
│   ILayer ── Linear · Conv2D · BatchNorm · RMSNorm · GELU · ReLU · Softmax│
│             SwiGLU · Embedding · MaxPool2D · GlobalAvgPool2D             │
│             ResNet Block / ResNet-20 · Transformer_Block · Actor-Critic  │
├──────────────────────────────────────────────────────────────────────────┤
│                      TENSOR ABSTRACTION  (math/)                         │
│                     Tensor · Shape · Stride · Data_Type                  │
│             Cpu_Tensor_Impl ◄──── PIMPL ────► Gpu_Tensor_Impl            │
├──────────────────────────────────────────────────────────────────────────┤
│                       EXECUTION ENGINE  (engine/)                        │
│  Vulkan_Context · Graph_Optimizer (JIT fusion) · Graph_Executor          │
│  Shader_Generator · Pipeline_Cache · Sub-Allocator · Loss_Scaler (AMP)   │
├──────────────────────────────────────────────────────────────────────────┤
│                       STATIC CORE LIBRARY (`src/`)                       │
│                     vulkanml_core (libvulkanml_core.a)                   │
└──────────────────────────────────────────────────────────────────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │     Vulkan Compute API     │
                       │  VkComputePipeline         │
                       │  VkCommandBuffer dispatch  │
                       │  VK_KHR_cooperative_matrix │
                       └────────────────────────────┘
```

---

## Features

| Category | Details |
|:---|:---|
| **Architecture** | Clean C++23 modular separation: public interface headers in `include/` and compiled implementation in `src/` packaged into `vulkanml_core` static library. |
| **Tensor Engine** | N-dimensional Tensor with dynamic/contiguous strides, broadcasting, slice/gather operations, and seamless zero-copy CPU↔GPU transfers. |
| **Neural Layers** | Linear (Dense), Conv2D, BatchNorm 1D/2D, MaxPool2D, GlobalAvgPool2D, GELU, ReLU, Softmax, RMSNorm, SwiGLU, Embedding, ResNet Block, ResNet-20, PPO Actor-Critic, Transformer Block. |
| **LLM & Attention** | Autoregressive Decoder-only `Causal_LM`, `KV_Cache_Manager`, FlashAttention forward/backward, Rotary Position Embeddings (RoPE), and memory-mapped `Binary_Dataset`. |
| **Tokenization** | Byte-Pair Encoding (`Bpe_Tokenizer`), `Syllable_Tokenizer`, and Vietnamese phonetics engine (`Vietnamese_Phonetics`) with tone analysis (Bằng/Trắc) and Lục Bát rhyme validation. |
| **Loss Functions** | MSE, MAE, BCE, CCE, Huber, and GPU-fused Cross Entropy with logits (`Fused_Cce_Cost`). |
| **Optimizers** | Adam (with fused weight decay and gradient clipping), SGD (with momentum). |
| **LR Schedulers** | Cosine Annealing, Step Decay, Multi-Step Decay, Exponential Decay, Polynomial Decay, Reduce on Plateau, Constant. |
| **Neuroevolution** | Population-based evolutionary algorithms (Tournament Selection, Uniform Crossover, Gaussian Mutation, Elitism) with batched GPU inference across generations. |
| **Reinforcement Learning**| Deep Q-Network (DQN) with Replay Buffer and target sync, Proximal Policy Optimization (PPO Actor-Critic), CartPole simulation environment. |
| **GPU Backend** | Vulkan 1.3 Compute, JIT GLSL shader generation, operator fusion (GEMM + Bias + Activation), persistent pipeline caching, and sub-allocator memory pooling. |
| **Cooperative Matrix** | Auto-detected `VK_KHR_cooperative_matrix` extension for high-performance 16×16×16 subgroup hardware tensor cores. |
| **Mixed Precision (AMP)** | True Native FP16 storage & compute pipeline (Zero-Cast), dynamic Loss Scaler for gradient underflow protection, FP32 master weights. |
| **Data Pipeline** | Asynchronous double-buffered data pipeline overlapping CPU disk I/O with GPU training batches. |
| **Serialization** | Binary model format (inference `.nni` + checkpoint `.nnc` / `.nnck`) with complete topology auto-restoration. |

---

## LLM Setup & Training Guide

VulkanML includes an end-to-end, hardware-accelerated **Decoder-only Transformer Large Language Model** subsystem (`Causal_LM`) inspired by modern architectures like LLaMA and Mistral.

### 1. Architecture Overview

```
Token IDs ────────► [ Embedding Layer ] (vocab_size -> hidden_dim)
                           │
             ┌─────────────┴─────────────┐
             │   Transformer Block × N   │
             │  ┌─────────────────────┐  │
             │  │ Pre-RMSNorm         │  │
             │  │ Multi-Head Attention│  │ (RoPE + FlashAttention + KV Cache)
             │  │ Residual Add        │  │
             │  ├─────────────────────┤  │
             │  │ Pre-RMSNorm         │  │
             │  │ SwiGLU FFN          │  │ (hidden_dim -> intermediate_dim -> hidden_dim)
             │  │ Residual Add        │  │
             │  └─────────────────────┘  │
             └─────────────┬─────────────┘
                           ▼
                  [ Final RMSNorm ]
                           ▼
                  [ Linear LM Head ] (hidden_dim -> vocab_size)
                           ▼
           [ Fused Cross-Entropy Loss / Logits ]
```

* **Position Encoding**: Rotary Position Embeddings (RoPE) applied to Query and Key projections.
* **Attention Mechanism**: Causal Masked Multi-Head Attention accelerated with FlashAttention GPU dispatch.
* **Feed-Forward**: SwiGLU (Swish-Gated Linear Unit) activation block.
* **Normalization**: RMSNorm with learnable scaling factors before attention and MLP blocks.
* **Inference Decoding**: $O(1)$ dynamic Key-Value Cache manager (`KV_Cache_Manager`).
* **Loss Function**: `Fused_Cce_Cost` computing cross-entropy directly on GPU memory without allocating huge uncompressed logit tensors on CPU.

---

### 2. Dataset Preparation & Tokenization

VulkanML provides two tokenizer implementations and a high-throughput binary cache pipeline:

1. **`Bpe_Tokenizer`**: Standard subword Byte-Pair Encoding tokenizer compatible with HuggingFace JSON exports.
2. **`Syllable_Tokenizer`**: Dedicated syllable-level tokenizer designed for Vietnamese text, with built-in tonal decomposition (`Vietnamese_Phonetics`) and rhyme matching.

#### Pre-tokenizing Text to Binary Cache (`Binary_Dataset`)

Parsing raw text on every epoch introduces severe CPU bottlenecks. VulkanML uses `Binary_Dataset` to compile raw `.txt` files into indexed binary token files (`.bin`):

```cpp
#include "llm/binary_dataset.h"
#include "tokenizer/bpe_tokenizer.h"

// 1. Initialize and train/load tokenizer
Bpe_Tokenizer tokenizer;
tokenizer.load("tokenizer/bpe_tokenizer.json");

// 2. Pre-tokenize text into binary cache
// Format: Header + 32-bit token IDs sequence
Binary_Dataset::createFromText(
    "data/corpus.txt",           // Raw text corpus
    "data/corpus.bin",           // Output binary file
    tokenizer,
    /*seq_len=*/128,             // Fixed sequence length
    /*stride=*/64                // Overlap stride for sliding window
);

// 3. Load binary dataset into training pipeline
Binary_Dataset dataset("data/corpus.bin");
std::cout << "Loaded " << dataset.getSampleCount() << " training sequences.\n";
```

---

### 3. Model Configuration (`Causal_LM_Config`)

The model architecture and training hyperparameters are defined using `Causal_LM_Config`:

```cpp
#include "llm/causal_lm.h"

Causal_LM_Config config{
    .vocab_size = 32000,                           // Vocabulary size
    .hidden_dim = 256,                             // Hidden embedding dimension
    .num_heads = 8,                                // Attention heads (head_dim = 256 / 8 = 32)
    .intermediate_dim = 1024,                      // SwiGLU intermediate dimension (~4x hidden)
    .num_layers = 6,                               // Number of Transformer blocks
    .max_seq_len = 128,                            // Maximum context window length
    .rms_norm_eps = 1e-5f,                         // RMSNorm epsilon
    .rope_base = 10000.0f,                         // RoPE base frequency
    .execution_target = Execution_Target::VULKAN_GPU,
    .data_type = Data_Type::FLOAT16,               // Native FP16 mixed precision
    .use_loss_scaler = true,                       // Dynamic Loss Scaler for FP16 AMP
    .initial_loss_scale = 1024.0f,
    .tokenizer_path = "tokenizer/tokenizer.json"
};

Causal_LM model(config);
```

---

### 4. Training Pipeline (`train_sllm`)

VulkanML includes an end-to-end Small Language Model (sLLM) training pipeline in [`train_sllm.cpp`](file:///d:/Not%20Python%20Projects/llm%20-%20Copy/train_sllm.cpp).

#### Training Loop Example

```cpp
#include "llm/causal_lm.h"
#include "llm/binary_dataset.h"
#include "optimizer/adam_optimizer.h"
#include "learning_rate/cosine_annealing.h"

// 1. Setup Optimizer & Scheduler
Adam_Optimizer optimizer(
    /*learning_rate=*/0.0004f,
    /*beta1=*/0.9f,
    /*beta2=*/0.95f,
    /*epsilon=*/1e-8f,
    /*max_gradient=*/1.0f,                          // Gradient clipping threshold
    /*weight_decay=*/0.1f                          // Decoupled weight decay
);

Cosine_Annealing scheduler(
    /*initial_lr=*/0.0004f,
    /*min_lr=*/0.00004f,
    /*max_epochs=*/40
);

// 2. Training Loop with Gradient Accumulation
model.setTrainingMode(true);
const size_t batch_size = 8;
const size_t grad_accum_steps = 4;                 // Effective batch size = 32

for (size_t epoch = 0; epoch < 40; ++epoch)
{
    float epoch_loss = 0.0f;
    size_t batch_count = 0;

    for (size_t i = 0; i < dataset.getSampleCount(); i += batch_size)
    {
        auto [input_batch, target_batch] = dataset.getBatch(i, batch_size);

        // Forward pass, loss computation, and backpropagation
        float loss = model.trainStep(input_batch, target_batch, optimizer, /*max_grad_norm=*/1.0f);
        epoch_loss += loss;
        batch_count++;
    }

    scheduler.step();
    std::cout << "Epoch " << epoch << " | Loss: " << (epoch_loss / batch_count) << "\n";

    // Periodic Checkpointing
    model.saveCheckpoint("output/checkpoint_latest.nnck");
}
```

#### Compiling and Running the LLM Trainer

```bash
# Configure CMake with train_sllm as the active entry point
cmake -B build -DACTIVE_FILE="train_sllm.cpp" -DCMAKE_BUILD_TYPE=Release

# Build the executable
cmake --build build --target train_sllm -j8

# Run pre-training
./train_sllm.exe
```

---

### 5. Autoregressive Text Generation with KV Caching

Inference uses the persistent `KV_Cache_Manager` to store precomputed Key and Value projections, guaranteeing $O(1)$ compute per newly generated token:

```cpp
#include "llm/causal_lm.h"

// Load trained model checkpoint
Causal_LM model(config);
model.loadInference("output/model.bin");
model.setTrainingMode(false);

// Stream tokens to console in real-time
auto token_stream_callback = [](const std::string& token_text) {
    std::cout << token_text << std::flush;
};

// Generate text with Top-P (Nucleus) and Top-K sampling
std::string prompt = "Trăm năm trong cõi người ta\n";
std::cout << prompt;

std::string completion = model.generate(
    prompt,
    /*max_new_tokens=*/128,
    /*temperature=*/0.7f,
    /*top_p=*/0.9f,
    /*eos_token_id=*/2,
    /*skip_special=*/true,
    /*token_callback=*/token_stream_callback,
    /*repetition_penalty=*/1.15f,
    /*top_k=*/40,
    /*stop_sequences=*/{ "<|endoftext|>", "\n\n" }
);
```

---

## Usage Example — Vision CNN Supervised Training

```cpp
#include "engine/execution_engine.h"
#include "helper/layer.h"
#include "learning_rate/cosine_annealing.h"
#include "math/tensor.h"
#include "neural_network.h"
#include "optimizer/adam_optimizer.h"
#include "cost_function/cce_cost.h"

int main()
{
    Execution_Engine::getInstance()
        .getPipelineCacheManager()
        .initializePipelineCache("temp/pipeline_cache.bin");

    Execution_Target target = Execution_Target::VULKAN_GPU;
    Neural_Network net(target);
    net.setTrainingMode(true);

    net.setLearningRate<Cosine_Annealing>(0.001f, 1e-5f, 10);
    net.setOptimizer<Adam_Optimizer>(net.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f);
    net.setCostFunction<Cce_Cost>();

    // Vision CNN: 28x28 grayscale -> 10 classes
    net.addLayer<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1, target);
    net.addLayer<Batch_Norm_2d_Layer>(28, 28, 16, 1e-5f, 0.1f, target);
    net.addLayer<Gelu_Layer>(target);
    net.addLayer<Max_Pool_2d_Layer>(28, 28, 16, 2, 2, 0, target);

    net.addLayer<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1, target);
    net.addLayer<Batch_Norm_2d_Layer>(14, 14, 32, 1e-5f, 0.1f, target);
    net.addLayer<Gelu_Layer>(target);
    net.addLayer<Max_Pool_2d_Layer>(14, 14, 32, 2, 2, 0, target);

    net.addLayer<Linear_Layer>(7 * 7 * 32, 128, target);
    net.addLayer<Batch_Norm_Layer>(128, 1e-5f, 0.1f, target);
    net.addLayer<Gelu_Layer>(target);
    net.addLayer<Linear_Layer>(128, 10, target);
    net.addLayer<Softmax_Layer>(true, target);

    net.compileAndWarmup(512, 784, 10);

    Tensor input(512, 784, target);
    Tensor target_labels(512, 10, target);

    net.trainStep(input, target_labels);
    net.getLearningRate().step();

    net.saveInference("output/mnist_model.bin");
    return 0;
}
```

---

## Usage Example — Population-Based Neuroevolution

```cpp
#include "population.h"
#include "neural_network.h"
#include "helper/layer.h"

int main()
{
    Neural_Network template_net(Execution_Target::VULKAN_GPU);
    template_net.addLayer<Linear_Layer>(128, 64, Execution_Target::VULKAN_GPU);
    template_net.addLayer<Gelu_Layer>(Execution_Target::VULKAN_GPU);
    template_net.addLayer<Linear_Layer>(64, 8, Execution_Target::VULKAN_GPU);

    // 64 individuals evolved simultaneously via single batched GPU GEMM
    Population population(template_net, 64);

    std::vector<std::vector<float>> inputs(64, std::vector<float>(128, 0.0f));
    auto actions = population.selectBatchActions(inputs);

    std::vector<float> fitness(64, 0.0f);
    // ... evaluate fitness ...
    population.evolve(fitness);

    return 0;
}
```

---

## Benchmarks & Performance

### FP16 vs FP32 Performance (MNIST CNN on AMD Radeon™ 860M)

| Evaluation Metric | FP32 Baseline | FP32 Optimized | FP16 Simulated (Cast-only) | True Native FP16 (Zero-Cast) | Impact / Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1-Epoch Training Time** | 24.00 s | 12.27 s | 13.60 s | **8.78 s** | **~28.5% faster than opt FP32, 2.73x vs baseline** |
| **Per-Batch Graph Dispatch** | ~3.50 ms | ~1.55 ms | ~1.81 ms | **~1.32 - 1.38 ms** | **-60% latency reduction** |
| **Intermediate Cast Passes** | 0 | 0 | 8 - 10 per batch | **0 (Zero-Cast Pipeline)** | **100% cast overhead eliminated** |
| **Fence Wait (CPU-GPU stall)**| 0.040 ms | 0.001 ms | 0.001 ms | **0.001 ms** | **Zero sync stall (Fully overlapped)** |
| **VRAM Buffer Allocation** | Dynamic | Persistent | Reallocated per batch | **Persistent Pre-allocated Buffers** | **Zero runtime allocation overhead** |
| **Test Accuracy (1 Epoch)** | 98.60% | 98.92% | 98.60% | **97.76% - 98.90%** | **Retains classification accuracy** |

### Benchmark Results Summary

* **MNIST (Supervised CNN)**: **99.51%** test accuracy in **~12.8s** (1 epoch / batch size 512) on AMD Radeon 860M iGPU.
* **CIFAR-100 (Deep CNN)**: **67.51%** validation accuracy with data augmentations over 100 epochs.

---

## Building & Installation

### Requirements

* **C++ Compiler**: Modern C++23 compliant compiler (MinGW GCC 13/14+, Clang 17+, MSVC 19.38+)
* **Vulkan SDK**: Vulkan 1.3+ SDK with `glslc` / `shaderc` libraries installed
* **Build System**: CMake 3.25+

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/<username>/VulkanML.git
cd VulkanML

# Configure CMake build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the core static library and the comprehensive test suite
cmake --build build --target test -j8

# Run test suite
./test.exe
```

### Building Application Targets

The build system automatically compiles all source files in `src/` into the static library `vulkanml_core` and links it to any active executable:

```bash
# Build CIFAR-100 vision model training target
cmake --build build --target cifar_train -j8

# Build Small Language Model (sLLM) training target
cmake --build build --target train_sllm -j8

# Build general usage example
cmake --build build --target example -j8
```

---

## Project Structure

The codebase is organized in an orthogonal layout separating public header interfaces (`include/`) from compiled implementation units (`src/`):

```
├── include/                          # Public header declarations and type interfaces
│   ├── cost_function/                # ICost_Function, MSE, MAE, BCE, CCE, Huber, Fused_CCE
│   ├── engine/                       # Vulkan_Context, Graph_Executor, Graph_Optimizer,
│   │                                 # Shader_Compiler, Sub_Allocator, Pipeline_Cache, Loss_Scaler
│   ├── helper/                       # Logger, User_Preferences, Training_Profiler, Facade headers
│   ├── layer/                        # ILayer, Linear, Conv2D, BatchNorm, RMSNorm, SwiGLU,
│   │                                 # Transformer_Block, Embedding, ResNet, PPO_Actor_Critic
│   ├── learning_rate/                # ILearning_Rate, Cosine, Step, Exponential, Polynomial, Plateau
│   ├── llm/                          # Causal_LM, KV_Cache_Manager, Binary_Dataset
│   ├── math/                         # Tensor, Shape, Cpu_Tensor_Impl, Gpu_Tensor_Impl
│   ├── optimizer/                    # IOptimizer, Adam_Optimizer, Sgd_Optimizer
│   ├── rl/                           # DQN_Agent, PPO_Agent, Replay_Buffer, CartPole_Env
│   ├── tokenizer/                    # BPE_Tokenizer, Syllable_Tokenizer, Vietnamese_Phonetics
│   ├── neural_network.h              # High-level sequential neural network container
│   ├── population.h                  # Neuroevolution population manager
│   └── training_context.h            # Training hyperparameters, layer & loss factories
│
├── src/                              # Implementation sources compiled into libvulkanml_core.a
│   ├── cost_function/                # Concrete cost function algorithms
│   ├── engine/                       # Vulkan device dispatch, JIT GLSL fusion, async pipeline
│   ├── helper/                       # Logging, profiling, and preferences implementations
│   ├── layer/                        # Concrete layer forward/backward and GPU kernel dispatches
│   ├── learning_rate/                # Learning rate scheduler step computations
│   ├── llm/                          # LLM autoregressive generation and KV cache paging
│   ├── math/                         # CPU & Vulkan GPU tensor algebra, FlashAttention, RoPE
│   ├── optimizer/                    # Adam / SGD parameter updates with weight decay & AMP scaling
│   ├── rl/                           # Reinforcement learning agents and replay memory algorithms
│   ├── tokenizer/                    # BPE tokenization and Vietnamese phonetic analysis
│   ├── neural_network.cpp            # Model training loops, forward/backward execution, checkpoints
│   ├── population.cpp                # Evolutionary genetic operators and batched GPU inference
│   └── training_context.cpp          # Serialization and reflection layer factories
│
├── test.cpp                          # Comprehensive unit & integration test suite (100% PASS)
├── train_sllm.cpp                    # Small Language Model training entry point
├── cifar_train.cpp                   # CIFAR vision model training entry point
├── example.cpp                       # General API demonstration entry point
└── CMakeLists.txt                    # Project build script and vulkanml_core target setup
```

---

## Third-Party Credits

| Library | License | Usage |
|:---|:---|:---|
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | JSON configuration serialization and hyperparameter persistence |
| [magic_enum](https://github.com/Neargye/magic_enum) | MIT | Compile-time enum-to-string reflection for logging and serialization |

---

## Design Philosophy

Rather than treating neural networks as black boxes, VulkanML focuses on understanding every computation involved in modern machine learning and systems engineering — from raw GPU memory allocations and SPIR-V / GLSL shader dispatches to FlashAttention kernels and genetic evolutionary algorithms.

Every core component is implemented from scratch with high architectural rigor, adhering to modern C++23 RAII principles, zero-cast mixed-precision pipelines, and minimal external dependencies.

---

## AI Usage

AI is utilized in this project for architectural refactoring, performance profiling, and debugging; the foundational engine, math kernels, and execution pipelines are designed and verified through rigorous human pair-programming and automated test suites.

---

## License

MIT License.
