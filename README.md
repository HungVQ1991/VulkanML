# VulkanML

> A high-performance, modular machine learning library built from scratch in modern C++23 with Vulkan Compute acceleration — supporting gradient-based deep learning (Vision CNNs, U-Net Segmentation, Transformer LLMs), 1.58-bit ternary quantization, gradient-free neuroevolution, and reinforcement learning on any Vulkan-capable GPU.

![Language](https://img.shields.io/badge/C%2B%2B-23-blue.svg)
![API](https://img.shields.io/badge/Vulkan-1.3-red.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-success.svg)
![Architecture](https://img.shields.io/badge/Architecture-Modular%20Shared%20Library-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

VulkanML is a self-contained, educational and production-oriented machine learning library written from scratch in C++23 with **Vulkan Compute** as the hardware accelerator. It does not depend on CUDA, ROCm, PyTorch, or TensorFlow.

The public API hides Vulkan SDK headers behind PIMPL abstractions (`Tensor`, `Neural_Network`, `Causal_LM`, `Runtime`, `LM_Trainer`, `Resource_Path`), linking `Vulkan::Vulkan` and `shaderc` as `PRIVATE` dependencies inside `vulkanml_core.dll`. User applications only need `#include "helper/vulkan_ml.h"` with zero direct Vulkan SDK include or link requirements.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     PUBLIC FACADE (`include/helper/vulkan_ml.h`)         │
│   Runtime · Neural_Network · Causal_LM · LM_Trainer · Population · RL   │
│         Tokenizers · User_Preferences · Resource_Path · Logger           │
├──────────────────────────────────────────────────────────────────────────┤
│                            LAYER SYSTEM (`include/layer/`)               │
│   ILayer ── Linear · Ternary_Linear · Conv2D · BatchNorm · InstanceNorm  │
│             RMSNorm · SwiGLU · Embedding · MaxPool2D · UpSample2D        │
│             ResNet Block · U_Net · NN_UNet · Transformer_Block · PPO     │
├──────────────────────────────────────────────────────────────────────────┤
│                      TENSOR ABSTRACTION (`include/math/tensor.h`)        │
│   Tensor · Shape · Stride · Data_Type (FLOAT32, FLOAT16, TERNARY_INT2)   │
├──────────────────────────────────────────────────────────────────────────┤
│                 PRIVATE VULKAN BACKEND (`src/engine/`, `src/ops/`)       │
│   Vulkan_Context · Graph_Optimizer (JIT fusion) · Graph_Executor         │
│   Vulkan_Network (207 SPIR-V pipelines) · Pipeline_Cache_Manager         │
│   Vulkan_Sub_Allocator (Best-Fit + Empty-Chunk Reclamation)              │
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
| :--- | :--- |
| **Zero-Vulkan Public API** | Public entry umbrella [`include/helper/vulkan_ml.h`](include/helper/vulkan_ml.h) exposes all high-level ML/LLM/RL/runtime types without leaking `<vulkan/vulkan.h>`. |
| **Tensor Engine** | N-dimensional `Tensor` with dynamic/contiguous strides, broadcasting, zero-copy views, FP32/FP16/TERNARY_INT2 storage. |
| **Neural Layers** | Linear, TernaryLinear, Conv2D, BatchNorm 1D/2D, InstanceNorm2D, RMSNorm, SwiGLU, Embedding, Pooling, UpSample2D, ResNet-18/20, U-Net, NN-UNet, Transformer Block, PPO Actor-Critic. |
| **LLM & Attention** | Autoregressive Decoder-only [`Causal_LM`](include/llm/causal_lm.h), [`KV_Cache_Manager`](include/llm/kv_cache_manager.h), FlashAttention-2 forward/backward (BSHD & CoopMat FP16), RoPE, and [`Binary_Token_Dataset`](include/llm/binary_dataset.h). |
| **Tokenization** | `Bpe_Tokenizer` (HuggingFace JSON, aliased as `Tokenizer`). |
| **Loss Functions** | MSE, MAE, BCE, CCE, Huber, ASL, Dice, Dice+BCE, and GPU-fused Cross Entropy (`Fused_Cross_Entropy`). |
| **Optimizers & LR** | AdamW (`Adam_Optimizer`), SGD, and schedulers (Cosine Annealing, Step, MultiStep, Exponential, Polynomial, Plateau, NoDecay). |
| **GPU Backend** | Vulkan 1.3 Compute, JIT GLSL operator fusion, `VK_KHR_cooperative_matrix` 16x16x16, unified device/driver-keyed `VkPipelineCache`, empty VRAM chunk reclamation (`reclaimEmptyChunks`), and `VkQueryPool` hardware timestamp profiling. |
| **Developer Experience** | RAII [`Runtime`](include/helper/runtime.h), [`LM_Trainer`](include/helper/trainer.h), [`Resource_Path`](include/helper/resource_path.h) layered path resolver, and feature-masked [`Logger`](include/helper/logger.h). |

---

## Quick Start — Public Facade (`Runtime` + `Neural_Network`)

```cpp
#include "helper/vulkan_ml.h"

int main(int argc, char *argv[])
{
    Logger::applyCliArgs(argc, argv);

    Runtime_Config rt_cfg;
    rt_cfg.enable_fp16 = true;
    rt_cfg.enable_cooperative_matrix = true;
    rt_cfg.enable_static_graph = true;
    Runtime runtime(rt_cfg);

    Neural_Network net(Execution_Target::VULKAN_GPU);
    net.setTrainingMode(true);
    net.setLearningRate<Cosine_Annealing>(0.001f, 1e-5f, 10);
    net.setOptimizer<Adam_Optimizer>(net.getLearningRate());
    net.setCostFunction<Cce_Cost>();

    net.addLayer<Linear_Layer>(784, 128, Execution_Target::VULKAN_GPU);
    net.addLayer<Gelu_Layer>(Execution_Target::VULKAN_GPU);
    net.addLayer<Linear_Layer>(128, 10, Execution_Target::VULKAN_GPU);
    net.addLayer<Softmax_Layer>(true, Execution_Target::VULKAN_GPU);

    net.compileAndWarmup(64, 784, 10);
    return 0;
}
```

## LLM Training & Generation (`Causal_LM` + `LM_Trainer`)

See [`apps/train_sllm.cpp`](apps/train_sllm.cpp) for the complete interactive and CLI training/generation workflow:

```bash
cmake --build build --target train_sllm -j 2
./build/bin/train_sllm.exe --train -b 8 --coop --static-graph --fused-gemm-adam
```

---

## Building, Shaders & Testing

### Requirements

* **C++ Compiler**: Modern C++23 compiler (MinGW-w64 UCRT64 GCC 13/14+, Clang 17+, or MSVC 19.38+)
* **Vulkan SDK**: Vulkan 1.3+ SDK with `glslc` and `shaderc`
* **Build System**: CMake 3.25+ and Python 3.11+ (for repository audit and shader tooling)

### Configure & Build (Memory-Safe `-j 2`)

All executables and `vulkanml_core.dll` are emitted to `build/bin/`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build --target compile_shaders -j 2

cmake --build build --target vulkanml_tests -j 2

cmake --build build --target train_sllm -j 2
cmake --build build --target train_chexpert -j 2
cmake --build build --target example -j 2
```

> **Note**: On memory-constrained Windows hosts without a pagefile, always build with `-j 2`.

### Running Tests (CTest)

The test suite in [`tests/`](tests) is partitioned into 10 isolated CTest suites returning non-zero exit codes on failure:

```bash
ctest --test-dir build --output-on-failure

ctest --test-dir build -L heavy --output-on-failure
ctest --test-dir build -LE heavy --output-on-failure

./build/bin/test.exe --list
./build/bin/test.exe --suite dx_infra
```

### Repository & Shader Verification

```bash
python scripts/generate_shaders.py
python scripts/verify_repo.py --compile-shaders
```

### Resource & Cache Configuration

`Resource_Path` resolves shader, preference, and pipeline cache locations automatically whether launched from the repository root, `build/bin/`, or an installed SDK (`vulkanml_sdk/bin/`), honoring explicit overrides (`VULKANML_ROOT`, `VULKANML_SHADER_DIR`, `VULKANML_SHADER_DICTIONARY`, `VULKANML_PREFERENCES_PATH`, `VULKANML_PIPELINE_CACHE_DIR`). Pipeline cache binaries default to `temp/pipeline_cache_<vendor>_<device>_d<driver>_<uuid>.bin`.

---

## Project Structure

```
├── include/                          # Public C++23 interface headers + private internal interfaces
│   ├── helper/                       # Public umbrella (vulkan_ml.h), Runtime, LM_Trainer, Logger, Resource_Path
│   ├── layer/, math/, llm/, cost_function/, optimizer/, learning_rate/, rl/, tokenizer/
│   ├── chexlocalize/                 # Medical X-ray Grad-CAM extension headers
│   ├── environment/                  # MuJoCo simulation headers
│   ├── ops/, engine/                 # Compute operator & engine headers (excluded from public SDK install)
│
├── src/                              # Implementations compiled into vulkanml_core.dll & chexlocalize_ext
├── apps/                             # Standalone training/inference/benchmark applications
│   ├── chexpert/                     # Modular CheXpert & U-Net application (train_chexpert target)
│   ├── train_sllm.cpp                # Small Language Model trainer & generator
│   ├── train_cifar.cpp, train_emnist.cpp, example.cpp
│   ├── validate_ternary_mnist.cpp, benchmark_ternary.cpp
│   └── train_humanoid.cpp, train_mujoco.cpp, verify_humanoid.cpp
├── tests/                            # Modular test suites registered as CTest targets
├── compute_shader/
│   ├── comp/                         # 141 GLSL compute shaders (.comp)
│   ├── spv/                          # 207 compiled SPIR-V kernels (.spv)
│   └── shader_dictionary.json        # Operator fusion snippet metadata
├── scripts/                          # Shader compilation/audit & automated verification scripts
├── cmake/                            # CMake package config template (Vulkan-free for downstream apps)
├── CMakeLists.txt                    # Build configuration, compile_shaders, CTest, and SDK install rules
└── ARCHITECTURE.md                   # Detailed architecture specification (Vietnamese)
```

---

## License

MIT License.
