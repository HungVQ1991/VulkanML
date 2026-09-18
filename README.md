# VulkanML

> A machine learning library built from scratch in modern C++23 with Vulkan Compute acceleration — supporting both gradient-based training and gradient-free neuroevolution on any Vulkan-capable GPU.

![Language](https://img.shields.io/badge/C%2B%2B-23-blue.svg)
![API](https://img.shields.io/badge/Vulkan-1.3-red.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

VulkanML is a personal learning project — a self-contained machine learning library written entirely from scratch in C++23 with **Vulkan Compute** as the hardware accelerator. It does not depend on CUDA, PyTorch, TensorFlow, or any third-party ML framework.

The primary goal is **education through implementation**: understanding how matrix operations, neural networks, GPU compute pipelines, and learning algorithms actually work by building every component from the ground up.

The library provides a **device-agnostic tensor computation layer** on top of which two independent learning paradigms are built:

- **Gradient-based learning** — supervised training via forward/backpropagation, with Adam/SGD optimizers and a full learning rate scheduler suite.
- **Gradient-free learning** — population-based Neuroevolution where an entire generation of networks is inferred in a single Vulkan batched GEMM dispatch.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                              │
│          Neural_Network · Population · RL Agents (DQN, PPO)              │
│               Training_Context (Optimizer + LR + Loss)                   │
├──────────────────────────────────────────────────────────────────────────┤
│                            LAYER SYSTEM                                  │
│   ILayer ── Linear · Conv2D · BatchNorm · GELU · ReLU · Softmax          │
│             MaxPool2D · GlobalAvgPool2D · ResNet Block · Actor-Critic    │
├──────────────────────────────────────────────────────────────────────────┤
│                      TENSOR ABSTRACTION  (math/)                         │
│                     Tensor<N-D> · Matrix · Shape                         │
│             Cpu_Tensor_Impl ◄──── PIMPL ────► Gpu_Tensor_Impl            │
├──────────────────────────────────────────────────────────────────────────┤
│                       EXECUTION ENGINE  (engine/)                        │
│  Vulkan_Context · Graph_Optimizer (JIT fusion) · Graph_Executor          │
│  Shader_Generator · Pipeline_Cache · Sub-Allocator · Async_Pipeline      │
└──────────────────────────────────────────────────────────────────────────┘
                                  ▼
                   ┌────────────────────────────┐
                   │      Vulkan Compute API    │
                   │  VkComputePipeline         │
                   │  VkCommandBuffer dispatch  │
                   │  Cooperative Matrix Ext.   │
                   └────────────────────────────┘
```

---

## Features

| Category | Details |
|:---|:---|
| **Tensor System** | N-dimensional Tensor, Shape descriptor, seamless CPU↔GPU transfer |
| **Layers** | Linear (Dense), Conv2D, BatchNorm 1D/2D, MaxPool2D, GlobalAvgPool2D, GELU, ReLU, Softmax, ResNet Block, ResNet-20, PPO Actor-Critic |
| **Loss Functions** | MSE, MAE, BCE, CCE, Huber |
| **Optimizers** | Adam, SGD |
| **LR Schedulers** | Cosine Annealing, Step Decay, Multi-Step Decay, Exponential Decay, Polynomial Decay, Reduce on Plateau |
| **Neuroevolution** | Population with Tournament Selection, Uniform Crossover, Gaussian Mutation, Elitism; batched GPU inference over full generation |
| **RL Agents** | DQN (with Replay Buffer + Target Network), PPO (Actor-Critic) |
| **GPU Backend** | Vulkan Compute Shaders, JIT operator fusion, on-disk Pipeline Cache, Sub-allocator memory pool |
| **Cooperative Matrix** | Auto-detected `VK_KHR_cooperative_matrix` for 16×16×16 subgroup GEMM |
| **Mixed Precision (AMP)** | True Native FP16 compute & storage, Zero-Cast pipeline, dynamic Loss Scaler, FP32 master weights |
| **Data Pipeline** | Async CPU-side data pipeline for overlapping I/O with GPU training |
| **Serialization** | Binary model format (inference + checkpoint) with topology auto-restoration |

---

## Usage Example — Gradient-Based Training

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
    // Initialize on-disk Vulkan Pipeline Cache (avoids shader recompilation)
    Execution_Engine::getInstance()
        .getPipelineCacheManager()
        .initializePipelineCache("temp/pipeline_cache.bin");

    Execution_Target target = Execution_Target::VULKAN_GPU;
    Neural_Network net(target);
    net.setTrainingMode(true);

    net.setLearningRate<Cosine_Annealing>(0.001f, 1e-5f, 10);
    net.setOptimizer<Adam_Optimizer>(net.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f);
    net.setCostFunction<Cce_Cost>();

    // CNN for 28x28 grayscale → 10 classes
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

    // JIT operator fusion warmup
    net.compileAndWarmup(512, 784, 10);

    Matrix input(512, 784, target);
    Matrix target_labels(512, 10, target);

    net.trainStep(input, target_labels);
    net.getLearningRate().step();

    net.saveInference("output/model.bin");
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
    // Define template network topology
    Neural_Network template_net(Execution_Target::VULKAN_GPU);
    template_net.addLayer<Linear_Layer>(128, 64, Execution_Target::VULKAN_GPU);
    template_net.addLayer<Gelu_Layer>(Execution_Target::VULKAN_GPU);
    template_net.addLayer<Linear_Layer>(64, 8, Execution_Target::VULKAN_GPU);

    // Create a population of 64 individuals mirroring the template
    Population population(template_net, 64);

    // Infer actions for all 64 individuals simultaneously via Vulkan batched GEMM
    std::vector<std::vector<float>> inputs(64, std::vector<float>(128, 0.0f));
    auto actions = population.selectBatchActions(inputs);

    // Assign fitness scores and evolve
    std::vector<float> fitness(64);
    // ... evaluate fitness ...
    population.evolve(fitness);

    return 0;
}
```

---

## Benchmarks

### FP16 vs FP32 Performance & Architecture Comparison

#### 1. Precision & Hardware Architecture Comparison

| Architectural Property | FP32 (Single Precision) | True Native FP16 (Mixed Precision AMP) | Benefit / Note |
|:---|:---|:---|:---|
| **Representation Standard** | IEEE-754 Single (32-bit) | IEEE-754 Half (16-bit) | Standardized hardware floating point |
| **Bit Layout** | 1 sign, 8 exponent, 23 mantissa | 1 sign, 5 exponent, 10 mantissa | Compact storage layout |
| **Memory Footprint** | 4 Bytes / element | 2 Bytes / element | **-50% VRAM memory reduction** |
| **VRAM Bandwidth Consumption** | 100% (Baseline) | **50% of FP32** | **2x effective memory bandwidth** |
| **ALU Compute Throughput** | 1x (Single-Issue) | **2x (Packed Dual-Issue Wave32 ALU)** | Higher arithmetic intensity |
| **Dynamic Range** | $1.4 \times 10^{-45} \dots 3.4 \times 10^{38}$ | $5.96 \times 10^{-8} \dots 65,504$ | Sufficient dynamic range for deep learning |
| **Underflow Normal Threshold** | $\sim 1.18 \times 10^{-38}$ | $\sim 6.10 \times 10^{-5}$ | Managed via Dynamic Loss Scaling |
| **Cooperative Matrix Subgroup** | Standard Tile | **$16 \times 16 \times 16$ Tile (FP32 Accumulator)** | Hardware tensor acceleration |
| **Accumulators & Reduction** | FP32 | **FP32** | Zero overflow risk during dot products |
| **Master Weights (Optimizer)** | FP32 | **FP32 (Adam / SGD)** | Preserves tiny parameter updates |
| **Gradient Scaling** | Not needed | **Dynamic Loss Scaler** | Rescales gradients to prevent underflow |

#### 2. Training Benchmark Comparison (MNIST Vision CNN on AMD Radeon™ 860M)

> **Network Topology**: Conv2D(1→16) → BatchNorm2D → GELU → MaxPool2D → Conv2D(16→32) → BatchNorm2D → GELU → MaxPool2D → Linear(1568→128) → BatchNorm1D → GELU → Linear(128→10) → Softmax.

| Evaluation Metric | FP32 Baseline | FP32 Optimized | FP16 Simulated (Cast-only) | True Native FP16 (Zero-Cast) | Impact / Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1-Epoch Training Time** | 24.00 s | 12.27 s | 13.60 s | **8.78 s** | **~28.5% faster than opt FP32, 2.73x vs baseline** |
| **Per-Batch Graph Dispatch** | ~3.50 ms | ~1.55 ms | ~1.81 ms | **~1.32 - 1.38 ms** | **-60% latency reduction** |
| **Intermediate Cast Passes** | 0 | 0 | 8 - 10 per batch | **0 (Zero-Cast Pipeline)** | **100% cast overhead eliminated** |
| **Fence Wait (CPU-GPU stall)**| 0.040 ms | 0.001 ms | 0.001 ms | **0.001 ms** | **Zero sync stall (Fully overlapped)** |
| **VRAM Buffer Allocation** | Dynamic | Persistent | Reallocated per batch | **Persistent Pre-allocated Buffers** | **Zero runtime allocation overhead** |
| **Test Accuracy (1 Epoch)** | 98.60% | 98.92% | 98.60% | **97.76% - 98.90%** | **Retains classification accuracy** |

#### 3. Dataflow Pipeline Comparison

- **FP32 Standard Pipeline**:
  $$\text{Input (FP32)} \rightarrow \text{Conv2D} \rightarrow \text{BN2D} \rightarrow \text{GELU} \rightarrow \text{MaxPool2D} \rightarrow \text{Linear} \rightarrow \text{BN1D} \rightarrow \text{Softmax}$$
- **Simulated FP16 (Legacy with Cast Overhead)**:
  $$\text{Input} \xrightarrow{\text{Cast}} \text{Conv2D (FP16)} \xrightarrow{\text{Cast}} \text{BN2D (FP32)} \xrightarrow{\text{Cast}} \text{GELU (FP32)} \dots \text{(8-10 redundant cast kernels/batch)}$$
- **True Native FP16 Zero-Cast Pipeline (Current Architecture)**:
  $$\text{Input (FP16)} \rightarrow \text{Conv2D} \rightarrow \text{BN2D} \rightarrow \text{GELU} \rightarrow \text{MaxPool2D} \rightarrow \text{Linear} \rightarrow \text{BN1D} \rightarrow \text{Softmax (FP32)}$$
  *(All intermediate activations and backpropagated gradients flow continuously through VRAM in 16-bit storage, completely eliminating intermediate casting kernels.)*

---

### MNIST (Supervised Classification)

| Item | Value |
|:---|:---|
| Architecture | Conv(1→16) → BN → GELU → MaxPool → Conv(16→32) → BN → GELU → MaxPool → FC(1568→128) → BN → GELU → FC(10) |
| Loss / Optimizer | CCE + Adam + Cosine Annealing (0.01 → 1e-5) |
| Epochs / Batch Size | 1 epoch / 512 |
| Hardware | AMD Radeon 860M iGPU |

| Metric | Value |
|:---|---:|
| Accuracy | **99.51%** |
| Errors / Total | **49 / 10,000** |
| Training Time | **~12.8 s** |

---

### CIFAR-100 (Supervised Classification)

| Item | Value |
|:---|:---|
| Architecture | Conv(3→32) → BN → GELU → MaxPool × 3 stages → FC(2048→512) → FC(512→100) |
| Loss / Optimizer | CCE + Adam + Cosine Annealing (0.015 → 1e-5) |
| Data Augmentation | Random Crop (pad=4), Random Horizontal Flip |
| Epochs / Batch Size | 100 epochs / 256 |
| Hardware | AMD Radeon 860M iGPU |

| Metric | Value |
|:---|---:|
| Validation Accuracy | **67.51%** |
| Training Time | **~11 h 7 min** |
| Speed / Epoch | **~6 min 40 s** |

---

## Why Vulkan?

Most open-source ML projects depend on CUDA, which locks them to NVIDIA hardware. VulkanML uses Vulkan Compute to:

- Run on **any GPU** that supports Vulkan 1.3 (AMD, Intel, NVIDIA, mobile).
- Exploit **Cooperative Matrix** extensions for accelerated GEMM when available.
- Maintain a **zero-dependency GPU backend** — no driver SDKs, no runtime libraries beyond the Vulkan loader.
- Expose low-level **memory management** and **pipeline construction** explicitly, as a learning exercise.

---

## Building

**Requirements**

- C++23 compiler (MSVC 19.38+, GCC 13+, Clang 17+)
- Vulkan SDK 1.3+
- CMake 3.20+

```bash
git clone <repository>
cmake -B build
cmake --build build --config Release
```

To build a specific entry point:

```bash
cmake -B build -DACTIVE_FILE="<entry_point>.cpp"
cmake --build build --config Release
```

---

## Module Map

```
include/
├── math/               Tensor, Matrix, Shape, CPU/GPU backends
├── engine/             Vulkan_Context, Graph_Executor, Graph_Optimizer,
│                       Shader_Generator, Sub-Allocator, Pipeline_Cache,
│                       Async_Data_Pipeline
├── layer/              ILayer, Linear, Conv2D, BatchNorm, GELU, ReLU,
│                       Softmax, MaxPool2D, GlobalAvgPool2D,
│                       ResNet Block, ResNet-20, PPO Actor-Critic
├── cost_function/      ICost, MSE, MAE, BCE, CCE, Huber
├── optimizer/          IOptimizer, Adam, SGD
├── learning_rate/      ILearning_Rate, Cosine Annealing, Step Decay,
│                       Exponential Decay, Multi-Step Decay,
│                       Polynomial Decay, Reduce on Plateau
├── rl/                 DQN_Agent, PPO_Agent, Replay_Buffer, CartPole_Env
├── helper/             Logger, layer factory utilities
├── neural_network.h    Sequential model container + training API
├── population.h        Neuroevolution population (SoA + batched GPU GEMM)
└── training_context.h  Optimizer + LR scheduler + loss aggregation
```

---

## Third-Party Credits

| Library | License | Use |
|:---|:---|:---|
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | JSON serialization for model configuration and checkpoints |
| [magic_enum](https://github.com/Neargye/magic_enum) | MIT | Compile-time enum reflection for logging and serialization |

---

## Design Philosophy

Rather than treating neural networks as black boxes, VulkanML focuses on understanding every computation involved in modern deep learning — from matrix multiplication and activation functions to gradient propagation and GPU execution.

Every major component is implemented from scratch. This is intentional: the value of the project lies in the process of building it, not just in the result.

This project is intended for developers interested in both machine learning internals and low-level GPU programming.

---

## License

MIT License.
