# VulkanML

> A lightweight deep learning framework built from scratch in modern C++23 with Vulkan Compute acceleration.

![Language](https://img.shields.io/badge/C%2B%2B-23-blue.svg)
![API](https://img.shields.io/badge/Vulkan-1.3-red.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

VulkanML is an educational deep learning framework designed to explore how modern neural networks work internally without relying on existing machine learning libraries such as PyTorch or TensorFlow.

The project implements the complete training pipeline from scratch, including tensor operations, neural network layers, automatic forward/backward propagation, optimizers, serialization, and GPU acceleration using Vulkan Compute Shaders.

Unlike production frameworks that hide implementation details behind large abstraction layers, VulkanML focuses on clarity, extensibility, and understanding every stage of the computation.

## Features

| Category                    | Features                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core**                    | Sequential neural network, Forward propagation, Backpropagation, Topology-aware model serialization (NNA1 format)                                 |
| **Layers**                  | Dense (Fully Connected), Conv2D, MaxPool2D, Batch Normalization                                                                                   |
| **Activation Functions**    | ReLU, GELU, Softmax                                                                                                                               |
| **Loss Functions**          | MSE, CCE, MAE, BCE                                                                                                                                |
| **Optimizers & Schedulers** | Stochastic Gradient Descent (SGD), Learning Rate Schedulers (Cosine Annealing, Step Decay, Multi-Step, Exponential, Polynomial, ReduceOnPlateau), |
|                             | Adam Optimizer                                                                                                                                    |
| **Data Augmentation**       | CPU-side Data Pipelines (Random Crop with Padding, Random Horizontal Flip, Random Shift)                                                          |
| **Math Library**            | Custom Matrix implementation, Matrix multiplication, Broadcasting, Element-wise operations                                                        |
| **Execution Backends**      | CPU backend, Vulkan Compute backend, Unified execution interface                                                                                  |
| **GPU Computing**           | Vulkan Compute Shaders, GPU memory abstraction, Compute pipeline execution                                                                        |
| **Utilities**               | Logging system, Self-contained binary model format with topology auto-restoration                                                                 |


---

## Project Architecture

```
                             Neural Network
                                   │
        ┌─────────────┼────────────┴──────────┼─────────────┐
        │             |                       |             │
     Layers     Loss Functions            Optimizer    LR Schedulers
        │                                                   │
        └───────────────────────────────────────────────────┘
                                   │
                               Matrix API
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
             CPU Backend                     Vulkan Backend
                  │                                 │
             Native C++                      Compute Shaders
```

---

## Example

```cpp
Execution_Target exec_target = Execution_Target::VULKAN_GPU;
Neural_Network model(exec_target);

model.setLearningRate(std::make_unique<Cosine_Annealing>(0.015f, 1e-5f, 30));
model.setOptimizer(std::make_unique<Adam_Optimizer>(model.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f));
model.setCostFunction(std::make_unique<CCE_Cost>());

model.addLayer(std::make_unique<Conv2d_Layer>(32, 32, 3, 32, 3, 1, 1));
model.addLayer(std::make_unique<Batch_Norm_Layer>(32 * 32 * 32, 1e-5f, 0.1f));
model.addLayer(std::make_unique<GeLU>());
model.addLayer(std::make_unique<MaxPool2d_Layer>(32, 32, 32, 2, 2, 0));

model.addLayer(std::make_unique<Linear_Layer>(2048, 512));
model.addLayer(std::make_unique<Batch_Norm_Layer>(512, 1e-5f, 0.1f));
model.addLayer(std::make_unique<GeLU>());
model.addLayer(std::make_unique<Linear_Layer>(512, 100));
model.addLayer(std::make_unique<Softmax>(true));

model.trainStep(input_mat, target_mat);
model.getLearningRate().step();

model.saveTrainingCheckpoint("output/checkpoint_epoch_0.nnck", 0);
model.saveInference("output/model.bin");
```

---

## Benchmarks

## MNIST

### Configuration

| Item                  | Value                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| Architecture          | Conv(1→16) → BN → GeLU → MaxPool → Conv(16→32) → BN → GeLU → FC(6272→128) → BN → GeLU → FC(10) |
| Loss                  | Categorical Cross Entropy                                                                      |
| Optimizer / Scheduler | Adam Optimizer + Cosine Annealing(0.01 -> 1e-5)                                                |
| Data Augmentation     | Random Shift + Padding                                                                         |
| Backend               | Vulkan Compute                                                                                 |
| Epochs / Batch Size   | 1 epochs / Batch size 512                                                                      |

### Results

| Metric        |                    Value |
| ------------- | -----------------------: |
| Accuracy      |               **99.51%** |
| Wrong / Total |          **49 / 10,000** |
| Training Time |              **~12.8 s** |
| Hardware      | **AMD Radeon 860M iGPU** |

---

## CIFAR-100

### Configuration

| Item                  | Value                                                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture          | Conv(3→32) → BN → GeLU → MaxPool → Conv(32→64) → BN → GeLU → MaxPool → Conv(64→128) → BN → GeLU → MaxPool → FC(2048→512) → FC(512→100) |
| Loss                  | Categorical Cross Entropy                                                                                                              |
| Optimizer / Scheduler | Adam + Cosine Annealing (0.015 → 1e-5)                                                                                                 |
| Data Augmentation     | Random Crop (Padding = 4), Random Horizontal Flip                                                                                      |
| Backend               | Vulkan Compute                                                                                                                         |
| Epochs / Batch size   | 100 epochs / Batch size 256                                                                                                            |

### Results

| Metric              |                    Value |
| ------------------- | -----------------------: |
| Validation Accuracy |               **67.51%** |
| Training Time       |         **≈ 11 h 7 min** |
| Speed / Epoch       |         **≈ 6 min 40 s** |
| Classes             |                  **100** |
| Hardware            | **AMD Radeon 860M iGPU** |


---

## Design Goals

The primary objective of this project is **education**, not competing with industrial deep learning frameworks.

The framework aims to demonstrate how:

* matrix operations are implemented,
* neural networks perform forward propagation,
* gradients are computed,
* optimizers update parameters,
* GPU compute pipelines execute deep learning workloads.

Every major component is implemented from scratch to maximize learning value.

---

## Future Work

Planned features include:

- Automatic differentiation (Autograd)
- Computational graph
- Dropout layer
- Vulkan kernel fusion
- FP16 support
- Descriptor pool optimization
- Memory pooling
---

## Why Vulkan?

Most educational machine learning projects rely on CUDA, which limits portability.

This project instead uses Vulkan Compute to

* support multiple GPU vendors,
* learn low-level GPU programming,
* understand compute shader optimization,
* explore GPU memory management,
* build a backend independent from proprietary APIs.

---

## Building

Requirements

* C++23 compiler
* Vulkan SDK
* CMake

```bash
git clone <repository>

mkdir build
cd build

cmake ..
cmake --build . --config Release
```

---

## Philosophy

Rather than treating neural networks as black boxes, VulkanML focuses on understanding every computation involved in modern deep learning—from matrix multiplication and activation functions to gradient propagation and GPU execution.

This project is intended for students, graphics programmers, and developers interested in both machine learning and GPU computing.

---

## License

MIT License.
