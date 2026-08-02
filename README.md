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

### Neural Network Components

* Fully-connected (Dense) layer
* GELU activation
* ReLU activation
* Softmax activation
* Mean Squared Error (MSE)
* Cross Entropy Loss
* SGD optimizer
* Forward propagation
* Backpropagation
* Model serialization/deserialization

### Execution Backends

* CPU backend
* Vulkan Compute backend
* Unified execution interface

### Math Library

* Custom Matrix implementation
* Matrix multiplication
* Broadcasting
* Element-wise operators
* Activation kernels
* GPU memory abstraction

### Utilities

* Logging system
* Binary model format
* MNIST dataset loader
* Evaluation utilities
* Confusion matrix generation

---

## Project Architecture

```
                 Neural Network
                        │
        ┌───────────────┴───────────────┐
        │                               │
     Layers                         Loss Functions
        │                               │
        └───────────────┬───────────────┘
                        │
                     Matrix API
                        │
        ┌───────────────┴───────────────┐
        │                               │
      CPU Backend                Vulkan Backend
        │                               │
   Native C++                  Compute Shaders
```

---

## Example

```cpp
Neural_Network model;

model.addLayer(std::make_unique<Linear>(784, 256));
model.addLayer(std::make_unique<GELU>());
model.addLayer(std::make_unique<Linear>(256, 10));
model.addLayer(std::make_unique<Softmax>(true));

model.train(images, labels);
```

---

## Current Performance

### MNIST

Configuration

* Architecture: 784 → Hidden → 10
* Activation: GELU
* Loss: Cross Entropy
* Optimizer: SGD
* Backend: Vulkan Compute
* Epochs: 10

Results

| Metric        |                Value |
| ------------- | -------------------: |
| Accuracy      |                 ~98% |
| Training Time |          ~20 seconds |
| Hardware      | AMD Radeon 860M iGPU |

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

Planned features include

* Tensor abstraction
* Automatic differentiation (Autograd)
* Computational graph
* Mini-batch training
* Adam optimizer
* Learning rate schedulers
* Batch Normalization
* Dropout
* Convolution layers
* Vulkan kernel fusion
* FP16 support
* Descriptor pool optimization
* Memory pooling
* Compute shader tiling
* Shared memory optimizations

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

## Repository Structure

```
include/
    layers/
    math/
    optimizer/
    loss/

src/
    layers/
    math/
    optimizer/
    vulkan/

shader/
    matmul.comp
    activation.comp

examples/

models/

dataset/
```

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
