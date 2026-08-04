#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <format>

#include "optimizer/adam_optimizer.h"
#include "learning_rate/cosine_annealing.h"
#include "layer/batch_norm_layer.h"
#include "neural_network.h"
#include "layer/linear_layer.h"
#include "layer/relu.h"
#include "layer/gelu.h"
#include "cost_function/cce_cost.h"
#include "cost_function/icost_function.h"
#include "math/matrix.h"
#include "helper/logger.h"
#include "layer/softmax.h"
#include "layer/conv2d_layer.h"
#include "layer/maxpool2d_layer.h"
#include "layer/globalavgpool2d_layer.h"

constexpr std::size_t INPUT_DIM = 784;
constexpr std::size_t OUTPUT_DIM = 10;
constexpr std::size_t BATCH_SIZE = 512;
constexpr std::size_t EPOCHS = 2;

uint32_t swapEndian(uint32_t val)
{
    return (val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24);
}

void augmentMnistImage(const float *src_ptr, float *dst_ptr, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> crop_dist(0, 4);

    int crop_x = crop_dist(rng);
    int crop_y = crop_dist(rng);

    for (int y = 0; y < 28; ++y)
    {
        int orig_y = y + crop_y - 2;
        for (int x = 0; x < 28; ++x)
        {
            int orig_x = x + crop_x - 2;
            std::size_t dst_idx = y * 28 + x;

            if (orig_y >= 0 && orig_y < 28 && orig_x >= 0 && orig_x < 28)
            {
                std::size_t src_idx = orig_y * 28 + orig_x;
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
            else
            {
                dst_ptr[dst_idx] = 0.0f;
            }
        }
    }
}

void loadMnistImages(const std::string &file_path, std::vector<float> &images_data, uint32_t &num_images)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        Logger::logMessage("loadMnistImages: Cannot open MNIST images file: " + file_path, LOG_ERROR);
        throw std::runtime_error("Cannot open MNIST images file");
    }

    uint32_t magic_number = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));

    magic_number = swapEndian(magic_number);
    if (magic_number != 2051)
    {
        Logger::logMessage("loadMnistImages: Invalid MNIST image magic number", LOG_ERROR);
        throw std::runtime_error("Invalid MNIST image magic number");
    }

    num_images = swapEndian(num_images);
    num_rows = swapEndian(num_rows);
    num_cols = swapEndian(num_cols);

    std::size_t total_pixels = num_images * num_rows * num_cols;
    std::vector<uint8_t> raw_pixels(total_pixels);
    file.read(reinterpret_cast<char *>(raw_pixels.data()), total_pixels);

    images_data.resize(total_pixels);
    for (std::size_t i = 0; i < total_pixels; ++i)
    {
        images_data[i] = static_cast<float>(raw_pixels[i]) / 255.0f;
    }
}

void loadMnistLabels(const std::string &file_path, std::vector<float> &labels_data, uint32_t num_images)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        Logger::logMessage("loadMnistLabels: Cannot open MNIST labels file: " + file_path, LOG_ERROR);
        throw std::runtime_error("Cannot open MNIST labels file");
    }

    uint32_t magic_number = 0;
    uint32_t num_items = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));

    magic_number = swapEndian(magic_number);
    if (magic_number != 2049)
    {
        Logger::logMessage("loadMnistLabels: Invalid MNIST label magic number", LOG_ERROR);
        throw std::runtime_error("Invalid MNIST label magic number");
    }

    num_items = swapEndian(num_items);

    std::vector<uint8_t> raw_labels(num_items);
    file.read(reinterpret_cast<char *>(raw_labels.data()), num_items);

    labels_data.assign(num_images * OUTPUT_DIM, 0.0f);
    for (std::size_t i = 0; i < num_images; ++i)
    {
        uint8_t label = raw_labels[i];
        labels_data[i * OUTPUT_DIM + label] = 1.0f;
    }
}

double runBenchmark(Execution_Target target,
                    const std::vector<float> &images_data,
                    const std::vector<float> &labels_data,
                    uint32_t num_images,
                    Neural_Network &nn,
                    const ICost_Function &cost_function,
                    ILearning_Rate &learning_rate,
                    IOptimizer &optimizer)
{
    std::size_t num_batches = num_images / BATCH_SIZE;

    Matrix input_mat(BATCH_SIZE, INPUT_DIM, target);
    Matrix target_mat(BATCH_SIZE, OUTPUT_DIM, target);

    std::vector<float> batch_x(BATCH_SIZE * INPUT_DIM);
    std::vector<float> batch_y(BATCH_SIZE * OUTPUT_DIM);

    std::random_device rd;
    std::mt19937 rng(rd());

    auto start_time = std::chrono::high_resolution_clock::now();

    for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch)
    {
        float current_rate = learning_rate.getCurrentRate();
        Logger::logMessage(std::format("Epoch {}: Current LR = {:.6f}", epoch, current_rate), LOG_INFO, true);

        for (std::size_t b = 0; b < num_batches; ++b)
        {
            std::size_t offset_x = b * BATCH_SIZE * INPUT_DIM;
            std::size_t offset_y = b * BATCH_SIZE * OUTPUT_DIM;

            for (std::size_t i = 0; i < BATCH_SIZE; ++i)
            {
                augmentMnistImage(images_data.data() + offset_x + i * INPUT_DIM,
                                  batch_x.data() + i * INPUT_DIM,
                                  rng);
            }

            std::copy(labels_data.begin() + offset_y,
                      labels_data.begin() + offset_y + (BATCH_SIZE * OUTPUT_DIM),
                      batch_y.begin());

            input_mat.uploadData(batch_x);
            target_mat.uploadData(batch_y);
            nn.trainStep(input_mat, target_mat, cost_function, current_rate, optimizer);
        }
        learning_rate.step();

        nn.saveTrainingCheckpoint(std::format("output/mnist/checkpoint_mnist_epoch_{}.nnck", epoch), epoch, cost_function, learning_rate, optimizer);
        Logger::logMessage(std::format("Checkpoint saved for epoch {}", epoch + 1), LOG_INFO, true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    return duration.count();
}

int main()
{
    Cosine_Annealing learning_rate(0.001f, 0.0f, static_cast<int>(EPOCHS));
    Adam_Optimizer optimizer(learning_rate, 0.9f, 0.999f, 1e-8f, 1.0f);
    CCE_Cost cost_function;

    std::vector<float> images_data;
    std::vector<float> labels_data;
    uint32_t num_images = 0;
    loadMnistImages("data/train-images.idx3-ubyte", images_data, num_images);
    loadMnistLabels("data/train-labels.idx1-ubyte", labels_data, num_images);

    Execution_Target target = Execution_Target::VULKAN_GPU;
    Neural_Network nn(target);

    nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 1, 32, 3, 1, 1));
    nn.addLayer(std::make_unique<Batch_Norm_Layer>(28 * 28 * 32, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 32, 32, 3, 1, 1));
    nn.addLayer(std::make_unique<Batch_Norm_Layer>(28 * 28 * 32, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<MaxPool2d_Layer>(28, 28, 32, 2, 2, 0));

    nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 32, 64, 3, 1, 1));
    nn.addLayer(std::make_unique<Batch_Norm_Layer>(14 * 14 * 64, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 64, 64, 3, 1, 1));
    nn.addLayer(std::make_unique<Batch_Norm_Layer>(14 * 14 * 64, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<MaxPool2d_Layer>(14, 14, 64, 2, 2, 0));

    nn.addLayer(std::make_unique<Linear_Layer>(3136, 128));
    nn.addLayer(std::make_unique<Batch_Norm_Layer>(128, 1e-5f, 0.1f));
    nn.addLayer(std::make_unique<GeLU>());

    nn.addLayer(std::make_unique<Linear_Layer>(128, 10));
    nn.addLayer(std::make_unique<Softmax>(true));

    Logger::logMessage("Starting training benchmark with Data Augmentation...", LOG_INFO, true);
    double duration = runBenchmark(target, images_data, labels_data, num_images, nn, cost_function, learning_rate, optimizer);
    Logger::logMessage("Training completed in " + std::to_string(duration / 1000.0) + " s", LOG_INFO, true);

    nn.saveInference("output/model.bin");
    Logger::logMessage("Inference model exported to model.bin", LOG_INFO, true);
}