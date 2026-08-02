#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <string>
#include <memory>
#include <cmath>

#include "neural_network.h"
#include "layer.h"
#include "relu.h"
#include "gelu.h"
#include "cost_function.h"
#include "math/matrix.h"
#include "math/logger.h"
#include "softmax.h"
#include "conv2d_layer.h"
#include "maxpool2d_layer.h"
#include "globalavgpool2d_layer.h"

constexpr std::size_t INPUT_DIM = 784;
constexpr std::size_t HIDDEN_DIM_1 = 256;
constexpr std::size_t HIDDEN_DIM_2 = 128;
constexpr std::size_t OUTPUT_DIM = 10;
constexpr std::size_t BATCH_SIZE = 512;
constexpr std::size_t EPOCHS = 20;
constexpr float LEARNING_RATE = 0.01f;

uint32_t swapEndian(uint32_t val)
{
    return (val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24);
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
                    Neural_Network &nn, const ICostFunction &cost_function)
{
    std::size_t num_batches = num_images / BATCH_SIZE;

    Matrix input_mat(BATCH_SIZE, INPUT_DIM, target);
    Matrix target_mat(BATCH_SIZE, OUTPUT_DIM, target);

    std::vector<float> batch_x(BATCH_SIZE * INPUT_DIM);
    std::vector<float> batch_y(BATCH_SIZE * OUTPUT_DIM);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch)
    {
        for (std::size_t b = 0; b < num_batches; ++b)
        {
            std::size_t offset_x = b * BATCH_SIZE * INPUT_DIM;
            std::size_t offset_y = b * BATCH_SIZE * OUTPUT_DIM;

            std::copy(images_data.begin() + offset_x,
                      images_data.begin() + offset_x + (BATCH_SIZE * INPUT_DIM),
                      batch_x.begin());

            std::copy(labels_data.begin() + offset_y,
                      labels_data.begin() + offset_y + (BATCH_SIZE * OUTPUT_DIM),
                      batch_y.begin());

            input_mat.uploadData(batch_x);
            target_mat.uploadData(batch_y);

            nn.trainStep(input_mat, target_mat, cost_function, LEARNING_RATE, 1.0f);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    return duration.count();
}

int main()
{
    const float GAIN = 2.0f;
    std::vector<float> images_data;
    std::vector<float> labels_data;
    uint32_t num_images = 0;
    loadMnistImages("data/train-images.idx3-ubyte", images_data, num_images);
    loadMnistLabels("data/train-labels.idx1-ubyte", labels_data, num_images);

    Execution_Target target = Execution_Target::VULKAN_GPU;
    Neural_Network nn(target);
    nn.addLayer(std::make_unique<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1, target));
    nn.addLayer(std::make_unique<GeLU>(target));
    nn.addLayer(std::make_unique<MaxPool2d_Layer>(28, 28, 16, 2, 2, 0, target));

    nn.addLayer(std::make_unique<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1, target));
    nn.addLayer(std::make_unique<GeLU>(target));

    // Classifier (Flatten: 14 * 14 * 32 = 6272)
    nn.addLayer(std::make_unique<Layer>(6272, 128, target));
    nn.addLayer(std::make_unique<GeLU>(target));
    nn.addLayer(std::make_unique<Layer>(128, 10, target));
    nn.addLayer(std::make_unique<Softmax>(true, target));

    Logger::logMessage("Starting training benchmark...", LOG_INFO, true);
    double duration = runBenchmark(target, images_data, labels_data, num_images, nn, CCE_Cost());
    Logger::logMessage("Training completed in " + std::to_string(duration / 1000) + " s", LOG_INFO, true);

    nn.saveModel("output/model_weights.bin");
    Logger::logMessage("Model exported to model_weights.bin", LOG_INFO, true);
}