#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cost_function/cce_cost.h"
#include "cost_function/icost_function.h"
#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "helper/layer.h"
#include "helper/logger.h"
#include "layer/ilayer.h"
#include "layer/linear_layer.h"
#include "layer/maxpool2d_layer.h"
#include "layer/relu.h"
#include "layer/softmax.h"
#include "learning_rate/cosine_annealing.h"
#include "math/matrix.h"
#include "neural_network.h"
#include "optimizer/adam_optimizer.h"

constexpr std::size_t INPUT_DIMENSION = 784;
constexpr std::size_t OUTPUT_DIMENSION = 10;
constexpr std::size_t BATCH_SIZE = 64;
constexpr std::size_t TOTAL_EPOCHS = 1;

std::uint32_t swapByteOrder(std::uint32_t _value)
{
    return (_value << 24) | ((_value << 8) & 0x00FF0000) | ((_value >> 8) & 0x0000FF00) | (_value >> 24);
}

void augmentMnistImage(const float *_source_pointer, float *_destination_pointer, std::mt19937 &_random_engine)
{
    std::uniform_int_distribution<int> crop_distribution(0, 4);

    int crop_x = crop_distribution(_random_engine);
    int crop_y = crop_distribution(_random_engine);

    for (int y = 0; y < 28; ++y)
    {
        int original_y = y + crop_y - 2;
        for (int x = 0; x < 28; ++x)
        {
            int original_x = x + crop_x - 2;
            std::size_t destination_index = y * 28 + x;

            if (original_y >= 0 && original_y < 28 && original_x >= 0 && original_x < 28)
            {
                std::size_t source_index = original_y * 28 + original_x;
                _destination_pointer[destination_index] = _source_pointer[source_index];
            }
            else
            {
                _destination_pointer[destination_index] = 0.0f;
            }
        }
    }
}

void loadMnistImages(const std::string &_file_path, std::vector<float> &_images_data, std::uint32_t &_images_count)
{
    std::ifstream input_file_stream(_file_path, std::ios::binary);
    if (!input_file_stream.is_open())
    {
        Logger::logMessage(std::format("loadMnistImages: Cannot open MNIST images file: {}", _file_path),
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
        throw std::runtime_error("Cannot open MNIST images file");
    }

    std::uint32_t magic_number = 0;
    std::uint32_t rows_count = 0;
    std::uint32_t columns_count = 0;

    input_file_stream.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    input_file_stream.read(reinterpret_cast<char *>(&_images_count), sizeof(_images_count));
    input_file_stream.read(reinterpret_cast<char *>(&rows_count), sizeof(rows_count));
    input_file_stream.read(reinterpret_cast<char *>(&columns_count), sizeof(columns_count));

    magic_number = swapByteOrder(magic_number);
    if (magic_number != 2051)
    {
        Logger::logMessage("loadMnistImages: Invalid MNIST image magic number",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
        throw std::runtime_error("Invalid MNIST image magic number");
    }

    _images_count = swapByteOrder(_images_count);
    rows_count = swapByteOrder(rows_count);
    columns_count = swapByteOrder(columns_count);

    std::size_t total_pixels = static_cast<std::size_t>(_images_count) * rows_count * columns_count;
    std::vector<std::uint8_t> raw_pixels(total_pixels);
    input_file_stream.read(reinterpret_cast<char *>(raw_pixels.data()), total_pixels);

    _images_data.resize(total_pixels);
    for (std::size_t i = 0; i < total_pixels; ++i)
    {
        _images_data[i] = static_cast<float>(raw_pixels[i]) / 255.0f;
    }
}

void loadMnistLabels(const std::string &_file_path, std::vector<float> &_labels_data, std::uint32_t _images_count)
{
    std::ifstream input_file_stream(_file_path, std::ios::binary);
    if (!input_file_stream.is_open())
    {
        Logger::logMessage(std::format("loadMnistLabels: Cannot open MNIST labels file: {}", _file_path),
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
        throw std::runtime_error("Cannot open MNIST labels file");
    }

    std::uint32_t magic_number = 0;
    std::uint32_t items_count = 0;

    input_file_stream.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    input_file_stream.read(reinterpret_cast<char *>(&items_count), sizeof(items_count));

    magic_number = swapByteOrder(magic_number);
    if (magic_number != 2049)
    {
        Logger::logMessage("loadMnistLabels: Invalid MNIST label magic number",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
        throw std::runtime_error("Invalid MNIST label magic number");
    }

    items_count = swapByteOrder(items_count);
    if (items_count < _images_count)
    {
        Logger::logMessage("loadMnistLabels: Label count mismatch",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DATA_PIPELINE);
        throw std::runtime_error("Label count mismatch");
    }

    std::vector<std::uint8_t> raw_labels(items_count);
    input_file_stream.read(reinterpret_cast<char *>(raw_labels.data()), items_count);

    _labels_data.assign(static_cast<std::size_t>(_images_count) * OUTPUT_DIMENSION, 0.0f);
    for (std::size_t i = 0; i < _images_count; ++i)
    {
        std::uint8_t label_value = raw_labels[i];
        if (label_value < OUTPUT_DIMENSION)
        {
            _labels_data[i * OUTPUT_DIMENSION + label_value] = 1.0f;
        }
    }
}

class Mnist_Data_Pipeline : public Async_Data_Pipeline
{
private:
    const std::vector<float> &images_data;
    const std::vector<float> &labels_data;
    std::uint32_t images_count = 0;
    std::size_t batch_size = 0;
    std::mt19937 random_engine{std::random_device{}()};
    bool is_augmentation_enabled = false;

    void prepareBatchHost(std::size_t _batch_step, std::vector<float> &_output_inputs, std::vector<float> &_output_targets) override
    {
        std::size_t batches_count = images_count / batch_size;
        std::size_t batch_index = _batch_step % batches_count;

        std::size_t offset_x = batch_index * batch_size * INPUT_DIMENSION;
        std::size_t offset_y = batch_index * batch_size * OUTPUT_DIMENSION;

        if (_output_inputs.size() != batch_size * INPUT_DIMENSION)
        {
            _output_inputs.resize(batch_size * INPUT_DIMENSION);
        }
        if (_output_targets.size() != batch_size * OUTPUT_DIMENSION)
        {
            _output_targets.resize(batch_size * OUTPUT_DIMENSION);
        }

        if (is_augmentation_enabled)
        {
            for (std::size_t i = 0; i < batch_size; ++i)
            {
                augmentMnistImage(images_data.data() + offset_x + i * INPUT_DIMENSION,
                                  _output_inputs.data() + i * INPUT_DIMENSION,
                                  random_engine);
            }
        }
        else
        {
            std::copy(images_data.begin() + offset_x,
                      images_data.begin() + offset_x + (batch_size * INPUT_DIMENSION),
                      _output_inputs.begin());
        }

        std::copy(labels_data.begin() + offset_y,
                  labels_data.begin() + offset_y + (batch_size * OUTPUT_DIMENSION),
                  _output_targets.begin());
    }

public:
    Mnist_Data_Pipeline(const std::vector<float> &_images_data,
                        const std::vector<float> &_labels_data,
                        std::uint32_t _images_count,
                        std::size_t _batch_size,
                        VkDevice _device = VK_NULL_HANDLE,
                        bool _is_augmentation_enabled = false)
        : Async_Data_Pipeline(_device),
          images_data(_images_data),
          labels_data(_labels_data),
          images_count(_images_count),
          batch_size(_batch_size),
          is_augmentation_enabled(_is_augmentation_enabled)
    {
    }

    [[nodiscard]] std::size_t getBatchSize() const override
    {
        return batch_size;
    }
};

double runBenchmark(Execution_Target _execution_target,
                    const std::vector<float> &_images_data,
                    const std::vector<float> &_labels_data,
                    std::uint32_t _images_count,
                    Neural_Network &_neural_network)
{
    std::size_t steps_per_epoch = _images_count / BATCH_SIZE;
    _neural_network.getLearningRate().setMaxEpoch(static_cast<int>(TOTAL_EPOCHS));

    Mnist_Data_Pipeline data_pipeline(
        _images_data,
        _labels_data,
        _images_count,
        BATCH_SIZE,
        Execution_Engine::getInstance().getContext().getDevice(),
        true);

    data_pipeline.initializeBuffers(BATCH_SIZE, INPUT_DIMENSION, OUTPUT_DIMENSION, _execution_target);

    auto start_time = std::chrono::high_resolution_clock::now();

    _neural_network.fit(data_pipeline, TOTAL_EPOCHS, steps_per_epoch, BATCH_SIZE, INPUT_DIMENSION, OUTPUT_DIMENSION);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_duration = end_time - start_time;

    return elapsed_duration.count();
}

void evaluateModel(Neural_Network &_neural_network,
                   const std::vector<float> &_test_images_data,
                   const std::vector<float> &_test_labels_data,
                   std::uint32_t _test_images_count,
                   Execution_Target _execution_target)
{
    Logger::logMessage("Evaluating model on test dataset...",
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::TRAINING);
    _neural_network.setTrainingMode(false);

    std::size_t correct_count = 0;
    std::array<std::array<std::size_t, OUTPUT_DIMENSION>, OUTPUT_DIMENSION> confusion_matrix{};

    std::size_t test_batch_size = BATCH_SIZE;
    std::size_t batches_count = (_test_images_count + test_batch_size - 1) / test_batch_size;

    for (std::size_t b = 0; b < batches_count; ++b)
    {
        std::size_t current_batch_size = std::min(test_batch_size, static_cast<std::size_t>(_test_images_count) - b * test_batch_size);

        std::vector<float> host_batch_inputs(current_batch_size * INPUT_DIMENSION);
        std::vector<float> host_batch_targets(current_batch_size * OUTPUT_DIMENSION);

        std::copy(_test_images_data.begin() + b * test_batch_size * INPUT_DIMENSION,
                  _test_images_data.begin() + (b * test_batch_size + current_batch_size) * INPUT_DIMENSION,
                  host_batch_inputs.begin());

        std::copy(_test_labels_data.begin() + b * test_batch_size * OUTPUT_DIMENSION,
                  _test_labels_data.begin() + (b * test_batch_size + current_batch_size) * OUTPUT_DIMENSION,
                  host_batch_targets.begin());

        Matrix input_matrix(current_batch_size, INPUT_DIMENSION, host_batch_inputs, _execution_target);
        Matrix target_matrix(current_batch_size, OUTPUT_DIMENSION, host_batch_targets, _execution_target);

        Matrix prediction_matrix = _neural_network.forward(input_matrix);

        if (_execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }

        std::vector<float> prediction_data = prediction_matrix.getData();

        for (std::size_t i = 0; i < current_batch_size; ++i)
        {
            std::size_t predicted_label = 0;
            float max_prediction_value = prediction_data[i * OUTPUT_DIMENSION];
            for (std::size_t c = 1; c < OUTPUT_DIMENSION; ++c)
            {
                float value = prediction_data[i * OUTPUT_DIMENSION + c];
                if (value > max_prediction_value)
                {
                    max_prediction_value = value;
                    predicted_label = c;
                }
            }

            std::size_t ground_truth_label = 0;
            float max_ground_truth_value = host_batch_targets[i * OUTPUT_DIMENSION];
            for (std::size_t c = 1; c < OUTPUT_DIMENSION; ++c)
            {
                float value = host_batch_targets[i * OUTPUT_DIMENSION + c];
                if (value > max_ground_truth_value)
                {
                    max_ground_truth_value = value;
                    ground_truth_label = c;
                }
            }

            if (predicted_label == ground_truth_label)
            {
                correct_count++;
            }

            confusion_matrix[ground_truth_label][predicted_label]++;
        }
    }

    std::size_t wrong_count = _test_images_count - correct_count;
    double accuracy_percentage = (static_cast<double>(correct_count) / _test_images_count) * 100.0;
    double error_rate_percentage = (static_cast<double>(wrong_count) / _test_images_count) * 100.0;

    std::string evaluation_report = "\n========== Evaluation ==========\n";
    evaluation_report += std::format("Samples   : {}\n", _test_images_count);
    evaluation_report += std::format("Correct   : {}\n", correct_count);
    evaluation_report += std::format("Wrong     : {}\n", wrong_count);
    evaluation_report += std::format("Accuracy  : {:.2f}%\n", accuracy_percentage);
    evaluation_report += std::format("Error     : {:.2f}%\n\n", error_rate_percentage);

    evaluation_report += "Confusion Matrix:\n[";
    for (std::size_t r = 0; r < OUTPUT_DIMENSION; ++r)
    {
        if (r > 0)
        {
            evaluation_report += " ";
        }
        evaluation_report += "[";
        for (std::size_t c = 0; c < OUTPUT_DIMENSION; ++c)
        {
            evaluation_report += std::format("{:4d}{}", confusion_matrix[r][c], (c == OUTPUT_DIMENSION - 1 ? "" : " "));
        }
        evaluation_report += std::format("]{}", (r == OUTPUT_DIMENSION - 1 ? "]" : "\n"));
    }

    std::cout << evaluation_report << "\n";
    Logger::logMessage(evaluation_report, Log_Level::LOG_INFO, true, 0, Log_Feature::TRAINING);

    _neural_network.setTrainingMode(true);
}

int main()
{
    Logger::setFileLogging(false);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);
    Logger::setConsoleOutput(true);

    std::vector<float> train_images_data;
    std::vector<float> train_labels_data;
    std::uint32_t train_images_count = 0;

    loadMnistImages("data/train-images.idx3-ubyte", train_images_data, train_images_count);
    loadMnistLabels("data/train-labels.idx1-ubyte", train_labels_data, train_images_count);

    std::vector<float> test_images_data;
    std::vector<float> test_labels_data;
    std::uint32_t test_images_count = 0;

    loadMnistImages("data/t10k-images.idx3-ubyte", test_images_data, test_images_count);
    loadMnistLabels("data/t10k-labels.idx1-ubyte", test_labels_data, test_images_count);

    Execution_Engine::getInstance()
        .getPipelineCacheManager()
        .initializePipelineCache("temp/pipeline_cache.bin");

    Execution_Target execution_target = Execution_Target::VULKAN_GPU;
    Neural_Network neural_network(execution_target);
    neural_network.setTrainingMode(true);

    neural_network.setLearningRate<Cosine_Annealing>(0.001f, 1e-5f, static_cast<int>(TOTAL_EPOCHS));
    neural_network.setOptimizer<Adam_Optimizer>(neural_network.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f);
    neural_network.setCostFunction<Cce_Cost>();

    neural_network.addLayer<Conv2d_Layer>(28, 28, 1, 16, 3, 1, 1);
    neural_network.addLayer<Batch_Norm_2d_Layer>(28, 28, 16);
    neural_network.addLayer<Gelu_Layer>();

    neural_network.addLayer<Max_Pool_2d_Layer>(28, 28, 16, 2, 2, 0);

    neural_network.addLayer<Conv2d_Layer>(14, 14, 16, 32, 3, 1, 1);
    neural_network.addLayer<Batch_Norm_2d_Layer>(14, 14, 32);
    neural_network.addLayer<Gelu_Layer>();

    neural_network.addLayer<Linear_Layer>(14 * 14 * 32, 128);
    neural_network.addLayer<Batch_Norm_Layer>(128);
    neural_network.addLayer<Gelu_Layer>();

    neural_network.addLayer<Linear_Layer>(128, OUTPUT_DIMENSION);
    neural_network.addLayer<Softmax_Layer>(true);

    neural_network.compileAndWarmup(BATCH_SIZE, INPUT_DIMENSION, OUTPUT_DIMENSION);
    Logger::logMessage("Starting training benchmark with Data Augmentation...",
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::TRAINING);
    double elapsed_duration_ms = runBenchmark(execution_target, train_images_data, train_labels_data, train_images_count, neural_network);
    std::string mes = std::format("Training completed in {:.4f} s", elapsed_duration_ms / 1000.0);
    std::cout << mes << "\n";
    Logger::logMessage(mes,
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::TRAINING);

    neural_network.setTrainingMode(false);
    evaluateModel(neural_network, test_images_data, test_labels_data, test_images_count, execution_target);

    std::filesystem::create_directories("output/mnist");

    neural_network.saveInference("output/mnist/model.bin");
    Logger::logMessage("Inference model exported to model.bin",
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    return 0;
}