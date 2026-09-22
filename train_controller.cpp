#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if __has_include("helper/vulkan_ml.h")
#include "helper/vulkan_ml.h"
#elif __has_include("vulkan_ml.h")
#include "vulkan_ml.h"
#endif

struct Audio_Sample
{
    std::string file_path;
    size_t label = 0;
};

std::vector<float> loadFullWav(const std::string& file_path)
{
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream.is_open())
    {
        return {};
    }

    char riff_tag[4];
    stream.read(riff_tag, 4);
    if (std::string_view(riff_tag, 4) != "RIFF")
    {
        return {};
    }

    uint32_t riff_size = 0;
    stream.read(reinterpret_cast<char*>(&riff_size), sizeof(riff_size));

    char wave_tag[4];
    stream.read(wave_tag, 4);
    if (std::string_view(wave_tag, 4) != "WAVE")
    {
        return {};
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;
    std::vector<std::int16_t> raw_pcm;

    while (stream)
    {
        char chunk_id[4];
        stream.read(chunk_id, 4);
        if (!stream)
        {
            break;
        }

        uint32_t chunk_size = 0;
        stream.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
        if (!stream)
        {
            break;
        }

        std::string_view id_view(chunk_id, 4);
        if (id_view == "fmt ")
        {
            stream.read(reinterpret_cast<char*>(&audio_format), sizeof(audio_format));
            stream.read(reinterpret_cast<char*>(&num_channels), sizeof(num_channels));
            stream.read(reinterpret_cast<char*>(&sample_rate), sizeof(sample_rate));
            stream.read(reinterpret_cast<char*>(&byte_rate), sizeof(byte_rate));
            stream.read(reinterpret_cast<char*>(&block_align), sizeof(block_align));
            stream.read(reinterpret_cast<char*>(&bits_per_sample), sizeof(bits_per_sample));

            if (chunk_size > 16)
            {
                stream.seekg(chunk_size - 16, std::ios::cur);
            }
        }
        else if (id_view == "data")
        {
            size_t sample_count = chunk_size / sizeof(std::int16_t);
            raw_pcm.resize(sample_count);
            stream.read(reinterpret_cast<char*>(raw_pcm.data()), chunk_size);
            break;
        }
        else
        {
            stream.seekg(chunk_size, std::ios::cur);
        }
    }

    if (raw_pcm.empty() || bits_per_sample != 16 || audio_format != 1)
    {
        return {};
    }

    std::vector<float> mono_samples;
    if (num_channels == 1)
    {
        mono_samples.resize(raw_pcm.size());
        for (size_t i = 0; i < raw_pcm.size(); ++i)
        {
            mono_samples[i] = static_cast<float>(raw_pcm[i]) / 32768.0f;
        }
    }
    else if (num_channels >= 2)
    {
        size_t frames = raw_pcm.size() / num_channels;
        mono_samples.resize(frames);
        for (size_t i = 0; i < frames; ++i)
        {
            float sum = 0.0f;
            for (size_t c = 0; c < num_channels; ++c)
            {
                sum += static_cast<float>(raw_pcm[i * num_channels + c]) / 32768.0f;
            }
            mono_samples[i] = sum / static_cast<float>(num_channels);
        }
    }

    return mono_samples;
}

std::vector<float> loadWav(const std::string& file_path)
{
    std::vector<float> mono_samples = loadFullWav(file_path);
    if (mono_samples.empty())
    {
        return std::vector<float>(16000, 0.0f);
    }

    std::vector<float> target_samples(16000, 0.0f);
    if (mono_samples.size() == 16000)
    {
        target_samples = std::move(mono_samples);
    }
    else if (mono_samples.size() < 16000)
    {
        size_t pad_left = (16000 - mono_samples.size()) / 2;
        std::copy(mono_samples.begin(), mono_samples.end(), target_samples.begin() + pad_left);
    }
    else
    {
        size_t offset = (mono_samples.size() - 16000) / 2;
        std::copy(mono_samples.begin() + offset, mono_samples.begin() + offset + 16000, target_samples.begin());
    }

    return target_samples;
}

class Audio_Feature_Extractor
{
private:
    static constexpr size_t FFT_SIZE = 512;
    static constexpr size_t HOP_SIZE = 499;
    static constexpr size_t TIME_FRAMES = 32;
    static constexpr size_t MEL_BINS = 32;
    static constexpr size_t SPECTRUM_BINS = FFT_SIZE / 2 + 1;

    std::array<float, FFT_SIZE> hann_window{};
    std::vector<std::vector<float>> mel_filterbank;

    static float hzToMel(float freq)
    {
        return 2595.0f * std::log10(1.0f + freq / 700.0f);
    }

    static float melToHz(float mel)
    {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    void initializeHannWindow()
    {
        for (size_t i = 0; i < FFT_SIZE; ++i)
        {
            float ratio = static_cast<float>(i) / static_cast<float>(FFT_SIZE - 1);
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> *ratio));
        }
    }

    void initializeMelFilterbank()
    {
        mel_filterbank.assign(MEL_BINS, std::vector<float>(SPECTRUM_BINS, 0.0f));

        float min_mel = hzToMel(20.0f);
        float max_mel = hzToMel(8000.0f);
        float mel_step = (max_mel - min_mel) / static_cast<float>(MEL_BINS + 1);

        std::vector<float> bin_positions(MEL_BINS + 2);
        for (size_t i = 0; i < MEL_BINS + 2; ++i)
        {
            float current_mel = min_mel + static_cast<float>(i) * mel_step;
            float current_hz = melToHz(current_mel);
            bin_positions[i] = std::floor(static_cast<float>(FFT_SIZE) * current_hz / 16000.0f);
        }

        for (size_t m = 0; m < MEL_BINS; ++m)
        {
            float left_bin = bin_positions[m];
            float center_bin = bin_positions[m + 1];
            float right_bin = bin_positions[m + 2];

            for (size_t k = 0; k < SPECTRUM_BINS; ++k)
            {
                float freq_bin = static_cast<float>(k);
                if (freq_bin >= left_bin && freq_bin <= center_bin && center_bin > left_bin)
                {
                    mel_filterbank[m][k] = (freq_bin - left_bin) / (center_bin - left_bin);
                }
                else if (freq_bin >= center_bin && freq_bin <= right_bin && right_bin > center_bin)
                {
                    mel_filterbank[m][k] = (right_bin - freq_bin) / (right_bin - center_bin);
                }
            }
        }
    }

    static void computeFft(std::vector<std::complex<float>>& buffer)
    {
        size_t n = buffer.size();
        for (size_t i = 1, j = 0; i < n; ++i)
        {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1)
            {
                j ^= bit;
            }
            j ^= bit;
            if (i < j)
            {
                std::swap(buffer[i], buffer[j]);
            }
        }

        for (size_t len = 2; len <= n; len <<= 1)
        {
            float angle = -2.0f * std::numbers::pi_v<float> / static_cast<float>(len);
            std::complex<float> wlen(std::cos(angle), std::sin(angle));
            for (size_t i = 0; i < n; i += len)
            {
                std::complex<float> w(1.0f, 0.0f);
                for (size_t j = 0; j < len / 2; ++j)
                {
                    std::complex<float> u = buffer[i + j];
                    std::complex<float> v = buffer[i + j + len / 2] * w;
                    buffer[i + j] = u + v;
                    buffer[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

public:
    Audio_Feature_Extractor()
    {
        initializeHannWindow();
        initializeMelFilterbank();
    }

    std::vector<float> compute(const std::vector<float>& audio_samples) const
    {
        std::vector<float> output_features(TIME_FRAMES * MEL_BINS, 0.0f);
        std::vector<std::complex<float>> fft_buffer(FFT_SIZE);

        for (size_t t = 0; t < TIME_FRAMES; ++t)
        {
            size_t start_index = t * HOP_SIZE;
            for (size_t i = 0; i < FFT_SIZE; ++i)
            {
                float sample_val = (start_index + i < audio_samples.size()) ? audio_samples[start_index + i] : 0.0f;
                fft_buffer[i] = std::complex<float>(sample_val * hann_window[i], 0.0f);
            }

            computeFft(fft_buffer);

            std::array<float, SPECTRUM_BINS> power_spectrum{};
            for (size_t k = 0; k < SPECTRUM_BINS; ++k)
            {
                float real = fft_buffer[k].real();
                float imag = fft_buffer[k].imag();
                power_spectrum[k] = (real * real + imag * imag) / static_cast<float>(FFT_SIZE);
            }

            for (size_t m = 0; m < MEL_BINS; ++m)
            {
                float energy_sum = 0.0f;
                for (size_t k = 0; k < SPECTRUM_BINS; ++k)
                {
                    energy_sum += power_spectrum[k] * mel_filterbank[m][k];
                }
                float log_energy = std::log(std::max(energy_sum, 1e-6f));
                output_features[t * MEL_BINS + m] = log_energy;
            }
        }

        float mean_val = 0.0f;
        for (float val : output_features)
        {
            mean_val += val;
        }
        mean_val /= static_cast<float>(output_features.size());

        float variance_val = 0.0f;
        for (float val : output_features)
        {
            float diff = val - mean_val;
            variance_val += diff * diff;
        }
        float std_dev = std::sqrt(variance_val / static_cast<float>(output_features.size()) + 1e-6f);

        for (float& val : output_features)
        {
            val = (val - mean_val) / std_dev;
        }

        return output_features;
    }
};

std::vector<float> augmentAudio(const std::vector<float>& raw_audio, std::mt19937& random_engine)
{
    std::uniform_int_distribution<int> shift_distribution(-800, 800);
    int time_shift = shift_distribution(random_engine);

    std::vector<float> shifted_audio(16000, 0.0f);
    for (int i = 0; i < 16000; ++i)
    {
        int source_index = i - time_shift;
        if (source_index >= 0 && source_index < 16000)
        {
            shifted_audio[i] = raw_audio[source_index];
        }
    }

    std::uniform_real_distribution<float> gain_distribution(0.8f, 1.2f);
    float gain = gain_distribution(random_engine);

    std::normal_distribution<float> noise_distribution(0.0f, 0.003f);
    for (float& val : shifted_audio)
    {
        val = std::clamp(val * gain + noise_distribution(random_engine), -1.0f, 1.0f);
    }

    return shifted_audio;
}

std::vector<std::vector<float>> loadBackgroundNoiseTracks(const std::string& dataset_root)
{
    std::vector<std::vector<float>> tracks;
    std::filesystem::path noise_dir = std::filesystem::path(dataset_root) / "_background_noise_";
    if (!std::filesystem::exists(noise_dir))
    {
        std::cout << "Notice: _background_noise_ directory not found, synthetic noise will be used.\n";
        return tracks;
    }

    for (const auto& entry : std::filesystem::directory_iterator(noise_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".wav")
        {
            auto track = loadFullWav(entry.path().string());
            if (track.size() >= 16000)
            {
                tracks.push_back(std::move(track));
            }
        }
    }
    std::cout << std::format("Loaded {} background noise tracks from {}\n", tracks.size(), noise_dir.string());
    return tracks;
}

std::vector<float> sampleNoiseClip(const std::vector<std::vector<float>>& tracks, std::mt19937& rng)
{
    std::vector<float> clip(16000, 0.0f);
    if (tracks.empty())
    {
        std::normal_distribution<float> noise_dist(0.0f, 0.005f);
        for (float& val : clip)
        {
            val = noise_dist(rng);
        }
        return clip;
    }

    std::uniform_real_distribution<float> silence_prob(0.0f, 1.0f);
    if (silence_prob(rng) < 0.15f)
    {
        std::normal_distribution<float> noise_dist(0.0f, 0.001f);
        for (float& val : clip)
        {
            val = noise_dist(rng);
        }
        return clip;
    }

    std::uniform_int_distribution<size_t> track_dist(0, tracks.size() - 1);
    const auto& track = tracks[track_dist(rng)];

    std::uniform_int_distribution<size_t> offset_dist(0, track.size() - 16000);
    size_t offset = offset_dist(rng);

    std::uniform_real_distribution<float> gain_dist(0.2f, 1.0f);
    float gain = gain_dist(rng);

    for (size_t i = 0; i < 16000; ++i)
    {
        clip[i] = std::clamp(track[offset + i] * gain, -1.0f, 1.0f);
    }
    return clip;
}

void injectBackgroundNoise(std::vector<float>& audio, const std::vector<std::vector<float>>& tracks, std::mt19937& rng)
{
    if (tracks.empty())
    {
        return;
    }

    std::uniform_real_distribution<float> inject_prob(0.0f, 1.0f);
    if (inject_prob(rng) > 0.5f)
    {
        return;
    }

    std::uniform_int_distribution<size_t> track_dist(0, tracks.size() - 1);
    const auto& track = tracks[track_dist(rng)];

    std::uniform_int_distribution<size_t> offset_dist(0, track.size() - 16000);
    size_t offset = offset_dist(rng);

    std::uniform_real_distribution<float> volume_dist(0.02f, 0.12f);
    float volume = volume_dist(rng);

    for (size_t i = 0; i < 16000; ++i)
    {
        audio[i] = std::clamp(audio[i] + track[offset + i] * volume, -1.0f, 1.0f);
    }
}

class Speech_Commands_Pipeline : public Async_Data_Pipeline
{
private:
    size_t batch_size = 64;
    std::vector<Audio_Sample> dataset_samples;
    std::vector<std::vector<float>> noise_tracks;
    Audio_Feature_Extractor feature_extractor;
    mutable std::mt19937 random_engine{ std::random_device{}() };

protected:
    void prepareBatchHost([[maybe_unused]] size_t batch_step, std::vector<float>& output_inputs, std::vector<float>& output_targets) override
    {
        output_inputs.resize(batch_size * 1024);
        output_targets.assign(batch_size * 6, 0.0f);

        std::uniform_int_distribution<size_t> sample_distribution(0, dataset_samples.size() - 1);

        for (size_t i = 0; i < batch_size; ++i)
        {
            size_t sample_index = sample_distribution(random_engine);
            const auto& sample = dataset_samples[sample_index];

            std::vector<float> audio;
            if (sample.label == 5)
            {
                audio = sampleNoiseClip(noise_tracks, random_engine);
            }
            else
            {
                auto raw_audio = loadWav(sample.file_path);
                audio = augmentAudio(raw_audio, random_engine);
                injectBackgroundNoise(audio, noise_tracks, random_engine);
            }

            auto features = feature_extractor.compute(audio);
            std::copy(features.begin(), features.end(), output_inputs.begin() + i * 1024);
            output_targets[i * 6 + sample.label] = 1.0f;
        }
    }

public:
    Speech_Commands_Pipeline(std::vector<Audio_Sample> _samples,
        std::vector<std::vector<float>> _noise_tracks,
        size_t _batch_size,
        VkDevice _device = VK_NULL_HANDLE,
        Execution_Target _target = Execution_Target::VULKAN_GPU)
        : Async_Data_Pipeline(_device, _target),
        batch_size(_batch_size),
        dataset_samples(std::move(_samples)),
        noise_tracks(std::move(_noise_tracks))
    {
        initializeBuffers(batch_size, 1024, 6, _target);
    }

    size_t getBatchSize() const override
    {
        return batch_size;
    }
};

std::vector<Audio_Sample> scanDataset(const std::string& dataset_root)
{
    const std::array<std::pair<std::string, size_t>, 5> label_mappings = { {
        { "stop", 0 },
        { "left", 1 },
        { "right", 2 },
        { "up", 3 },
        { "down", 4 }
    } };

    std::vector<Audio_Sample> collected_samples;
    for (const auto& [folder_name, label] : label_mappings)
    {
        std::filesystem::path folder_path = std::filesystem::path(dataset_root) / folder_name;
        if (!std::filesystem::exists(folder_path))
        {
            std::cout << std::format("Warning: Directory does not exist: {}\n", folder_path.string());
            continue;
        }

        for (const auto& entry : std::filesystem::directory_iterator(folder_path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".wav")
            {
                collected_samples.push_back(Audio_Sample{ entry.path().string(), label });
            }
        }
    }

    size_t samples_per_class = collected_samples.size() / 5;
    for (size_t i = 0; i < samples_per_class; ++i)
    {
        collected_samples.push_back(Audio_Sample{ "", 5 });
    }

    return collected_samples;
}

int main(int argc, char* argv[])
{
    Logger::setFileLogging(true);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);
    Logger::setConsoleOutput(false);

    std::string dataset_path = "data/GSCD";
    std::string output_model_path = "output/controller/model.bin";
    size_t total_epochs = 100;
    size_t batch_size = 8;
    float learning_rate = 0.001f;
    bool is_dry_run = false;
    Execution_Target execution_target = Execution_Target::VULKAN_GPU;

    for (int i = 1; i < argc; ++i)
    {
        std::string_view argument = argv[i];
        if (argument == "--data" && i + 1 < argc)
        {
            dataset_path = argv[++i];
        }
        else if (argument == "--output" && i + 1 < argc)
        {
            output_model_path = argv[++i];
        }
        else if (argument == "--epochs" && i + 1 < argc)
        {
            total_epochs = std::stoull(argv[++i]);
        }
        else if (argument == "--batch-size" && i + 1 < argc)
        {
            batch_size = std::stoull(argv[++i]);
        }
        else if (argument == "--lr" && i + 1 < argc)
        {
            learning_rate = std::stof(argv[++i]);
        }
        else if (argument == "--cpu")
        {
            execution_target = Execution_Target::CPU;
        }
        else if (argument == "--dry-run")
        {
            is_dry_run = true;
        }
    }

    std::cout << "Scanning Speech Commands dataset from: " << dataset_path << '\n';
    auto all_samples = scanDataset(dataset_path);
    if (all_samples.empty())
    {
        std::cerr << "Error: No .wav samples found in " << dataset_path << '\n';
        return 1;
    }

    auto noise_tracks = loadBackgroundNoiseTracks(dataset_path);

    std::mt19937 split_rng(42);
    std::ranges::shuffle(all_samples, split_rng);

    size_t validation_count = all_samples.size() / 10;
    std::vector<Audio_Sample> validation_samples(all_samples.begin(), all_samples.begin() + validation_count);
    std::vector<Audio_Sample> training_samples(all_samples.begin() + validation_count, all_samples.end());

    std::cout << std::format("Total samples: {} (Train: {}, Val: {})\n",
        all_samples.size(), training_samples.size(), validation_samples.size());

    size_t steps_per_epoch = is_dry_run ? 2 : (training_samples.size() / batch_size);
    if (is_dry_run)
    {
        total_epochs = 1;
        std::cout << "Executing dry-run mode (1 epoch, 2 steps)\n";
    }

    Neural_Network network(execution_target);
    // network.enableMixedPrecision();

    network.addLayer<Conv2d_Layer>(32, 32, 1, 16, 3, 1, 1, execution_target);
    network.addLayer<Batch_Norm_2d_Layer>(32, 32, 16, 1e-5f, 0.1f, execution_target);
    network.addLayer<Gelu_Layer>(execution_target);
    network.addLayer<Max_Pool_2d_Layer>(32, 32, 16, 2, 2, 0, execution_target);

    network.addLayer<Conv2d_Layer>(16, 16, 16, 32, 3, 1, 1, execution_target);
    network.addLayer<Batch_Norm_2d_Layer>(16, 16, 32, 1e-5f, 0.1f, execution_target);
    network.addLayer<Gelu_Layer>(execution_target);
    network.addLayer<Max_Pool_2d_Layer>(16, 16, 32, 2, 2, 0, execution_target);
    network.addLayer<Max_Pool_2d_Layer>(8, 8, 32, 2, 2, 0, execution_target);

    network.addLayer<Linear_Layer>(512, 64, execution_target);
    network.addLayer<Batch_Norm_Layer>(64, 1e-5f, 0.1f, execution_target);
    network.addLayer<Gelu_Layer>(execution_target);
    network.addLayer<Linear_Layer>(64, 6, execution_target);
    network.addLayer<Softmax_Layer>(true, execution_target);

    network.setCostFunction<Cce_Cost>(1e-7f, execution_target);
    network.setLearningRate<Cosine_Annealing>(learning_rate, 1e-5f, static_cast<int>(total_epochs));
    network.setOptimizer<Adam_Optimizer>(network.getLearningRate(), 0.9f, 0.999f, 1e-8f, 1.0f);

    Speech_Commands_Pipeline pipeline(std::move(training_samples), noise_tracks, batch_size, VK_NULL_HANDLE, execution_target);

    std::cout << "Starting neural network training...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    network.fit(pipeline, total_epochs, steps_per_epoch, batch_size, 1024, 6, "");

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << std::format("Training completed in {:.2f} seconds.\n", elapsed_seconds);

    std::cout << "Evaluating model on validation set...\n";
    network.setTrainingMode(false);
    Audio_Feature_Extractor evaluation_extractor;

    std::array<std::array<size_t, 6>, 6> confusion_matrix{};
    size_t correct_count = 0;
    size_t evaluated_samples = is_dry_run ? std::min<size_t>(validation_samples.size(), 24) : validation_samples.size();

    for (size_t i = 0; i < evaluated_samples; ++i)
    {
        const auto& sample = validation_samples[i];
        std::vector<float> audio;
        if (sample.label == 5)
        {
            audio = sampleNoiseClip(noise_tracks, split_rng);
        }
        else
        {
            audio = loadWav(sample.file_path);
        }
        auto features = evaluation_extractor.compute(audio);

        Tensor input_tensor(1, 1024, features, execution_target);
        Tensor prediction = network.forward(input_tensor);

        auto output_data = prediction.getData();
        size_t predicted_class = static_cast<size_t>(std::distance(
            output_data.begin(),
            std::max_element(output_data.begin(), output_data.end())));

        confusion_matrix[sample.label][predicted_class]++;
        if (predicted_class == sample.label)
        {
            correct_count++;
        }
    }

    float accuracy = evaluated_samples > 0 ? (static_cast<float>(correct_count) / static_cast<float>(evaluated_samples) * 100.0f) : 0.0f;
    std::cout << std::format("Validation Accuracy: {:.2f}% ({}/{})\n\n", accuracy, correct_count, evaluated_samples);

    const std::array<std::string_view, 6> class_names = { "stop", "left", "right", "up", "down", "noise" };
    std::cout << "Confusion Matrix (Row: Ground Truth, Col: Prediction):\n";
    std::cout << std::format("{:<8}", "");
    for (auto name : class_names)
    {
        std::cout << std::format("{:>8}", name);
    }
    std::cout << '\n';

    for (size_t r = 0; r < 6; ++r)
    {
        std::cout << std::format("{:<8}", class_names[r]);
        for (size_t c = 0; c < 6; ++c)
        {
            std::cout << std::format("{:>8}", confusion_matrix[r][c]);
        }
        std::cout << '\n';
    }

    std::filesystem::path output_path(output_model_path);
    if (output_path.has_parent_path())
    {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::cout << "\nSaving model inference checkpoint to: " << output_model_path << '\n';
    network.saveInference(output_model_path);
    std::cout << "Model saved successfully.\n";

    return 0;
}