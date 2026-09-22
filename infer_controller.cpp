#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if __has_include("helper/vulkan_ml.h")
#include "helper/vulkan_ml.h"
#elif __has_include("vulkan_ml.h")
#include "vulkan_ml.h"
#endif

using Pfn_Wave_In_Open = MMRESULT(WINAPI*)(LPHWAVEIN, UINT, LPCWAVEFORMATEX, DWORD_PTR, DWORD_PTR, DWORD);
using Pfn_Wave_In_Close = MMRESULT(WINAPI*)(HWAVEIN);
using Pfn_Wave_In_Prepare_Header = MMRESULT(WINAPI*)(HWAVEIN, LPWAVEHDR, UINT);
using Pfn_Wave_In_Unprepare_Header = MMRESULT(WINAPI*)(HWAVEIN, LPWAVEHDR, UINT);
using Pfn_Wave_In_Add_Buffer = MMRESULT(WINAPI*)(HWAVEIN, LPWAVEHDR, UINT);
using Pfn_Wave_In_Start = MMRESULT(WINAPI*)(HWAVEIN);
using Pfn_Wave_In_Stop = MMRESULT(WINAPI*)(HWAVEIN);
using Pfn_Wave_In_Reset = MMRESULT(WINAPI*)(HWAVEIN);
using Pfn_Wave_In_Get_Num_Devs = UINT(WINAPI*)(void);

struct Command_Info
{
    std::string_view keyword;
    std::string_view translation;
};

const std::array<Command_Info, 6> COMMAND_CATALOG = { {
    { "stop", "Dừng lại" },
    { "left", "Rẽ trái" },
    { "right", "Rẽ phải" },
    { "up", "Tiến lên" },
    { "down", "Lùi lại" },
    { "noise", "Tiếng ồn / Im lặng" }
} };

std::vector<float> loadWav(const std::string& file_path)
{
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream.is_open())
    {
        return std::vector<float>(16000, 0.0f);
    }

    char riff_tag[4];
    stream.read(riff_tag, 4);
    if (std::string_view(riff_tag, 4) != "RIFF")
    {
        return std::vector<float>(16000, 0.0f);
    }

    uint32_t riff_size = 0;
    stream.read(reinterpret_cast<char*>(&riff_size), sizeof(riff_size));

    char wave_tag[4];
    stream.read(wave_tag, 4);
    if (std::string_view(wave_tag, 4) != "WAVE")
    {
        return std::vector<float>(16000, 0.0f);
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
        return std::vector<float>(16000, 0.0f);
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

    std::vector<float> compute(std::span<const float> audio_samples) const
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

float computeRms(std::span<const float> audio_samples)
{
    float sum_squares = 0.0f;
    for (float sample : audio_samples)
    {
        sum_squares += sample * sample;
    }
    return std::sqrt(sum_squares / static_cast<float>(audio_samples.size()));
}

class Win_Mm_Audio_Capture
{
private:
    static constexpr size_t BUFFER_COUNT = 4;
    static constexpr size_t SAMPLES_PER_BUFFER = 3200;

    HMODULE winmm_module = nullptr;
    HWAVEIN wave_in_handle = nullptr;

    Pfn_Wave_In_Open pfn_wave_in_open = nullptr;
    Pfn_Wave_In_Close pfn_wave_in_close = nullptr;
    Pfn_Wave_In_Prepare_Header pfn_wave_in_prepare_header = nullptr;
    Pfn_Wave_In_Unprepare_Header pfn_wave_in_unprepare_header = nullptr;
    Pfn_Wave_In_Add_Buffer pfn_wave_in_add_buffer = nullptr;
    Pfn_Wave_In_Start pfn_wave_in_start = nullptr;
    Pfn_Wave_In_Stop pfn_wave_in_stop = nullptr;
    Pfn_Wave_In_Reset pfn_wave_in_reset = nullptr;
    Pfn_Wave_In_Get_Num_Devs pfn_wave_in_get_num_devs = nullptr;

    std::array<std::vector<std::int16_t>, BUFFER_COUNT> buffer_storage;
    std::array<WAVEHDR, BUFFER_COUNT> wave_headers{};
    std::vector<float> sliding_window;
    std::atomic<bool> is_capturing{ false };

    bool loadLibrarySymbols()
    {
        winmm_module = LoadLibraryA("winmm.dll");
        if (!winmm_module)
        {
            return false;
        }

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
        pfn_wave_in_open = reinterpret_cast<Pfn_Wave_In_Open>(GetProcAddress(winmm_module, "waveInOpen"));
        pfn_wave_in_close = reinterpret_cast<Pfn_Wave_In_Close>(GetProcAddress(winmm_module, "waveInClose"));
        pfn_wave_in_prepare_header = reinterpret_cast<Pfn_Wave_In_Prepare_Header>(GetProcAddress(winmm_module, "waveInPrepareHeader"));
        pfn_wave_in_unprepare_header = reinterpret_cast<Pfn_Wave_In_Unprepare_Header>(GetProcAddress(winmm_module, "waveInUnprepareHeader"));
        pfn_wave_in_add_buffer = reinterpret_cast<Pfn_Wave_In_Add_Buffer>(GetProcAddress(winmm_module, "waveInAddBuffer"));
        pfn_wave_in_start = reinterpret_cast<Pfn_Wave_In_Start>(GetProcAddress(winmm_module, "waveInStart"));
        pfn_wave_in_stop = reinterpret_cast<Pfn_Wave_In_Stop>(GetProcAddress(winmm_module, "waveInStop"));
        pfn_wave_in_reset = reinterpret_cast<Pfn_Wave_In_Reset>(GetProcAddress(winmm_module, "waveInReset"));
        pfn_wave_in_get_num_devs = reinterpret_cast<Pfn_Wave_In_Get_Num_Devs>(GetProcAddress(winmm_module, "waveInGetNumDevs"));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

        return pfn_wave_in_open && pfn_wave_in_close && pfn_wave_in_prepare_header &&
            pfn_wave_in_unprepare_header && pfn_wave_in_add_buffer && pfn_wave_in_start &&
            pfn_wave_in_stop && pfn_wave_in_reset && pfn_wave_in_get_num_devs;
    }

public:
    Win_Mm_Audio_Capture()
        : sliding_window(16000, 0.0f)
    {
    }

    ~Win_Mm_Audio_Capture()
    {
        stop();
        if (winmm_module)
        {
            FreeLibrary(winmm_module);
            winmm_module = nullptr;
        }
    }

    bool initialize()
    {
        if (!loadLibrarySymbols())
        {
            std::cerr << "Error: Failed to load required symbols from winmm.dll\n";
            return false;
        }

        if (pfn_wave_in_get_num_devs() == 0)
        {
            std::cerr << "Error: No audio recording devices found\n";
            return false;
        }

        WAVEFORMATEX wave_format{
            .wFormatTag = WAVE_FORMAT_PCM,
            .nChannels = 1,
            .nSamplesPerSec = 16000,
            .nAvgBytesPerSec = 16000 * sizeof(std::int16_t),
            .nBlockAlign = sizeof(std::int16_t),
            .wBitsPerSample = 16,
            .cbSize = 0 };

        MMRESULT open_result = pfn_wave_in_open(&wave_in_handle, WAVE_MAPPER, &wave_format, 0, 0, CALLBACK_NULL);
        if (open_result != MMSYSERR_NOERROR)
        {
            std::cerr << std::format("Error: waveInOpen failed with error code {}\n", open_result);
            return false;
        }

        for (size_t i = 0; i < BUFFER_COUNT; ++i)
        {
            buffer_storage[i].assign(SAMPLES_PER_BUFFER, 0);
            wave_headers[i] = WAVEHDR{
                .lpData = reinterpret_cast<LPSTR>(buffer_storage[i].data()),
                .dwBufferLength = static_cast<DWORD>(SAMPLES_PER_BUFFER * sizeof(std::int16_t)),
                .dwBytesRecorded = 0,
                .dwUser = 0,
                .dwFlags = 0,
                .dwLoops = 0,
                .lpNext = nullptr,
                .reserved = 0 };

            pfn_wave_in_prepare_header(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
            pfn_wave_in_add_buffer(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
        }

        return true;
    }

    bool start()
    {
        if (!wave_in_handle || is_capturing.load())
        {
            return false;
        }
        MMRESULT start_result = pfn_wave_in_start(wave_in_handle);
        if (start_result == MMSYSERR_NOERROR)
        {
            is_capturing.store(true);
            return true;
        }
        return false;
    }

    void stop()
    {
        if (!wave_in_handle || !is_capturing.load())
        {
            return;
        }
        is_capturing.store(false);
        pfn_wave_in_stop(wave_in_handle);
        pfn_wave_in_reset(wave_in_handle);

        for (size_t i = 0; i < BUFFER_COUNT; ++i)
        {
            pfn_wave_in_unprepare_header(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
        }

        pfn_wave_in_close(wave_in_handle);
        wave_in_handle = nullptr;
    }

    bool pollNewAudio()
    {
        bool has_new_data = false;
        for (size_t i = 0; i < BUFFER_COUNT; ++i)
        {
            if (wave_headers[i].dwFlags & WHDR_DONE)
            {
                size_t recorded_samples = wave_headers[i].dwBytesRecorded / sizeof(std::int16_t);
                if (recorded_samples > 0)
                {
                    std::copy(sliding_window.begin() + recorded_samples, sliding_window.end(), sliding_window.begin());
                    for (size_t s = 0; s < recorded_samples; ++s)
                    {
                        sliding_window[16000 - recorded_samples + s] = static_cast<float>(buffer_storage[i][s]) / 32768.0f;
                    }
                    has_new_data = true;
                }

                pfn_wave_in_unprepare_header(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
                wave_headers[i].dwFlags = 0;
                wave_headers[i].dwBytesRecorded = 0;
                pfn_wave_in_prepare_header(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
                pfn_wave_in_add_buffer(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
            }
        }
        return has_new_data;
    }

    std::span<const float> getSlidingWindow() const noexcept
    {
        return sliding_window;
    }

    std::vector<float> recordSingleClip()
    {
        std::vector<float> clip_samples(16000, 0.0f);
        size_t accumulated_samples = 0;

        while (accumulated_samples < 16000)
        {
            for (size_t i = 0; i < BUFFER_COUNT; ++i)
            {
                if (wave_headers[i].dwFlags & WHDR_DONE)
                {
                    size_t recorded = wave_headers[i].dwBytesRecorded / sizeof(std::int16_t);
                    size_t copy_count = std::min(recorded, 16000 - accumulated_samples);

                    for (size_t s = 0; s < copy_count; ++s)
                    {
                        clip_samples[accumulated_samples + s] = static_cast<float>(buffer_storage[i][s]) / 32768.0f;
                    }
                    accumulated_samples += copy_count;

                    pfn_wave_in_unprepare_header(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
                    wave_headers[i].dwFlags = 0;
                    wave_headers[i].dwBytesRecorded = 0;
                    pfn_wave_in_prepare_header(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));
                    pfn_wave_in_add_buffer(wave_in_handle, &wave_headers[i], sizeof(WAVEHDR));

                    if (accumulated_samples >= 16000)
                    {
                        break;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        return clip_samples;
    }
};

void runFileInference(Neural_Network& network,
    const Audio_Feature_Extractor& feature_extractor,
    const std::string& wav_path,
    Execution_Target execution_target)
{
    std::cout << std::format("Evaluating audio file: {}\n", wav_path);
    auto raw_samples = loadWav(wav_path);
    auto features = feature_extractor.compute(raw_samples);

    Tensor input_tensor(1, 1024, features, execution_target);
    Tensor prediction = network.forward(input_tensor);
    auto probabilities = prediction.getData();

    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    size_t predicted_idx = static_cast<size_t>(std::distance(probabilities.begin(), max_it));
    float confidence = *max_it;

    std::cout << "\n--- Classification Result ---\n";
    for (size_t i = 0; i < COMMAND_CATALOG.size(); ++i)
    {
        std::cout << std::format("  [{:<5}]: {:>6.2f}%\n",
            COMMAND_CATALOG[i].keyword,
            probabilities[i] * 100.0f);
    }
    if (predicted_idx < 5)
    {
        std::cout << std::format("\nIdentified command: '{}' with confidence {:.2f}%\n",
            COMMAND_CATALOG[predicted_idx].keyword,
            confidence * 100.0f);
    }
    else
    {
        std::cout << std::format("\nNo command detected: Sound classified as '{}' with confidence {:.2f}%\n",
            COMMAND_CATALOG[predicted_idx].keyword,
            confidence * 100.0f);
    }
}

void runLiveStreamingInference(Neural_Network& network,
    const Audio_Feature_Extractor& feature_extractor,
    Win_Mm_Audio_Capture& audio_capture,
    float confidence_threshold,
    Execution_Target execution_target)
{
    std::cout << std::format("Confidence threshold: {:.2f} \n\n", confidence_threshold);

    auto last_trigger_time = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    constexpr float SILENCE_RMS_THRESHOLD = 0.015f;

    while (true)
    {
        if (audio_capture.pollNewAudio())
        {
            auto current_window = audio_capture.getSlidingWindow();
            float current_rms = computeRms(current_window);

            if (current_rms >= SILENCE_RMS_THRESHOLD)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_since_trigger = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_trigger_time).count();

                if (elapsed_since_trigger >= 750)
                {
                    auto features = feature_extractor.compute(current_window);
                    Tensor input_tensor(1, 1024, features, execution_target);
                    Tensor prediction = network.forward(input_tensor);
                    auto probabilities = prediction.getData();

                    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
                    size_t predicted_idx = static_cast<size_t>(std::distance(probabilities.begin(), max_it));
                    float max_probability = *max_it;

                    if (predicted_idx < 5 && max_probability >= confidence_threshold)
                    {
                        last_trigger_time = now;
                        std::cout << std::format("{:<6} | Confidence: {:>5.1f}% | Energy: {:.4f}\n",
                            COMMAND_CATALOG[predicted_idx].keyword,
                            max_probability * 100.0f,
                            current_rms);
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

int main(int argc, char* argv[])
{
    Logger::setFileLogging(true);
    Logger::setOnlyActiveFeatures(Log_Feature::NONE);
    Logger::setConsoleOutput(false);

    std::string model_path = "output/controller/model.bin";
    std::string test_wav_path = "";
    float confidence_threshold = 0.99f;
    bool is_single_record_mode = false;
    Execution_Target execution_target = Execution_Target::VULKAN_GPU;

    for (int i = 1; i < argc; ++i)
    {
        std::string_view argument = argv[i];
        if (argument == "--model" && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (argument == "--test-wav" && i + 1 < argc)
        {
            test_wav_path = argv[++i];
        }
        else if (argument == "--threshold" && i + 1 < argc)
        {
            confidence_threshold = std::stof(argv[++i]);
        }
        else if (argument == "--record")
        {
            is_single_record_mode = true;
        }
        else if (argument == "--cpu")
        {
            execution_target = Execution_Target::CPU;
        }
    }

    std::cout << std::format("Loading inference model from: {}\n", model_path);
    Neural_Network network(execution_target);
    network.loadInference(model_path, execution_target);
    network.setTrainingMode(false);

    Audio_Feature_Extractor feature_extractor;

    if (!test_wav_path.empty())
    {
        runFileInference(network, feature_extractor, test_wav_path, execution_target);
        return 0;
    }

    Win_Mm_Audio_Capture audio_capture;
    if (!audio_capture.initialize())
    {
        std::cerr << "Fatal Error: Failed to initialize Windows Multimedia audio capture.\n";
        return 1;
    }

    if (!audio_capture.start())
    {
        std::cerr << "Fatal Error: Failed to start audio capture stream.\n";
        return 1;
    }
    else
    {
        runLiveStreamingInference(network, feature_extractor, audio_capture, confidence_threshold, execution_target);
    }

    audio_capture.stop();
    return 0;
}