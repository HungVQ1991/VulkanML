#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "engine/data_type.h"
#include "llm/causal_lm.h"
#include "tokenizer/bpe_tokenizer.h"

namespace weights
{

enum class Weight_Format : uint32_t
{
    AUTO = 0,
    SAFETENSORS = 1,
    GGUF = 2,
    ONNX = 3,
    PTH = 4
};

enum class Weight_Dtype : uint32_t
{
    UNKNOWN = 0,
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    F64 = 4,
    I8 = 5,
    U8 = 6,
    I16 = 7,
    I32 = 8,
    I64 = 9,
    Q4_0 = 10,
    Q4_1 = 11,
    Q5_0 = 12,
    Q5_1 = 13,
    Q8_0 = 14,
    Q4_K = 15,
    Q6_K = 16
};

struct Weight_Tensor_Info
{
    std::string name;
    std::vector<size_t> shape;
    Weight_Dtype dtype = Weight_Dtype::UNKNOWN;
    uint64_t byte_size = 0;

    size_t elementCount() const noexcept;
};

struct Weight_Metadata
{
    using Value = std::variant<
        bool,
        int64_t,
        uint64_t,
        double,
        std::string,
        std::vector<int64_t>,
        std::vector<double>,
        std::vector<std::string>>;

    std::unordered_map<std::string, Value> kv;

    bool has(const std::string& key) const noexcept;
    int64_t getInt(const std::string& key, int64_t def = 0) const noexcept;
    uint64_t getUint(const std::string& key, uint64_t def = 0) const noexcept;
    double getDouble(const std::string& key, double def = 0.0) const noexcept;
    bool getBool(const std::string& key, bool def = false) const noexcept;
    std::string getString(const std::string& key, const std::string& def = "") const;
    std::vector<std::string> getStringArray(const std::string& key) const;
    std::vector<int64_t> getIntArray(const std::string& key) const;
};

float f16ToF32(uint16_t h) noexcept;
uint16_t f32ToF16(float f) noexcept;
float bf16ToF32(uint16_t b) noexcept;
uint16_t f32ToBf16(float f) noexcept;
std::vector<float> transpose2D(const std::vector<float>& src, size_t rows, size_t cols);
std::string weightFormatName(Weight_Format fmt);
std::string weightDtypeName(Weight_Dtype dtype);
Weight_Format detectWeightFormat(const std::filesystem::path& path);

class IWeight_Source
{
public:
    virtual ~IWeight_Source() = default;
    virtual Weight_Format format() const noexcept = 0;
    virtual bool hasTensor(const std::string& name) const noexcept = 0;
    virtual std::vector<std::string> tensorNames() const = 0;
    virtual Weight_Tensor_Info getTensorInfo(const std::string& name) const = 0;
    virtual std::vector<float> readTensorF32(const std::string& name) const = 0;
    virtual const Weight_Metadata& metadata() const noexcept = 0;
};

std::unique_ptr<IWeight_Source> openSafetensors(const std::filesystem::path& path);
std::unique_ptr<IWeight_Source> openGguf(const std::filesystem::path& path);
std::unique_ptr<IWeight_Source> openOnnx(const std::filesystem::path& path);
std::unique_ptr<IWeight_Source> openPth(const std::filesystem::path& path);
std::unique_ptr<IWeight_Source> openWeightSource(const std::filesystem::path& path, Weight_Format fmt = Weight_Format::AUTO);

struct Load_Model_Options
{
    Execution_Target target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT32;
    size_t max_seq_len = 0;
    Weight_Format format = Weight_Format::AUTO;
    std::filesystem::path config_path;
    std::filesystem::path tokenizer_path;
    std::unordered_map<std::string, std::string> name_aliases;
    bool transpose_linear_weights = true;
    bool strict_missing_tensors = true;
};

struct Loaded_Causal_LM
{
    Causal_LM model;
    int32_t config_eos_id = -1;
    Weight_Format detected_format = Weight_Format::AUTO;
    std::string architecture;
};

Loaded_Causal_LM loadCausalLM(const std::filesystem::path& path, const Load_Model_Options& opts = {});

size_t loadNamedTensors(const IWeight_Source& source,
                        const std::unordered_map<std::string, Tensor*>& targets,
                        const std::unordered_map<std::string, bool>& transpose_2d_flags = {});

}
