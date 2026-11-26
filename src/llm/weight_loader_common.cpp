#include "llm/weight_loader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/json.hpp"

namespace weights
{

namespace
{

std::string toLowerCopy(std::string s)
{
    for (char& ch : s)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

std::vector<std::string> candidateNamesFor(const std::string& canonical_name,
                                           const std::unordered_map<std::string, std::string>& aliases)
{
    std::vector<std::string> out;
    auto it = aliases.find(canonical_name);
    if (it != aliases.end())
    {
        out.push_back(it->second);
    }
    out.push_back(canonical_name);
    out.push_back("/" + canonical_name);

    if (canonical_name.rfind("model.", 0) == 0)
    {
        out.push_back(canonical_name.substr(6));
        out.push_back("transformer." + canonical_name.substr(6));
    }

    if (canonical_name == "model.embed_tokens.weight")
    {
        out.push_back("token_embd.weight");
        out.push_back("tok_embeddings.weight");
        out.push_back("transformer.wte.weight");
        return out;
    }
    if (canonical_name == "model.norm.weight")
    {
        out.push_back("output_norm.weight");
        out.push_back("norm.weight");
        out.push_back("transformer.ln_f.weight");
        return out;
    }
    if (canonical_name == "lm_head.weight")
    {
        out.push_back("output.weight");
        out.push_back("model.lm_head.weight");
        return out;
    }

    const std::string layer_pfx = "model.layers.";
    if (canonical_name.rfind(layer_pfx, 0) == 0)
    {
        const size_t dot_pos = canonical_name.find('.', layer_pfx.size());
        if (dot_pos != std::string::npos)
        {
            const std::string layer_idx = canonical_name.substr(layer_pfx.size(), dot_pos - layer_pfx.size());
            const std::string suffix = canonical_name.substr(dot_pos + 1);
            const std::string blk = "blk." + layer_idx + ".";

            static const std::unordered_map<std::string, std::string> gguf_map = {
                { "input_layernorm.weight", "attn_norm.weight" },
                { "self_attn.q_proj.weight", "attn_q.weight" },
                { "self_attn.k_proj.weight", "attn_k.weight" },
                { "self_attn.v_proj.weight", "attn_v.weight" },
                { "self_attn.o_proj.weight", "attn_output.weight" },
                { "self_attn.q_proj.bias", "attn_q.bias" },
                { "self_attn.k_proj.bias", "attn_k.bias" },
                { "self_attn.v_proj.bias", "attn_v.bias" },
                { "self_attn.o_proj.bias", "attn_output.bias" },
                { "self_attn.q_norm.weight", "attn_q_norm.weight" },
                { "self_attn.k_norm.weight", "attn_k_norm.weight" },
                { "post_attention_layernorm.weight", "ffn_norm.weight" },
                { "mlp.gate_proj.weight", "ffn_gate.weight" },
                { "mlp.up_proj.weight", "ffn_up.weight" },
                { "mlp.down_proj.weight", "ffn_down.weight" },
                { "mlp.gate_proj.bias", "ffn_gate.bias" },
                { "mlp.up_proj.bias", "ffn_up.bias" },
                { "mlp.down_proj.bias", "ffn_down.bias" }
            };
            auto git = gguf_map.find(suffix);
            if (git != gguf_map.end())
            {
                out.push_back(blk + git->second);
            }
        }
    }
    return out;
}

std::string resolveTensorName(const IWeight_Source& src,
                              const std::string& canonical_name,
                              const std::unordered_map<std::string, std::string>& aliases,
                              bool required)
{
    for (const auto& cand : candidateNamesFor(canonical_name, aliases))
    {
        if (src.hasTensor(cand))
        {
            return cand;
        }
    }
    if (required)
    {
        throw std::runtime_error("loadCausalLM: Missing required tensor '" + canonical_name + "'");
    }
    return {};
}

Tensor makeTypedTensor(const Shape& shape, std::vector<float>&& data, Execution_Target target, Data_Type dtype)
{
    Tensor t(shape, std::move(data), target);
    if (dtype == Data_Type::FLOAT16 && t.getDataType() != Data_Type::FLOAT16)
    {
        Tensor t16;
        t.to(Data_Type::FLOAT16, t16);
        return t16;
    }
    return t;
}

void assignLinearWeight(Linear_Layer& layer,
                        const IWeight_Source& src,
                        const std::string& actual_name,
                        size_t expected_in,
                        size_t expected_out,
                        bool transpose_weights)
{
    const auto info = src.getTensorInfo(actual_name);
    if (info.shape.size() != 2)
    {
        throw std::runtime_error(std::format("assignLinearWeight: Expected 2D tensor for '{}'", actual_name));
    }
    std::vector<float> raw = src.readTensorF32(actual_name);
    if (transpose_weights)
    {
        if (info.shape[0] != expected_out || info.shape[1] != expected_in)
        {
            throw std::runtime_error(std::format("assignLinearWeight: Shape mismatch for '{}' (expected [{}, {}], got [{}, {}])",
                                                 actual_name, expected_out, expected_in, info.shape[0], info.shape[1]));
        }
        raw = transpose2D(raw, expected_out, expected_in);
    }
    else if (info.shape[0] != expected_in || info.shape[1] != expected_out)
    {
        throw std::runtime_error(std::format("assignLinearWeight: Shape mismatch for '{}' (expected [{}, {}], got [{}, {}])",
                                             actual_name, expected_in, expected_out, info.shape[0], info.shape[1]));
    }
    const Data_Type w_dtype = layer.getWeights().getDataType();
    layer.setWeights(makeTypedTensor(Shape{ expected_in, expected_out }, std::move(raw), layer.getExecutionTarget(), w_dtype));
}

void assignLinearBias(Linear_Layer& layer,
                      const IWeight_Source& src,
                      const std::string& actual_name,
                      size_t expected_out)
{
    const auto info = src.getTensorInfo(actual_name);
    if (info.shape.size() != 1 || info.shape[0] != expected_out)
    {
        throw std::runtime_error(std::format("assignLinearBias: Shape mismatch for '{}' (expected [{}])",
                                             actual_name, expected_out));
    }
    const Data_Type b_dtype = layer.getBiases().getDataType();
    layer.setBiases(makeTypedTensor(Shape{ 1, expected_out }, src.readTensorF32(actual_name), layer.getExecutionTarget(), b_dtype));
}

void assignNormWeight(RMSNorm_Layer& layer,
                      const IWeight_Source& src,
                      const std::string& actual_name,
                      size_t expected_dim)
{
    const auto info = src.getTensorInfo(actual_name);
    if (info.shape.size() != 1 || info.shape[0] != expected_dim)
    {
        throw std::runtime_error(std::format("assignNormWeight: Shape mismatch for '{}' (expected [{}])",
                                             actual_name, expected_dim));
    }
    auto params = layer.getParametersAndGradients();
    if (params.empty() || params[0].first == nullptr)
    {
        throw std::runtime_error("assignNormWeight: Norm layer has no writable weight");
    }
    params[0].first->uploadData(src.readTensorF32(actual_name));
}

std::filesystem::path findCompanionFile(const std::filesystem::path& model_path,
                                        const std::filesystem::path& override_path,
                                        const std::string& filename)
{
    if (!override_path.empty())
    {
        return override_path;
    }
    std::error_code ec;
    if (std::filesystem::is_directory(model_path, ec))
    {
        const auto cand = model_path / filename;
        if (std::filesystem::exists(cand, ec))
        {
            return cand;
        }
    }
    else
    {
        const auto cand = model_path.parent_path() / filename;
        if (std::filesystem::exists(cand, ec))
        {
            return cand;
        }
    }
    return {};
}

Causal_LM_Config buildConfigFromJson(const std::filesystem::path& cfg_path,
                                     const IWeight_Source& src,
                                     const Load_Model_Options& opts,
                                     int32_t& out_eos_id,
                                     std::string& out_arch)
{
    std::ifstream cfg_in(cfg_path);
    if (!cfg_in.is_open())
    {
        throw std::runtime_error("loadCausalLM: Failed to open config JSON '" + cfg_path.string() + "'");
    }
    nlohmann::json j = nlohmann::json::parse(cfg_in);

    Causal_LM_Config cfg;
    cfg.vocab_size = j.at("vocab_size").get<size_t>();
    cfg.hidden_dim = j.at("hidden_size").get<size_t>();
    cfg.num_heads = j.at("num_attention_heads").get<size_t>();
    cfg.num_kv_heads = j.value("num_key_value_heads", cfg.num_heads);
    if (j.contains("head_dim") && j["head_dim"].is_number_integer())
    {
        cfg.head_dim = j["head_dim"].get<size_t>();
    }
    else
    {
        cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
    }
    cfg.intermediate_dim = j.at("intermediate_size").get<size_t>();
    cfg.num_layers = j.at("num_hidden_layers").get<size_t>();
    const size_t cfg_max_pos = j.value("max_position_embeddings", size_t{ 2048 });
    cfg.max_seq_len = (opts.max_seq_len > 0) ? std::min(cfg_max_pos, opts.max_seq_len) : cfg_max_pos;
    cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    cfg.rope_base = j.value("rope_theta", 1000000.0f);
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", false);
    cfg.residual_scale = 1.0f;
    cfg.use_fused_projections = false;
    cfg.use_loss_scaler = false;
    cfg.execution_target = opts.target;
    cfg.data_type = opts.data_type;

    out_arch = j.value("model_type", "");
    const bool has_q_norm = !resolveTensorName(src, "model.layers.0.self_attn.q_norm.weight", opts.name_aliases, false).empty();
    const bool has_q_bias = !resolveTensorName(src, "model.layers.0.self_attn.q_proj.bias", opts.name_aliases, false).empty();
    const bool has_gate_bias = !resolveTensorName(src, "model.layers.0.mlp.gate_proj.bias", opts.name_aliases, false).empty();
    cfg.use_qk_norm = (out_arch == "qwen3") || has_q_norm;
    cfg.attention_bias = j.value("attention_bias", false) || has_q_bias;
    cfg.mlp_bias = j.value("mlp_bias", false) || has_gate_bias;

    if (j.contains("rope_scaling") && j["rope_scaling"].is_object())
    {
        const auto& rs = j["rope_scaling"];
        const std::string rtype = rs.value("rope_type", rs.value("type", ""));
        if (rtype == "yarn")
        {
            cfg.yarn.enabled = true;
            cfg.yarn.factor = rs.value("factor", 1.0f);
            cfg.yarn.original_max_position_embeddings = rs.value("original_max_position_embeddings", 32768u);
            cfg.yarn.beta_fast = rs.value("beta_fast", 32.0f);
            cfg.yarn.beta_slow = rs.value("beta_slow", 1.0f);
            cfg.yarn.attention_factor = rs.value("attention_factor", 0.0f);
        }
    }

    out_eos_id = -1;
    if (j.contains("eos_token_id"))
    {
        if (j["eos_token_id"].is_number_integer())
        {
            out_eos_id = j["eos_token_id"].get<int32_t>();
        }
        else if (j["eos_token_id"].is_array() && !j["eos_token_id"].empty())
        {
            out_eos_id = j["eos_token_id"].back().get<int32_t>();
        }
    }
    return cfg;
}

Causal_LM_Config buildConfigFromMetadata(const IWeight_Source& src,
                                         const Load_Model_Options& opts,
                                         int32_t& out_eos_id,
                                         std::string& out_arch)
{
    const auto& m = src.metadata();
    out_arch = m.getString("general.architecture", "qwen3");
    const std::string p = out_arch + ".";

    const std::string emb_name = resolveTensorName(src, "model.embed_tokens.weight", opts.name_aliases, true);
    const auto emb_info = src.getTensorInfo(emb_name);
    if (emb_info.shape.size() != 2)
    {
        throw std::runtime_error("loadCausalLM: Embedding tensor must be 2D");
    }

    Causal_LM_Config cfg;
    cfg.vocab_size = static_cast<size_t>(m.getUint(p + "vocab_size", emb_info.shape[0]));
    if (cfg.vocab_size == 0)
    {
        cfg.vocab_size = emb_info.shape[0];
    }
    cfg.hidden_dim = static_cast<size_t>(m.getUint(p + "embedding_length", emb_info.shape[1]));
    cfg.num_heads = static_cast<size_t>(m.getUint(p + "attention.head_count", 0));
    if (cfg.num_heads == 0)
    {
        throw std::runtime_error("loadCausalLM: Missing GGUF metadata '" + p + "attention.head_count'");
    }
    cfg.num_kv_heads = static_cast<size_t>(m.getUint(p + "attention.head_count_kv", cfg.num_heads));
    cfg.head_dim = static_cast<size_t>(m.getUint(p + "attention.key_length", cfg.hidden_dim / cfg.num_heads));
    cfg.intermediate_dim = static_cast<size_t>(m.getUint(p + "feed_forward_length", 0));
    cfg.num_layers = static_cast<size_t>(m.getUint(p + "block_count", 0));
    if (cfg.intermediate_dim == 0 || cfg.num_layers == 0)
    {
        throw std::runtime_error("loadCausalLM: Missing GGUF block_count or feed_forward_length");
    }
    const size_t cfg_max_pos = static_cast<size_t>(m.getUint(p + "context_length", 2048));
    cfg.max_seq_len = (opts.max_seq_len > 0) ? std::min(cfg_max_pos, opts.max_seq_len) : cfg_max_pos;
    cfg.rms_norm_eps = static_cast<float>(m.getDouble(p + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg.rope_base = static_cast<float>(m.getDouble(p + "rope.freq_base", 1000000.0));

    const bool has_lm_head = !resolveTensorName(src, "lm_head.weight", opts.name_aliases, false).empty();
    cfg.tie_word_embeddings = !has_lm_head;
    cfg.use_qk_norm = (out_arch == "qwen3") || !resolveTensorName(src, "model.layers.0.self_attn.q_norm.weight", opts.name_aliases, false).empty();
    cfg.attention_bias = !resolveTensorName(src, "model.layers.0.self_attn.q_proj.bias", opts.name_aliases, false).empty();
    cfg.mlp_bias = !resolveTensorName(src, "model.layers.0.mlp.gate_proj.bias", opts.name_aliases, false).empty();
    cfg.residual_scale = 1.0f;
    cfg.use_fused_projections = false;
    cfg.use_loss_scaler = false;
    cfg.execution_target = opts.target;
    cfg.data_type = opts.data_type;

    out_eos_id = static_cast<int32_t>(m.getInt("tokenizer.ggml.eos_token_id", -1));
    return cfg;
}

}

size_t Weight_Tensor_Info::elementCount() const noexcept
{
    if (shape.empty())
    {
        return 0;
    }
    size_t n = 1;
    for (size_t d : shape)
    {
        n *= d;
    }
    return n;
}

bool Weight_Metadata::has(const std::string& key) const noexcept
{
    return kv.find(key) != kv.end();
}

int64_t Weight_Metadata::getInt(const std::string& key, int64_t def) const noexcept
{
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (const auto* v = std::get_if<int64_t>(&it->second)) return *v;
    if (const auto* u = std::get_if<uint64_t>(&it->second)) return static_cast<int64_t>(*u);
    if (const auto* d = std::get_if<double>(&it->second)) return static_cast<int64_t>(*d);
    if (const auto* b = std::get_if<bool>(&it->second)) return *b ? 1 : 0;
    return def;
}

uint64_t Weight_Metadata::getUint(const std::string& key, uint64_t def) const noexcept
{
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (const auto* u = std::get_if<uint64_t>(&it->second)) return *u;
    if (const auto* v = std::get_if<int64_t>(&it->second)) return (*v >= 0) ? static_cast<uint64_t>(*v) : def;
    if (const auto* d = std::get_if<double>(&it->second)) return (*d >= 0.0) ? static_cast<uint64_t>(*d) : def;
    return def;
}

double Weight_Metadata::getDouble(const std::string& key, double def) const noexcept
{
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (const auto* d = std::get_if<double>(&it->second)) return *d;
    if (const auto* v = std::get_if<int64_t>(&it->second)) return static_cast<double>(*v);
    if (const auto* u = std::get_if<uint64_t>(&it->second)) return static_cast<double>(*u);
    return def;
}

bool Weight_Metadata::getBool(const std::string& key, bool def) const noexcept
{
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (const auto* b = std::get_if<bool>(&it->second)) return *b;
    if (const auto* v = std::get_if<int64_t>(&it->second)) return *v != 0;
    if (const auto* u = std::get_if<uint64_t>(&it->second)) return *u != 0;
    return def;
}

std::string Weight_Metadata::getString(const std::string& key, const std::string& def) const
{
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
    return def;
}

std::vector<std::string> Weight_Metadata::getStringArray(const std::string& key) const
{
    auto it = kv.find(key);
    if (it == kv.end()) return {};
    if (const auto* arr = std::get_if<std::vector<std::string>>(&it->second)) return *arr;
    return {};
}

std::vector<int64_t> Weight_Metadata::getIntArray(const std::string& key) const
{
    auto it = kv.find(key);
    if (it == kv.end()) return {};
    if (const auto* arr = std::get_if<std::vector<int64_t>>(&it->second)) return *arr;
    return {};
}

float f16ToF32(uint16_t h) noexcept
{
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;
    uint32_t bits = 0;

    if (exp == 0u)
    {
        if (mant == 0u)
        {
            bits = sign;
        }
        else
        {
            uint32_t e = 113u;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0u)
            {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    }
    else if (exp == 31u)
    {
        bits = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t f32ToF16(float f) noexcept
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t abs_bits = bits & 0x7FFFFFFFu;
    if (abs_bits > 0x7F800000u)
    {
        return static_cast<uint16_t>(sign | 0x7E00u);
    }
    if (abs_bits >= 0x47800000u)
    {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (abs_bits < 0x38800000u)
    {
        if (abs_bits < 0x33000000u)
        {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t shift = 113u - (abs_bits >> 23);
        const uint32_t mant = (abs_bits & 0x007FFFFFu) | 0x00800000u;
        const uint32_t rounded = (mant + (1u << (shift + 12u))) >> (shift + 13u);
        return static_cast<uint16_t>(sign | rounded);
    }
    const uint32_t exp = ((abs_bits >> 23) - 112u) << 10;
    const uint32_t mant = (abs_bits & 0x007FFFFFu) + 0x00001000u;
    return static_cast<uint16_t>(sign | (exp + (mant >> 13)));
}

float bf16ToF32(uint16_t b) noexcept
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t f32ToBf16(float f) noexcept
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
    {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    const uint32_t round_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + round_bias) >> 16);
}

std::vector<float> transpose2D(const std::vector<float>& src, size_t rows, size_t cols)
{
    if (src.size() != rows * cols)
    {
        throw std::invalid_argument("weights::transpose2D: element count mismatch");
    }
    std::vector<float> dst(rows * cols, 0.0f);
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return dst;
}

std::string weightFormatName(Weight_Format fmt)
{
    switch (fmt)
    {
    case Weight_Format::SAFETENSORS: return "SAFETENSORS";
    case Weight_Format::GGUF: return "GGUF";
    case Weight_Format::ONNX: return "ONNX";
    case Weight_Format::PTH: return "PTH";
    default: return "AUTO";
    }
}

std::string weightDtypeName(Weight_Dtype dtype)
{
    switch (dtype)
    {
    case Weight_Dtype::F32: return "F32";
    case Weight_Dtype::F16: return "F16";
    case Weight_Dtype::BF16: return "BF16";
    case Weight_Dtype::F64: return "F64";
    case Weight_Dtype::I8: return "I8";
    case Weight_Dtype::U8: return "U8";
    case Weight_Dtype::I16: return "I16";
    case Weight_Dtype::I32: return "I32";
    case Weight_Dtype::I64: return "I64";
    case Weight_Dtype::Q4_0: return "Q4_0";
    case Weight_Dtype::Q4_1: return "Q4_1";
    case Weight_Dtype::Q5_0: return "Q5_0";
    case Weight_Dtype::Q5_1: return "Q5_1";
    case Weight_Dtype::Q8_0: return "Q8_0";
    case Weight_Dtype::Q4_K: return "Q4_K";
    case Weight_Dtype::Q6_K: return "Q6_K";
    default: return "UNKNOWN";
    }
}

Weight_Format detectWeightFormat(const std::filesystem::path& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
        throw std::runtime_error("weights::detectWeightFormat: path does not exist: '" + path.string() + "'");
    }
    if (std::filesystem::is_directory(path, ec))
    {
        if (std::filesystem::exists(path / "model.safetensors.index.json", ec) ||
            std::filesystem::exists(path / "model.safetensors", ec))
        {
            return Weight_Format::SAFETENSORS;
        }
        for (const auto& entry : std::filesystem::directory_iterator(path, ec))
        {
            if (entry.is_regular_file() && toLowerCopy(entry.path().extension().string()) == ".safetensors")
            {
                return Weight_Format::SAFETENSORS;
            }
        }
        throw std::runtime_error("weights::detectWeightFormat: no recognized weight files in directory '" + path.string() + "'");
    }

    std::ifstream in(path, std::ios::binary);
    std::array<char, 4> magic{};
    if (in.is_open() && in.read(magic.data(), 4))
    {
        if (magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F')
        {
            return Weight_Format::GGUF;
        }
        if (static_cast<unsigned char>(magic[0]) == 'P' &&
            static_cast<unsigned char>(magic[1]) == 'K' &&
            static_cast<unsigned char>(magic[2]) == 0x03 &&
            static_cast<unsigned char>(magic[3]) == 0x04)
        {
            return Weight_Format::PTH;
        }
    }

    const std::string ext = toLowerCopy(path.extension().string());
    if (ext == ".safetensors") return Weight_Format::SAFETENSORS;
    if (ext == ".gguf") return Weight_Format::GGUF;
    if (ext == ".onnx") return Weight_Format::ONNX;
    if (ext == ".pth" || ext == ".pt" || ext == ".bin") return Weight_Format::PTH;

    throw std::runtime_error("weights::detectWeightFormat: unsupported weight format for '" + path.string() + "'");
}

std::unique_ptr<IWeight_Source> openWeightSource(const std::filesystem::path& path, Weight_Format fmt)
{
    const Weight_Format resolved = (fmt == Weight_Format::AUTO) ? detectWeightFormat(path) : fmt;
    switch (resolved)
    {
    case Weight_Format::SAFETENSORS: return openSafetensors(path);
    case Weight_Format::GGUF: return openGguf(path);
    case Weight_Format::ONNX: return openOnnx(path);
    case Weight_Format::PTH: return openPth(path);
    default: throw std::runtime_error("weights::openWeightSource: unknown format");
    }
}

size_t loadNamedTensors(const IWeight_Source& source,
                        const std::unordered_map<std::string, Tensor*>& targets,
                        const std::unordered_map<std::string, bool>& transpose_2d_flags)
{
    size_t loaded = 0;
    for (const auto& [name, dst] : targets)
    {
        if (!dst || !source.hasTensor(name))
        {
            continue;
        }
        const auto info = source.getTensorInfo(name);
        std::vector<float> data = source.readTensorF32(name);
        bool do_transpose = false;
        if (auto it = transpose_2d_flags.find(name); it != transpose_2d_flags.end())
        {
            do_transpose = it->second;
        }
        if (do_transpose && info.shape.size() == 2)
        {
            data = transpose2D(data, info.shape[0], info.shape[1]);
        }
        if (data.size() != dst->getTotalElements())
        {
            throw std::runtime_error(std::format("loadNamedTensors: Element count mismatch for '{}' (expected {}, got {})",
                                                 name, dst->getTotalElements(), data.size()));
        }
        dst->uploadData(data);
        ++loaded;
    }
    return loaded;
}

Loaded_Causal_LM loadCausalLM(const std::filesystem::path& path, const Load_Model_Options& opts)
{
    const Weight_Format fmt = (opts.format == Weight_Format::AUTO) ? detectWeightFormat(path) : opts.format;
    auto source = openWeightSource(path, fmt);

    const std::filesystem::path cfg_json_path = findCompanionFile(path, opts.config_path, "config.json");
    const std::filesystem::path tok_json_path = findCompanionFile(path, opts.tokenizer_path, "tokenizer.json");

    Causal_LM_Config cfg;
    int32_t eos_from_cfg = -1;
    std::string arch;
    if (!cfg_json_path.empty())
    {
        cfg = buildConfigFromJson(cfg_json_path, *source, opts, eos_from_cfg, arch);
    }
    else if (fmt == Weight_Format::GGUF)
    {
        cfg = buildConfigFromMetadata(*source, opts, eos_from_cfg, arch);
    }
    else
    {
        throw std::runtime_error("loadCausalLM: Missing config.json for format " + weightFormatName(fmt));
    }

    if (!tok_json_path.empty())
    {
        cfg.tokenizer_path = tok_json_path.string();
    }
    else
    {
        cfg.tokenizer_path.clear();
    }

    Causal_LM model(cfg);

    if (tok_json_path.empty() && fmt == Weight_Format::GGUF)
    {
        const auto& m = source->metadata();
        const auto gguf_tokens = m.getStringArray("tokenizer.ggml.tokens");
        if (!gguf_tokens.empty())
        {
            const auto gguf_merges = m.getStringArray("tokenizer.ggml.merges");
            const auto gguf_ttypes_i64 = m.getIntArray("tokenizer.ggml.token_type");
            std::vector<int32_t> gguf_ttypes;
            gguf_ttypes.reserve(gguf_ttypes_i64.size());
            for (int64_t t : gguf_ttypes_i64)
            {
                gguf_ttypes.push_back(static_cast<int32_t>(t));
            }
            const std::string tok_model = m.getString("tokenizer.ggml.model", "gpt2");
            const bool is_byte_level = (tok_model == "gpt2" || tok_model == "bpe");
            const int32_t bos_id = m.has("tokenizer.ggml.bos_token_id")
                ? static_cast<int32_t>(m.getInt("tokenizer.ggml.bos_token_id", -1))
                : -1;
            const int32_t eos_id = m.has("tokenizer.ggml.eos_token_id")
                ? static_cast<int32_t>(m.getInt("tokenizer.ggml.eos_token_id", -1))
                : eos_from_cfg;
            const int32_t unk_id = m.has("tokenizer.ggml.unknown_token_id")
                ? static_cast<int32_t>(m.getInt("tokenizer.ggml.unknown_token_id", 0))
                : 0;
            const bool add_bos = m.getBool("tokenizer.ggml.add_bos_token", false);
            model.getTokenizer().loadFromVocabAndMerges(
                gguf_tokens, gguf_merges, gguf_ttypes, is_byte_level, bos_id, eos_id, unk_id, add_bos);
        }
    }

    {
        const std::string emb_name = resolveTensorName(*source, "model.embed_tokens.weight", opts.name_aliases, opts.strict_missing_tensors);
        if (!emb_name.empty())
        {
            const auto info = source->getTensorInfo(emb_name);
            if (info.shape.size() != 2 || info.shape[0] != cfg.vocab_size || info.shape[1] != cfg.hidden_dim)
            {
                throw std::runtime_error(std::format("loadCausalLM: '{}' shape mismatch (expected [{}, {}])",
                                                     emb_name, cfg.vocab_size, cfg.hidden_dim));
            }
            model.getTokenEmbedding().getWeight().uploadData(source->readTensorF32(emb_name));
        }
    }

    const size_t q_dim = cfg.num_heads * cfg.head_dim;
    const size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    auto& blocks = model.getBlocks();
    for (size_t i = 0; i < cfg.num_layers; ++i)
    {
        const std::string pfx = "model.layers." + std::to_string(i) + ".";
        Transformer_Block& blk = *blocks[i];

        auto loadReqNorm = [&](RMSNorm_Layer& layer, const std::string& cname, size_t dim) {
            const std::string actual = resolveTensorName(*source, cname, opts.name_aliases, opts.strict_missing_tensors);
            if (!actual.empty()) assignNormWeight(layer, *source, actual, dim);
        };
        auto loadReqLinear = [&](Linear_Layer& layer, const std::string& cname, size_t in_d, size_t out_d) {
            const std::string actual = resolveTensorName(*source, cname, opts.name_aliases, opts.strict_missing_tensors);
            if (!actual.empty()) assignLinearWeight(layer, *source, actual, in_d, out_d, opts.transpose_linear_weights);
        };
        auto loadOptBias = [&](Linear_Layer& layer, const std::string& cname, size_t out_d) {
            const std::string actual = resolveTensorName(*source, cname, opts.name_aliases, false);
            if (!actual.empty()) assignLinearBias(layer, *source, actual, out_d);
        };

        loadReqNorm(blk.getInputLayernorm(), pfx + "input_layernorm.weight", cfg.hidden_dim);
        loadReqLinear(blk.getQProj(), pfx + "self_attn.q_proj.weight", cfg.hidden_dim, q_dim);
        loadReqLinear(blk.getKProj(), pfx + "self_attn.k_proj.weight", cfg.hidden_dim, kv_dim);
        loadReqLinear(blk.getVProj(), pfx + "self_attn.v_proj.weight", cfg.hidden_dim, kv_dim);
        loadReqLinear(blk.getOProj(), pfx + "self_attn.o_proj.weight", q_dim, cfg.hidden_dim);

        if (cfg.attention_bias)
        {
            loadOptBias(blk.getQProj(), pfx + "self_attn.q_proj.bias", q_dim);
            loadOptBias(blk.getKProj(), pfx + "self_attn.k_proj.bias", kv_dim);
            loadOptBias(blk.getVProj(), pfx + "self_attn.v_proj.bias", kv_dim);
            loadOptBias(blk.getOProj(), pfx + "self_attn.o_proj.bias", cfg.hidden_dim);
        }

        if (cfg.use_qk_norm)
        {
            loadReqNorm(*blk.getQNorm(), pfx + "self_attn.q_norm.weight", cfg.head_dim);
            loadReqNorm(*blk.getKNorm(), pfx + "self_attn.k_norm.weight", cfg.head_dim);
        }

        loadReqNorm(blk.getPostAttentionLayernorm(), pfx + "post_attention_layernorm.weight", cfg.hidden_dim);
        loadReqLinear(blk.getGateProj(), pfx + "mlp.gate_proj.weight", cfg.hidden_dim, cfg.intermediate_dim);
        loadReqLinear(blk.getUpProj(), pfx + "mlp.up_proj.weight", cfg.hidden_dim, cfg.intermediate_dim);
        loadReqLinear(blk.getDownProj(), pfx + "mlp.down_proj.weight", cfg.intermediate_dim, cfg.hidden_dim);

        if (cfg.mlp_bias)
        {
            loadOptBias(blk.getGateProj(), pfx + "mlp.gate_proj.bias", cfg.intermediate_dim);
            loadOptBias(blk.getUpProj(), pfx + "mlp.up_proj.bias", cfg.intermediate_dim);
            loadOptBias(blk.getDownProj(), pfx + "mlp.down_proj.bias", cfg.hidden_dim);
        }
    }

    {
        const std::string fn_name = resolveTensorName(*source, "model.norm.weight", opts.name_aliases, opts.strict_missing_tensors);
        if (!fn_name.empty())
        {
            assignNormWeight(model.getFinalNorm(), *source, fn_name, cfg.hidden_dim);
        }
    }
    if (!cfg.tie_word_embeddings)
    {
        const std::string head_name = resolveTensorName(*source, "lm_head.weight", opts.name_aliases, opts.strict_missing_tensors);
        if (!head_name.empty())
        {
            assignLinearWeight(model.getLMHead(), *source, head_name, cfg.hidden_dim, cfg.vocab_size, opts.transpose_linear_weights);
        }
    }

    if (opts.target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    model.invalidateWeightCaches();
    if (opts.target == Execution_Target::VULKAN_GPU)
    {
        model.warmupWeightCaches();
    }
    model.setTrainingMode(false);
    return Loaded_Causal_LM{ std::move(model), eos_from_cfg, fmt, arch };
}

}
