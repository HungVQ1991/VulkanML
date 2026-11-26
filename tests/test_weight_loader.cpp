#include "test_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "helper/json.hpp"
#include "llm/sealed_read.h"
#include "llm/weight_loader.h"

namespace
{

void appendU16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}

void appendU32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void appendU64(std::vector<uint8_t>& out, uint64_t v)
{
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void appendVarint(std::vector<uint8_t>& out, uint64_t v)
{
    while (v >= 0x80ULL)
    {
        out.push_back(static_cast<uint8_t>((v & 0x7FULL) | 0x80ULL));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

void appendTag(std::vector<uint8_t>& out, uint32_t field, uint32_t wire_type)
{
    appendVarint(out, (static_cast<uint64_t>(field) << 3) | wire_type);
}

void appendBytesField(std::vector<uint8_t>& out, uint32_t field, const std::vector<uint8_t>& data)
{
    appendTag(out, field, 2);
    appendVarint(out, data.size());
    out.insert(out.end(), data.begin(), data.end());
}

void appendStringField(std::vector<uint8_t>& out, uint32_t field, const std::string& s)
{
    appendTag(out, field, 2);
    appendVarint(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

struct Tensor_Spec
{
    std::string name;
    std::vector<size_t> shape;
    std::vector<float> data;
};

void writeSafetensorsFile(const std::filesystem::path& path, const std::vector<Tensor_Spec>& tensors, const std::string& dtype_str)
{
    nlohmann::json header = nlohmann::json::object();
    header["__metadata__"] = { { "format", "pt" } };
    uint64_t off = 0;
    const uint64_t elem_b = (dtype_str == "F32") ? 4ULL : 2ULL;
    for (const auto& t : tensors)
    {
        const uint64_t bytes = static_cast<uint64_t>(t.data.size()) * elem_b;
        header[t.name] = {
            { "dtype", dtype_str },
            { "shape", t.shape },
            { "data_offsets", { off, off + bytes } }
        };
        off += bytes;
    }
    std::string hdr = header.dump();
    while ((hdr.size() % 8) != 0) hdr.push_back(' ');
    const uint64_t hlen = hdr.size();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&hlen), 8);
    out.write(hdr.data(), static_cast<std::streamsize>(hlen));
    for (const auto& t : tensors)
    {
        if (dtype_str == "F32")
        {
            out.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size() * 4));
        }
        else if (dtype_str == "F16")
        {
            std::vector<uint16_t> raw(t.data.size());
            for (size_t i = 0; i < t.data.size(); ++i) raw[i] = weights::f32ToF16(t.data[i]);
            out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 2));
        }
        else
        {
            std::vector<uint16_t> raw(t.data.size());
            for (size_t i = 0; i < t.data.size(); ++i) raw[i] = weights::f32ToBf16(t.data[i]);
            out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 2));
        }
    }
}

struct Gguf_Raw_Tensor
{
    std::string name;
    std::vector<size_t> shape;
    uint32_t ggml_type = 0;
    std::vector<uint8_t> raw_bytes;
};

void writeGgufFile(const std::filesystem::path& path,
                   const std::vector<std::tuple<std::string, uint32_t, std::vector<uint8_t>>>& kvs,
                   const std::vector<Gguf_Raw_Tensor>& tensors)
{
    std::vector<uint8_t> hdr;
    appendU32(hdr, 0x46554747u);
    appendU32(hdr, 3u);
    appendU64(hdr, tensors.size());
    appendU64(hdr, kvs.size());

    auto pushGgufStr = [&](const std::string& s) {
        appendU64(hdr, s.size());
        hdr.insert(hdr.end(), s.begin(), s.end());
    };

    for (const auto& [k, t, val] : kvs)
    {
        pushGgufStr(k);
        appendU32(hdr, t);
        hdr.insert(hdr.end(), val.begin(), val.end());
    }

    uint64_t rel_off = 0;
    for (const auto& ten : tensors)
    {
        pushGgufStr(ten.name);
        appendU32(hdr, static_cast<uint32_t>(ten.shape.size()));
        for (size_t d = ten.shape.size(); d > 0; --d)
        {
            appendU64(hdr, ten.shape[d - 1]);
        }
        appendU32(hdr, ten.ggml_type);
        appendU64(hdr, rel_off);
        rel_off += ten.raw_bytes.size();
    }

    while ((hdr.size() % 32) != 0) hdr.push_back(0);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
    for (const auto& ten : tensors)
    {
        out.write(reinterpret_cast<const char*>(ten.raw_bytes.data()), static_cast<std::streamsize>(ten.raw_bytes.size()));
    }
}

std::vector<uint8_t> makeGgufStringVal(const std::string& s)
{
    std::vector<uint8_t> v;
    appendU64(v, s.size());
    v.insert(v.end(), s.begin(), s.end());
    return v;
}

std::vector<uint8_t> makeGgufU32Val(uint32_t x)
{
    std::vector<uint8_t> v;
    appendU32(v, x);
    return v;
}

std::vector<uint8_t> makeGgufF32Val(float x)
{
    std::vector<uint8_t> v(4);
    std::memcpy(v.data(), &x, 4);
    return v;
}

std::vector<uint8_t> makeGgufStringArrayVal(const std::vector<std::string>& arr)
{
    std::vector<uint8_t> v;
    appendU32(v, 8u);
    appendU64(v, arr.size());
    for (const auto& s : arr)
    {
        appendU64(v, s.size());
        v.insert(v.end(), s.begin(), s.end());
    }
    return v;
}

std::vector<uint8_t> makeGgufI32ArrayVal(const std::vector<int32_t>& arr)
{
    std::vector<uint8_t> v;
    appendU32(v, 5u);
    appendU64(v, arr.size());
    for (int32_t x : arr)
    {
        appendU32(v, static_cast<uint32_t>(x));
    }
    return v;
}

std::vector<uint8_t> encodeZipArchive(const std::vector<std::tuple<std::string, std::vector<uint8_t>, uint16_t>>& files)
{
    std::vector<uint8_t> archive;
    std::vector<uint8_t> cd;
    for (const auto& [name, content, method] : files)
    {
        std::vector<uint8_t> payload;
        if (method == 0)
        {
            payload = content;
        }
        else
        {
            if (content.size() > 65535) throw std::runtime_error("test DEFLATE helper only for < 64KB");
            payload.push_back(0x01u);
            const uint16_t len = static_cast<uint16_t>(content.size());
            const uint16_t nlen = static_cast<uint16_t>(len ^ 0xFFFFu);
            appendU16(payload, len);
            appendU16(payload, nlen);
            payload.insert(payload.end(), content.begin(), content.end());
        }
        const uint32_t lfh_off = static_cast<uint32_t>(archive.size());
        appendU32(archive, 0x04034b50u);
        appendU16(archive, 20u);
        appendU16(archive, 0u);
        appendU16(archive, method);
        appendU16(archive, 0u);
        appendU16(archive, 0u);
        appendU32(archive, 0u);
        appendU32(archive, static_cast<uint32_t>(payload.size()));
        appendU32(archive, static_cast<uint32_t>(content.size()));
        appendU16(archive, static_cast<uint16_t>(name.size()));
        appendU16(archive, 0u);
        archive.insert(archive.end(), name.begin(), name.end());
        archive.insert(archive.end(), payload.begin(), payload.end());

        appendU32(cd, 0x02014b50u);
        appendU16(cd, 20u);
        appendU16(cd, 20u);
        appendU16(cd, 0u);
        appendU16(cd, method);
        appendU16(cd, 0u);
        appendU16(cd, 0u);
        appendU32(cd, 0u);
        appendU32(cd, static_cast<uint32_t>(payload.size()));
        appendU32(cd, static_cast<uint32_t>(content.size()));
        appendU16(cd, static_cast<uint16_t>(name.size()));
        appendU16(cd, 0u);
        appendU16(cd, 0u);
        appendU16(cd, 0u);
        appendU16(cd, 0u);
        appendU32(cd, 0u);
        appendU32(cd, lfh_off);
        cd.insert(cd.end(), name.begin(), name.end());
    }
    const uint32_t cd_off = static_cast<uint32_t>(archive.size());
    archive.insert(archive.end(), cd.begin(), cd.end());
    appendU32(archive, 0x06054b50u);
    appendU16(archive, 0u);
    appendU16(archive, 0u);
    appendU16(archive, static_cast<uint16_t>(files.size()));
    appendU16(archive, static_cast<uint16_t>(files.size()));
    appendU32(archive, static_cast<uint32_t>(cd.size()));
    appendU32(archive, cd_off);
    appendU16(archive, 0u);
    return archive;
}

std::vector<uint8_t> buildStateDictPickle(const std::vector<std::tuple<std::string, std::string, std::string, size_t, size_t, std::vector<size_t>, std::vector<int64_t>>>& entries)
{
    std::vector<uint8_t> p;
    p.push_back(0x80u);
    p.push_back(2u);
    p.push_back('}');
    p.push_back('(');

    auto pushStr = [&](const std::string& s) {
        p.push_back('U');
        p.push_back(static_cast<uint8_t>(s.size()));
        p.insert(p.end(), s.begin(), s.end());
    };
    auto pushInt = [&](int64_t v) {
        p.push_back('J');
        appendU32(p, static_cast<uint32_t>(v));
    };

    for (const auto& [tname, storage_cls, key, storage_off, storage_len, shape, stride] : entries)
    {
        pushStr(tname);

        p.push_back('c');
        const std::string rb = "torch._utils\n_rebuild_tensor_v2\n";
        p.insert(p.end(), rb.begin(), rb.end());

        p.push_back('(');
        p.push_back('(');
        pushStr("storage");
        p.push_back('c');
        const std::string st = "torch\n" + storage_cls + "\n";
        p.insert(p.end(), st.begin(), st.end());
        pushStr(key);
        pushStr("cpu");
        pushInt(static_cast<int64_t>(storage_len));
        p.push_back('t');
        p.push_back('Q');

        pushInt(static_cast<int64_t>(storage_off));

        p.push_back('(');
        for (size_t d : shape) pushInt(static_cast<int64_t>(d));
        p.push_back('t');

        p.push_back('(');
        for (int64_t s : stride) pushInt(s);
        p.push_back('t');

        p.push_back(0x89u);
        p.push_back('}');
        p.push_back('t');
        p.push_back('R');
    }
    p.push_back('u');
    p.push_back('.');
    return p;
}

std::vector<uint8_t> buildOnnxTensorProto(const Tensor_Spec& t,
                                          int32_t onnx_dt,
                                          bool use_external,
                                          const std::string& ext_loc,
                                          uint64_t ext_off,
                                          uint64_t ext_len)
{
    std::vector<uint8_t> tp;
    for (size_t d : t.shape)
    {
        appendTag(tp, 1, 0);
        appendVarint(tp, d);
    }
    appendTag(tp, 2, 0);
    appendVarint(tp, static_cast<uint64_t>(onnx_dt));
    appendStringField(tp, 8, t.name);
    if (use_external)
    {
        auto addKv = [&](const std::string& k, const std::string& v) {
            std::vector<uint8_t> kv;
            appendStringField(kv, 1, k);
            appendStringField(kv, 2, v);
            appendBytesField(tp, 13, kv);
        };
        addKv("location", ext_loc);
        addKv("offset", std::to_string(ext_off));
        addKv("length", std::to_string(ext_len));
        appendTag(tp, 14, 0);
        appendVarint(tp, 1);
    }
    else
    {
        std::vector<uint8_t> raw;
        if (onnx_dt == 1)
        {
            raw.resize(t.data.size() * 4);
            std::memcpy(raw.data(), t.data.data(), raw.size());
        }
        else if (onnx_dt == 10)
        {
            raw.resize(t.data.size() * 2);
            for (size_t i = 0; i < t.data.size(); ++i)
            {
                const uint16_t h = weights::f32ToF16(t.data[i]);
                std::memcpy(raw.data() + i * 2, &h, 2);
            }
        }
        else if (onnx_dt == 16)
        {
            raw.resize(t.data.size() * 2);
            for (size_t i = 0; i < t.data.size(); ++i)
            {
                const uint16_t b = weights::f32ToBf16(t.data[i]);
                std::memcpy(raw.data() + i * 2, &b, 2);
            }
        }
        appendBytesField(tp, 9, raw);
    }
    return tp;
}

void writeOnnxFile(const std::filesystem::path& path,
                   const std::vector<std::vector<uint8_t>>& init_protos)
{
    std::vector<uint8_t> graph;
    for (const auto& tp : init_protos)
    {
        appendBytesField(graph, 5, tp);
    }
    std::vector<uint8_t> model;
    appendTag(model, 1, 0);
    appendVarint(model, 9);
    appendStringField(model, 4, "vulkanml_test");
    appendBytesField(model, 7, graph);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
}

bool testSafetensorsSingleAndSharded()
{
    const char* tag = "WL-Safetensors";
    const auto dir = makeUniqueTestTempPath("wl_st", "");
    Scoped_Temp_File_Guard guard({}, { dir });
    std::filesystem::create_directories(dir);

    const std::vector<float> v1 = { 1.0f, -2.5f, 3.25f, 0.5f };
    const std::vector<float> v2 = { -0.75f, 4.0f, 1.5f, -3.0f };
    writeSafetensorsFile(dir / "model-00001-of-00002.safetensors", { { "w_f16", { 2, 2 }, v1 } }, "F16");
    writeSafetensorsFile(dir / "model-00002-of-00002.safetensors", { { "w_bf16", { 4 }, v2 } }, "BF16");

    nlohmann::json idx = {
        { "weight_map", {
            { "w_f16", "model-00001-of-00002.safetensors" },
            { "w_bf16", "model-00002-of-00002.safetensors" }
        } }
    };
    {
        std::ofstream out(dir / "model.safetensors.index.json");
        out << idx.dump(2);
    }

    auto src = weights::openWeightSource(dir);
    if (src->format() != weights::Weight_Format::SAFETENSORS || !src->hasTensor("w_f16") || !src->hasTensor("w_bf16"))
    {
        return failTestStep(tag, "Sharded safetensors discovery failed");
    }
    if (!checkVectorsClose(src->readTensorF32("w_f16"), v1, 1e-3f, tag, "F16 mismatch") ||
        !checkVectorsClose(src->readTensorF32("w_bf16"), v2, 1e-2f, tag, "BF16 mismatch"))
    {
        return false;
    }
    return true;
}

bool testGgufParserAndDequant()
{
    const char* tag = "WL-GGUF";
    const auto path = makeUniqueTestTempPath("wl_gguf", ".gguf");
    Scoped_Temp_File_Guard guard({ path.string() });

    const std::vector<std::string> vocab = { "!", "H", "e", "l", "o", "He", "llo", "<|im_start|>", "<|im_end|>" };
    const std::vector<std::string> merges = { "H e", "l l", "ll o" };
    const std::vector<int32_t> ttypes = { 1, 1, 1, 1, 1, 1, 1, 3, 3 };

    const float scale = 0.25f;
    const uint16_t d16 = weights::f32ToF16(scale);

    std::vector<uint8_t> q8_bytes(34);
    std::memcpy(q8_bytes.data(), &d16, 2);
    std::vector<float> q8_exp(32);
    for (int i = 0; i < 32; ++i)
    {
        const int8_t q = static_cast<int8_t>(i - 16);
        q8_bytes[2 + i] = static_cast<uint8_t>(q);
        q8_exp[i] = scale * static_cast<float>(q);
    }

    std::vector<uint8_t> q4_0_bytes(18);
    std::memcpy(q4_0_bytes.data(), &d16, 2);
    std::vector<float> q4_0_exp(32);
    for (size_t j = 0; j < 16; ++j)
    {
        const uint8_t lo = static_cast<uint8_t>(j % 16);
        const uint8_t hi = static_cast<uint8_t>((j + 3) % 16);
        q4_0_bytes[2 + j] = static_cast<uint8_t>(lo | (hi << 4));
        q4_0_exp[j] = scale * static_cast<float>(static_cast<int>(lo) - 8);
        q4_0_exp[j + 16] = scale * static_cast<float>(static_cast<int>(hi) - 8);
    }

    std::vector<uint8_t> q4k_bytes(144, 0u);
    const uint16_t d16_1 = weights::f32ToF16(1.0f);
    const uint16_t dm16_0 = weights::f32ToF16(0.5f);
    std::memcpy(q4k_bytes.data(), &d16_1, 2);
    std::memcpy(q4k_bytes.data() + 2, &dm16_0, 2);
    for (int i = 0; i < 8; ++i) q4k_bytes[4 + i] = 2u;
    for (int i = 8; i < 12; ++i) q4k_bytes[4 + i] = 0x22u;
    for (int i = 0; i < 128; ++i) q4k_bytes[16 + i] = 0x35u;
    std::vector<float> q4k_exp(256);
    for (int blk = 0; blk < 256; blk += 64)
    {
        for (int l = 0; l < 32; ++l) q4k_exp[blk + l] = 2.0f * 5.0f - 1.0f;
        for (int l = 0; l < 32; ++l) q4k_exp[blk + 32 + l] = 2.0f * 3.0f - 1.0f;
    }

    std::vector<uint8_t> q6k_bytes(210, 0u);
    std::memcpy(q6k_bytes.data() + 208, &d16_1, 2);
    std::fill(q6k_bytes.data() + 192, q6k_bytes.data() + 208, static_cast<uint8_t>(2));
    std::vector<float> q6k_exp(256, 2.0f * (-32.0f));

    writeGgufFile(path,
        {
            { "general.architecture", 8u, makeGgufStringVal("qwen3") },
            { "tokenizer.ggml.model", 8u, makeGgufStringVal("gpt2") },
            { "tokenizer.ggml.tokens", 9u, makeGgufStringArrayVal(vocab) },
            { "tokenizer.ggml.merges", 9u, makeGgufStringArrayVal(merges) },
            { "tokenizer.ggml.token_type", 9u, makeGgufI32ArrayVal(ttypes) },
            { "tokenizer.ggml.eos_token_id", 4u, makeGgufU32Val(8u) }
        },
        {
            { "t_q8", { 32 }, 8u, q8_bytes },
            { "t_q4_0", { 32 }, 2u, q4_0_bytes },
            { "t_q4_k", { 256 }, 12u, q4k_bytes },
            { "t_q6_k", { 256 }, 14u, q6k_bytes }
        });

    auto src = weights::openWeightSource(path);
    if (src->format() != weights::Weight_Format::GGUF ||
        src->metadata().getString("general.architecture") != "qwen3")
    {
        return failTestStep(tag, "GGUF metadata parsing failed");
    }
    if (!checkVectorsClose(src->readTensorF32("t_q8"), q8_exp, 1e-3f, tag, "Q8_0 dequant mismatch") ||
        !checkVectorsClose(src->readTensorF32("t_q4_0"), q4_0_exp, 1e-3f, tag, "Q4_0 dequant mismatch") ||
        !checkVectorsClose(src->readTensorF32("t_q4_k"), q4k_exp, 1e-2f, tag, "Q4_K dequant mismatch") ||
        !checkVectorsClose(src->readTensorF32("t_q6_k"), q6k_exp, 1e-2f, tag, "Q6_K dequant mismatch"))
    {
        return false;
    }

    Bpe_Tokenizer bpe;
    std::vector<int32_t> tt_conv;
    for (int64_t x : src->metadata().getIntArray("tokenizer.ggml.token_type")) tt_conv.push_back(static_cast<int32_t>(x));
    if (!bpe.loadFromVocabAndMerges(
            src->metadata().getStringArray("tokenizer.ggml.tokens"),
            src->metadata().getStringArray("tokenizer.ggml.merges"),
            tt_conv, true, -1, 8, 0, false))
    {
        return failTestStep(tag, "Bpe_Tokenizer::loadFromVocabAndMerges failed");
    }
    const auto ids = bpe.encode("<|im_start|>Hello<|im_end|>", false);
    if (ids.empty() || ids.front() != 7 || ids.back() != 8 ||
        bpe.decode(ids, false, true) != "<|im_start|>Hello<|im_end|>")
    {
        return failTestStep(tag, "GGUF embedded tokenizer round-trip failed");
    }
    return true;
}

bool testOnnxInlineAndExternalData()
{
    const char* tag = "WL-ONNX";
    const auto dir = makeUniqueTestTempPath("wl_onnx", "");
    Scoped_Temp_File_Guard guard({}, { dir });
    std::filesystem::create_directories(dir);

    const std::vector<float> w1 = { 0.5f, -1.25f, 2.0f, 3.5f };
    const std::vector<float> w2 = { -4.0f, 0.25f, 1.5f, -0.5f };
    {
        std::ofstream ext(dir / "weights.onnx_data", std::ios::binary | std::ios::trunc);
        const std::vector<uint8_t> pad(16, 0xAAu);
        ext.write(reinterpret_cast<const char*>(pad.data()), 16);
        ext.write(reinterpret_cast<const char*>(w2.data()), 16);
    }

    writeOnnxFile(dir / "model.onnx", {
        buildOnnxTensorProto({ "inline_f16", { 2, 2 }, w1 }, 10, false, "", 0, 0),
        buildOnnxTensorProto({ "ext_f32", { 4 }, w2 }, 1, true, "weights.onnx_data", 16, 16)
    });

    auto src = weights::openWeightSource(dir / "model.onnx");
    if (src->format() != weights::Weight_Format::ONNX ||
        !checkVectorsClose(src->readTensorF32("inline_f16"), w1, 1e-3f, tag, "ONNX inline F16 mismatch") ||
        !checkVectorsClose(src->readTensorF32("ext_f32"), w2, 1e-5f, tag, "ONNX external F32 mismatch"))
    {
        return false;
    }

    writeOnnxFile(dir / "bad.onnx", {
        buildOnnxTensorProto({ "bad_ext", { 4 }, w2 }, 1, true, "../escape.bin", 0, 16)
    });
    if (!expectThrows([&]() { (void)weights::openWeightSource(dir / "bad.onnx"); }))
    {
        return failTestStep(tag, "ONNX failed to block external_data path traversal");
    }
    return true;
}

bool testPthZipAndSafePickle()
{
    const char* tag = "WL-PTH";
    const auto pth_path = makeUniqueTestTempPath("wl_pth", ".pth");
    const auto bad_path = makeUniqueTestTempPath("wl_pth_bad", ".pth");
    Scoped_Temp_File_Guard guard({ pth_path.string(), bad_path.string() });

    const std::vector<float> f32_all = { 99.0f, 1.5f, -2.0f, 3.0f, 4.5f };
    std::vector<uint8_t> f32_bytes(f32_all.size() * 4);
    std::memcpy(f32_bytes.data(), f32_all.data(), f32_bytes.size());

    const std::vector<float> f16_orig = { -1.0f, 2.0f, 0.5f, -3.5f };
    std::vector<uint8_t> f16_bytes(f16_orig.size() * 2);
    for (size_t i = 0; i < f16_orig.size(); ++i)
    {
        const uint16_t h = weights::f32ToF16(f16_orig[i]);
        std::memcpy(f16_bytes.data() + i * 2, &h, 2);
    }

    const auto pkl = buildStateDictPickle({
        { "t_offset", "FloatStorage", "0", 1, 5, { 2, 2 }, { 2, 1 } },
        { "t_deflate_f16", "HalfStorage", "1", 0, 4, { 2, 2 }, { 1, 2 } }
    });
    const auto zip_bytes = encodeZipArchive({
        { "archive/data.pkl", pkl, 0 },
        { "archive/data/0", f32_bytes, 0 },
        { "archive/data/1", f16_bytes, 8 }
    });
    {
        std::ofstream out(pth_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(zip_bytes.data()), static_cast<std::streamsize>(zip_bytes.size()));
    }

    auto src = weights::openWeightSource(pth_path);
    const std::vector<float> exp_off = { 1.5f, -2.0f, 3.0f, 4.5f };
    const std::vector<float> exp_tr = { -1.0f, 0.5f, 2.0f, -3.5f };
    if (src->format() != weights::Weight_Format::PTH ||
        !checkVectorsClose(src->readTensorF32("t_offset"), exp_off, 1e-5f, tag, "PTH offset read mismatch") ||
        !checkVectorsClose(src->readTensorF32("t_deflate_f16"), exp_tr, 1e-3f, tag, "PTH DEFLATE strided F16 mismatch"))
    {
        return false;
    }

    std::vector<uint8_t> unsafe_pkl = { 0x80u, 2u, 'c' };
    const std::string bad_g = "os\nsystem\n";
    unsafe_pkl.insert(unsafe_pkl.end(), bad_g.begin(), bad_g.end());
    unsafe_pkl.push_back('.');
    const auto bad_zip = encodeZipArchive({ { "archive/data.pkl", unsafe_pkl, 0 } });
    {
        std::ofstream out(bad_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bad_zip.data()), static_cast<std::streamsize>(bad_zip.size()));
    }
    if (!expectThrows([&]() { (void)weights::openWeightSource(bad_path); }))
    {
        return failTestStep(tag, "Safe Pickle VM failed to reject unsafe global");
    }
    return true;
}

bool testCrossFormatCausalLMParity()
{
    const char* tag = "WL-Parity";
    const auto root_dir = makeUniqueTestTempPath("wl_parity", "");
    Scoped_Temp_File_Guard guard({}, { root_dir });
    const auto st_dir = root_dir / "st_model";
    const auto onnx_dir = root_dir / "onnx_model";
    const auto pth_dir = root_dir / "pth_model";
    std::filesystem::create_directories(st_dir);
    std::filesystem::create_directories(onnx_dir);
    std::filesystem::create_directories(pth_dir);

    const size_t vocab = 32;
    const size_t hidden = 32;
    const size_t heads = 4;
    const size_t head_dim = 16;
    const size_t kv_heads = 2;
    const size_t inter = 48;
    const size_t layers = 2;
    const size_t q_dim = heads * head_dim;
    const size_t kv_dim = kv_heads * head_dim;

    nlohmann::json cfg_json = {
        { "model_type", "qwen3" },
        { "vocab_size", vocab },
        { "hidden_size", hidden },
        { "num_attention_heads", heads },
        { "num_key_value_heads", kv_heads },
        { "head_dim", head_dim },
        { "intermediate_size", inter },
        { "num_hidden_layers", layers },
        { "max_position_embeddings", 64 },
        { "rms_norm_eps", 1e-6 },
        { "rope_theta", 1000000.0 },
        { "tie_word_embeddings", true },
        { "attention_bias", false },
        { "mlp_bias", false },
        { "eos_token_id", 28 }
    };
    const std::string cfg_str = cfg_json.dump(2);
    for (const auto& d : { st_dir, onnx_dir, pth_dir })
    {
        std::ofstream out(d / "config.json");
        out << cfg_str;
    }

    auto det = [](size_t n, uint32_t seed) {
        return generateRandomVector(n, -0.25f, 0.25f, seed);
    };
    std::vector<Tensor_Spec> specs;
    uint32_t seed = 31337u;
    specs.push_back({ "model.embed_tokens.weight", { vocab, hidden }, det(vocab * hidden, seed++) });
    for (size_t l = 0; l < layers; ++l)
    {
        const std::string pfx = "model.layers." + std::to_string(l) + ".";
        specs.push_back({ pfx + "input_layernorm.weight", { hidden }, std::vector<float>(hidden, 1.0f) });
        specs.push_back({ pfx + "self_attn.q_proj.weight", { q_dim, hidden }, det(q_dim * hidden, seed++) });
        specs.push_back({ pfx + "self_attn.k_proj.weight", { kv_dim, hidden }, det(kv_dim * hidden, seed++) });
        specs.push_back({ pfx + "self_attn.v_proj.weight", { kv_dim, hidden }, det(kv_dim * hidden, seed++) });
        specs.push_back({ pfx + "self_attn.o_proj.weight", { hidden, q_dim }, det(hidden * q_dim, seed++) });
        specs.push_back({ pfx + "self_attn.q_norm.weight", { head_dim }, std::vector<float>(head_dim, 1.0f) });
        specs.push_back({ pfx + "self_attn.k_norm.weight", { head_dim }, std::vector<float>(head_dim, 1.0f) });
        specs.push_back({ pfx + "post_attention_layernorm.weight", { hidden }, std::vector<float>(hidden, 1.0f) });
        specs.push_back({ pfx + "mlp.gate_proj.weight", { inter, hidden }, det(inter * hidden, seed++) });
        specs.push_back({ pfx + "mlp.up_proj.weight", { inter, hidden }, det(inter * hidden, seed++) });
        specs.push_back({ pfx + "mlp.down_proj.weight", { hidden, inter }, det(hidden * inter, seed++) });
    }
    specs.push_back({ "model.norm.weight", { hidden }, std::vector<float>(hidden, 1.0f) });

    writeSafetensorsFile(st_dir / "model.safetensors", specs, "F32");

    std::vector<std::vector<uint8_t>> onnx_inits;
    for (const auto& sp : specs)
    {
        onnx_inits.push_back(buildOnnxTensorProto(sp, 1, false, "", 0, 0));
    }
    writeOnnxFile(onnx_dir / "model.onnx", onnx_inits);

    std::vector<std::tuple<std::string, std::string, std::string, size_t, size_t, std::vector<size_t>, std::vector<int64_t>>> pkl_entries;
    std::vector<std::tuple<std::string, std::vector<uint8_t>, uint16_t>> zip_files;
    for (size_t i = 0; i < specs.size(); ++i)
    {
        const auto& sp = specs[i];
        const std::string key = std::to_string(i);
        std::vector<int64_t> stride(sp.shape.size(), 1);
        if (sp.shape.size() == 2) stride[0] = static_cast<int64_t>(sp.shape[1]);
        pkl_entries.emplace_back(sp.name, "FloatStorage", key, 0, sp.data.size(), sp.shape, stride);
        std::vector<uint8_t> raw(sp.data.size() * 4);
        std::memcpy(raw.data(), sp.data.data(), raw.size());
        zip_files.emplace_back("archive/data/" + key, std::move(raw), 0);
    }
    zip_files.insert(zip_files.begin(), { "archive/data.pkl", buildStateDictPickle(pkl_entries), 0 });
    {
        const auto zb = encodeZipArchive(zip_files);
        std::ofstream out(pth_dir / "model.pth", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(zb.data()), static_cast<std::streamsize>(zb.size()));
    }

    auto toGgufName = [](const std::string& hf) -> std::string {
        if (hf == "model.embed_tokens.weight") return "token_embd.weight";
        if (hf == "model.norm.weight") return "output_norm.weight";
        const size_t dot = hf.find('.', 13);
        const std::string idx = hf.substr(13, dot - 13);
        const std::string suf = hf.substr(dot + 1);
        const std::string b = "blk." + idx + ".";
        if (suf == "input_layernorm.weight") return b + "attn_norm.weight";
        if (suf == "self_attn.q_proj.weight") return b + "attn_q.weight";
        if (suf == "self_attn.k_proj.weight") return b + "attn_k.weight";
        if (suf == "self_attn.v_proj.weight") return b + "attn_v.weight";
        if (suf == "self_attn.o_proj.weight") return b + "attn_output.weight";
        if (suf == "self_attn.q_norm.weight") return b + "attn_q_norm.weight";
        if (suf == "self_attn.k_norm.weight") return b + "attn_k_norm.weight";
        if (suf == "post_attention_layernorm.weight") return b + "ffn_norm.weight";
        if (suf == "mlp.gate_proj.weight") return b + "ffn_gate.weight";
        if (suf == "mlp.up_proj.weight") return b + "ffn_up.weight";
        if (suf == "mlp.down_proj.weight") return b + "ffn_down.weight";
        return hf;
    };
    std::vector<Gguf_Raw_Tensor> gguf_tensors;
    for (const auto& sp : specs)
    {
        std::vector<uint8_t> raw(sp.data.size() * 4);
        std::memcpy(raw.data(), sp.data.data(), raw.size());
        gguf_tensors.push_back(Gguf_Raw_Tensor{ toGgufName(sp.name), sp.shape, 0u, std::move(raw) });
    }
    std::vector<std::string> vocab_strs;
    for (size_t i = 0; i < vocab; ++i) vocab_strs.push_back("tok_" + std::to_string(i));
    vocab_strs[28] = "<|im_end|>";
    vocab_strs[29] = "<|im_start|>";
    const auto gguf_path = root_dir / "qwen3_tiny.gguf";
    writeGgufFile(gguf_path,
        {
            { "general.architecture", 8u, makeGgufStringVal("qwen3") },
            { "qwen3.block_count", 4u, makeGgufU32Val(static_cast<uint32_t>(layers)) },
            { "qwen3.embedding_length", 4u, makeGgufU32Val(static_cast<uint32_t>(hidden)) },
            { "qwen3.attention.head_count", 4u, makeGgufU32Val(static_cast<uint32_t>(heads)) },
            { "qwen3.attention.head_count_kv", 4u, makeGgufU32Val(static_cast<uint32_t>(kv_heads)) },
            { "qwen3.attention.key_length", 4u, makeGgufU32Val(static_cast<uint32_t>(head_dim)) },
            { "qwen3.feed_forward_length", 4u, makeGgufU32Val(static_cast<uint32_t>(inter)) },
            { "qwen3.context_length", 4u, makeGgufU32Val(64u) },
            { "qwen3.attention.layer_norm_rms_epsilon", 6u, makeGgufF32Val(1e-6f) },
            { "qwen3.rope.freq_base", 6u, makeGgufF32Val(1000000.0f) },
            { "tokenizer.ggml.model", 8u, makeGgufStringVal("gpt2") },
            { "tokenizer.ggml.tokens", 9u, makeGgufStringArrayVal(vocab_strs) },
            { "tokenizer.ggml.eos_token_id", 4u, makeGgufU32Val(28u) }
        },
        gguf_tensors);

    const std::vector<int32_t> prompt = { 2, 5, 11, 17, 23 };
    auto evalFormat = [&](const std::filesystem::path& p, Execution_Target target) {
        weights::Load_Model_Options o;
        o.target = target;
        o.data_type = Data_Type::FLOAT32;
        auto b = weights::loadCausalLM(p, o);
        return b.model.forward(prompt, false).getData();
    };

    const auto ref_st = evalFormat(st_dir, Execution_Target::CPU);
    if (!checkVectorsClose(ref_st, evalFormat(gguf_path, Execution_Target::CPU), 1e-5f, tag, "GGUF vs Safetensors CPU mismatch") ||
        !checkVectorsClose(ref_st, evalFormat(onnx_dir / "model.onnx", Execution_Target::CPU), 1e-5f, tag, "ONNX vs Safetensors CPU mismatch") ||
        !checkVectorsClose(ref_st, evalFormat(pth_dir / "model.pth", Execution_Target::CPU), 1e-5f, tag, "PTH vs Safetensors CPU mismatch"))
    {
        return false;
    }

    if (hasVulkanDevice())
    {
        Scoped_Coop_Mat_Guard coop;
        Execution_Engine::getInstance().setCooperativeMatrixEnabled(false);
        const auto gpu_gguf = evalFormat(gguf_path, Execution_Target::VULKAN_GPU);
        if (!checkVectorsClose(ref_st, gpu_gguf, 2e-3f, tag, "GGUF GPU vs CPU mismatch"))
        {
            return false;
        }
    }

    weights::Load_Model_Options sr_opts;
    sr_opts.target = Execution_Target::CPU;
    auto gguf_bundle = weights::loadCausalLM(gguf_path, sr_opts);
    sealedread::Sample s;
    s.qid = "q_wl";
    s.question = "Where was Alice born?";
    s.answers = { "Paris" };
    s.segments = {
        sealedread::Segment{ 101, "D1", "Intro", "Bob lives in Rome.", "D1", false },
        sealedread::Segment{ 205, "D2", "Intro", "Alice was born in Paris.", "D2", true }
    };
    s.hops = { sealedread::Hop{ { 205 }, "Paris" } };
    const sealedread::Proto_Config pcfg;
    const sealedread::Prompts prompts;
    const sealedread::Byte_Tokenizer btok;
    sealedread::Scripted_Policy sp(0.0f, 0.0f, 1);
    const auto eps = sealedread::runEpisodes(sp, { s }, 1, sealedread::Driver_Config{ pcfg, "model", false, false, 32 }, 1);
    const auto seq_test = sealedread::buildSeq(s, eps[0].steps, btok, prompts, sealedread::View_Mode::CHAIN, sealedread::Branch::SEALEDREAD);
    (void)seq_test;

    Causal_LM& gguf_lm = gguf_bundle.model;
    const std::vector<sealedread::Step> short_steps = {
        sealedread::Step{ "reason", "need", {}, "model", {} },
        sealedread::Step{ "locate", "", { 205 }, "model", {} },
        sealedread::Step{ "knowledge", "", { 205 }, "engine", {} },
        sealedread::Step{ "answer", "tok_3", {}, "model", {} }
    };
    class Tiny_Token_Tok final : public sealedread::ISealed_Read_Tokenizer
    {
    public:
        std::string name() const override { return "tiny32"; }
        size_t vocabSize() const noexcept override { return 32; }
        std::string eosText() const override { return "<|im_end|>"; }
        std::vector<int32_t> encode(const std::string& text) const override
        {
            std::vector<int32_t> out;
            for (unsigned char c : text) out.push_back(static_cast<int32_t>(c % 27));
            return out;
        }
        std::string decode(const std::vector<int32_t>&) const override { return {}; }
        std::unordered_set<int32_t> eosIds() const override { return { 28 }; }
    } tiny_tok;
    sealedread::Prompts compact_prompts;
    compact_prompts.instruction = "Q";
    compact_prompts.system_prompt = "S";
    const auto inv = sealedread::probeInvariance(
        gguf_lm, s, short_steps, tiny_tok, compact_prompts,
        sealedread::Mask_Opts{ sealedread::Branch::SEALEDREAD, true, sealedread::Locate_History::ALL }, 0);
    if (inv.positions == 0 || inv.kl_max > 1e-6f)
    {
        return failTestStep(tag, "Loaded GGUF model failed SealedRead Property 1 invariance");
    }
    return true;
}

}

void runWeightLoaderSuite(Test_Result_Counter& counter)
{
    counter.record("  Safetensors Single & Sharded:     ", testSafetensorsSingleAndSharded());
    counter.record("  GGUF Parser, Dequant & Tokenizer: ", testGgufParserAndDequant());
    counter.record("  ONNX Protobuf & External Data:    ", testOnnxInlineAndExternalData());
    counter.record("  PTH ZIP, DEFLATE & Safe Pickle:   ", testPthZipAndSafePickle());
    counter.record("  Cross-Format Parity & SealedRead: ", testCrossFormatCausalLMParity());
}
