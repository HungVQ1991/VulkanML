#include "llm/weight_loader.h"
#include "weight_loader_internal.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace weights
{

namespace
{

inline uint16_t u16Le(const uint8_t* p) noexcept { return internal::readU16Le(p); }
inline uint32_t u32Le(const uint8_t* p) noexcept { return internal::readU32Le(p); }
inline uint64_t u64Le(const uint8_t* p) noexcept { return internal::readU64Le(p); }

class Bit_Reader
{
private:
    const uint8_t* data = nullptr;
    size_t size_bytes = 0;
    size_t byte_pos = 0;
    uint32_t bit_buf = 0;
    int bit_count = 0;

public:
    Bit_Reader(const uint8_t* d, size_t n) : data(d), size_bytes(n) {}

    uint32_t readBits(int n)
    {
        while (bit_count < n)
        {
            if (byte_pos >= size_bytes)
            {
                throw std::runtime_error("PTH DEFLATE: unexpected end of compressed stream");
            }
            bit_buf |= static_cast<uint32_t>(data[byte_pos++]) << bit_count;
            bit_count += 8;
        }
        const uint32_t val = bit_buf & ((1u << n) - 1u);
        bit_buf >>= n;
        bit_count -= n;
        return val;
    }

    void alignToByte() noexcept
    {
        const int discard = bit_count & 7;
        bit_buf >>= discard;
        bit_count -= discard;
    }

    uint16_t readU16Aligned()
    {
        alignToByte();
        if (byte_pos + 2 > size_bytes)
        {
            throw std::runtime_error("PTH DEFLATE: truncated uncompressed block header");
        }
        const uint16_t v = u16Le(data + byte_pos);
        byte_pos += 2;
        return v;
    }

    void copyRaw(size_t len, std::vector<uint8_t>& out)
    {
        alignToByte();
        if (byte_pos + len > size_bytes)
        {
            throw std::runtime_error("PTH DEFLATE: truncated stored block payload");
        }
        out.insert(out.end(), data + byte_pos, data + byte_pos + len);
        byte_pos += len;
    }
};

struct Huff_Table
{
    static constexpr int MAX_BITS = 15;
    std::array<int, MAX_BITS + 1> count{};
    std::vector<int> symbols;

    void build(const int* lengths, int num_syms)
    {
        count.fill(0);
        int max_len = 0;
        for (int i = 0; i < num_syms; ++i)
        {
            if (lengths[i] > MAX_BITS)
            {
                throw std::runtime_error("PTH DEFLATE: invalid Huffman code length");
            }
            count[lengths[i]]++;
            max_len = std::max(max_len, lengths[i]);
        }
        count[0] = 0;
        std::array<int, MAX_BITS + 1> offsets{};
        int total = 0;
        for (int b = 1; b <= MAX_BITS; ++b)
        {
            offsets[b] = total;
            total += count[b];
        }
        symbols.assign(total, 0);
        for (int s = 0; s < num_syms; ++s)
        {
            const int l = lengths[s];
            if (l != 0)
            {
                symbols[offsets[l]++] = s;
            }
        }
    }

    int decode(Bit_Reader& br) const
    {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= MAX_BITS; ++len)
        {
            code |= static_cast<int>(br.readBits(1));
            const int cnt = count[len];
            if (code - cnt < first)
            {
                return symbols[index + (code - first)];
            }
            index += cnt;
            first += cnt;
            first <<= 1;
            code <<= 1;
        }
        throw std::runtime_error("PTH DEFLATE: invalid Huffman symbol");
    }
};

std::vector<uint8_t> inflateDeflate(const uint8_t* src, size_t len, size_t expected_size)
{
    static const std::array<int, 29> len_base = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
        35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
    };
    static const std::array<int, 29> len_extra = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
    };
    static const std::array<int, 30> dist_base = {
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
        257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
    };
    static const std::array<int, 30> dist_extra = {
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
        7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
    };
    static const std::array<int, 19> cl_order = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    };

    Bit_Reader br(src, len);
    std::vector<uint8_t> out;
    if (expected_size > 0)
    {
        out.reserve(expected_size);
    }

    uint32_t bfinal = 0;
    do
    {
        bfinal = br.readBits(1);
        const uint32_t btype = br.readBits(2);
        if (btype == 0)
        {
            const uint16_t blen = br.readU16Aligned();
            const uint16_t nlen = br.readU16Aligned();
            if (static_cast<uint16_t>(blen ^ 0xFFFFu) != nlen)
            {
                throw std::runtime_error("PTH DEFLATE: stored block length complement mismatch");
            }
            br.copyRaw(blen, out);
        }
        else if (btype == 1 || btype == 2)
        {
            Huff_Table lit_table;
            Huff_Table dist_table;
            if (btype == 1)
            {
                std::array<int, 288> lit_lens{};
                for (int i = 0; i <= 143; ++i) lit_lens[i] = 8;
                for (int i = 144; i <= 255; ++i) lit_lens[i] = 9;
                for (int i = 256; i <= 279; ++i) lit_lens[i] = 7;
                for (int i = 280; i <= 287; ++i) lit_lens[i] = 8;
                std::array<int, 32> dist_lens{};
                dist_lens.fill(5);
                lit_table.build(lit_lens.data(), 288);
                dist_table.build(dist_lens.data(), 32);
            }
            else
            {
                const int hlit = static_cast<int>(br.readBits(5)) + 257;
                const int hdist = static_cast<int>(br.readBits(5)) + 1;
                const int hclen = static_cast<int>(br.readBits(4)) + 4;
                std::array<int, 19> cl_lens{};
                for (int i = 0; i < hclen; ++i)
                {
                    cl_lens[cl_order[i]] = static_cast<int>(br.readBits(3));
                }
                Huff_Table cl_table;
                cl_table.build(cl_lens.data(), 19);

                std::vector<int> all_lens(hlit + hdist, 0);
                int idx = 0;
                while (idx < hlit + hdist)
                {
                    const int sym = cl_table.decode(br);
                    if (sym < 16)
                    {
                        all_lens[idx++] = sym;
                    }
                    else if (sym == 16)
                    {
                        if (idx == 0) throw std::runtime_error("PTH DEFLATE: invalid repeat");
                        const int rep = static_cast<int>(br.readBits(2)) + 3;
                        const int prev = all_lens[idx - 1];
                        for (int r = 0; r < rep && idx < hlit + hdist; ++r) all_lens[idx++] = prev;
                    }
                    else if (sym == 17)
                    {
                        const int rep = static_cast<int>(br.readBits(3)) + 3;
                        idx += rep;
                    }
                    else
                    {
                        const int rep = static_cast<int>(br.readBits(7)) + 11;
                        idx += rep;
                    }
                }
                lit_table.build(all_lens.data(), hlit);
                dist_table.build(all_lens.data() + hlit, hdist);
            }

            while (true)
            {
                const int sym = lit_table.decode(br);
                if (sym < 256)
                {
                    out.push_back(static_cast<uint8_t>(sym));
                }
                else if (sym == 256)
                {
                    break;
                }
                else
                {
                    const int lidx = sym - 257;
                    if (lidx < 0 || lidx >= 29) throw std::runtime_error("PTH DEFLATE: length code out of range");
                    const size_t match_len = static_cast<size_t>(len_base[lidx] + static_cast<int>(br.readBits(len_extra[lidx])));
                    const int didx = dist_table.decode(br);
                    if (didx < 0 || didx >= 30) throw std::runtime_error("PTH DEFLATE: dist code out of range");
                    const size_t dist = static_cast<size_t>(dist_base[didx] + static_cast<int>(br.readBits(dist_extra[didx])));
                    if (dist == 0 || dist > out.size())
                    {
                        throw std::runtime_error("PTH DEFLATE: invalid back-reference distance");
                    }
                    for (size_t k = 0; k < match_len; ++k)
                    {
                        out.push_back(out[out.size() - dist]);
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error("PTH DEFLATE: reserved block type");
        }
    } while (bfinal == 0);
    return out;
}

struct Zip_Entry
{
    std::string name;
    uint16_t method = 0;
    uint64_t comp_size = 0;
    uint64_t uncomp_size = 0;
    uint64_t local_header_offset = 0;
    uint64_t data_offset = 0;
};

class Zip_Archive
{
private:
    std::filesystem::path path;
    std::unordered_map<std::string, Zip_Entry> entries;

    uint64_t resolveDataOffset(std::ifstream& in, uint64_t lfh_off) const
    {
        in.seekg(static_cast<std::streamoff>(lfh_off), std::ios::beg);
        uint8_t hdr[30]{};
        if (!in.read(reinterpret_cast<char*>(hdr), 30) || u32Le(hdr) != 0x04034b50u)
        {
            throw std::runtime_error("PTH ZIP: invalid local file header");
        }
        const uint16_t fn_len = u16Le(hdr + 26);
        const uint16_t ex_len = u16Le(hdr + 28);
        return lfh_off + 30ULL + fn_len + ex_len;
    }

public:
    explicit Zip_Archive(const std::filesystem::path& p)
        : path(p)
    {
        std::ifstream in(p, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("PTH: failed to open '" + p.string() + "'");
        }
        in.seekg(0, std::ios::end);
        const uint64_t fsize = static_cast<uint64_t>(in.tellg());
        if (fsize < 22)
        {
            throw std::runtime_error("PTH: file too small to be a PyTorch ZIP archive");
        }
        const size_t tail_len = static_cast<size_t>(std::min<uint64_t>(fsize, 65558ULL));
        std::vector<uint8_t> tail(tail_len);
        in.seekg(static_cast<std::streamoff>(fsize - tail_len), std::ios::beg);
        in.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail_len));

        int64_t eocd_pos_rel = -1;
        for (int64_t i = static_cast<int64_t>(tail_len) - 22; i >= 0; --i)
        {
            if (u32Le(tail.data() + i) == 0x06054b50u)
            {
                eocd_pos_rel = i;
                break;
            }
        }
        if (eocd_pos_rel < 0)
        {
            throw std::runtime_error("PTH: not a PyTorch ZIP archive (legacy tar/pickle checkpoints require conversion to .safetensors)");
        }

        const uint8_t* eocd = tail.data() + eocd_pos_rel;
        uint64_t total_entries = u16Le(eocd + 10);
        uint64_t cd_size = u32Le(eocd + 12);
        uint64_t cd_offset = u32Le(eocd + 16);

        if (total_entries == 0xFFFFu || cd_offset == 0xFFFFFFFFu || cd_size == 0xFFFFFFFFu)
        {
            const uint64_t eocd_abs = (fsize - tail_len) + static_cast<uint64_t>(eocd_pos_rel);
            if (eocd_abs >= 20)
            {
                uint8_t loc[20]{};
                in.seekg(static_cast<std::streamoff>(eocd_abs - 20), std::ios::beg);
                in.read(reinterpret_cast<char*>(loc), 20);
                if (u32Le(loc) == 0x07064b50u)
                {
                    const uint64_t z64_off = u64Le(loc + 8);
                    uint8_t z64[56]{};
                    in.seekg(static_cast<std::streamoff>(z64_off), std::ios::beg);
                    in.read(reinterpret_cast<char*>(z64), 56);
                    if (u32Le(z64) == 0x06064b50u)
                    {
                        total_entries = u64Le(z64 + 32);
                        cd_size = u64Le(z64 + 40);
                        cd_offset = u64Le(z64 + 48);
                    }
                }
            }
        }

        std::vector<uint8_t> cd(static_cast<size_t>(cd_size));
        in.seekg(static_cast<std::streamoff>(cd_offset), std::ios::beg);
        if (!cd.empty() && !in.read(reinterpret_cast<char*>(cd.data()), static_cast<std::streamsize>(cd.size())))
        {
            throw std::runtime_error("PTH ZIP: failed reading central directory");
        }

        size_t pos = 0;
        for (uint64_t i = 0; i < total_entries && pos + 46 <= cd.size(); ++i)
        {
            if (u32Le(cd.data() + pos) != 0x02014b50u)
            {
                throw std::runtime_error("PTH ZIP: invalid central directory signature");
            }
            Zip_Entry e;
            e.method = u16Le(cd.data() + pos + 10);
            e.comp_size = u32Le(cd.data() + pos + 20);
            e.uncomp_size = u32Le(cd.data() + pos + 24);
            const uint16_t fn_len = u16Le(cd.data() + pos + 28);
            const uint16_t ex_len = u16Le(cd.data() + pos + 30);
            const uint16_t cm_len = u16Le(cd.data() + pos + 32);
            e.local_header_offset = u32Le(cd.data() + pos + 42);
            e.name.assign(reinterpret_cast<const char*>(cd.data() + pos + 46), fn_len);

            if (ex_len >= 4)
            {
                const uint8_t* ex = cd.data() + pos + 46 + fn_len;
                size_t ep = 0;
                while (ep + 4 <= ex_len)
                {
                    const uint16_t hid = u16Le(ex + ep);
                    const uint16_t hsz = u16Le(ex + ep + 2);
                    if (hid == 0x0001u)
                    {
                        size_t fp = ep + 4;
                        if (e.uncomp_size == 0xFFFFFFFFu && fp + 8 <= ep + 4 + hsz) { e.uncomp_size = u64Le(ex + fp); fp += 8; }
                        if (e.comp_size == 0xFFFFFFFFu && fp + 8 <= ep + 4 + hsz) { e.comp_size = u64Le(ex + fp); fp += 8; }
                        if (e.local_header_offset == 0xFFFFFFFFu && fp + 8 <= ep + 4 + hsz) { e.local_header_offset = u64Le(ex + fp); fp += 8; }
                    }
                    ep += 4 + hsz;
                }
            }
            e.data_offset = resolveDataOffset(in, e.local_header_offset);
            entries.emplace(e.name, e);
            pos += 46 + fn_len + ex_len + cm_len;
        }
    }

    const Zip_Entry* findDataPkl() const noexcept
    {
        for (const auto& [name, e] : entries)
        {
            if (name == "data.pkl" || (name.size() > 9 && name.compare(name.size() - 9, 9, "/data.pkl") == 0))
            {
                return &e;
            }
        }
        return nullptr;
    }

    const Zip_Entry* findStorageKey(const std::string& key) const noexcept
    {
        const std::string suffix = "/data/" + key;
        const std::string direct = "data/" + key;
        for (const auto& [name, e] : entries)
        {
            if (name == direct || (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0))
            {
                return &e;
            }
        }
        return nullptr;
    }

    std::vector<uint8_t> readEntry(const Zip_Entry& e) const
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("PTH ZIP: cannot reopen '" + path.string() + "'");
        }
        in.seekg(static_cast<std::streamoff>(e.data_offset), std::ios::beg);
        std::vector<uint8_t> raw(static_cast<size_t>(e.comp_size));
        if (!raw.empty() && !in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
        {
            throw std::runtime_error("PTH ZIP: failed reading entry '" + e.name + "'");
        }
        if (e.method == 0)
        {
            return raw;
        }
        if (e.method == 8)
        {
            return inflateDeflate(raw.data(), raw.size(), static_cast<size_t>(e.uncomp_size));
        }
        throw std::runtime_error("PTH ZIP: unsupported compression method " + std::to_string(e.method));
    }

    const std::filesystem::path& archivePath() const noexcept { return path; }
};

struct Pkl_Storage
{
    std::string storage_type;
    std::string key;
    Weight_Dtype dtype = Weight_Dtype::UNKNOWN;
    size_t num_elements = 0;
};

struct Pkl_Tensor
{
    Pkl_Storage storage;
    size_t storage_offset = 0;
    std::vector<size_t> shape;
    std::vector<int64_t> stride;
};

struct Pkl_Global
{
    std::string module;
    std::string name;
};

struct Pkl_Val;
using Pkl_List = std::vector<std::shared_ptr<Pkl_Val>>;
using Pkl_Dict = std::vector<std::pair<std::shared_ptr<Pkl_Val>, std::shared_ptr<Pkl_Val>>>;

struct Pkl_Val
{
    enum class Kind
    {
        NONE,
        BOOL,
        INT,
        FLOAT,
        STRING,
        BYTES,
        TUPLE,
        LIST,
        DICT,
        GLOBAL,
        STORAGE,
        TENSOR,
        MARK
    };

    Kind kind = Kind::NONE;
    bool b = false;
    int64_t i = 0;
    double f = 0.0;
    std::string s;
    std::vector<uint8_t> bytes;
    Pkl_List items;
    Pkl_Dict dict;
    Pkl_Global global;
    Pkl_Storage storage;
    Pkl_Tensor tensor;
};

using Pkl_Ptr = std::shared_ptr<Pkl_Val>;

Pkl_Ptr makeVal(Pkl_Val::Kind k)
{
    auto p = std::make_shared<Pkl_Val>();
    p->kind = k;
    return p;
}

Weight_Dtype storageNameToDtype(const std::string& sname)
{
    if (sname == "FloatStorage") return Weight_Dtype::F32;
    if (sname == "HalfStorage") return Weight_Dtype::F16;
    if (sname == "BFloat16Storage") return Weight_Dtype::BF16;
    if (sname == "DoubleStorage") return Weight_Dtype::F64;
    if (sname == "CharStorage") return Weight_Dtype::I8;
    if (sname == "ByteStorage") return Weight_Dtype::U8;
    if (sname == "ShortStorage") return Weight_Dtype::I16;
    if (sname == "IntStorage") return Weight_Dtype::I32;
    if (sname == "LongStorage") return Weight_Dtype::I64;
    return Weight_Dtype::UNKNOWN;
}

bool isWhitelistedGlobal(const std::string& mod, const std::string& name)
{
    static const std::vector<std::pair<std::string, std::string>> allow = {
        { "collections", "OrderedDict" },
        { "torch._utils", "_rebuild_tensor_v2" },
        { "torch._utils", "_rebuild_tensor" },
        { "torch._utils", "_rebuild_parameter" },
        { "torch.storage", "_load_from_bytes" },
        { "torch", "FloatStorage" },
        { "torch", "HalfStorage" },
        { "torch", "BFloat16Storage" },
        { "torch", "DoubleStorage" },
        { "torch", "IntStorage" },
        { "torch", "LongStorage" },
        { "torch", "ShortStorage" },
        { "torch", "CharStorage" },
        { "torch", "ByteStorage" },
        { "torch", "UntypedStorage" },
        { "builtins", "bytearray" },
        { "builtins", "slice" },
        { "_codecs", "encode" }
    };
    for (const auto& [m, n] : allow)
    {
        if (m == mod && n == name) return true;
    }
    return false;
}

class Safe_Pickle_Vm
{
private:
    const std::vector<uint8_t>& buf;
    size_t pos = 0;
    std::vector<Pkl_Ptr> stack;
    std::unordered_map<int64_t, Pkl_Ptr> memo;
    int64_t next_memo_id = 0;

    uint8_t readU8()
    {
        if (pos >= buf.size()) throw std::runtime_error("Pickle VM: unexpected EOF");
        return buf[pos++];
    }

    uint16_t readU16()
    {
        if (pos + 2 > buf.size()) throw std::runtime_error("Pickle VM: unexpected EOF");
        const uint16_t v = u16Le(buf.data() + pos);
        pos += 2;
        return v;
    }

    uint32_t readU32()
    {
        if (pos + 4 > buf.size()) throw std::runtime_error("Pickle VM: unexpected EOF");
        const uint32_t v = u32Le(buf.data() + pos);
        pos += 4;
        return v;
    }

    uint64_t readU64()
    {
        if (pos + 8 > buf.size()) throw std::runtime_error("Pickle VM: unexpected EOF");
        const uint64_t v = u64Le(buf.data() + pos);
        pos += 8;
        return v;
    }

    std::string readLine()
    {
        const size_t nl = static_cast<size_t>(std::find(buf.begin() + static_cast<std::ptrdiff_t>(pos), buf.end(), '\n') - buf.begin());
        if (nl >= buf.size()) throw std::runtime_error("Pickle VM: missing newline");
        std::string out(reinterpret_cast<const char*>(buf.data() + pos), nl - pos);
        pos = nl + 1;
        return out;
    }

    std::string readStr(size_t n)
    {
        if (pos + n > buf.size()) throw std::runtime_error("Pickle VM: string EOF");
        std::string out(reinterpret_cast<const char*>(buf.data() + pos), n);
        pos += n;
        return out;
    }

    Pkl_Ptr popVal()
    {
        if (stack.empty()) throw std::runtime_error("Pickle VM: stack underflow");
        auto v = std::move(stack.back());
        stack.pop_back();
        return v;
    }

    size_t findMark() const
    {
        for (size_t i = stack.size(); i > 0; --i)
        {
            if (stack[i - 1]->kind == Pkl_Val::Kind::MARK) return i - 1;
        }
        throw std::runtime_error("Pickle VM: MARK not found on stack");
    }

    Pkl_List popToMark()
    {
        const size_t m = findMark();
        Pkl_List out(stack.begin() + static_cast<std::ptrdiff_t>(m + 1), stack.end());
        stack.resize(m);
        return out;
    }

    Pkl_Ptr applyReduce(const Pkl_Ptr& fn, const Pkl_Ptr& args)
    {
        if (fn->kind != Pkl_Val::Kind::GLOBAL)
        {
            throw std::runtime_error("Pickle VM: REDUCE target is not a safe global");
        }
        const std::string& mod = fn->global.module;
        const std::string& name = fn->global.name;

        if (mod == "collections" && name == "OrderedDict")
        {
            auto d = makeVal(Pkl_Val::Kind::DICT);
            if (!args->items.empty() && args->items[0]->kind == Pkl_Val::Kind::LIST)
            {
                for (const auto& pair_v : args->items[0]->items)
                {
                    if (pair_v->items.size() == 2)
                    {
                        d->dict.emplace_back(pair_v->items[0], pair_v->items[1]);
                    }
                }
            }
            return d;
        }
        if (mod == "torch._utils" && (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor"))
        {
            if (args->items.size() < 4 || args->items[0]->kind != Pkl_Val::Kind::STORAGE)
            {
                throw std::runtime_error("Pickle VM: malformed _rebuild_tensor arguments");
            }
            auto tv = makeVal(Pkl_Val::Kind::TENSOR);
            tv->tensor.storage = args->items[0]->storage;
            tv->tensor.storage_offset = static_cast<size_t>(std::max<int64_t>(0, args->items[1]->i));
            for (const auto& d : args->items[2]->items)
            {
                tv->tensor.shape.push_back(static_cast<size_t>(std::max<int64_t>(0, d->i)));
            }
            for (const auto& st : args->items[3]->items)
            {
                tv->tensor.stride.push_back(st->i);
            }
            return tv;
        }
        if (mod == "torch._utils" && name == "_rebuild_parameter")
        {
            if (args->items.empty()) throw std::runtime_error("Pickle VM: empty _rebuild_parameter");
            return args->items[0];
        }
        if (mod == "_codecs" && name == "encode" && args->items.size() >= 1)
        {
            return args->items[0];
        }
        if (mod == "builtins" && name == "bytearray")
        {
            return makeVal(Pkl_Val::Kind::BYTES);
        }
        return makeVal(Pkl_Val::Kind::NONE);
    }

    Pkl_Ptr applyPersid(const Pkl_Ptr& pid)
    {
        if (pid->kind == Pkl_Val::Kind::TUPLE && pid->items.size() >= 5 && pid->items[0]->s == "storage")
        {
            auto sv = makeVal(Pkl_Val::Kind::STORAGE);
            if (pid->items[1]->kind == Pkl_Val::Kind::GLOBAL)
            {
                sv->storage.storage_type = pid->items[1]->global.name;
                sv->storage.dtype = storageNameToDtype(sv->storage.storage_type);
            }
            sv->storage.key = pid->items[2]->s;
            sv->storage.num_elements = static_cast<size_t>(std::max<int64_t>(0, pid->items[4]->i));
            return sv;
        }
        throw std::runtime_error("Pickle VM: unsupported persistent_id payload");
    }

public:
    explicit Safe_Pickle_Vm(const std::vector<uint8_t>& b) : buf(b) {}

    Pkl_Ptr run()
    {
        while (pos < buf.size())
        {
            const uint8_t op = readU8();
            switch (op)
            {
            case 0x80: (void)readU8(); break;
            case 0x95: (void)readU64(); break;
            case '(': stack.push_back(makeVal(Pkl_Val::Kind::MARK)); break;
            case '.': return stack.empty() ? makeVal(Pkl_Val::Kind::NONE) : stack.back();
            case '0': (void)popVal(); break;
            case '1': (void)popToMark(); break;
            case '2':
            {
                if (stack.empty()) throw std::runtime_error("Pickle VM: DUP underflow");
                stack.push_back(stack.back());
                break;
            }
            case 'N': stack.push_back(makeVal(Pkl_Val::Kind::NONE)); break;
            case 0x88:
            {
                auto v = makeVal(Pkl_Val::Kind::BOOL);
                v->b = true;
                stack.push_back(v);
                break;
            }
            case 0x89:
            {
                auto v = makeVal(Pkl_Val::Kind::BOOL);
                v->b = false;
                stack.push_back(v);
                break;
            }
            case 'I':
            {
                const std::string line = readLine();
                if (line == "01") { auto v = makeVal(Pkl_Val::Kind::BOOL); v->b = true; stack.push_back(v); }
                else if (line == "00") { auto v = makeVal(Pkl_Val::Kind::BOOL); v->b = false; stack.push_back(v); }
                else { auto v = makeVal(Pkl_Val::Kind::INT); v->i = std::stoll(line); stack.push_back(v); }
                break;
            }
            case 'J':
            {
                auto v = makeVal(Pkl_Val::Kind::INT);
                v->i = static_cast<int32_t>(readU32());
                stack.push_back(v);
                break;
            }
            case 'K':
            {
                auto v = makeVal(Pkl_Val::Kind::INT);
                v->i = static_cast<int64_t>(readU8());
                stack.push_back(v);
                break;
            }
            case 'M':
            {
                auto v = makeVal(Pkl_Val::Kind::INT);
                v->i = static_cast<int64_t>(readU16());
                stack.push_back(v);
                break;
            }
            case 0x8A:
            case 0x8B:
            {
                const size_t n = (op == 0x8A) ? readU8() : readU32();
                int64_t val = 0;
                if (n > 0)
                {
                    if (n > 8) throw std::runtime_error("Pickle VM: LONG exceeds int64");
                    uint64_t u = 0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        u |= static_cast<uint64_t>(readU8()) << (8 * i);
                    }
                    if ((u >> (8 * n - 1)) & 1ULL && n < 8)
                    {
                        u |= (~0ULL) << (8 * n);
                    }
                    val = static_cast<int64_t>(u);
                }
                auto v = makeVal(Pkl_Val::Kind::INT);
                v->i = val;
                stack.push_back(v);
                break;
            }
            case 'G':
            {
                uint64_t be = 0;
                for (int i = 0; i < 8; ++i) be = (be << 8) | readU8();
                double d = 0.0;
                std::memcpy(&d, &be, 8);
                auto v = makeVal(Pkl_Val::Kind::FLOAT);
                v->f = d;
                stack.push_back(v);
                break;
            }
            case 'U':
            {
                auto v = makeVal(Pkl_Val::Kind::STRING);
                v->s = readStr(readU8());
                stack.push_back(v);
                break;
            }
            case 'T':
            case 'X':
            {
                auto v = makeVal(Pkl_Val::Kind::STRING);
                v->s = readStr(readU32());
                stack.push_back(v);
                break;
            }
            case 0x8C:
            {
                auto v = makeVal(Pkl_Val::Kind::STRING);
                v->s = readStr(readU8());
                stack.push_back(v);
                break;
            }
            case 0x8D:
            {
                auto v = makeVal(Pkl_Val::Kind::STRING);
                v->s = readStr(static_cast<size_t>(readU64()));
                stack.push_back(v);
                break;
            }
            case 'C':
            case 0x8E:
            case 'B':
            {
                const size_t n = (op == 'C') ? readU8() : (op == 'B' ? readU32() : static_cast<size_t>(readU64()));
                auto v = makeVal(Pkl_Val::Kind::BYTES);
                if (pos + n > buf.size()) throw std::runtime_error("Pickle VM: bytes EOF");
                v->bytes.assign(buf.begin() + static_cast<std::ptrdiff_t>(pos), buf.begin() + static_cast<std::ptrdiff_t>(pos + n));
                pos += n;
                stack.push_back(v);
                break;
            }
            case ')': stack.push_back(makeVal(Pkl_Val::Kind::TUPLE)); break;
            case 't':
            {
                auto v = makeVal(Pkl_Val::Kind::TUPLE);
                v->items = popToMark();
                stack.push_back(v);
                break;
            }
            case 0x85:
            {
                auto v = makeVal(Pkl_Val::Kind::TUPLE);
                v->items.push_back(popVal());
                stack.push_back(v);
                break;
            }
            case 0x86:
            {
                auto b = popVal();
                auto a = popVal();
                auto v = makeVal(Pkl_Val::Kind::TUPLE);
                v->items = { std::move(a), std::move(b) };
                stack.push_back(v);
                break;
            }
            case 0x87:
            {
                auto c = popVal();
                auto b = popVal();
                auto a = popVal();
                auto v = makeVal(Pkl_Val::Kind::TUPLE);
                v->items = { std::move(a), std::move(b), std::move(c) };
                stack.push_back(v);
                break;
            }
            case ']': stack.push_back(makeVal(Pkl_Val::Kind::LIST)); break;
            case 'l':
            {
                auto v = makeVal(Pkl_Val::Kind::LIST);
                v->items = popToMark();
                stack.push_back(v);
                break;
            }
            case 'a':
            {
                auto item = popVal();
                if (stack.empty() || stack.back()->kind != Pkl_Val::Kind::LIST)
                {
                    throw std::runtime_error("Pickle VM: APPEND target is not a list");
                }
                stack.back()->items.push_back(std::move(item));
                break;
            }
            case 'e':
            {
                auto items = popToMark();
                if (stack.empty() || stack.back()->kind != Pkl_Val::Kind::LIST)
                {
                    throw std::runtime_error("Pickle VM: APPENDS target is not a list");
                }
                stack.back()->items.insert(stack.back()->items.end(), items.begin(), items.end());
                break;
            }
            case '}': stack.push_back(makeVal(Pkl_Val::Kind::DICT)); break;
            case 'd':
            {
                auto items = popToMark();
                auto d = makeVal(Pkl_Val::Kind::DICT);
                for (size_t i = 0; i + 1 < items.size(); i += 2)
                {
                    d->dict.emplace_back(items[i], items[i + 1]);
                }
                stack.push_back(d);
                break;
            }
            case 's':
            {
                auto val = popVal();
                auto key = popVal();
                if (stack.empty() || stack.back()->kind != Pkl_Val::Kind::DICT)
                {
                    throw std::runtime_error("Pickle VM: SETITEM target is not a dict");
                }
                stack.back()->dict.emplace_back(std::move(key), std::move(val));
                break;
            }
            case 'u':
            {
                auto items = popToMark();
                if (stack.empty() || stack.back()->kind != Pkl_Val::Kind::DICT)
                {
                    throw std::runtime_error("Pickle VM: SETITEMS target is not a dict");
                }
                for (size_t i = 0; i + 1 < items.size(); i += 2)
                {
                    stack.back()->dict.emplace_back(items[i], items[i + 1]);
                }
                break;
            }
            case 'p':
            {
                const int64_t mid = std::stoll(readLine());
                memo[mid] = stack.back();
                break;
            }
            case 'q': memo[readU8()] = stack.back(); break;
            case 'r': memo[readU32()] = stack.back(); break;
            case 0x94: memo[next_memo_id++] = stack.back(); break;
            case 'g': stack.push_back(memo.at(std::stoll(readLine()))); break;
            case 'h': stack.push_back(memo.at(readU8())); break;
            case 'j': stack.push_back(memo.at(readU32())); break;
            case 'c':
            {
                const std::string mod = readLine();
                const std::string name = readLine();
                if (!isWhitelistedGlobal(mod, name))
                {
                    throw std::runtime_error("Pickle VM: disallowed global '" + mod + "." + name + "'");
                }
                auto g = makeVal(Pkl_Val::Kind::GLOBAL);
                g->global = Pkl_Global{ mod, name };
                stack.push_back(std::move(g));
                break;
            }
            case 0x93:
            {
                auto name_v = popVal();
                auto mod_v = popVal();
                if (!isWhitelistedGlobal(mod_v->s, name_v->s))
                {
                    throw std::runtime_error("Pickle VM: disallowed global '" + mod_v->s + "." + name_v->s + "'");
                }
                auto g = makeVal(Pkl_Val::Kind::GLOBAL);
                g->global = Pkl_Global{ mod_v->s, name_v->s };
                stack.push_back(std::move(g));
                break;
            }
            case 'Q':
            {
                auto pid = popVal();
                stack.push_back(applyPersid(pid));
                break;
            }
            case 'R':
            {
                auto args = popVal();
                auto fn = popVal();
                stack.push_back(applyReduce(fn, args));
                break;
            }
            case 0x81:
            {
                auto args = popVal();
                auto cls = popVal();
                stack.push_back(applyReduce(cls, args));
                break;
            }
            case 'b':
            {
                auto state = popVal();
                if (stack.empty()) throw std::runtime_error("Pickle VM: BUILD underflow");
                if (stack.back()->kind == Pkl_Val::Kind::DICT && state->kind == Pkl_Val::Kind::DICT)
                {
                    stack.back()->dict.insert(stack.back()->dict.end(), state->dict.begin(), state->dict.end());
                }
                break;
            }
            default:
                throw std::runtime_error(std::string("Pickle VM: unsupported opcode 0x") +
                    [](uint8_t b) {
                        const char* hex = "0123456789ABCDEF";
                        return std::string{ hex[b >> 4], hex[b & 0xF] };
                    }(op));
            }
        }
        throw std::runtime_error("Pickle VM: STOP opcode not reached");
    }
};

const Pkl_Val* unwrapStateDict(const Pkl_Ptr& root)
{
    if (!root || root->kind != Pkl_Val::Kind::DICT)
    {
        throw std::runtime_error("PTH: root pickle object is not a state_dict mapping");
    }
    auto hasAnyTensor = [](const Pkl_Val* d) {
        for (const auto& [k, v] : d->dict)
        {
            if (k && k->kind == Pkl_Val::Kind::STRING && v && v->kind == Pkl_Val::Kind::TENSOR)
            {
                return true;
            }
        }
        return false;
    };
    if (hasAnyTensor(root.get()))
    {
        return root.get();
    }
    for (const char* sub_key : { "state_dict", "model", "model_state_dict" })
    {
        for (const auto& [k, v] : root->dict)
        {
            if (k && k->kind == Pkl_Val::Kind::STRING && k->s == sub_key && v && v->kind == Pkl_Val::Kind::DICT)
            {
                if (hasAnyTensor(v.get()))
                {
                    return v.get();
                }
            }
        }
    }
    return root.get();
}

bool isRowMajorContiguous(const std::vector<size_t>& shape, const std::vector<int64_t>& stride) noexcept
{
    if (shape.empty()) return true;
    if (stride.size() != shape.size()) return true;
    int64_t expected = 1;
    for (size_t i = shape.size(); i > 0; --i)
    {
        const size_t dim = shape[i - 1];
        if (dim > 1 && stride[i - 1] != expected)
        {
            return false;
        }
        expected *= static_cast<int64_t>(dim);
    }
    return true;
}

class Pth_Weight_Source final : public IWeight_Source
{
private:
    struct Entry
    {
        Weight_Tensor_Info info;
        Pkl_Tensor pt;
    };

    Zip_Archive zip;
    std::unordered_map<std::string, Entry> entries;
    std::vector<std::string> ordered_names;
    Weight_Metadata meta;

public:
    explicit Pth_Weight_Source(const std::filesystem::path& path)
        : zip(path)
    {
        const Zip_Entry* pkl_entry = zip.findDataPkl();
        if (!pkl_entry)
        {
            throw std::runtime_error("PTH: data.pkl not found inside ZIP archive '" + path.string() + "'");
        }
        const auto pkl_bytes = zip.readEntry(*pkl_entry);
        Safe_Pickle_Vm vm(pkl_bytes);
        Pkl_Ptr root = vm.run();
        const Pkl_Val* dict = unwrapStateDict(root);

        for (const auto& [k, v] : dict->dict)
        {
            if (!k || k->kind != Pkl_Val::Kind::STRING || !v || v->kind != Pkl_Val::Kind::TENSOR)
            {
                continue;
            }
            Entry e;
            e.info.name = k->s;
            e.info.shape = v->tensor.shape;
            e.info.dtype = v->tensor.storage.dtype;
            const size_t elem_bytes = internal::denseDtypeByteSize(e.info.dtype);
            e.info.byte_size = static_cast<uint64_t>(e.info.elementCount()) * elem_bytes;
            e.pt = v->tensor;
            if (entries.find(k->s) == entries.end())
            {
                ordered_names.push_back(k->s);
            }
            entries.insert_or_assign(k->s, std::move(e));
        }
    }

    Weight_Format format() const noexcept override { return Weight_Format::PTH; }

    bool hasTensor(const std::string& name) const noexcept override
    {
        return entries.find(name) != entries.end();
    }

    std::vector<std::string> tensorNames() const override { return ordered_names; }

    Weight_Tensor_Info getTensorInfo(const std::string& name) const override
    {
        auto it = entries.find(name);
        if (it == entries.end())
        {
            throw std::runtime_error("PTH: missing tensor '" + name + "'");
        }
        return it->second.info;
    }

    const Weight_Metadata& metadata() const noexcept override { return meta; }

    std::vector<float> readTensorF32(const std::string& name) const override
    {
        auto it = entries.find(name);
        if (it == entries.end())
        {
            throw std::runtime_error("PTH: missing tensor '" + name + "'");
        }
        const Entry& e = it->second;
        const size_t elem_bytes = internal::denseDtypeByteSize(e.info.dtype);
        if (elem_bytes == 0)
        {
            throw std::runtime_error("PTH: unsupported storage type for '" + name + "'");
        }
        const Zip_Entry* ze = zip.findStorageKey(e.pt.storage.key);
        if (!ze)
        {
            throw std::runtime_error("PTH: missing ZIP storage entry for key '" + e.pt.storage.key + "'");
        }

        const size_t total = e.info.elementCount();
        if (total == 0)
        {
            return {};
        }

        if (ze->method == 0 && isRowMajorContiguous(e.pt.shape, e.pt.stride))
        {
            const uint64_t byte_off = e.pt.storage_offset * elem_bytes;
            const uint64_t read_bytes = total * elem_bytes;
            if (byte_off + read_bytes > ze->uncomp_size)
            {
                throw std::runtime_error("PTH: tensor storage slice out of range for '" + name + "'");
            }
            std::ifstream in(zip.archivePath(), std::ios::binary);
            in.seekg(static_cast<std::streamoff>(ze->data_offset + byte_off), std::ios::beg);
            std::vector<uint8_t> raw(read_bytes);
            if (!in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(read_bytes)))
            {
                throw std::runtime_error("PTH: failed reading stored tensor '" + name + "'");
            }
            return internal::convertDenseRawToF32(raw.data(), total, e.info.dtype, "PTH");
        }

        std::vector<float> out(total, 0.0f);
        const std::vector<uint8_t> storage_bytes = zip.readEntry(*ze);
        const size_t rank = e.pt.shape.size();
        std::vector<size_t> coords(rank, 0);
        for (size_t flat = 0; flat < total; ++flat)
        {
            int64_t elem_idx = static_cast<int64_t>(e.pt.storage_offset);
            for (size_t d = 0; d < rank; ++d)
            {
                const int64_t st = (d < e.pt.stride.size()) ? e.pt.stride[d] : 1;
                elem_idx += static_cast<int64_t>(coords[d]) * st;
            }
            const size_t byte_idx = static_cast<size_t>(elem_idx) * elem_bytes;
            if (elem_idx < 0 || byte_idx + elem_bytes > storage_bytes.size())
            {
                throw std::runtime_error("PTH: strided index out of bounds for '" + name + "'");
            }
            out[flat] = internal::convertScalarToF32(storage_bytes.data() + byte_idx, e.info.dtype);

            for (size_t d = rank; d > 0; --d)
            {
                coords[d - 1]++;
                if (coords[d - 1] < e.pt.shape[d - 1])
                {
                    break;
                }
                coords[d - 1] = 0;
            }
        }
        return out;
    }
};

}

std::unique_ptr<IWeight_Source> openPth(const std::filesystem::path& path)
{
    return std::make_unique<Pth_Weight_Source>(path);
}

}
