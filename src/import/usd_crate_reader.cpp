// ---------------------------------------------------------------------------
// nuka::import - Self-written USDC (crate) binary mesh reader implementation
// ---------------------------------------------------------------------------
//
// A from-scratch port of the public OpenUSD crate format (no pxr/OpenUSD code).
// The structure follows OpenUSD's pxr/usd/sdf/{crateFile,integerCoding,
// fastCompression}.cpp, re-implemented for plain byte parsing.
//
// Pipeline:
//   bootstrap (magic + version + tocOffset)
//     -> TOC (section name/offset/size table)
//       -> TOKENS  (LZ4-compressed null-separated token strings)
//       -> FIELDS  (numFields; integer-compressed token-index array;
//                   LZ4-compressed uint64 ValueRep array)
//       -> FIELDSETS (integer-compressed field-index runs, -1 sentinel)
//       -> PATHS   (count written TWICE; integer-compressed pathIndexes /
//                   elementTokenIndexes (sign=>property) / jumps; rebuilt to
//                   a path tree)
//       -> SPECS   (integer-compressed pathIndex / fieldSetIndex / specType)
//   -> map "/Model.points" / ".faceVertexIndices" / ".faceVertexCounts" to
//      their `default` field ValueRep -> read the arrays at the ValueRep payload
//      offset (raw float32 for points; integer-compressed for the int arrays).
// ---------------------------------------------------------------------------

#include "import/usd_crate_reader.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace nuka::import {

namespace {

// ---------------------------------------------------------------------------
// Bounds-checked little-endian primitive reads over the whole file buffer.
// ---------------------------------------------------------------------------
class ByteReader {
public:
    explicit ByteReader(const std::vector<uint8_t>& data) : data_(data) {}

    std::size_t size() const { return data_.size(); }

    void RequireRange(std::size_t off, std::size_t len) const {
        if (off > data_.size() || len > data_.size() - off) {
            std::ostringstream msg;
            msg << "USDC: read out of bounds (offset " << off << " len " << len
                << " file " << data_.size() << ")";
            throw std::runtime_error(msg.str());
        }
    }

    uint64_t U64(std::size_t off) const {
        RequireRange(off, 8);
        uint64_t v;
        std::memcpy(&v, data_.data() + off, 8);
        return v;
    }
    int64_t I64(std::size_t off) const { return static_cast<int64_t>(U64(off)); }

    uint32_t U32(std::size_t off) const {
        RequireRange(off, 4);
        uint32_t v;
        std::memcpy(&v, data_.data() + off, 4);
        return v;
    }

    float F32(std::size_t off) const {
        RequireRange(off, 4);
        float v;
        std::memcpy(&v, data_.data() + off, 4);
        return v;
    }

    const uint8_t* Ptr(std::size_t off, std::size_t len) const {
        RequireRange(off, len);
        return data_.data() + off;
    }

private:
    const std::vector<uint8_t>& data_;
};

// ---------------------------------------------------------------------------
// LZ4 block decompression (the raw LZ4 block format -- no frame, no magic).
// Faithful to the standard format that pxrLZ4 produces.
// ---------------------------------------------------------------------------
std::vector<uint8_t> Lz4BlockDecompress(const uint8_t* src, std::size_t n,
                                        std::size_t expected) {
    std::vector<uint8_t> out;
    out.reserve(expected);
    std::size_t i = 0;
    auto need = [&](std::size_t k) {
        if (i + k > n) {
            throw std::runtime_error("USDC: LZ4 block truncated");
        }
    };
    while (i < n) {
        need(1);
        const uint8_t token = src[i++];
        std::size_t lit = token >> 4;
        if (lit == 15) {
            while (true) {
                need(1);
                const uint8_t b = src[i++];
                lit += b;
                if (b != 255) break;
            }
        }
        need(lit);
        out.insert(out.end(), src + i, src + i + lit);
        i += lit;
        if (i >= n) break;  // last sequence: literals only, no match.
        need(2);
        const std::size_t offset =
            static_cast<std::size_t>(src[i]) | (static_cast<std::size_t>(src[i + 1]) << 8);
        i += 2;
        std::size_t match = token & 0x0F;
        if (match == 15) {
            while (true) {
                need(1);
                const uint8_t b = src[i++];
                match += b;
                if (b != 255) break;
            }
        }
        match += 4;
        if (offset == 0 || offset > out.size()) {
            throw std::runtime_error("USDC: LZ4 invalid match offset");
        }
        std::size_t start = out.size() - offset;
        for (std::size_t k = 0; k < match; ++k) {
            out.push_back(out[start + k]);  // overlap-safe (byte-at-a-time)
        }
    }
    return out;
}

// USD's TfFastCompression wrapper: first byte = number of chunks. 0 => a single
// LZ4 block follows directly. >0 => that many [int32 chunkSize][lz4 block].
std::vector<uint8_t> UsdDecompress(const uint8_t* blob, std::size_t blobSize,
                                   std::size_t uncompressedSize) {
    if (blobSize == 0) {
        return {};
    }
    const uint8_t nChunks = blob[0];
    if (nChunks == 0) {
        return Lz4BlockDecompress(blob + 1, blobSize - 1, uncompressedSize);
    }
    std::vector<uint8_t> out;
    out.reserve(uncompressedSize);
    std::size_t p = 1;
    for (uint8_t c = 0; c < nChunks; ++c) {
        if (p + 4 > blobSize) {
            throw std::runtime_error("USDC: chunk header truncated");
        }
        int32_t csz;
        std::memcpy(&csz, blob + p, 4);
        p += 4;
        if (csz < 0 || p + static_cast<std::size_t>(csz) > blobSize) {
            throw std::runtime_error("USDC: chunk body truncated");
        }
        auto part = Lz4BlockDecompress(blob + p, static_cast<std::size_t>(csz), uncompressedSize);
        out.insert(out.end(), part.begin(), part.end());
        p += static_cast<std::size_t>(csz);
    }
    return out;
}

// ---------------------------------------------------------------------------
// USD integer-coding (32-bit) decode of an already-DECOMPRESSED buffer.
//   layout: int32 commonValue, then (numInts*2+7)/8 bytes of 2-bit codes,
//           then a variable-width section of deltas.
//   code 0 = commonValue, 1 = int8, 2 = int16, 3 = int32.
//   every value is a delta added to the running previous value (starts at 0).
// ---------------------------------------------------------------------------
std::vector<int32_t> DecodeInts32(const uint8_t* buf, std::size_t bufLen,
                                  std::size_t numInts) {
    std::vector<int32_t> out;
    if (numInts == 0) {
        return out;
    }
    auto req = [&](std::size_t off, std::size_t len) {
        if (off + len > bufLen) {
            throw std::runtime_error("USDC: integer-coded buffer truncated");
        }
    };
    req(0, 4);
    int32_t commonValue;
    std::memcpy(&commonValue, buf, 4);
    const std::size_t codesOff = 4;
    const std::size_t numCodesBytes = (numInts * 2 + 7) / 8;
    req(codesOff, numCodesBytes);
    std::size_t vp = codesOff + numCodesBytes;

    out.reserve(numInts);
    int32_t prev = 0;
    for (std::size_t i = 0; i < numInts; ++i) {
        const uint8_t codeByte = buf[codesOff + (i >> 2)];
        const int code = (codeByte >> (2 * (i & 3))) & 3;
        switch (code) {
        case 0:
            prev += commonValue;
            break;
        case 1: {
            req(vp, 1);
            int8_t v;
            std::memcpy(&v, buf + vp, 1);
            vp += 1;
            prev += v;
            break;
        }
        case 2: {
            req(vp, 2);
            int16_t v;
            std::memcpy(&v, buf + vp, 2);
            vp += 2;
            prev += v;
            break;
        }
        default: {
            req(vp, 4);
            int32_t v;
            std::memcpy(&v, buf + vp, 4);
            vp += 4;
            prev += v;
            break;
        }
        }
        out.push_back(prev);
    }
    return out;
}

// Working-space / encoded-buffer size for a 32-bit integer-coded buffer,
// matching Sdf_IntegerCompression::GetDecompressionWorkingSpaceSize.
std::size_t EncodedBufferSize32(std::size_t numInts) {
    if (numInts == 0) return 0;
    return 4 /*commonValue*/ + ((numInts * 2 + 7) / 8) /*codes*/ + numInts * 4 /*max ints*/;
}

// Read a USD "_ReadCompressedInts" sub-buffer starting at `off`: int64
// compressedSize, then that many bytes (LZ4-wrapped, integer-coded). Advances
// `off`. Returns the decoded int32 array of length `numInts`.
std::vector<int32_t> ReadCompressedInts(const ByteReader& r, std::size_t& off,
                                        std::size_t numInts) {
    const int64_t compSize = r.I64(off);
    off += 8;
    if (compSize < 0) {
        throw std::runtime_error("USDC: negative compressed-int size");
    }
    const uint8_t* blob = r.Ptr(off, static_cast<std::size_t>(compSize));
    off += static_cast<std::size_t>(compSize);
    auto dec = UsdDecompress(blob, static_cast<std::size_t>(compSize),
                             EncodedBufferSize32(numInts));
    return DecodeInts32(dec.data(), dec.size(), numInts);
}

// ---------------------------------------------------------------------------
// ValueRep (64-bit packed): bit63 isArray, bit62 isInlined, bit61 isCompressed,
// bits48..55 type enum, bits0..47 payload (file offset for non-inlined arrays).
// ---------------------------------------------------------------------------
struct ValueRep {
    uint64_t data = 0;
    bool IsArray()      const { return (data >> 63) & 1; }
    bool IsInlined()    const { return (data >> 62) & 1; }
    bool IsCompressed() const { return (data >> 61) & 1; }
    uint32_t Type()     const { return static_cast<uint32_t>((data >> 48) & 0xFF); }
    uint64_t Payload()  const { return data & ((1ull << 48) - 1); }
};

// Crate value-type enums, copied verbatim from the OpenUSD source of truth:
//   pxr/usd/sdf/crateDataTypes.h  (_SDF_CRATE_DATA_TYPES X-macro list).
// The numeric IDs are part of the on-disk crate ABI and are STABLE across crate
// versions (the reader guards crate version >= 0.7; the cup is 0.8.0). They are
// 1-based in file order: Bool=1, UChar=2, Int=3, ..., Vec3f=24. A future spec
// only APPENDS new IDs, so these two stay valid; if that ever changes the version
// guard above (version_minor) is the trip-wire.
constexpr uint32_t kTypeInt   = 3;   // SdfValueTypeNames->Int  (TypeInt)
constexpr uint32_t kTypeVec3f = 24;  // SdfValueTypeNames->Float3 (TypeVec3f: point3f/normal3f/float3)

constexpr uint32_t kFieldSetSentinel = 0xFFFFFFFFu;  // default FieldIndex

// MinCompressedArraySize from crateFile.cpp: arrays below this are stored raw.
constexpr std::size_t kMinCompressedArraySize = 16;

// Sanity cap on a count read from the (untrusted) crate payload, used only where
// a byte bound is impossible (integer-compressed arrays expand from far fewer
// on-disk bytes than n*4). Generously above any real mesh's element count; a
// count past it is treated as corruption and rejected before any allocation.
constexpr std::size_t kMaxArrayElements = 1ull << 30;  // 1 Gi elements

// ---------------------------------------------------------------------------
// Crate document: parsed structural sections.
// ---------------------------------------------------------------------------
struct Crate {
    ByteReader r;
    std::vector<std::string> tokens;
    std::vector<uint32_t>    field_token;   // per field: token index (property/meta name)
    std::vector<ValueRep>    field_rep;     // per field: value rep
    std::vector<int32_t>     fieldsets;     // flat runs, -1 sentinel terminated
    std::vector<std::string> paths;         // path string per path index
    std::vector<uint32_t>    spec_path;     // per spec: path index
    std::vector<uint32_t>    spec_fieldset; // per spec: fieldset start index
    uint8_t version_major = 0;
    uint8_t version_minor = 0;

    explicit Crate(const std::vector<uint8_t>& data) : r(data) {}
};

void ReadTokens(Crate& c, std::size_t off) {
    const uint64_t numTokens = c.r.U64(off);
    const uint64_t uncompSize = c.r.U64(off + 8);
    const uint64_t compSize = c.r.U64(off + 16);
    const uint8_t* blob = c.r.Ptr(off + 24, static_cast<std::size_t>(compSize));
    auto dec = UsdDecompress(blob, static_cast<std::size_t>(compSize),
                             static_cast<std::size_t>(uncompSize));
    if (dec.size() < uncompSize) {
        throw std::runtime_error("USDC: token blob under-decompressed");
    }
    // Null-separated strings.
    c.tokens.clear();
    c.tokens.reserve(static_cast<std::size_t>(numTokens));
    std::size_t start = 0;
    for (std::size_t i = 0; i < uncompSize && c.tokens.size() < numTokens; ++i) {
        if (dec[i] == 0) {
            c.tokens.emplace_back(reinterpret_cast<const char*>(dec.data() + start), i - start);
            start = i + 1;
        }
    }
    if (c.tokens.size() != numTokens) {
        throw std::runtime_error("USDC: token count mismatch");
    }
}

void ReadFields(Crate& c, std::size_t off) {
    const uint64_t numFields = c.r.U64(off);
    off += 8;
    auto tokenIdx = ReadCompressedInts(c.r, off, static_cast<std::size_t>(numFields));
    // Compressed value reps: uint64 repsSize, then LZ4 of numFields*8 bytes
    // (NOT integer-coded -- a straight LZ4 of the raw uint64 array).
    const int64_t repsSize = c.r.I64(off);
    off += 8;
    if (repsSize < 0) {
        throw std::runtime_error("USDC: negative reps size");
    }
    const uint8_t* blob = c.r.Ptr(off, static_cast<std::size_t>(repsSize));
    auto raw = UsdDecompress(blob, static_cast<std::size_t>(repsSize),
                             static_cast<std::size_t>(numFields) * 8);
    if (raw.size() < numFields * 8) {
        throw std::runtime_error("USDC: value-rep blob under-decompressed");
    }
    c.field_token.resize(static_cast<std::size_t>(numFields));
    c.field_rep.resize(static_cast<std::size_t>(numFields));
    for (std::size_t i = 0; i < numFields; ++i) {
        c.field_token[i] = static_cast<uint32_t>(tokenIdx[i]);
        uint64_t rep;
        std::memcpy(&rep, raw.data() + i * 8, 8);
        c.field_rep[i].data = rep;
    }
}

void ReadFieldSets(Crate& c, std::size_t off) {
    const uint64_t numFieldSets = c.r.U64(off);
    off += 8;
    c.fieldsets = ReadCompressedInts(c.r, off, static_cast<std::size_t>(numFieldSets));
}

void ReadPaths(Crate& c, std::size_t off) {
    // The path count is written TWICE: _ReadPaths reads it to size the table,
    // then _ReadCompressedPaths reads it again before the arrays.
    const uint64_t numPaths = c.r.U64(off);
    off += 8;
    const uint64_t numPaths2 = c.r.U64(off);
    off += 8;
    if (numPaths != numPaths2) {
        throw std::runtime_error("USDC: PATHS count mismatch");
    }
    const std::size_t n = static_cast<std::size_t>(numPaths);
    auto pathIndexes = ReadCompressedInts(c.r, off, n);
    auto elemTokens = ReadCompressedInts(c.r, off, n);  // int32; sign => property
    auto jumps = ReadCompressedInts(c.r, off, n);       // int32

    c.paths.assign(n, std::string());

    // Rebuild the path tree, matching _BuildDecompressedPathsImpl. Iterative
    // explicit stack to avoid deep recursion.
    struct Frame { std::size_t curIndex; std::string parent; bool root; };
    std::vector<Frame> stack;
    stack.push_back({0, std::string(), true});
    while (!stack.empty()) {
        Frame fr = stack.back();
        stack.pop_back();
        std::size_t curIndex = fr.curIndex;
        std::string parent = fr.parent;
        bool isRootStart = fr.root;
        while (true) {
            const std::size_t thisIndex = curIndex++;
            if (thisIndex >= n) {
                throw std::runtime_error("USDC: corrupt paths encoding");
            }
            const uint32_t pi = static_cast<uint32_t>(pathIndexes[thisIndex]);
            if (pi >= n) {
                throw std::runtime_error("USDC: path index out of range");
            }
            std::string thisPath;
            if (isRootStart) {
                thisPath = "/";
                isRootStart = false;
            } else {
                int32_t ti = elemTokens[thisIndex];
                const bool isProperty = ti < 0;
                if (ti < 0) ti = -ti;
                if (static_cast<std::size_t>(ti) >= c.tokens.size()) {
                    throw std::runtime_error("USDC: path token index out of range");
                }
                const std::string& elem = c.tokens[static_cast<std::size_t>(ti)];
                std::string base = (parent == "/") ? std::string() : parent;
                thisPath = base + (isProperty ? "." : "/") + elem;
            }
            c.paths[pi] = thisPath;

            const int32_t jump = jumps[thisIndex];
            const bool hasChild = (jump > 0) || (jump == -1);
            const bool hasSibling = (jump >= 0);
            if (hasChild) {
                if (hasSibling) {
                    const std::size_t sibling = thisIndex + static_cast<std::size_t>(jump);
                    stack.push_back({sibling, parent, false});
                }
                parent = thisPath;  // descend into child (next in stream)
            }
            if (!(hasChild || hasSibling)) {
                break;
            }
        }
    }
}

void ReadSpecs(Crate& c, std::size_t off) {
    const uint64_t numSpecs = c.r.U64(off);
    off += 8;
    const std::size_t n = static_cast<std::size_t>(numSpecs);
    auto pathIdx = ReadCompressedInts(c.r, off, n);
    auto fsIdx = ReadCompressedInts(c.r, off, n);
    (void)ReadCompressedInts(c.r, off, n);  // specType (unused for geometry)
    c.spec_path.resize(n);
    c.spec_fieldset.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        c.spec_path[i] = static_cast<uint32_t>(pathIdx[i]);
        c.spec_fieldset[i] = static_cast<uint32_t>(fsIdx[i]);
    }
}

// Resolve the `default` field ValueRep for a given property path. Returns true
// and fills `out` on success.
bool DefaultRepForPath(const Crate& c, const std::string& path, ValueRep& out) {
    // Find the spec whose path matches.
    for (std::size_t s = 0; s < c.spec_path.size(); ++s) {
        if (c.spec_path[s] >= c.paths.size()) continue;
        if (c.paths[c.spec_path[s]] != path) continue;
        // Walk this spec's fieldset run until the -1 sentinel.
        for (std::size_t j = c.spec_fieldset[s]; j < c.fieldsets.size(); ++j) {
            const uint32_t fi = static_cast<uint32_t>(c.fieldsets[j]);
            if (fi == kFieldSetSentinel) break;
            if (fi >= c.field_token.size()) {
                throw std::runtime_error("USDC: field index out of range");
            }
            const uint32_t tok = c.field_token[fi];
            if (tok < c.tokens.size() && c.tokens[tok] == "default") {
                out = c.field_rep[fi];
                return true;
            }
        }
        return false;  // matched the spec but it had no `default` field
    }
    return false;
}

// Read a Vec3f array (point3f[] / normal3f[]) at a ValueRep. For version >= 0.7
// the on-disk layout at the payload offset is: uint64 count, then count*3 raw
// little-endian float32. Vec3f arrays are not integer-compressed.
std::vector<float> ReadVec3fArray(const Crate& c, const ValueRep& rep) {
    if (rep.Payload() == 0) {
        return {};  // empty array sentinel
    }
    if (rep.IsCompressed()) {
        throw std::runtime_error("USDC: unexpected compressed float array (out of scope)");
    }
    std::size_t o = static_cast<std::size_t>(rep.Payload());
    const uint64_t count = c.r.U64(o);
    o += 8;  // o <= size() here (U64 already RequireRange'd o+8).
    // Validate the count BEFORE allocating: a garbage-large count must fail with
    // the file's runtime_error, not bad_alloc/length_error. Bound in division form
    // so an untrusted count*12 (or count*3) cannot wrap past the check. Vec3f is
    // always raw little-endian f32 here, so the exact byte cost is count*12.
    if (count > (c.r.size() - o) / 12u) {
        throw std::runtime_error("USDC: vec3f array count exceeds remaining payload");
    }
    std::vector<float> out;
    out.resize(static_cast<std::size_t>(count) * 3);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = c.r.F32(o + i * 4);
    }
    return out;
}

// Read an int[] array at a ValueRep. Layout at payload: uint64 count, then
// either raw int32 (count < 16 or not compressed) or [int64 compSize][...]
// integer-compressed ints.
std::vector<int> ReadIntArray(const Crate& c, const ValueRep& rep) {
    if (rep.Payload() == 0) {
        return {};
    }
    std::size_t o = static_cast<std::size_t>(rep.Payload());
    const uint64_t count = c.r.U64(o);
    o += 8;  // o <= size() here (U64 already RequireRange'd o+8).
    const bool raw = (!rep.IsCompressed() || count < kMinCompressedArraySize);
    // Validate the count BEFORE reserving: a garbage count must fail with the
    // file's runtime_error, not bad_alloc/length_error. The raw path consumes n*4
    // bytes, so bound it exactly in division form (overflow-safe). The compressed
    // path expands from far fewer on-disk bytes than n*4, so no byte bound is
    // possible there -- fall back to a sane max-element cap (this also shields the
    // downstream EncodedBufferSize32(n) reserve in ReadCompressedInts).
    if (raw) {
        if (count > (c.r.size() - o) / 4u) {
            throw std::runtime_error("USDC: int array count exceeds remaining payload");
        }
    } else if (count > kMaxArrayElements) {
        throw std::runtime_error("USDC: compressed int array count exceeds sanity cap");
    }
    const std::size_t n = static_cast<std::size_t>(count);
    std::vector<int> out;
    out.reserve(n);
    if (raw) {
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(static_cast<int>(c.r.U32(o + i * 4)));
        }
        return out;
    }
    auto decoded = ReadCompressedInts(c.r, o, n);
    for (int32_t v : decoded) {
        out.push_back(static_cast<int>(v));
    }
    return out;
}

} // namespace

CrateMesh ReadUsdcMesh(const std::string& path, const std::string& target_prim_path) {
    // -- Load whole file ----------------------------------------------------
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("USDC: failed to open file: " + path);
    }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    Crate c(data);

    // -- Bootstrap ----------------------------------------------------------
    static const char kMagic[8] = {'P', 'X', 'R', '-', 'U', 'S', 'D', 'C'};
    if (data.size() < 88 || std::memcmp(data.data(), kMagic, 8) != 0) {
        throw std::runtime_error("USDC: not a crate file (bad magic): " + path);
    }
    c.version_major = data[8];
    c.version_minor = data[9];
    // We target the 0.x line that uses compressed structural sections (>= 0.4)
    // and the >= 0.7 array-count width. The cup is 0.8.0.
    if (c.version_major != 0 || c.version_minor < 7) {
        std::ostringstream msg;
        msg << "USDC: unsupported crate version " << int(c.version_major) << "."
            << int(c.version_minor) << " (scoped reader targets 0.7+): " << path;
        throw std::runtime_error(msg.str());
    }
    const int64_t tocOffset = c.r.I64(16);
    if (tocOffset < 0 || static_cast<std::size_t>(tocOffset) >= data.size()) {
        throw std::runtime_error("USDC: bad TOC offset");
    }

    // -- TOC ----------------------------------------------------------------
    std::size_t p = static_cast<std::size_t>(tocOffset);
    const int64_t numSections = c.r.I64(p);
    p += 8;
    if (numSections < 0 || numSections > 64) {
        throw std::runtime_error("USDC: implausible section count");
    }
    std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> sections;
    for (int64_t i = 0; i < numSections; ++i) {
        const uint8_t* namePtr = c.r.Ptr(p, 16);
        std::size_t nameLen = 0;
        while (nameLen < 16 && namePtr[nameLen] != 0) ++nameLen;
        std::string name(reinterpret_cast<const char*>(namePtr), nameLen);
        p += 16;
        const int64_t sOff = c.r.I64(p);
        p += 8;
        const int64_t sSz = c.r.I64(p);
        p += 8;
        if (sOff < 0 || sSz < 0) {
            throw std::runtime_error("USDC: bad section bounds");
        }
        sections[name] = {static_cast<std::size_t>(sOff), static_cast<std::size_t>(sSz)};
    }
    auto require = [&](const char* name) -> std::size_t {
        auto it = sections.find(name);
        if (it == sections.end()) {
            throw std::runtime_error(std::string("USDC: missing section ") + name);
        }
        return it->second.first;
    };

    // -- Structural sections ------------------------------------------------
    ReadTokens(c, require("TOKENS"));
    ReadFields(c, require("FIELDS"));
    ReadFieldSets(c, require("FIELDSETS"));
    ReadPaths(c, require("PATHS"));
    ReadSpecs(c, require("SPECS"));

    // -- Resolve the target Mesh prim --------------------------------------
    std::string mesh_path = target_prim_path;
    if (mesh_path.empty()) {
        // First path that has a `.points` property.
        for (const auto& pth : c.paths) {
            if (pth.size() > 7 && pth.compare(pth.size() - 7, 7, ".points") == 0) {
                mesh_path = pth.substr(0, pth.size() - 7);
                break;
            }
        }
    }
    if (mesh_path.empty()) {
        throw std::runtime_error("USDC: no Mesh prim with points found: " + path);
    }

    CrateMesh out;
    out.prim_path = mesh_path;

    ValueRep ptsRep;
    if (!DefaultRepForPath(c, mesh_path + ".points", ptsRep)) {
        throw std::runtime_error("USDC: target Mesh has no points: " + mesh_path);
    }
    if (!ptsRep.IsArray() || ptsRep.Type() != kTypeVec3f) {
        throw std::runtime_error("USDC: points is not a point3f[] array");
    }
    out.points = ReadVec3fArray(c, ptsRep);

    // faceVertexIndices / faceVertexCounts: present for a real Mesh; if absent
    // the caller may still cook a convex hull from points alone.
    ValueRep fviRep;
    if (DefaultRepForPath(c, mesh_path + ".faceVertexIndices", fviRep)) {
        if (!fviRep.IsArray() || fviRep.Type() != kTypeInt) {
            throw std::runtime_error("USDC: faceVertexIndices is not an int[] array");
        }
        out.face_vertex_indices = ReadIntArray(c, fviRep);
    }
    ValueRep fvcRep;
    if (DefaultRepForPath(c, mesh_path + ".faceVertexCounts", fvcRep)) {
        if (!fvcRep.IsArray() || fvcRep.Type() != kTypeInt) {
            throw std::runtime_error("USDC: faceVertexCounts is not an int[] array");
        }
        out.face_vertex_counts = ReadIntArray(c, fvcRep);
    }

    return out;
}

} // namespace nuka::import
