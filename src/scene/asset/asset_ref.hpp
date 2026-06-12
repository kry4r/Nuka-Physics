#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::AssetRef - reference into a .nka asset container.
//
// A reference is "<nka_path>#<TAG><index>" where TAG is a 4-char FourCC chunk
// tag (e.g. "SDF0") and index selects the asset within that chunk family. The
// .nka container itself lands in M2c; this header is only the reference type
// that components (VisualMesh, CollisionShape, SoftBody, RenderMaterial
// textures, ...) embed.
//
//   "cup.nka#SDF0/0"  ->  { nka_path="cup.nka", fourcc='SDF0', index=0 }
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

namespace nuka::scene {

struct AssetRef {
    std::string nka_path;       // container file, e.g. "cup.nka"
    uint32_t    fourcc = 0;     // 4-char chunk tag packed big-endian (see MakeFourCC)
    uint32_t    index  = 0;     // which asset of that family within the chunk

    bool operator==(const AssetRef& o) const {
        return nka_path == o.nka_path && fourcc == o.fourcc && index == o.index;
    }
    bool operator!=(const AssetRef& o) const { return !(*this == o); }

    bool Empty() const { return nka_path.empty() && fourcc == 0; }
};

// Pack a 4-char tag into a uint32 (big-endian: tag[0] is the most-significant
// byte) so ToString round-trips it back to the same character order.
inline uint32_t MakeFourCC(char a, char b, char c, char d) {
    return (static_cast<uint32_t>(static_cast<unsigned char>(a)) << 24) |
           (static_cast<uint32_t>(static_cast<unsigned char>(b)) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(c)) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(d)));
}

// Render the 4-char tag back out (most-significant byte first).
inline std::string FourCCToString(uint32_t fourcc) {
    std::string s;
    s.push_back(static_cast<char>((fourcc >> 24) & 0xFF));
    s.push_back(static_cast<char>((fourcc >> 16) & 0xFF));
    s.push_back(static_cast<char>((fourcc >> 8) & 0xFF));
    s.push_back(static_cast<char>(fourcc & 0xFF));
    return s;
}

// Parse "cup.nka#SDF0/0". On any malformed input the path is taken verbatim and
// fourcc/index default to 0 (a best-effort, never-throws parser).
inline AssetRef ParseAssetRef(const std::string& text) {
    AssetRef ref;
    const auto hash = text.find('#');
    if (hash == std::string::npos) {
        ref.nka_path = text;
        return ref;
    }
    ref.nka_path = text.substr(0, hash);

    std::string suffix = text.substr(hash + 1);  // "SDF0/0"
    std::string tag;
    std::string idx;
    const auto slash = suffix.find('/');
    if (slash == std::string::npos) {
        tag = suffix;
    } else {
        tag = suffix.substr(0, slash);
        idx = suffix.substr(slash + 1);
    }

    if (tag.size() == 4) {
        ref.fourcc = MakeFourCC(tag[0], tag[1], tag[2], tag[3]);
    }
    if (!idx.empty()) {
        // best-effort decimal parse; leaves index=0 on garbage
        uint32_t value = 0;
        bool any = false;
        for (char ch : idx) {
            if (ch < '0' || ch > '9') { any = false; break; }
            value = value * 10u + static_cast<uint32_t>(ch - '0');
            any = true;
        }
        if (any) {
            ref.index = value;
        }
    }
    return ref;
}

inline std::string ToString(const AssetRef& ref) {
    std::string out = ref.nka_path;
    if (ref.fourcc != 0 || ref.index != 0) {
        out.push_back('#');
        out += FourCCToString(ref.fourcc);
        out.push_back('/');
        out += std::to_string(ref.index);
    }
    return out;
}

} // namespace nuka::scene
