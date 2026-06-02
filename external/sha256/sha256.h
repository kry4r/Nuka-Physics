// ---------------------------------------------------------------------------
// Minimal public-domain SHA-256 (single-header)
// ---------------------------------------------------------------------------
// Vendored for the Nuka Physics asset cooker to content-hash decomposition
// inputs (mesh bytes + serialized params). This is a small, self-contained,
// public-domain implementation: no external dependencies, no allocation.
//
// Origin: derived from the widely-used public-domain SHA-256 reference
// implementation by Brad Conte (https://github.com/B-Con/crypto-algorithms),
// which the author released into the public domain. Reformatted into a
// single header and namespaced for this project. See external/sha256/LICENSE.
//
// Usage:
//   nuka::sha256::Hasher h;
//   h.Update(ptr, len);
//   std::array<uint8_t, 32> digest = h.Final();
//   std::string hex = nuka::sha256::ToHex(digest);
// ---------------------------------------------------------------------------
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace nuka::sha256 {

class Hasher {
public:
    Hasher() { Reset(); }

    void Reset() {
        datalen_ = 0;
        bitlen_ = 0;
        state_[0] = 0x6a09e667u;
        state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u;
        state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu;
        state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu;
        state_[7] = 0x5be0cd19u;
    }

    void Update(const void* data, size_t len) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            buffer_[datalen_++] = bytes[i];
            if (datalen_ == 64) {
                Transform(buffer_);
                bitlen_ += 512;
                datalen_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> Final() {
        uint32_t i = datalen_;

        // Pad whatever data is left in the buffer.
        if (datalen_ < 56) {
            buffer_[i++] = 0x80;
            while (i < 56) buffer_[i++] = 0x00;
        } else {
            buffer_[i++] = 0x80;
            while (i < 64) buffer_[i++] = 0x00;
            Transform(buffer_);
            for (uint32_t j = 0; j < 56; ++j) buffer_[j] = 0x00;
        }

        // Append the total message length in bits and transform the final block.
        bitlen_ += static_cast<uint64_t>(datalen_) * 8u;
        buffer_[63] = static_cast<uint8_t>(bitlen_);
        buffer_[62] = static_cast<uint8_t>(bitlen_ >> 8);
        buffer_[61] = static_cast<uint8_t>(bitlen_ >> 16);
        buffer_[60] = static_cast<uint8_t>(bitlen_ >> 24);
        buffer_[59] = static_cast<uint8_t>(bitlen_ >> 32);
        buffer_[58] = static_cast<uint8_t>(bitlen_ >> 40);
        buffer_[57] = static_cast<uint8_t>(bitlen_ >> 48);
        buffer_[56] = static_cast<uint8_t>(bitlen_ >> 56);
        Transform(buffer_);

        // SHA uses big endian byte ordering.
        std::array<uint8_t, 32> digest{};
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t k = 0; k < 8; ++k) {
                digest[j + (k * 4)] =
                    static_cast<uint8_t>((state_[k] >> (24 - j * 8)) & 0xff);
            }
        }
        return digest;
    }

private:
    static uint32_t Rotr(uint32_t a, uint32_t b) {
        return (a >> b) | (a << (32 - b));
    }

    void Transform(const uint8_t* data) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
            0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        uint32_t m[64];
        uint32_t i = 0;
        for (uint32_t j = 0; i < 16; ++i, j += 4) {
            m[i] = (static_cast<uint32_t>(data[j]) << 24) |
                   (static_cast<uint32_t>(data[j + 1]) << 16) |
                   (static_cast<uint32_t>(data[j + 2]) << 8) |
                   (static_cast<uint32_t>(data[j + 3]));
        }
        for (; i < 64; ++i) {
            const uint32_t s0 =
                Rotr(m[i - 15], 7) ^ Rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
            const uint32_t s1 =
                Rotr(m[i - 2], 17) ^ Rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
            m[i] = m[i - 16] + s0 + m[i - 7] + s1;
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

        for (i = 0; i < 64; ++i) {
            const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + m[i];
            const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint8_t  buffer_[64]{};
    uint32_t datalen_ = 0;
    uint64_t bitlen_  = 0;
    uint32_t state_[8]{};
};

inline std::string ToHex(const std::array<uint8_t, 32>& digest) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t byte : digest) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0f]);
    }
    return out;
}

// Convenience one-shot helper.
inline std::string HashHex(const void* data, size_t len) {
    Hasher h;
    h.Update(data, len);
    return ToHex(h.Final());
}

} // namespace nuka::sha256
