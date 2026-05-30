// go2_policy_check.cpp
//
// Standalone C++ inference-validation harness for the trained Go2 locomotion
// policy. Loads the TorchScript model (an MLP 48 -> 512 -> 256 -> 128 -> 12,
// ELU), runs it on CPU, and validates the 12-dim action outputs against a
// golden JSON produced by the reference Python pipeline.
//
// Correctness bar:
//   (1) the model file's SHA-256 must match golden["sha256_model"] -- so we
//       know we loaded the exact asset the golden was generated from, and
//   (2) every case's max |y_cpp - y_golden| must be < 1e-4 (we expect ~1e-5).
//
// This harness links ONLY libtorch. It deliberately does NOT link any Nuka
// engine library: the pip libtorch is built pre-CXX11-ABI
// (-D_GLIBCXX_USE_CXX11_ABI=0) while the engine is CXX11-ABI=1, so mixing them
// would clash at the std::string / linker ABI boundary. See README.md.
//
// Usage:
//   go2_policy_check [model.pt] [golden_io.json]
// Defaults point at the PR62 asset paths below.

#include <torch/script.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kDefaultModel  = "/root/third_party/go2_pr62/motion.pt";
constexpr const char* kDefaultGolden = "/root/third_party/go2_pr62/golden_io.json";
constexpr int   kInDim    = 48;
constexpr int   kOutDim   = 12;
constexpr double kTol     = 1e-4;  // hard pass/fail threshold

// ---------------------------------------------------------------------------
// Minimal, self-contained SHA-256 (public-domain style implementation).
// Used to confirm we loaded the same .pt the golden was made from.
// ---------------------------------------------------------------------------
class Sha256 {
 public:
  Sha256() { reset(); }

  void update(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buffer_[buffer_len_++] = data[i];
      if (buffer_len_ == 64) {
        transform(buffer_);
        bit_len_ += 512;
        buffer_len_ = 0;
      }
    }
  }

  std::string hex_digest() {
    uint8_t hash[32];
    finalize(hash);
    static const char* hx = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 32; ++i) {
      out.push_back(hx[hash[i] >> 4]);
      out.push_back(hx[hash[i] & 0xF]);
    }
    return out;
  }

 private:
  uint32_t state_[8];
  uint8_t  buffer_[64];
  uint32_t buffer_len_;
  uint64_t bit_len_;

  static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

  void reset() {
    state_[0] = 0x6a09e667; state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372; state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f; state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab; state_[7] = 0x5be0cd19;
    buffer_len_ = 0;
    bit_len_    = 0;
  }

  void transform(const uint8_t* p) {
    static const uint32_t k[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(p[i*4]) << 24) | (uint32_t(p[i*4+1]) << 16) |
             (uint32_t(p[i*4+2]) << 8) | uint32_t(p[i*4+3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
      uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19)  ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
    uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + k[i] + w[i];
      uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
    state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
  }

  void finalize(uint8_t out[32]) {
    uint64_t total_bits = bit_len_ + uint64_t(buffer_len_) * 8;
    // append 0x80 then pad with zeros, leaving room for 8-byte length
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0x00;
    while (buffer_len_ != 56) update(&zero, 1);
    uint8_t lenbytes[8];
    for (int i = 0; i < 8; ++i)
      lenbytes[7-i] = uint8_t((total_bits >> (i*8)) & 0xFF);
    update(lenbytes, 8);  // triggers final transform
    for (int i = 0; i < 8; ++i) {
      out[i*4]   = uint8_t((state_[i] >> 24) & 0xFF);
      out[i*4+1] = uint8_t((state_[i] >> 16) & 0xFF);
      out[i*4+2] = uint8_t((state_[i] >> 8)  & 0xFF);
      out[i*4+3] = uint8_t( state_[i]        & 0xFF);
    }
  }
};

std::string sha256_of_file(const std::string& path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { *ok = false; return {}; }
  Sha256 h;
  std::vector<char> buf(1 << 16);
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    std::streamsize got = f.gcount();
    if (got > 0) h.update(reinterpret_cast<const uint8_t*>(buf.data()),
                          static_cast<size_t>(got));
  }
  *ok = true;
  return h.hex_digest();
}

// ---------------------------------------------------------------------------
// Tiny tolerant JSON reader for this known-shape golden file.
// We don't implement a general JSON parser; we only need:
//   - a string value:  find_string_value("sha256_model")
//   - the list of cases, each with "input":[...] and "output":[...] arrays.
// The scanner walks the raw text; numbers are read with strtod.
// ---------------------------------------------------------------------------
struct Case {
  std::string name;
  std::vector<float> input;
  std::vector<float> output;
};

std::string read_file(const std::string& path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { *ok = false; return {}; }
  std::ostringstream ss;
  ss << f.rdbuf();
  *ok = true;
  return ss.str();
}

// Return the string value for "key": "value" (first occurrence). Empty if none.
std::string find_string_value(const std::string& js, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  size_t k = js.find(needle);
  if (k == std::string::npos) return {};
  size_t colon = js.find(':', k + needle.size());
  if (colon == std::string::npos) return {};
  size_t q1 = js.find('"', colon + 1);
  if (q1 == std::string::npos) return {};
  size_t q2 = js.find('"', q1 + 1);
  if (q2 == std::string::npos) return {};
  return js.substr(q1 + 1, q2 - q1 - 1);
}

// Parse the JSON array that begins at js[pos] (pos must index a '['). Reads
// floats until the matching ']'. Returns the values and advances *pos to one
// past the ']'. Returns false on malformed input.
bool parse_float_array(const std::string& js, size_t* pos, std::vector<float>* out) {
  size_t i = *pos;
  if (i >= js.size() || js[i] != '[') return false;
  ++i;
  out->clear();
  while (i < js.size()) {
    while (i < js.size() && (std::isspace((unsigned char)js[i]) || js[i] == ',')) ++i;
    if (i < js.size() && js[i] == ']') { *pos = i + 1; return true; }
    const char* start = js.c_str() + i;
    char* end = nullptr;
    double v = std::strtod(start, &end);
    if (end == start) return false;  // no number consumed
    out->push_back(static_cast<float>(v));
    i += static_cast<size_t>(end - start);
  }
  return false;  // unterminated
}

// Find the next `"key": [` after `from`; set *arr_pos to the '['. Returns npos
// if not found.
size_t find_array_key(const std::string& js, const std::string& key, size_t from,
                      size_t* arr_pos) {
  std::string needle = "\"" + key + "\"";
  size_t k = js.find(needle, from);
  if (k == std::string::npos) return std::string::npos;
  size_t br = js.find('[', k + needle.size());
  if (br == std::string::npos) return std::string::npos;
  *arr_pos = br;
  return k;
}

bool parse_golden(const std::string& js, std::string* sha_out,
                  std::vector<Case>* cases_out) {
  *sha_out = find_string_value(js, "sha256_model");
  if (sha_out->empty()) return false;

  // Walk the "cases" array by repeatedly finding "input" / "output" arrays in
  // order. Each case has exactly one of each. We anchor on "input" because it
  // appears before "output" in each object.
  size_t cursor = js.find("\"cases\"");
  if (cursor == std::string::npos) return false;

  cases_out->clear();
  while (true) {
    size_t in_arr = 0;
    size_t in_key = find_array_key(js, "input", cursor, &in_arr);
    if (in_key == std::string::npos) break;  // no more cases

    Case c;
    // Optional: grab the case "name" that precedes this input (best-effort).
    {
      size_t nm = js.rfind("\"name\"", in_key);
      if (nm != std::string::npos && nm > cursor) {
        size_t colon = js.find(':', nm);
        size_t q1 = (colon == std::string::npos) ? std::string::npos
                                                  : js.find('"', colon + 1);
        size_t q2 = (q1 == std::string::npos) ? std::string::npos
                                              : js.find('"', q1 + 1);
        if (q2 != std::string::npos) c.name = js.substr(q1 + 1, q2 - q1 - 1);
      }
    }

    size_t p = in_arr;
    if (!parse_float_array(js, &p, &c.input)) return false;

    size_t out_arr = 0;
    size_t out_key = find_array_key(js, "output", p, &out_arr);
    if (out_key == std::string::npos) return false;  // input without output
    size_t q = out_arr;
    if (!parse_float_array(js, &q, &c.output)) return false;

    cases_out->push_back(std::move(c));
    cursor = q;  // continue scanning after this output array
  }
  return !cases_out->empty();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_path  = (argc > 1) ? argv[1] : kDefaultModel;
  const std::string golden_path = (argc > 2) ? argv[2] : kDefaultGolden;

  std::cout << "Go2 policy C++ inference validation (CPU, libtorch)\n";
  std::cout << "  model : " << model_path  << "\n";
  std::cout << "  golden: " << golden_path << "\n\n";

  // --- Load golden ---------------------------------------------------------
  bool ok = false;
  std::string js = read_file(golden_path, &ok);
  if (!ok) { std::cerr << "ERROR: cannot read golden: " << golden_path << "\n"; return 2; }

  std::string golden_sha;
  std::vector<Case> cases;
  if (!parse_golden(js, &golden_sha, &cases)) {
    std::cerr << "ERROR: failed to parse golden JSON\n";
    return 2;
  }
  std::cout << "Parsed " << cases.size() << " golden case(s).\n";

  // --- SHA-256 of the model asset -----------------------------------------
  bool sha_ok = false;
  std::string file_sha = sha256_of_file(model_path, &sha_ok);
  if (!sha_ok) { std::cerr << "ERROR: cannot read model file: " << model_path << "\n"; return 2; }

  const bool sha_match = (file_sha == golden_sha);
  std::cout << "SHA-256 model : " << file_sha << "\n";
  std::cout << "SHA-256 golden: " << golden_sha << "\n";
  std::cout << "SHA-256 match : " << (sha_match ? "YES" : "NO") << "\n\n";

  // --- Load TorchScript model on CPU --------------------------------------
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path, torch::kCPU);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: torch::jit::load failed: " << e.what() << "\n";
    return 2;
  }
  module.eval();
  module.to(torch::kCPU);
  torch::NoGradGuard no_grad;

  // --- Run each case -------------------------------------------------------
  std::cout << std::left << std::setw(24) << "case"
            << std::setw(10) << "in_dim"
            << std::setw(10) << "out_dim"
            << std::setw(14) << "max_err"
            << "result\n";
  std::cout << std::string(64, '-') << "\n";

  bool all_pass = true;
  double overall_max_err = 0.0;

  for (const auto& c : cases) {
    if (static_cast<int>(c.input.size()) != kInDim ||
        static_cast<int>(c.output.size()) != kOutDim) {
      std::cout << std::left << std::setw(24) << (c.name.empty() ? "?" : c.name)
                << std::setw(10) << c.input.size()
                << std::setw(10) << c.output.size()
                << std::setw(14) << "-"
                << "FAIL (dim)\n";
      all_pass = false;
      continue;
    }

    torch::Tensor x = torch::from_blob(
        const_cast<float*>(c.input.data()), {1, kInDim}, torch::kFloat32).clone();

    torch::Tensor y;
    try {
      std::vector<torch::jit::IValue> inputs{x};
      y = module.forward(inputs).toTensor().to(torch::kCPU).contiguous();
    } catch (const std::exception& e) {
      std::cout << std::left << std::setw(24) << (c.name.empty() ? "?" : c.name)
                << "FAIL (forward: " << e.what() << ")\n";
      all_pass = false;
      continue;
    }

    if (y.numel() != kOutDim) {
      std::cout << std::left << std::setw(24) << (c.name.empty() ? "?" : c.name)
                << "FAIL (out numel=" << y.numel() << ")\n";
      all_pass = false;
      continue;
    }

    const float* yp = y.data_ptr<float>();
    double max_err = 0.0;
    for (int i = 0; i < kOutDim; ++i) {
      double e = std::fabs(static_cast<double>(yp[i]) -
                           static_cast<double>(c.output[i]));
      max_err = std::max(max_err, e);
    }
    overall_max_err = std::max(overall_max_err, max_err);
    const bool pass = (max_err < kTol);
    all_pass = all_pass && pass;

    std::ostringstream errs;
    errs << std::scientific << std::setprecision(3) << max_err;
    std::cout << std::left << std::setw(24) << (c.name.empty() ? "?" : c.name)
              << std::setw(10) << kInDim
              << std::setw(10) << kOutDim
              << std::setw(14) << errs.str()
              << (pass ? "PASS" : "FAIL") << "\n";
  }

  std::cout << std::string(64, '-') << "\n";
  std::cout << "overall max_err = " << std::scientific << std::setprecision(3)
            << overall_max_err << "  (tol " << kTol << ")\n";

  const bool overall_ok = all_pass && sha_match;
  std::cout << "\nOVERALL: " << (overall_ok ? "PASS" : "FAIL")
            << "  (golden_match=" << (all_pass ? "yes" : "no")
            << ", sha256_match=" << (sha_match ? "yes" : "no") << ")\n";

  return overall_ok ? 0 : 1;
}
