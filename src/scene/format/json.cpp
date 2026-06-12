// ---------------------------------------------------------------------------
// nuka::scene::json - minimal JSON value/writer/parser implementation (M2c).
// ---------------------------------------------------------------------------
// HOST-ONLY, dependency-free. Determinism: insertion-ordered object keys, "%.9g"
// floats (binary32-lossless). The parser is a hand-written recursive descent
// with 1-based line/column error reporting.
// ---------------------------------------------------------------------------

#include "scene/format/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace nuka::scene::json {

// ---------------------------------------------------------------------------
// scalar accessors
// ---------------------------------------------------------------------------
bool Value::AsBool() const {
    if (type_ != Type::Bool) throw ParseError("json: value is not a bool");
    return bool_;
}

int64_t Value::AsInt() const {
    if (type_ == Type::Int) return int_;
    if (type_ == Type::Float) return static_cast<int64_t>(float_);
    throw ParseError("json: value is not a number");
}

double Value::AsDouble() const {
    if (type_ == Type::Int) return static_cast<double>(int_);
    if (type_ == Type::Float) return float_;
    throw ParseError("json: value is not a number");
}

const std::string& Value::AsString() const {
    if (type_ != Type::String) throw ParseError("json: value is not a string");
    return str_;
}

// ---------------------------------------------------------------------------
// object / array ops
// ---------------------------------------------------------------------------
void Value::Set(const std::string& key, Value value) {
    object_.emplace_back(key, std::move(value));
}

bool Value::Has(const std::string& key) const { return Find(key) != nullptr; }

const Value* Value::Find(const std::string& key) const {
    for (const auto& kv : object_) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

const Value& Value::At(const std::string& key) const {
    const Value* v = Find(key);
    if (!v) throw ParseError("json: missing key '" + key + "'");
    return *v;
}

void Value::PushBack(Value value) { array_.push_back(std::move(value)); }

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------
std::string EscapeString(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 2);
    for (unsigned char c : raw) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<int>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

namespace {
void Indent(std::string& out, int indent, int depth) {
    if (indent > 0) {
        out.push_back('\n');
        out.append(static_cast<size_t>(indent) * static_cast<size_t>(depth), ' ');
    }
}

// Render a double via "%.9g" (binary32-lossless) and guarantee the text reads
// back as a float (append ".0" if it rendered as a bare integer, so the type is
// unambiguous on re-parse and the writer is self-consistent).
std::string FloatToString(double d) {
    if (std::isnan(d)) return "0";          // JSON has no NaN; cooked data never has it
    if (std::isinf(d)) return d < 0 ? "-1e308" : "1e308";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", d);
    std::string s = buf;
    // If it looks like an integer (no '.', 'e', 'n'(an/inf)), append ".0".
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos && s.find('n') == std::string::npos &&
        s.find('N') == std::string::npos) {
        s += ".0";
    }
    return s;
}
}  // namespace

void Value::DumpTo(std::string& out, int indent, int depth) const {
    switch (type_) {
        case Type::Null:   out += "null"; break;
        case Type::Bool:   out += bool_ ? "true" : "false"; break;
        case Type::Int:    out += std::to_string(int_); break;
        case Type::Float:  out += FloatToString(float_); break;
        case Type::String: out.push_back('"'); out += EscapeString(str_); out.push_back('"'); break;
        case Type::Array: {
            if (array_.empty()) { out += "[]"; break; }
            out.push_back('[');
            for (size_t i = 0; i < array_.size(); ++i) {
                if (i) out.push_back(',');
                Indent(out, indent, depth + 1);
                array_[i].DumpTo(out, indent, depth + 1);
            }
            Indent(out, indent, depth);
            out.push_back(']');
            break;
        }
        case Type::Object: {
            if (object_.empty()) { out += "{}"; break; }
            out.push_back('{');
            for (size_t i = 0; i < object_.size(); ++i) {
                if (i) out.push_back(',');
                Indent(out, indent, depth + 1);
                out.push_back('"');
                out += EscapeString(object_[i].first);
                out += indent > 0 ? "\": " : "\":";
                object_[i].second.DumpTo(out, indent, depth + 1);
            }
            Indent(out, indent, depth);
            out.push_back('}');
            break;
        }
    }
}

std::string Value::Dump(int indent) const {
    std::string out;
    DumpTo(out, indent, 0);
    return out;
}

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value Parse() {
        SkipWs();
        Value v = ParseValue();
        SkipWs();
        if (pos_ != s_.size()) {
            Fail("trailing characters after JSON value");
        }
        return v;
    }

private:
    [[noreturn]] void Fail(const std::string& msg) {
        // Compute 1-based line/column at pos_.
        size_t line = 1, col = 1;
        for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
            if (s_[i] == '\n') { ++line; col = 1; } else { ++col; }
        }
        throw ParseError("json parse error at line " + std::to_string(line) +
                         " col " + std::to_string(col) + ": " + msg);
    }

    char Peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
    char Get() { return pos_ < s_.size() ? s_[pos_++] : '\0'; }
    bool Eof() const { return pos_ >= s_.size(); }

    void SkipWs() {
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    void Expect(char c) {
        if (Get() != c) {
            --pos_;
            Fail(std::string("expected '") + c + "'");
        }
    }

    Value ParseValue() {
        SkipWs();
        if (Eof()) Fail("unexpected end of input");
        const char c = Peek();
        switch (c) {
            case '{': return ParseObject();
            case '[': return ParseArray();
            case '"': return Value::Str(ParseString());
            case 't': case 'f': return ParseBool();
            case 'n': return ParseNull();
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();
                Fail("unexpected character");
        }
    }

    Value ParseObject() {
        Expect('{');
        Value obj = Value::Object();
        SkipWs();
        if (Peek() == '}') { Get(); return obj; }
        while (true) {
            SkipWs();
            if (Peek() != '"') Fail("expected object key string");
            std::string key = ParseString();
            SkipWs();
            Expect(':');
            Value val = ParseValue();
            obj.Set(key, std::move(val));
            SkipWs();
            const char c = Get();
            if (c == ',') continue;
            if (c == '}') break;
            --pos_;
            Fail("expected ',' or '}' in object");
        }
        return obj;
    }

    Value ParseArray() {
        Expect('[');
        Value arr = Value::Array();
        SkipWs();
        if (Peek() == ']') { Get(); return arr; }
        while (true) {
            Value val = ParseValue();
            arr.PushBack(std::move(val));
            SkipWs();
            const char c = Get();
            if (c == ',') continue;
            if (c == ']') break;
            --pos_;
            Fail("expected ',' or ']' in array");
        }
        return arr;
    }

    Value ParseBool() {
        if (s_.compare(pos_, 4, "true") == 0) { pos_ += 4; return Value::Bool(true); }
        if (s_.compare(pos_, 5, "false") == 0) { pos_ += 5; return Value::Bool(false); }
        Fail("invalid literal");
    }

    Value ParseNull() {
        if (s_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Value::Null(); }
        Fail("invalid literal");
    }

    Value ParseNumber() {
        const size_t start = pos_;
        bool is_float = false;
        if (Peek() == '-') Get();
        while (Peek() >= '0' && Peek() <= '9') Get();
        if (Peek() == '.') { is_float = true; Get(); while (Peek() >= '0' && Peek() <= '9') Get(); }
        if (Peek() == 'e' || Peek() == 'E') {
            is_float = true;
            Get();
            if (Peek() == '+' || Peek() == '-') Get();
            while (Peek() >= '0' && Peek() <= '9') Get();
        }
        const std::string tok = s_.substr(start, pos_ - start);
        if (tok.empty() || tok == "-") Fail("invalid number");
        if (is_float) {
            return Value::Float(std::strtod(tok.c_str(), nullptr));
        }
        return Value::Int(static_cast<int64_t>(std::strtoll(tok.c_str(), nullptr, 10)));
    }

    std::string ParseString() {
        Expect('"');
        std::string out;
        while (true) {
            if (Eof()) Fail("unterminated string");
            const char c = Get();
            if (c == '"') break;
            if (c == '\\') {
                if (Eof()) Fail("unterminated escape");
                const char e = Get();
                switch (e) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'n':  out.push_back('\n'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'u': {
                        // Parse 4 hex digits => a BMP codepoint, encode as UTF-8.
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            const char h = Get();
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
                            else { --pos_; Fail("invalid \\u escape"); }
                        }
                        if (cp < 0x80) {
                            out.push_back(static_cast<char>(cp));
                        } else if (cp < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default:
                        --pos_;
                        Fail("invalid escape sequence");
                }
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

    const std::string& s_;
    size_t pos_ = 0;
};

}  // namespace

Value Value::Parse(const std::string& text) {
    Parser parser(text);
    return parser.Parse();
}

} // namespace nuka::scene::json
