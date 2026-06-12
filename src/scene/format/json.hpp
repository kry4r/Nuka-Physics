#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::json - minimal self-written JSON value + writer + parser (M2c).
//
// A tiny, dependency-free JSON subset sufficient for the .nks scene format:
//   - objects, arrays, strings, numbers, bools, null
//   - string escapes: \" \\ \n \t \r \/ \b \f and \uXXXX (BMP)
//
// Determinism: object keys serialize in INSERTION ORDER (not sorted), and
// floats serialize via "%.9g" so a binary32 round-trips losslessly and a given
// value always renders to the same text. This makes the .nks byte-identical for
// identical scenes (the M2c roundtrip gate).
//
// Parser: recursive-descent; on malformed input it throws json::ParseError with
// 1-based line/column info. Numbers parse as double (the writer chooses int vs
// float rendering by the stored value's kind).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::scene::json {

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& msg) : std::runtime_error(msg) {}
};

// A JSON value. Objects preserve insertion order (a vector of key/value pairs)
// so serialization is deterministic and authoring order is meaningful.
class Value {
public:
    enum class Type { Null, Bool, Int, Float, String, Array, Object };

    Value() : type_(Type::Null) {}

    static Value Null() { return Value(); }
    static Value Bool(bool b) { Value v; v.type_ = Type::Bool; v.bool_ = b; return v; }
    static Value Int(int64_t i) { Value v; v.type_ = Type::Int; v.int_ = i; return v; }
    static Value Float(double d) { Value v; v.type_ = Type::Float; v.float_ = d; return v; }
    static Value Str(std::string s) { Value v; v.type_ = Type::String; v.str_ = std::move(s); return v; }
    static Value Array() { Value v; v.type_ = Type::Array; return v; }
    static Value Object() { Value v; v.type_ = Type::Object; return v; }

    Type type() const { return type_; }
    bool IsNull() const { return type_ == Type::Null; }
    bool IsObject() const { return type_ == Type::Object; }
    bool IsArray() const { return type_ == Type::Array; }

    // -- scalar accessors (throw on type mismatch) --------------------------
    bool        AsBool() const;
    int64_t     AsInt() const;     // accepts Int or Float (truncates a Float)
    double      AsDouble() const;  // accepts Int or Float
    float       AsFloat() const { return static_cast<float>(AsDouble()); }
    const std::string& AsString() const;

    // -- object ops ---------------------------------------------------------
    // Append a key/value (insertion order preserved). Does NOT dedup; the writer
    // never produces duplicate keys.
    void Set(const std::string& key, Value value);
    bool Has(const std::string& key) const;
    const Value* Find(const std::string& key) const;   // null if absent
    const Value& At(const std::string& key) const;      // throws if absent
    const std::vector<std::pair<std::string, Value>>& Items() const { return object_; }

    // -- array ops ----------------------------------------------------------
    void PushBack(Value value);
    const std::vector<Value>& Elements() const { return array_; }
    size_t Size() const { return array_.size(); }

    // -- serialize / parse --------------------------------------------------
    // Pretty-print with 2-space indentation; deterministic.
    std::string Dump(int indent = 2) const;
    static Value Parse(const std::string& text);

private:
    void DumpTo(std::string& out, int indent, int depth) const;

    Type type_;
    bool        bool_ = false;
    int64_t     int_  = 0;
    double      float_ = 0.0;
    std::string str_;
    std::vector<Value>                          array_;
    std::vector<std::pair<std::string, Value>>  object_;
};

// Escape a raw string into JSON string-literal contents (without the quotes).
std::string EscapeString(const std::string& raw);

} // namespace nuka::scene::json
