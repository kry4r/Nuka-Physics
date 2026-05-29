#pragma once

// Project C++ wrapper "expected" contract.
//
// libstdc++ only exposes std::expected under C++23 (gated behind
// __cpp_lib_expected); under the project's -std=c++20 build the <expected>
// header is present but empty, so __has_include(<expected>) is not a reliable
// probe. To keep the nuka.hpp RAII wrapper usable under -std=c++20, alias to
// std::expected when the standard library actually provides it, and otherwise
// ship a minimal C++20 fallback that is sufficient for the wrapper contract:
// move-only value types, an expected<void, E> specialization, and unexpected<E>
// with CTAD so `nuka::unexpected(err)` deduces.

#include <version>

#if defined(__cpp_lib_expected)

#include <expected>

namespace nuka {

template <class T, class E>
using expected = std::expected<T, E>;

template <class E>
using unexpected = std::unexpected<E>;

}  // namespace nuka

#else  // C++20 fallback (no std::expected)

#include <new>
#include <type_traits>
#include <utility>

namespace nuka {

template <class E>
class unexpected {
public:
    constexpr explicit unexpected(E error) noexcept(
        std::is_nothrow_move_constructible_v<E>)
        : error_(std::move(error)) {}

    constexpr const E& error() const& noexcept { return error_; }
    constexpr E& error() & noexcept { return error_; }
    constexpr E&& error() && noexcept { return std::move(error_); }

private:
    E error_;
};

// Allow `nuka::unexpected(err)` to deduce, mirroring std::unexpected CTAD.
template <class E>
unexpected(E) -> unexpected<E>;

template <class T, class E>
class expected {
public:
    using value_type = T;
    using error_type = E;
    using unexpected_type = unexpected<E>;

    expected(const T& value) : has_value_(true) {
        ::new (static_cast<void*>(&storage_)) T(value);
    }
    expected(T&& value) : has_value_(true) {
        ::new (static_cast<void*>(&storage_)) T(std::move(value));
    }

    expected(const unexpected<E>& err) : has_value_(false) {
        ::new (static_cast<void*>(&storage_)) E(err.error());
    }
    expected(unexpected<E>&& err) : has_value_(false) {
        ::new (static_cast<void*>(&storage_)) E(std::move(err).error());
    }

    expected(const expected&) = delete;
    expected& operator=(const expected&) = delete;

    expected(expected&& other) noexcept(
        std::is_nothrow_move_constructible_v<T> &&
        std::is_nothrow_move_constructible_v<E>)
        : has_value_(other.has_value_) {
        if (has_value_) {
            ::new (static_cast<void*>(&storage_)) T(std::move(*other.value_ptr()));
        } else {
            ::new (static_cast<void*>(&storage_)) E(std::move(*other.error_ptr()));
        }
    }

    ~expected() {
        if (has_value_) {
            value_ptr()->~T();
        } else {
            error_ptr()->~E();
        }
    }

    bool has_value() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }

    T& operator*() & noexcept { return *value_ptr(); }
    const T& operator*() const& noexcept { return *value_ptr(); }
    T&& operator*() && noexcept { return std::move(*value_ptr()); }

    T* operator->() noexcept { return value_ptr(); }
    const T* operator->() const noexcept { return value_ptr(); }

    T& value() & noexcept { return *value_ptr(); }
    const T& value() const& noexcept { return *value_ptr(); }

    E& error() & noexcept { return *error_ptr(); }
    const E& error() const& noexcept { return *error_ptr(); }
    E&& error() && noexcept { return std::move(*error_ptr()); }

private:
    T* value_ptr() noexcept {
        return std::launder(reinterpret_cast<T*>(&storage_));
    }
    const T* value_ptr() const noexcept {
        return std::launder(reinterpret_cast<const T*>(&storage_));
    }
    E* error_ptr() noexcept {
        return std::launder(reinterpret_cast<E*>(&storage_));
    }
    const E* error_ptr() const noexcept {
        return std::launder(reinterpret_cast<const E*>(&storage_));
    }

    std::aligned_union_t<0, T, E> storage_;
    bool has_value_;
};

template <class E>
class expected<void, E> {
public:
    using value_type = void;
    using error_type = E;
    using unexpected_type = unexpected<E>;

    constexpr expected() noexcept : has_value_(true) {}

    expected(const unexpected<E>& err) : has_value_(false) {
        ::new (static_cast<void*>(&storage_)) E(err.error());
    }
    expected(unexpected<E>&& err) : has_value_(false) {
        ::new (static_cast<void*>(&storage_)) E(std::move(err).error());
    }

    expected(const expected&) = delete;
    expected& operator=(const expected&) = delete;

    expected(expected&& other) noexcept(
        std::is_nothrow_move_constructible_v<E>)
        : has_value_(other.has_value_) {
        if (!has_value_) {
            ::new (static_cast<void*>(&storage_)) E(std::move(*other.error_ptr()));
        }
    }

    ~expected() {
        if (!has_value_) {
            error_ptr()->~E();
        }
    }

    bool has_value() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }

    void value() const noexcept {}

    E& error() & noexcept { return *error_ptr(); }
    const E& error() const& noexcept { return *error_ptr(); }
    E&& error() && noexcept { return std::move(*error_ptr()); }

private:
    E* error_ptr() noexcept {
        return std::launder(reinterpret_cast<E*>(&storage_));
    }
    const E* error_ptr() const noexcept {
        return std::launder(reinterpret_cast<const E*>(&storage_));
    }

    std::aligned_union_t<0, E> storage_;
    bool has_value_;
};

}  // namespace nuka

#endif  // __cpp_lib_expected
