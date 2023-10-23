#pragma once

#include <sstream>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <filesystem>

namespace s21 {

struct SStr {
  template <const char Delimiter, class... Args >
  static std::string Fill(const Args &... args) {
    std::ostringstream sstr;
    int i = 0;
    ((sstr << std::dec << args << (++i == sizeof...(args) ? '\0' : Delimiter)), ...);
    return sstr.str();
  }

  template <class... Args >
  static std::string Fill(const Args &... args) {
    std::ostringstream sstr;
    ((sstr << std::dec << args), ...);
    return sstr.str();
  }

  template <const char Delimiter = '\n', class... Args >
  static void Print(const Args &... args) {
    int i = 0;
    ([&] (const auto &arg) {
      Print(std::cout, args);
      std::cout << (++i == sizeof...(args) ? '\0' : Delimiter);
    } (args), ...);
  }

  struct PrintStyle {
    inline static std::string container_delimiter = ", ";
    inline static std::string pair_delimiter = ", ";

    template<class T>
    static typename std::enable_if<std::is_same<T, std::string>::value, std::string>::type
    Wrapper(const T &l) {
      return "\"" + l + "\"";
    }

    template<class T>
    static typename std::enable_if<! std::is_same<T, std::string>::value, const T&>::type
    Wrapper(const T &l) {
      return l;
    }
  };

  template <typename T>
  struct is_container {
    template <typename U>
    static constexpr bool is_container_helper(
        U*,
        decltype(std::declval<U>().begin())* = nullptr,
        decltype(std::declval<U>().end())* = nullptr) {
      return true;
    }

    template <typename>
    static constexpr bool is_container_helper(...) {
      return false;
    }

    static constexpr bool value = is_container_helper<T>(nullptr);
  };

  template <class Container>
  static typename std::enable_if<is_container<Container>::value, void>::type
  Print(std::ostream &os, const Container &c) {
    auto i = c.begin();
    if (i == c.end()) {
      return;
    }
    os << '{';
    while (i != --c.end()) {
      os << PrintStyle::Wrapper(*(i++)) << PrintStyle::container_delimiter;
    }
    os << PrintStyle::Wrapper(*i) << "}\n";
  }

  template<class T>
  static typename std::enable_if<!is_container<T>::value, void>::type
  Print(std::ostream &os, const T &data) {
    os << data;
  }

  template<class A, class B>
  static void Print(std::ostream &os, const std::pair<A, B> &p) {
    os << '[' << p.first << PrintStyle::pair_delimiter << p.second << "]\n";
  }

  template<class A>
  static void Print(std::ostream &os, const std::pair<A, A> &p) {
    os << '[' << p.first << PrintStyle::pair_delimiter << p.second << "]\n";
  }

  static std::string RelativePath(const char *FILE, const std::string &path_from_source) {
    using namespace std::filesystem;
    return relative(path(FILE).parent_path(), current_path()).string() + path_from_source;
  }
};

} // namespace s21