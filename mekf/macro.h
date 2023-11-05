#pragma once
#include <tuple>

// clang-format off
#define DeclareBundleElementTypes(BundleVariableName, Scalar, ...) \
using BundleVariableName = manif::Bundle<Scalar, __VA_ARGS__>;

#define DeclareBundleElementNames(BundleVariableName, ...) \
struct BundleVariableName##Element{ \
  enum{Leader = -1, __VA_ARGS__, Total}; \
  static_assert(size_t( BundleVariableName::BundleSize)== Total, "Type and Name number should match"); \
};

#define DeclareListElementTypes(ListName, ...) \
using ListName = std::tuple<__VA_ARGS__>;

#define DeclareListElementNames(ListName, ...) \
struct ListName##Element{ \
  enum{ Leader = -1, __VA_ARGS__, Total }; \
  static_assert(size_t( std::tuple_size<ListName>::value)== Total, "Type and Name number should match"); \
};
// clang-format on
