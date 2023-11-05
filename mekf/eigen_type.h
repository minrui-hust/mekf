#pragma once
#include "eigen3/Eigen/Core"
#include "tl/optional.hpp"

namespace mekf {

template <typename _Scalar, size_t _dim>
using Vector = Eigen::Matrix<_Scalar, _dim, 1>;

template <size_t _dim> using Vectorf = Vector<float, _dim>;
using Vector1f = Vectorf<1>;

template <size_t _dim> using Vectord = Vector<double, _dim>;
using Vector1d = Vectord<1>;

template <size_t _rows, size_t _cols>
using Matrixf = Eigen::Matrix<float, _rows, _cols>;

template <size_t _rows, size_t _cols>
using Matrixd = Eigen::Matrix<double, _rows, _cols>;

template <typename _Scalar, size_t _dim>
using SquareMatrix = Eigen::Matrix<_Scalar, _dim, _dim>;

template <size_t _dim> using SquareMatrixf = SquareMatrix<float, _dim>;

template <size_t _dim> using SquareMatrixd = SquareMatrix<double, _dim>;

template <typename _FX, typename _X>
using Jacobian = Eigen::Matrix<typename _X::Scalar, _FX::DoF, _X::DoF>;

template <typename _FX, typename _X>
using OptJacobianRef = tl::optional<Eigen::Ref<Jacobian<_FX, _X>>>;

} // namespace mekf
