#pragma once
#include <cstddef>

#include "eigen_type.h"
#include "manif/manif.h"

namespace mekf {

// variable manif::LieGroup element
template <typename _Derived> using Variable = manif::LieGroupBase<_Derived>;

template <typename _T, size_t _N> using VariableArray = manif::Array<_T, _N>;

// RandomVariable
template <typename _Variable, template <typename, size_t> class _UncertaintyT>
struct RandomVariable {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  using Scalar = typename _Variable::Scalar;
  using Variable = _Variable;
  using Uncertainty = _UncertaintyT<Scalar, Variable::DoF>;

  template <typename _Scalar, size_t _Dof>
  using UncertaintyT = _UncertaintyT<_Scalar, _Dof>;

  static constexpr int DoF = Variable::DoF;
  static constexpr int Dim = Variable::Dim;

  RandomVariable() = default;
  RandomVariable(const Variable &mean, const Uncertainty &uncertainty)
      : mean_(mean), uncertainty_(uncertainty) {}

  Variable &mean() { return mean_; }
  const Variable &mean() const { return mean_; }

  Uncertainty &uncertainty() { return uncertainty_; }
  const Uncertainty &uncertainty() const { return uncertainty_; }

protected:
  Variable mean_;
  Uncertainty uncertainty_;
};

// forward declare of traits
namespace internal {
template <typename _T> struct Traits;
}

// Uncertainty of an random variable, which can be covariance, sqrt covariance
template <typename _Derived> struct Uncertainty {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using Scalar = typename internal::Traits<_Derived>::Scalar;
  static constexpr int Dim = internal::Traits<_Derived>::Dim;

  template <typename _EigenDerived>
  void setCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<_Derived &>(*this).setCovariance(m);
  }

  template <typename _EigenDerived>
  void setSqrtCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<_Derived &>(*this).setSqrtCovariance(m);
  }

  template <typename _EigenDerived>
  void setInformation(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<_Derived &>(*this).setInformation(m);
  }

  template <typename _EigenDerived>
  static _Derived FromCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    _Derived uncertainty;
    uncertainty.setCovariance(m);
    return uncertainty;
  }

  template <typename _EigenDerived>
  static _Derived
  FromSqrtCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    _Derived uncertainty;
    uncertainty.setSqrtCovariance(m);
    return uncertainty;
  }

  template <typename _EigenDerived>
  static _Derived FromInformation(const Eigen::MatrixBase<_EigenDerived> &m) {
    _Derived uncertainty;
    uncertainty.setInformation(m);
    return uncertainty;
  }

  SquareMatrix<Scalar, Dim> getSqrtCovariance() const {
    assert(false);
    return SquareMatrix<Scalar, Dim>();
  }
};

template <typename _Scalar, size_t _dim>
struct Covariance : public Uncertainty<Covariance<_Scalar, _dim>>,
                    public SquareMatrix<_Scalar, _dim> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  using UncertaintyBase = Uncertainty<Covariance<_Scalar, _dim>>;
  using EigenBase = SquareMatrix<_Scalar, _dim>;

  using EigenBase::EigenBase;
  using EigenBase::operator=;

  template <typename _EigenDerived>
  void setCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<EigenBase &>(*this) = m;
  }

  template <typename _EigenDerived>
  void setSqrtCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<EigenBase &>(*this) = m.transpose() * m;
  }

  template <typename _EigenDerived>
  void setInformation(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<EigenBase &>(*this) = m.inverse();
  }

  SquareMatrix<_Scalar, _dim> getSqrtCovariance() const {
    // std::cout << "get cov:\n" << (*this) << std::endl;
    if ((this->array() != 0.0).any()) {
      return this->sqrt();
    } else {
      return SquareMatrix<_Scalar, _dim>::Zero();
    }
  }
};

namespace internal {
template <typename _Scalar, size_t _dim>
struct Traits<Covariance<_Scalar, _dim>> {
  using Scalar = _Scalar;
  static constexpr int Dim = _dim;
};
} // namespace internal

template <size_t _dim> using Covarianced = Covariance<double, _dim>;

template <size_t _dim> using Covariancef = Covariance<float, _dim>;

template <typename _Scalar, size_t _dim>
struct SqrtCovariance : public Uncertainty<SqrtCovariance<_Scalar, _dim>>,
                        public Eigen::LLT<SquareMatrix<_Scalar, _dim>> {
  using UncertaintyBase = Uncertainty<SqrtCovariance<_Scalar, _dim>>;
  using EigenBase = Eigen::LLT<SquareMatrix<_Scalar, _dim>>;

  template <typename _EigenDerived>
  void setCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<EigenBase &>(*this).compute(m);
  }

  template <typename _EigenDerived>
  void setSqrtCovariance(const Eigen::MatrixBase<_EigenDerived> &m) {
    this->m_matrix = m.transpose().template triangularView<Eigen::Lower>();
    this->m_isInitialized = true;
  }

  template <typename _EigenDerived>
  void setInformation(const Eigen::MatrixBase<_EigenDerived> &m) {
    static_cast<EigenBase &>(*this).compute(m.inverse());
  }

  template <typename _EigenDerived>
  SqrtCovariance &operator=(const Eigen::MatrixBase<_EigenDerived> &m) {
    setSqrtCovariance(m);
    return (*this);
  }
};

namespace internal {
template <typename _Scalar, size_t _dim>
struct Traits<SqrtCovariance<_Scalar, _dim>> {
  using Scalar = _Scalar;
  static constexpr int Dim = _dim;
};
} // namespace internal

template <size_t _dim> using SqrtCovarianced = SqrtCovariance<double, _dim>;

template <size_t _dim> using SqrtCovariancef = SqrtCovariance<float, _dim>;

} // namespace mekf
