#pragma once

#include "Eigen/Core"

#include "traits.h"

namespace mekf {

template <typename Derived> struct SystemModelBase {
  using Scalar = typename internal::traits<Derived>::Scalar;
  using StateVector = typename internal::traits<Derived>::StateVector;
  using NoiseVector = typename internal::traits<Derived>::NoiseVector;

  const static int StateDoF = StateVector::DoF;
  const static int StateRepSize = StateVector::RepSize;

  const static int NoiseDoF = NoiseVector::DoF;
  const static int NoiseRepSize = NoiseVector::RepSize;

  StateVector predict(const StateVector &sv, const NoiseVector &nv,
                      Eigen::Matrix<Scalar, StateDoF, StateDoF> *A,
                      Eigen::Matrix<Scalar, StateDoF, NoiseDoF> *L) {
    return derived().predict(sv, nv, A, L);
  }

protected:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }
};

} // namespace mekf
