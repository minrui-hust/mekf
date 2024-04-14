#pragma once

#include "ceres/ceres.h"

#include "macro.h"
#include "system_model_base.h"

namespace mekf {

template <typename Functor> struct AutodiffSystemModel;

namespace internal {

template <typename Functor> struct traits<AutodiffSystemModel<Functor>> {
  using Scalar = typename Functor::Scalar;
  using StateVector = typename Functor::StateVector;
  using NoiseVector = typename Functor::NoiseVector;
};

} // namespace internal

template <typename Functor>
struct AutodiffSystemModel
    : public SystemModelBase<AutodiffSystemModel<Functor>> {
  using Base = SystemModelBase<AutodiffSystemModel<Functor>>;

  SYSTEM_MODEL_TYPEDEF
  SYSTEM_MODEL_PROPERTIES

  StateVector predict(const StateVector &sv, const NoiseVector &nv,
                      Eigen::Matrix<Scalar, StateDoF, StateDoF> *A,
                      Eigen::Matrix<Scalar, StateDoF, NoiseDoF> *L) {

    if (!A && !L) { // None of A,L
      return functor_(sv, nv);
    } else if (A && L) { // both A,L
      using JetT = ceres::Jet<Scalar, StateRepSize + NoiseRepSize>;
      auto sv_out_jet =
          functor_(toJet<JetT>(sv, 0), toJet<JetT>(nv, StateRepSize));
      auto jac = getJacobian(sv_out_jet);

      // TODO jac global to local

      return sv_out_jet.template cast<Scalar>();
    } else if (A) { // only A
      using JetT = ceres::Jet<Scalar, StateRepSize>;
      auto sv_out_jet =
          functor_(toJet<JetT>(sv, 0), toJet<JetT>(nv, StateRepSize));
      auto jac = getJacobian(sv_out_jet);

      // TODO jac global to local

      return sv_out_jet.template cast<Scalar>();
    } else { // only L
      using JetT = ceres::Jet<Scalar, NoiseRepSize>;
      auto sv_out_jet =
          functor_(toJet<JetT>(sv, NoiseRepSize), toJet<JetT>(nv, 0));
      auto jac = getJacobian(sv_out_jet);

      // TODO jac global to local

      return sv_out_jet.template cast<Scalar>();
    }
  }

protected:
  template <typename _StateT, typename _JetT>
  auto toJet(const _StateT &src, const int offset = 0) {
    auto dst = src.template cast<_JetT>();
    for (auto i = 0; i < _StateT::RepSize; ++i) {
      if (i + offset < _JetT::DIMENSION) {
        dst.coeffs()[i] = _JetT(src.coeffs()[i], i + offset);
      }
    }
    return dst;
  }

  template <typename _StateT> auto getJacobian(const _StateT &sv_jet) {
    Eigen::Matrix<Scalar, _StateT::RepSize, _StateT::Scalar::DIMENSION> jac;
    for (auto r = 0; r < jac.rows(); ++r) {
      jac.row(r) = sv_jet.coeffs()[r].v;
    }
    return jac;
  }

protected:
  Functor functor_;
};

} // namespace mekf
