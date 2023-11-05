#pragma once
#include "random_variable.h"

namespace ekf {

// NOTE: require partial states are leading states in full states
template <typename _PartialState, typename _FullState>
void PartialUpdate(const RandomVariable<_PartialState, Covariance>& prior,
                   const RandomVariable<_PartialState, Covariance>& posterior,
                   RandomVariable<_FullState, Covariance>& full_state) {
  using Scalar = typename _PartialState::Scalar;
  using FullTangent = typename _FullState::Tangent;

  constexpr int PartialDof = _PartialState::DoF;
  constexpr int RemainDof = _FullState::DoF - _PartialState::DoF;
  constexpr int FullDof = _FullState::DoF;

  Eigen::Matrix<Scalar, PartialDof, PartialDof> Pxx =
      full_state.uncertainty().template topLeftCorner<PartialDof, PartialDof>();
  Eigen::Matrix<Scalar, PartialDof, RemainDof> Pxy =
      0.5 * (full_state.uncertainty().template topRightCorner<PartialDof, RemainDof>() +
             full_state.uncertainty().template bottomLeftCorner<RemainDof, PartialDof>().transpose());
  Eigen::Matrix<Scalar, RemainDof, RemainDof> Pyy =
      full_state.uncertainty().template bottomRightCorner<RemainDof, RemainDof>();
  Eigen::Matrix<Scalar, PartialDof, PartialDof> Pxx_inv = Pxx.inverse();

  Eigen::Vector<Scalar, FullDof> delta;
  auto delta_x = posterior.mean().rminus(prior.mean()).coeffs();
  delta.template head<PartialDof>() = delta_x;
  delta.template tail<RemainDof>() = Pxy.transpose() * Pxx_inv * delta_x;
  full_state.mean() = full_state.mean().rplus(FullTangent(delta));

  Eigen::Matrix<Scalar, PartialDof, PartialDof> Pxx_new = posterior.uncertainty();
  Eigen::Matrix<Scalar, PartialDof, RemainDof> Pxy_new = Pxx_new * Pxx_inv * Pxy;
  Eigen::Matrix<Scalar, RemainDof, RemainDof> Pyy_new =
      Pyy +
      Pxy.transpose() * Pxx_inv * (Pxx_new * Pxx_inv.transpose() - Eigen::Matrix<Scalar, PartialDof, PartialDof>::Identity()) * Pxy;

  full_state.uncertainty().template topLeftCorner<PartialDof, PartialDof>() = Pxx_new;
  full_state.uncertainty().template topRightCorner<PartialDof, RemainDof>() = Pxy_new;
  full_state.uncertainty().template bottomLeftCorner<RemainDof, PartialDof>() = Pxy_new.transpose();
  full_state.uncertainty().template bottomRightCorner<RemainDof, RemainDof>() = Pyy_new;
}

}  // namespace ekf
