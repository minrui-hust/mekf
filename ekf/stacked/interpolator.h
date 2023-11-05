#pragma once

#include "random_variable.h"
namespace ekf {

struct InterpolateInfo {
  int curr;
  int next;
  int dist_curr;
  int dist_next;
  int valid_idx;
};

template <typename _Derived>
struct Interpolator {};

struct LinearInterpolator : Interpolator<LinearInterpolator> {
  static constexpr size_t Span = 2;

  template <typename _State>
  auto interpolate(const manif::ArrayBase<_State>& state, InterpolateInfo& info,
                   OptJacobianRef<typename _State::Element, _State> jac = {}) const {
    using SingleState = typename _State::Element::LieGroup;
    using Scalar = typename SingleState::Scalar;

    info.valid_idx = info.curr * SingleState::DoF;

    Scalar ratio = Scalar(info.dist_curr) / Scalar(info.dist_next + info.dist_curr);

    SingleState interp_state;
    if (jac) {
      // d: diff, a: curr state, b: next state
      typename SingleState::Jacobian jac_d_a;
      typename SingleState::Jacobian jac_d_b;
      typename SingleState::Jacobian jac_o_a;
      typename SingleState::Jacobian jac_o_d;
      interp_state = state.element(info.curr).rplus(
          ratio * state.element(info.next).rminus(state.element(info.curr), jac_d_a, jac_d_b), jac_o_a, jac_o_d);
      jac->template middleCols<SingleState::DoF>(info.curr * SingleState::DoF) = jac_o_a + ratio * jac_o_d * jac_d_a;
      jac->template middleCols<SingleState::DoF>(info.next * SingleState::DoF) = ratio * jac_o_d * jac_d_b;
    } else {
      interp_state = state.element(info.curr).rplus(ratio * state.element(info.next).rminus(state.element(info.curr)));
    }

    return interp_state;
  }
};

struct NearstInterpolator : Interpolator<NearstInterpolator> {
  static constexpr size_t Span = 1;

  template <typename _State>
  auto interpolate(const manif::ArrayBase<_State>& state, InterpolateInfo& info,
                   OptJacobianRef<typename _State::Element, _State> jac = {}) const {
    using SingleState = typename _State::Element::LieGroup;

    info.valid_idx = info.curr * SingleState::DoF;

    SingleState interp_state = state.element(info.curr);

    if (jac) {
      jac->template middleCols<SingleState::DoF>(info.curr * SingleState::DoF).setIdentity();
    }

    return interp_state;
  }
};

}  // namespace ekf
