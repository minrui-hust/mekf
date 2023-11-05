#pragma once
#include <cstddef>
#include "transformer.h"

namespace ekf {

// Stacked transition model
template <typename _SingleModel, size_t _N>
struct StackedTransitionModel : public TransitionModel<StackedTransitionModel<_SingleModel, _N>> {
 public:
  using Base = TransitionModel<StackedTransitionModel<_SingleModel, _N>>;
  using SingleModel = _SingleModel;

  static constexpr size_t StackedSize = _N;

  template <typename _State, typename _Input, typename _Noise>
  auto predict(
      const manif::ArrayBase<_State>& state, const Variable<_Input>& input, const Variable<_Noise>& noise,
      OptJacobianRef<typename _State::Element, typename _State::Element> jac_state = {},
      OptJacobianRef<typename _State::Element, _Noise> jac_noise = {}) const {  // Note: jacobian is of single state
    using State = typename _State::LieGroup;

    constexpr size_t RemainRepSize = State::RepSize - State::Element::RepSize;

    State next_state;
    next_state.element(0) = single_model_.predict(state.element(0), input, noise, jac_state, jac_noise);
    next_state.coeffs().template tail<RemainRepSize>() = state.coeffs().template head<RemainRepSize>();

    return next_state;
  }

 protected:
  SingleModel single_model_;
};

struct StackedCovTransformer : public Transformer<StackedCovTransformer> {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Input,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto transform(const RandomVariable<_StateVariable, _StateUncertainty>& state, const Variable<_Input>& input,
                 const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
                 const TransitionModel<_Model>& model) const {
    using StateRandomVariable = RandomVariable<_StateVariable, _StateUncertainty>;
    using StateVariable = _StateVariable;
    using NoiseVariable = _NoiseVariable;

    using SingleStateVariable = typename StateVariable::Element;

    constexpr size_t SingleDoF = SingleStateVariable::DoF;
    constexpr size_t RemainDoF = StateVariable::DoF - SingleDoF;

    StateRandomVariable state_new;
    Jacobian<SingleStateVariable, SingleStateVariable> A;
    Jacobian<SingleStateVariable, NoiseVariable> L;

    state_new.mean() = static_cast<const _Model&>(model).predict(state.mean(), input, noise.mean(), A, L);

    state_new.uncertainty().template topLeftCorner<SingleDoF, SingleDoF>() =
        A * state.uncertainty().template topLeftCorner<SingleDoF, SingleDoF>() * A.transpose() +
        L * noise.uncertainty() * L.transpose();

    if (RemainDoF != 0) {
      // fill remain states' covariance
      state_new.uncertainty().template bottomRightCorner<RemainDoF, RemainDoF>() =
          state.uncertainty().template topLeftCorner<RemainDoF, RemainDoF>();

      // fill cov between current state and remained states
      auto cov = A * state.uncertainty().template topLeftCorner<SingleDoF, RemainDoF>();
      state_new.uncertainty().template topRightCorner<SingleDoF, RemainDoF>() = cov;
      state_new.uncertainty().template bottomLeftCorner<RemainDoF, SingleDoF>() = cov.transpose();
    }

    return state_new;
  }
};

}  // namespace ekf
