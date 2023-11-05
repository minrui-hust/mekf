#pragma once
#include "random_variable.h"

namespace ekf {

// TransitionModel
template <typename _Derived>
struct TransitionModel {
  template <typename _State, typename _Input, typename _Noise>
  typename _State::LieGroup predict(const Variable<_State>& state, const Variable<_Input>& input,
                                    const Variable<_Noise>& noise, OptJacobianRef<_State, _State> jac_state = {},
                                    OptJacobianRef<_State, _Noise> jac_noise = {}) const {
    return static_cast<const _Derived&>(*this).predict(static_cast<const _State&>(state),
                                                       static_cast<const _Input&>(input),
                                                       static_cast<const _Noise&>(noise), jac_state, jac_noise);
  }
};

// Transformer, used to trasform an RandomVariable through TransitionModel
template <typename _Derived>
struct Transformer {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Input,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto transform(const RandomVariable<_StateVariable, _StateUncertainty>& state, const Variable<_Input>& input,
                 const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
                 const TransitionModel<_Model>& model) const {
    return static_cast<const _Derived&>(*this).transform(state, static_cast<const _Input&>(input), noise,
                                                         static_cast<const _Model&>(model));
  }
};

struct CovTransformer : public Transformer<CovTransformer> {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Input,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto transform(const RandomVariable<_StateVariable, _StateUncertainty>& state, const Variable<_Input>& input,
                 const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
                 const TransitionModel<_Model>& model) const {
    using StateRandomVariable = RandomVariable<_StateVariable, _StateUncertainty>;
    using StateVariable = _StateVariable;
    using NoiseVariable = _NoiseVariable;

    StateRandomVariable state_new;
    Jacobian<StateVariable, StateVariable> A;
    Jacobian<StateVariable, NoiseVariable> L;

    state_new.mean() = model.predict(state.mean(), input, NoiseVariable::Identity(), A, L);
    state_new.uncertainty() = A * state.uncertainty() * A.transpose() + L * noise.uncertainty() * L.transpose();

    return state_new;
  }
};

struct SqrtCovTransformer : public Transformer<SqrtCovTransformer> {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Input,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto transform(const RandomVariable<_StateVariable, _StateUncertainty>& state, const Variable<_Input>& input,
                 const RandomVariable<_NoiseVariable, _StateUncertainty>& noise,
                 const TransitionModel<_Model>& model) const {
    using StateRandomVariable = RandomVariable<_StateVariable, _StateUncertainty>;
    using StateVariable = _StateVariable;
    using NoiseVariable = _NoiseVariable;

    using Scalar = typename StateVariable::Scalar;

    static constexpr size_t StateDof = StateVariable::DoF;
    static constexpr size_t NoiseDof = NoiseVariable::DoF;

    StateRandomVariable state_new;
    Jacobian<StateVariable, StateVariable> A;
    Jacobian<StateVariable, NoiseVariable> L;

    // Predict the mean
    state_new.mean() = model.predict(state.mean(), input, NoiseVariable::Identity(), A, L);

    // Calculate the agumented matrix
    Eigen::Matrix<Scalar, StateDof + NoiseDof, StateDof> AP_LN;
    AP_LN.template topRows<StateDof>() = state.uncertainty().matrixU() * A.transpose();
    AP_LN.template bottomRows<NoiseDof>() = noise.uncertainty().matrixU() * L.transpose();

    // Calculate uncertainty based on QR-decomposition
    state_new.uncertainty() = AP_LN.householderQr().matrixQR().template topRightCorner<StateDof, StateDof>();

    return state_new;
  }
};

}  // namespace ekf
