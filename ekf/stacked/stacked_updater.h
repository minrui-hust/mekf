#pragma once
#include "stacked/interpolator.h"
#include "updater.h"

namespace ekf {

template <typename _SingleModel, size_t _N, typename _Interpolator>
struct StackedMeasurementModel : public MeasurementModel<StackedMeasurementModel<_SingleModel, _N, _Interpolator>,
                                                         typename _SingleModel::Measurement> {
  using Measurement = typename _SingleModel::Measurement;

  static constexpr size_t Span = _Interpolator::Span;

  // given state and noise, predict the measurement
  template <typename _State, typename _Noise>
  auto measure(const Variable<_State>& state, const Variable<_Noise>& noise, InterpolateInfo& info,
               OptJacobianRef<Measurement, _State> jac_state = {},
               OptJacobianRef<Measurement, _Noise> jac_noise = {}) const {
    using State = _State;
    using SingleState = typename State::Element;

    static constexpr int ValidSize = SingleState::DoF * Span;

    // first interpolate the state
    Jacobian<SingleState, State> jac_interp;
    OptJacobianRef<SingleState, State> opt_jac_interp;
    if (jac_state) {
      opt_jac_interp = jac_interp;
    }
    auto interp_state = interpolator_.interpolate(static_cast<const _State&>(state), info, opt_jac_interp);

    // then do measurement
    Jacobian<Measurement, SingleState> jac_state_single;
    OptJacobianRef<Measurement, SingleState> opt_jac_state_single;
    if (jac_state) {
      opt_jac_state_single = jac_state_single;
    }
    auto measurement = single_model_.measure(interp_state, noise, opt_jac_state_single, jac_noise);

    // finally compose jacobian, valid info is store in info
    if (jac_state) {
      // only set value which is valid, leave others unchanged (usually not zero)
      jac_state->template middleCols<ValidSize>(info.valid_idx) =
          jac_state_single * jac_interp.template middleCols<ValidSize>(info.valid_idx);
    }

    return measurement;
  }

 protected:
  _SingleModel single_model_;
  _Interpolator interpolator_;
};

struct StackedCovUpdater : public Updater<StackedCovUpdater> {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Measurement,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto update(const RandomVariable<_StateVariable, _StateUncertainty>& state,
              const Variable<_Measurement>& measurement_sample,
              const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
              const MeasurementModel<_Model, _Measurement>& measurement_model, InterpolateInfo& info) const {
    using StateVariable = _StateVariable;
    using StateUncertainty = _StateUncertainty<typename StateVariable::Scalar, StateVariable::DoF>;
    using StateRandomVariable = RandomVariable<_StateVariable, _StateUncertainty>;
    using StateTangent = typename StateVariable::Tangent;

    using MeasurementVariable = _Measurement;
    // using MeasurementUncertainty = Covariance<typename _Measurement::Scalar, _Measurement::DoF>;
    using MeasurementRandomVariable = RandomVariable<MeasurementVariable, Covariance>;

    using NoiseVariable = _NoiseVariable;
    // using NoiseUncertainty = _NoiseUncertainty<typename NoiseVariable::Scalar, NoiseVariable::DoF>;
    // using NoiseRandomVariable = RandomVariable<_NoiseVariable, _NoiseUncertainty>;

    using SingleStateVariable = typename StateVariable::Element;

    // Measurement margin distribution
    MeasurementRandomVariable measurement;
    Jacobian<MeasurementVariable, StateVariable> H;
    Jacobian<MeasurementVariable, NoiseVariable> L;

    static constexpr size_t ValidSize = SingleStateVariable::DoF * _Model::Span;

    measurement.mean() = static_cast<const _Model&>(measurement_model).measure(state.mean(), noise.mean(), info, H, L);

    auto h = H.template middleCols<ValidSize>(info.valid_idx);
    measurement.uncertainty() =
        h * state.uncertainty().template block<ValidSize, ValidSize>(info.valid_idx, info.valid_idx) * h.transpose() +
        L * noise.uncertainty() * L.transpose();

    // Covariance between state and menasurement
    Jacobian<StateVariable, MeasurementVariable> xz_covariance =
        state.uncertainty().template middleCols<ValidSize>(info.valid_idx) * h.transpose();

    // Kalman gain
    Jacobian<StateVariable, MeasurementVariable> K = xz_covariance * measurement.uncertainty().inverse();

    // Error-state, in tangent space
    StateTangent dx = K * measurement_sample.rminus(measurement.mean()).coeffs();

    // Error-state covariance
    StateUncertainty dx_cov = state.uncertainty() - K * measurement.uncertainty() * K.transpose();

    // Merge error-state to global state
    StateRandomVariable state_new;
    state_new.mean() = state.mean().rplus(dx);
    state_new.uncertainty() = dx.rjac() * 0.5 * (dx_cov + dx_cov.transpose()) * dx.rjac().transpose();

    return state_new;
  }
};

template <typename _List, size_t _N, typename _Interpolator>
struct StackedMeasurementModelList;

template <typename... _Model, size_t _N, typename _Interpolator>
struct StackedMeasurementModelList<std::tuple<_Model...>, _N, _Interpolator> {
  using type = std::tuple<StackedMeasurementModel<_Model, _N, _Interpolator>...>;
};

template <typename _List, size_t _N, typename _Interpolator>
using StackedMeasurementModelList_t = typename StackedMeasurementModelList<_List, _N, _Interpolator>::type;

}  // namespace ekf
