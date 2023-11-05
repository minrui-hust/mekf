#pragma once
#include <utility>

#include "random_variable.h"

namespace ekf {

// MeasurementModel
template <typename _Derived, typename _Measurement>
struct MeasurementModel {
  using Measurement = _Measurement;

  // given state and noise, predict the measurement
  template <typename _State, typename _Noise>
  auto measure(const Variable<_State>& state, const Variable<_Noise>& noise,
               OptJacobianRef<_Measurement, _State> jac_state = {},
               OptJacobianRef<_Measurement, _Noise> jac_noise = {}) const {
    return static_cast<const _Derived&>(*this).measure(static_cast<const _State&>(state), static_cast<_Noise>(noise),
                                                       jac_state, jac_noise);
  }
};

// Updater, used to update an RandomVariable by measurement and MeasurmentModel
template <typename _Derived>
struct Updater {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Measurement,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto update(const RandomVariable<_StateVariable, _StateUncertainty>& state, const Variable<_Measurement>& measurement,
              const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
              const MeasurementModel<_Model, _Measurement>& measurement_model) const {
    return static_cast<const _Derived&>(*this).update(state, static_cast<_Measurement&>(measurement), noise,
                                                      static_cast<_Model&>(measurement_model));
  }
};

struct CovUpdater : public Updater<CovUpdater> {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Measurement,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto update(const RandomVariable<_StateVariable, _StateUncertainty>& state,
              const Variable<_Measurement>& measurement_sample,
              const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
              const MeasurementModel<_Model, _Measurement>& measurement_model) const {
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

    // Measurement margin distribution
    MeasurementRandomVariable measurement;
    Jacobian<MeasurementVariable, StateVariable> H;
    Jacobian<MeasurementVariable, NoiseVariable> L;

    measurement.mean() = measurement_model.measure(state.mean(), NoiseVariable::Identity(), H, L);
    measurement.uncertainty() = H * state.uncertainty() * H.transpose() + L * noise.uncertainty() * L.transpose();

    // Covariance between state and menasurement
    Jacobian<StateVariable, MeasurementVariable> xz_covariance = state.uncertainty() * H.transpose();

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

struct SqrtCovUpdater : public Updater<SqrtCovUpdater> {
  template <typename _StateVariable, template <typename, size_t> class _StateUncertainty, typename _Measurement,
            typename _NoiseVariable, template <typename, size_t> class _NoiseUncertainty, typename _Model>
  auto update(const RandomVariable<_StateVariable, _StateUncertainty>& state,
              const Variable<_Measurement>& measurement_sample,
              const RandomVariable<_NoiseVariable, _NoiseUncertainty>& noise,
              const MeasurementModel<_Model, _Measurement>& measurement_model) const {
    using StateVariable = _StateVariable;
    // using StateUncertainty = _StateUncertainty<typename StateVariable::Scalar, StateVariable::DoF>;
    using StateRandomVariable = RandomVariable<_StateVariable, _StateUncertainty>;
    using StateTangent = typename StateVariable::Tangent;

    using MeasurementVariable = _Measurement;
    // using MeasurementUncertainty = Covariance<typename _Measurement::Scalar, _Measurement::DoF>;
    using MeasurementRandomVariable = RandomVariable<MeasurementVariable, Covariance>;

    using NoiseVariable = _NoiseVariable;
    // using NoiseUncertainty = _NoiseUncertainty<typename NoiseVariable::Scalar, NoiseVariable::DoF>;
    // using NoiseRandomVariable = RandomVariable<_NoiseVariable, _NoiseUncertainty>;

    using Scalar = typename StateVariable::Scalar;

    static constexpr size_t MeasurementDof = MeasurementVariable::DoF;
    static constexpr size_t StateDof = StateVariable::DoF;
    static constexpr size_t NoiseDof = NoiseVariable::DoF;

    // Calculate measurement margin distribution
    MeasurementRandomVariable measurement;
    Jacobian<MeasurementVariable, StateVariable> H;
    Jacobian<MeasurementVariable, NoiseVariable> L;

    // Measurement mean
    measurement.mean() = measurement_model.measure(state.mean(), NoiseVariable::Identity(), H, L);

    // Measurement uncertainty
    // Agumented matrix
    Eigen::Matrix<Scalar, StateDof + NoiseDof, MeasurementDof> HP_LN;
    HP_LN.template topRows<StateDof>() = state.uncertainty().matrixU() * H.transpose();
    HP_LN.template bottomRows<NoiseDof>() = noise.uncertainty().matrixU() * L.transpose();

    // Calculate Sqrt-Covariance based on QR-decomposition
    measurement.uncertainty() =
        HP_LN.householderQr().matrixQR().template topRightCorner<MeasurementDof, MeasurementDof>();

    // Covariance between state and menasurement
    Jacobian<StateVariable, MeasurementVariable> xz_covariance =
        state.uncertainty().reconstructedMatrix() * H.transpose();

    // Calculate Kalman gain by back subtitution
    Jacobian<StateVariable, MeasurementVariable> K =
        measurement.uncertainty().solve(xz_covariance.transpose()).transpose();

    // Error-state, in tangent space
    StateTangent dx = K * measurement_sample.rminus(measurement.mean()).coeffs();

    // Error-state covariance, by LLT Rank-one Update
    auto dx_cov = state.uncertainty();
    Jacobian<StateVariable, MeasurementVariable> dP = K * measurement.uncertainty().matrixL();
    for (int i = 0; i < dP.cols(); ++i) {
      dx_cov.rankUpdate(dP.col(i), -1);
    }

    // Merge error-state into global state
    StateRandomVariable state_new;
    state_new.mean() = state.mean().rplus(dx);
    state_new.uncertainty() = (dx_cov.matrixU() * dx.rjac().transpose())
                                  .householderQr()
                                  .matrixQR()
                                  .template topRightCorner<StateDof, StateDof>();

    return state_new;
  }
};

}  // namespace ekf
