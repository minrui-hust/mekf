#pragma once

#include "random_variable.h"
#include "stacked/stacked_transformer.h"
#include "stacked/stacked_updater.h"
#include "transformer.h"
#include "updater.h"

namespace mekf {

template <typename _TransitionModel, typename _MeasurementModelList,
          typename _Transformer, typename _Updater>
struct EkfBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using TransitionModel = _TransitionModel;
  using MeasurementModelList = _MeasurementModelList;
  using Transformer = _Transformer;
  using Updater = _Updater;

  // predict full version
  template <typename _StateVariable,
            template <typename, size_t> class _StateUncertainty,
            typename _Input, typename _NoiseVariable,
            template <typename, size_t> class _NoiseUncertainty>
  RandomVariable<_StateVariable, _StateUncertainty> predict(
      const RandomVariable<_StateVariable, _StateUncertainty> &state,
      const Variable<_Input> &input,
      const RandomVariable<_NoiseVariable, _NoiseUncertainty> &noise) const {
    return transformer_.transform(state, static_cast<const _Input &>(input),
                                  noise, transition_model_);
  }

  // predict simple version
  template <typename _State, typename _Input, typename _Noise>
  _State predict(const Variable<_State> &state, const Variable<_Input> &input,
                 const Variable<_Noise> &noise) const {
    return transition_model_.predict(
        static_cast<const _State &>(state), static_cast<const _Input &>(input),
        static_cast<const _Noise &>(noise), {}, {});
  }

  // measurement update
  template <size_t _meas_id, typename _StateVariable,
            template <typename, size_t> class _StateUncertainty,
            typename _Measurement, typename _NoiseVariable,
            template <typename, size_t> class _NoiseUncertainty>
  RandomVariable<_StateVariable, _StateUncertainty> measure(
      const RandomVariable<_StateVariable, _StateUncertainty> &state,
      const Variable<_Measurement> &measurement,
      const RandomVariable<_NoiseVariable, _NoiseUncertainty> &noise) const {
    return updater_.update(state,
                           static_cast<const _Measurement &>(measurement),
                           noise, std::get<_meas_id>(measurement_model_list_));
  }

  // measurement update with interpolation info
  template <size_t _meas_id, typename _StateVariable,
            template <typename, size_t> class _StateUncertainty,
            typename _Measurement, typename _NoiseVariable,
            template <typename, size_t> class _NoiseUncertainty>
  RandomVariable<_StateVariable, _StateUncertainty>
  measure(const RandomVariable<_StateVariable, _StateUncertainty> &state,
          const Variable<_Measurement> &measurement,
          const RandomVariable<_NoiseVariable, _NoiseUncertainty> &noise,
          InterpolateInfo &interp_info) const {
    return updater_.update(
        state, static_cast<const _Measurement &>(measurement), noise,
        std::get<_meas_id>(measurement_model_list_), interp_info);
  }

  // get the measurement model
  template <size_t _meas_id> auto &measurementModel() {
    return std::get<_meas_id>(measurement_model_list_);
  }

  // get the measurement model, const version
  template <size_t _meas_id> const auto &measurementModel() const {
    return std::get<_meas_id>(measurement_model_list_);
  }

protected:
  TransitionModel transition_model_;
  MeasurementModelList measurement_model_list_;
  Transformer transformer_;
  Updater updater_;
};

template <typename _TransitionModel, typename _MeasurementModelList>
using Ekf = EkfBase<_TransitionModel, _MeasurementModelList, CovTransformer,
                    CovUpdater>;

template <typename _TransitionModel, typename _MeasurementModelList, size_t _N,
          typename _Interpolator>
using StackEkf = EkfBase<
    StackedTransitionModel<_TransitionModel, _N>,
    StackedMeasurementModelList_t<_MeasurementModelList, _N, _Interpolator>,
    StackedCovTransformer, StackedCovUpdater>;

template <typename _TransitionModel, typename _MeasurementModelList>
using SqrtEkf = EkfBase<_TransitionModel, _MeasurementModelList,
                        SqrtCovTransformer, SqrtCovUpdater>;

} // namespace mekf
