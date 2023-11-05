#include "ekf.h"

using namespace manif;
using namespace ekf;

DeclareBundleElementTypes(Rigid2D, float, R2, SO2);
DeclareBundleElementNames(Rigid2D, Pos, Rot);

struct MeasPos : public MeasurementModel<MeasPos, R2f> {
  template <typename _State, typename _Noise>
  R2f measure(const Variable<_State>& state, const Variable<_Noise>& noise, OptJacobianRef<R2f, _State> jac_state = {},
              OptJacobianRef<R2f, _Noise> jac_noise = {}) const {
    return R2f();
  }
};

struct MeasRot : public MeasurementModel<MeasRot, SO2f> {
  template <typename _State, typename _Noise>
  SO2f measure(const Variable<_State>& state, const Variable<_Noise>& noise,
               OptJacobianRef<SO2f, _State> jac_state = {}, OptJacobianRef<SO2f, _Noise> jac_noise = {}) const {
    return SO2f();
  }
};

DeclareListElementTypes(MeasList, MeasPos, MeasRot);
DeclareListElementNames(MeasList, MeasPos, MeasRot);

struct Rigid2DModel : public TransitionModel<Rigid2DModel> {  //, Rigid2D, R1f, R2f> {
  template <typename _State, typename _Input, typename _Noise>
  Rigid2D predict(const Variable<_State>& state, const Variable<_Input>& input,
                  const Variable<_Noise>& noise = _Noise::Identity(), OptJacobianRef<_State, _State> jac_state = {},
                  OptJacobianRef<_State, _Noise> jac_noise = {}) const {
    return Rigid2D();
  }
};

using StackedRigi2DModel = StackedTransitionModel<Rigid2DModel, 2>;

using StackedMeasRot = StackedMeasurementModel<MeasRot, 2, LinearInterpolator>;

using Rigid2DEkf = Ekf<Rigid2DModel, MeasList>;

using StackedRigid2DEkf = StackEkf<Rigid2DModel, MeasList, 2, LinearInterpolator>;

using StackedRigid2DEkfNearest = StackEkf<Rigid2DModel, MeasList, 2, NearstInterpolator>;

int main() {
  Rigid2DEkf ekf;

  auto& meas = ekf.measurementModel<MeasListElement::MeasPos>();

  Rigid2D state;
  R1f dt;
  RandomVariable<R2f, Covariance> noise_rv;

  RandomVariable<Rigid2D, Covariance> state_rv;

  state = ekf.predict(state, dt, noise_rv.mean());

  state_rv = ekf.predict(state_rv, dt, noise_rv);

  R2f pos_meas;
  RandomVariable<R2f, Covariance> pos_meas_n;

  SO2f rot_meas;
  RandomVariable<SO2f, Covariance> rot_meas_n;

  state_rv = ekf.measure<MeasListElement::MeasPos>(state_rv, pos_meas, pos_meas_n);
  state_rv = ekf.measure<MeasListElement::MeasRot>(state_rv, rot_meas, rot_meas_n);

  StackedRigid2DEkf stack_ekf;
  StackedRigid2DEkfNearest stack_ekf_nearest;

  InterpolateInfo interp_info = {0, 1, 5, 5, 0};
  VariableArray<Rigid2D, 2> stacked_state;
  RandomVariable<VariableArray<Rigid2D, 2>, Covariance> stacked_state_rv;

  stacked_state = stack_ekf.predict(stacked_state, dt, noise_rv.mean());

  stacked_state_rv = stack_ekf.predict(stacked_state_rv, dt, noise_rv);

  stacked_state_rv = stack_ekf.measure<MeasListElement::MeasPos>(stacked_state_rv, pos_meas, pos_meas_n, interp_info);
  stacked_state_rv = stack_ekf.measure<MeasListElement::MeasRot>(stacked_state_rv, rot_meas, rot_meas_n, interp_info);

  stacked_state_rv =
      stack_ekf_nearest.measure<MeasListElement::MeasRot>(stacked_state_rv, rot_meas, rot_meas_n, interp_info);
  stacked_state_rv =
      stack_ekf_nearest.measure<MeasListElement::MeasRot>(stacked_state_rv, rot_meas, rot_meas_n, interp_info);

  return 0;
}
