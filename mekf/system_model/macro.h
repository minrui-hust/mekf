#pragma once

#define SYSTEM_MODEL_PROPERTIES                                                \
  using Base::StateDoF;                                                        \
  using Base::StateRepSize;                                                    \
  using Base::NoiseDoF;                                                        \
  using Base::NoiseRepSize;

#define SYSTEM_MODEL_TYPEDEF                                                   \
  using Scalar = typename Base::Scalar;                                        \
  using StateVector = typename Base::StateVector;                              \
  using NoiseVector = typename Base::NoiseVector;
