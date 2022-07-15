//
// Created by amit on 7/12/22.
//

#ifndef TORCH_ML_MODEL_H
#define TORCH_ML_MODEL_H

#include "KIM_ModelDriverHeaders.hpp"

extern "C" {
int model_driver_create(KIM::ModelDriverCreate * const modelDriverCreate,
                        KIM::LengthUnit const requestedLengthUnit,
                        KIM::EnergyUnit const requestedEnergyUnit,
                        KIM::ChargeUnit const requestedChargeUnit,
                        KIM::TemperatureUnit const requestedTemperatureUnit,
                        KIM::TimeUnit const requestedTimeUnit);
}

//class TorchMLModelImplementation;

class TorchMLModel
{
 public:
    double m, c, inflDist;
  TorchMLModel(KIM::ModelDriverCreate * const modelDriverCreate,
                  KIM::LengthUnit const requestedLengthUnit,
                  KIM::EnergyUnit const requestedEnergyUnit,
                  KIM::ChargeUnit const requestedChargeUnit,
                  KIM::TemperatureUnit const requestedTemperatureUnit,
                  KIM::TimeUnit const requestedTimeUnit,
                  int * const ier);
//  ~TorchMLModel();

  // no need to make these "extern" since KIM will only access them
  // via function pointers.  "static" is required so that there is not
  // an implicit this pointer added to the prototype by the C++ compiler
  static int Destroy(KIM::ModelDestroy * const modelDestroy);
  static int Refresh(KIM::ModelRefresh * const modelRefresh);
  static int
  Compute(KIM::ModelCompute const * const modelCompute,
          KIM::ModelComputeArguments const * const modelComputeArguments);
  static int ComputeArgumentsCreate(
      KIM::ModelCompute const * const modelCompute,
      KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate);
  static int ComputeArgumentsDestroy(
      KIM::ModelCompute const * const modelCompute,
      KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy);

// private:
//  TorchMLModelImplementation * implementation_;
};

#endif //TORCH_ML_MODEL_H
