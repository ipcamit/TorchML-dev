#ifndef TORCH_ML_MODEL_DRIVER_HPP
#define TORCH_ML_MODEL_DRIVER_HPP

#include "KIM_ModelDriverHeaders.hpp"

extern "C" {
int model_driver_create(KIM::ModelDriverCreate *modelDriverCreate,
                        KIM::LengthUnit requestedLengthUnit,
                        KIM::EnergyUnit requestedEnergyUnit,
                        KIM::ChargeUnit requestedChargeUnit,
                        KIM::TemperatureUnit requestedTemperatureUnit,
                        KIM::TimeUnit requestedTimeUnit);
}

class TorchMLModelDriverImplementation;

class TorchMLModelDriver {
public:
    TorchMLModelDriver(KIM::ModelDriverCreate *modelDriverCreate,
                       KIM::LengthUnit requestedLengthUnit,
                       KIM::EnergyUnit requestedEnergyUnit,
                       KIM::ChargeUnit requestedChargeUnit,
                       KIM::TemperatureUnit requestedTemperatureUnit,
                       KIM::TimeUnit requestedTimeUnit,
                       int *ier);

    static int Destroy(KIM::ModelDestroy *modelDestroy);

    static int Refresh(KIM::ModelRefresh *modelRefresh);

    static int Compute(KIM::ModelCompute const *modelCompute,
                       KIM::ModelComputeArguments const *modelComputeArguments);

    static int ComputeArgumentsCreate(
            KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArgumentsCreate *modelComputeArgumentsCreate);

    static int ComputeArgumentsDestroy(
            KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArgumentsDestroy *modelComputeArgumentsDestroy);

    ~TorchMLModelDriver();

private:
    TorchMLModelDriverImplementation *implementation_;

};

#endif