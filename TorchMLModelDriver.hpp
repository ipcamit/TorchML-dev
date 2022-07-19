//
// Created by amit on 7/12/22.
// TODO PIMPL implementation
// Scheme: snake_case for implemented variables
// Methods : camelCase
//

#ifndef TORCH_ML_MODEL_H
#define TORCH_ML_MODEL_H

#include "KIM_ModelDriverHeaders.hpp"
#include "MLModel.hpp"

extern "C" {
int model_driver_create(KIM::ModelDriverCreate *modelDriverCreate,
                        KIM::LengthUnit requestedLengthUnit,
                        KIM::EnergyUnit requestedEnergyUnit,
                        KIM::ChargeUnit requestedChargeUnit,
                        KIM::TemperatureUnit requestedTemperatureUnit,
                        KIM::TimeUnit requestedTimeUnit);
}

//class TorchMLModelImplementation;

class TorchMLModelDriver {
public:
    double influence_distance;
    // TODO fic CamelCase
    int n_elements;
    std::vector<std::string> elements_list;
    std::string preprocessing;
    std::string model_name;

    TorchMLModelDriver(KIM::ModelDriverCreate *modelDriverCreate,
                       KIM::LengthUnit requestedLengthUnit,
                       KIM::EnergyUnit requestedEnergyUnit,
                       KIM::ChargeUnit requestedChargeUnit,
                       KIM::TemperatureUnit requestedTemperatureUnit,
                       KIM::TimeUnit requestedTimeUnit,
                       int *ier);
//  ~TorchMLModelDriver();

    // no need to make these "extern" since KIM will only access them
    // via function pointers.  "static" is required so that there is not
    // an implicit this pointer added to the prototype by the C++ compiler
    static int Destroy(KIM::ModelDestroy *modelDestroy);

    static int Refresh(KIM::ModelRefresh *modelRefresh);

    static int
    Compute(KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArguments const *modelComputeArguments);

    static int ComputeArgumentsCreate(
            KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArgumentsCreate *modelComputeArgumentsCreate);

    static int ComputeArgumentsDestroy(
            KIM::ModelCompute const *modelCompute,
            KIM::ModelComputeArgumentsDestroy *modelComputeArgumentsDestroy);

private:
    MLModel *torchModel;
    std::vector<int> num_neighbors_;
    std::vector<int> neighbor_list;

    void updateNeighborList(KIM::ModelComputeArguments const *modelComputeArguments, int numberOfParticles);

    void setInputs(KIM::ModelComputeArguments const *modelComputeArguments);

    void readParameters(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    static void unitConversion(KIM::ModelDriverCreate *modelDriverCreate,
                               KIM::LengthUnit requestedLengthUnit,
                               KIM::EnergyUnit requestedEnergyUnit,
                               KIM::ChargeUnit requestedChargeUnit,
                               KIM::TemperatureUnit requestedTemperatureUnit,
                               KIM::TimeUnit requestedTimeUnit,
                               int *ier);

    void setSpecies(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    static void registerFunctionPointers(KIM::ModelDriverCreate *modelDriverCreate, int *ier);
//  TorchMLModelImplementation * implementation_;
};

#endif //TORCH_ML_MODEL_H