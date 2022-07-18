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
int model_driver_create(KIM::ModelDriverCreate *const modelDriverCreate,
                        KIM::LengthUnit const requestedLengthUnit,
                        KIM::EnergyUnit const requestedEnergyUnit,
                        KIM::ChargeUnit const requestedChargeUnit,
                        KIM::TemperatureUnit const requestedTemperatureUnit,
                        KIM::TimeUnit const requestedTimeUnit);
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

    TorchMLModelDriver(KIM::ModelDriverCreate *const modelDriverCreate,
                       KIM::LengthUnit const requestedLengthUnit,
                       KIM::EnergyUnit const requestedEnergyUnit,
                       KIM::ChargeUnit const requestedChargeUnit,
                       KIM::TemperatureUnit const requestedTemperatureUnit,
                       KIM::TimeUnit const requestedTimeUnit,
                       int *const ier);
//  ~TorchMLModelDriver();

    // no need to make these "extern" since KIM will only access them
    // via function pointers.  "static" is required so that there is not
    // an implicit this pointer added to the prototype by the C++ compiler
    static int Destroy(KIM::ModelDestroy *const modelDestroy);

    static int Refresh(KIM::ModelRefresh *const modelRefresh);

    static int
    Compute(KIM::ModelCompute const *const modelCompute,
            KIM::ModelComputeArguments const *const modelComputeArguments);

    static int ComputeArgumentsCreate(
            KIM::ModelCompute const *const modelCompute,
            KIM::ModelComputeArgumentsCreate *const modelComputeArgumentsCreate);

    static int ComputeArgumentsDestroy(
            KIM::ModelCompute const *const modelCompute,
            KIM::ModelComputeArgumentsDestroy *const modelComputeArgumentsDestroy);

 private:
    MLModel *torchModel;
    std::vector<int> num_neighbors_;
    std::vector<int> neighbor_list;

//    void updateNeighborList();
//    void setInputs(TorchMLModelDriver &modelObject);
    void readParameters(KIM::ModelDriverCreate * const modelDriverCreate, int * const ier);
    void unitConversion(KIM::ModelDriverCreate *const modelDriverCreate,
                 KIM::LengthUnit const requestedLengthUnit,
                 KIM::EnergyUnit const requestedEnergyUnit,
                 KIM::ChargeUnit const requestedChargeUnit,
                 KIM::TemperatureUnit const requestedTemperatureUnit,
                 KIM::TimeUnit const requestedTimeUnit,
                 int *const ier);
    void setSpecies(KIM::ModelDriverCreate * const modelDriverCreate, int * const ier);
    void registerFunctionPointers(KIM::ModelDriverCreate * const modelDriverCreate, int * const ier);
//  TorchMLModelImplementation * implementation_;
};

#endif //TORCH_ML_MODEL_H
