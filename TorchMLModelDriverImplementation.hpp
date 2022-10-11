//
// Created by amit on 7/12/22.
//

#ifndef TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP
#define TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP

#include "KIM_ModelDriverHeaders.hpp"
#include "MLModel.hpp"
#include "Descriptors.hpp"
#include <torch/torch.h>

using namespace Descriptor;

class TorchMLModelDriverImplementation {
public:
    // All file params are public
    double influence_distance, cutoff_distance;
    int n_elements, n_layers;
    std::vector<std::string> elements_list;
    std::string preprocessing;
    std::string model_name;
    bool returns_forces;
    std::string descriptor_name = "None";
    std::string descriptor_param_file = "None";

    TorchMLModelDriverImplementation(KIM::ModelDriverCreate *modelDriverCreate,
                                     KIM::LengthUnit requestedLengthUnit,
                                     KIM::EnergyUnit requestedEnergyUnit,
                                     KIM::ChargeUnit requestedChargeUnit,
                                     KIM::TemperatureUnit requestedTemperatureUnit,
                                     KIM::TimeUnit requestedTimeUnit,
                                     int *ier);

    ~TorchMLModelDriverImplementation();

    static int Refresh(KIM::ModelRefresh *modelRefresh);

    int Compute(
            KIM::ModelComputeArguments const *modelComputeArguments);

    static int ComputeArgumentsCreate(
            KIM::ModelComputeArgumentsCreate *modelComputeArgumentsCreate);

    static int ComputeArgumentsDestroy(
            KIM::ModelComputeArgumentsDestroy *modelComputeArgumentsDestroy);

private:
    // Derived or assigned variables are private
    int modelWillNotRequestNeighborsOfNoncontributingParticles_;
    int n_contributing_atoms;
    int number_of_inputs;
    int *species_atomic_number;
    int64_t *contraction_array;

    MLModel *mlModel;

    AvailableDescriptor descriptor_kind;
    DescriptorKind *descriptor;

    std::vector<int> num_neighbors_;
    std::vector<int> neighbor_list;
    std::vector<int> z_map;

    double *descriptor_array;
    long **graph_edge_indices;

    void updateNeighborList(KIM::ModelComputeArguments const *modelComputeArguments, int numberOfParticles);

    void setDefaultInputs(const KIM::ModelComputeArguments *modelComputeArguments);

    void setDescriptorInputs(const KIM::ModelComputeArguments *modelComputeArguments);

    void setGraphInputs(const KIM::ModelComputeArguments *modelComputeArguments);

    void readParametersFile(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    static void unitConversion(KIM::ModelDriverCreate *modelDriverCreate,
                               KIM::LengthUnit requestedLengthUnit,
                               KIM::EnergyUnit requestedEnergyUnit,
                               KIM::ChargeUnit requestedChargeUnit,
                               KIM::TemperatureUnit requestedTemperatureUnit,
                               KIM::TimeUnit requestedTimeUnit,
                               int *ier);

    void setSpecies(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    static void registerFunctionPointers(KIM::ModelDriverCreate *modelDriverCreate, int *ier);

    void preprocessInputs(KIM::ModelComputeArguments const *modelComputeArguments);

    void postprocessOutputs(c10::IValue &, KIM::ModelComputeArguments const *);

    void Run(KIM::ModelComputeArguments const *modelComputeArguments);

    void contributingAtomCounts(KIM::ModelComputeArguments const *modelComputeArguments);

    void graphSetToGraphArray(std::vector<std::set<std::tuple<long, long> > > &);
    // TorchMLModelImplementation * implementation_;
};

int sym_to_z(std::string &);

#endif //TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP