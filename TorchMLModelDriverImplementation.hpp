//
// Created by amit on 7/12/22.
//

#ifndef TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP
#define TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP

#include "KIM_ModelDriverHeaders.hpp"
#include "MLModel.hpp"
#include <torch/torch.h>

#ifdef USE_LIBDESC
#include "Descriptors.hpp"

// TODO TEMP WORKAROUND TILL ENZYME FIXES BUG OR I SHIFT TO CLAD etc
// SEE DESTRUCTOR FOR MORE DETAILS
//#include "SymmetryFunctions.hpp"
//#include "Bispectrum.hpp"
// --------------------------------------------

using namespace Descriptor;
#endif

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
    int64_t *species_atomic_number;
    int64_t *contraction_array;

    MLModel *ml_model;

#ifdef USE_LIBDESC
    AvailableDescriptor descriptor_kind;
    DescriptorKind *descriptor;
#endif
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

// For hashing unordered_set of pairs
// https://arxiv.org/pdf/2105.10752.pdf
class SymmetricCantorPairing {
public:

  int64_t operator()(const std::array<long,2> &t) const {
    int64_t k1 = t[0];
    int64_t k2 = t[1];
    int64_t kmin = std::min(k1, k2);
    int64_t ksum = k1 + k2 + 1;

    return ((ksum * ksum - ksum%2) + kmin)/4;
  }
};

#endif //TORCH_ML_MODEL_DRIVER_IMPLEMENTATION_HPP